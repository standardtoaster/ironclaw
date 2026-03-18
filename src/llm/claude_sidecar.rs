//! Claude Code CLI sidecar provider.
//!
//! Manages a warm `claude -p --output-format stream-json` subprocess and
//! translates IronClaw's LlmProvider calls to CLI stdin/stdout exchanges.
//! One instance per lens. Lazy spawn on first call, auto-restart on crash.
//!
//! ## Session-based conversations
//!
//! The sidecar maintains persistent sessions per thread. Instead of sending
//! full conversation history every time, only new messages are sent after the
//! initial bootstrap. This reduces token usage and latency for escalated
//! conversations.
//!
//! - On first call for a thread: full history is sent as the bootstrap prompt
//! - On subsequent calls for the same thread: only new (delta) messages are sent
//! - On de-escalation: session is cleaned up via `end_session()`
//! - If the process crashes: sessions are invalidated and re-bootstrapped

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::llm::costs;
use crate::llm::error::LlmError;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    Role, ToolCall, ToolCompletionRequest, ToolCompletionResponse,
};

/// Configuration for the Claude sidecar process.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Claude model alias ("sonnet", "opus", or full model ID).
    pub model: String,
    /// Optional text appended to the system prompt.
    pub system_prompt_append: Option<String>,
    /// Path to MCP config JSON for tool access (Phase 3).
    pub mcp_config_path: Option<String>,
    /// Working directory for the sidecar process.
    pub working_dir: Option<String>,
    /// Path to the claude binary (default: "claude").
    pub claude_binary: String,
    /// Timeout for spawning the process (seconds).
    pub spawn_timeout_secs: u64,
    /// Timeout for each request (seconds).
    pub request_timeout_secs: u64,
    /// Skip permission checks (`--dangerously-skip-permissions`).
    /// When false, the sidecar will proxy approval requests from Claude.
    /// Default: true (for backward compatibility with existing escalation flow).
    pub skip_permissions: bool,
}

/// A tool approval request from the Claude CLI process.
///
/// When `skip_permissions` is false, Claude may pause and request approval
/// for tool executions. This struct represents that request, which callers
/// can respond to via `ClaudeSidecarProvider::approve()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarApproval {
    /// Unique request ID for this approval.
    pub request_id: String,
    /// Name of the tool requesting approval.
    pub tool_name: String,
    /// Tool parameters.
    pub parameters: serde_json::Value,
}

/// Result of an exchange with the sidecar.
///
/// Either a completed response or a tool approval request that needs
/// user confirmation before the exchange can continue.
#[derive(Debug)]
pub enum ExchangeResult {
    /// The exchange completed with a response.
    Complete {
        content: String,
        tool_calls: Vec<ToolCall>,
        input_tokens: u32,
        output_tokens: u32,
    },
    /// The CLI is waiting for tool approval before continuing.
    NeedApproval(SidecarApproval),
}

/// Tracks the state of an active sidecar session for a specific thread.
#[derive(Debug, Clone)]
struct SidecarSession {
    /// The thread UUID that owns this session (used in Debug output).
    #[allow(dead_code)]
    thread_id: Uuid,
    /// The CLI session ID returned in the `system/init` message.
    cli_session_id: Option<String>,
    /// Number of messages sent so far (used to compute deltas).
    messages_sent: usize,
}

/// Internal state of the managed subprocess.
struct ManagedProcess {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout_reader: BufReader<tokio::process::ChildStdout>,
    /// Captured from the CLI's `system/init` message.
    session_id: Option<String>,
}

/// Messages received from the Claude CLI stdout (stream-json format).
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum SidecarMessage {
    #[serde(rename = "system")]
    System {
        #[serde(default)]
        session_id: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        message: serde_json::Value,
    },
    #[serde(rename = "result")]
    Result {
        result: String,
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default)]
        input_tokens: Option<u32>,
        #[serde(default)]
        output_tokens: Option<u32>,
    },
    /// Tool permission request from Claude (when not using --dangerously-skip-permissions).
    /// The CLI pauses execution and waits for approval on stdin.
    #[serde(rename = "tool_use_permission")]
    ToolUsePermission {
        /// Unique ID for this permission request.
        #[serde(default)]
        id: Option<String>,
        /// Name of the tool requesting permission.
        #[serde(default)]
        tool_name: Option<String>,
        /// Tool input parameters.
        #[serde(default)]
        input: Option<serde_json::Value>,
    },
}

/// An LLM provider that manages a warm Claude Code CLI subprocess.
///
/// Lazy-spawns `claude -p --output-format stream-json` on first call.
/// One instance per lens. Restarts on crash with exponential backoff.
///
/// Supports session-based conversations: the first call for a thread sends
/// full history, subsequent calls send only new messages. Pass `thread_id`
/// in the request's `metadata` map to enable session tracking.
pub struct ClaudeSidecarProvider {
    config: SidecarConfig,
    process: Arc<Mutex<Option<ManagedProcess>>>,
    model_label: String,
    restart_count: Arc<std::sync::atomic::AtomicU32>,
    /// Active sessions keyed by thread UUID.
    /// The current process can only serve one session at a time (single-session mode).
    /// If a different thread arrives, the old session is ended and the process is
    /// restarted for the new thread.
    sessions: Arc<Mutex<HashMap<Uuid, SidecarSession>>>,
}

impl ClaudeSidecarProvider {
    /// Create a new sidecar provider with the given config.
    ///
    /// The subprocess is not spawned until the first `complete()` call.
    pub fn new(config: SidecarConfig) -> Self {
        let model_label = format!("claude-sidecar/{}", config.model);
        Self {
            config,
            process: Arc::new(Mutex::new(None)),
            model_label,
            restart_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check if the sidecar process is currently held (non-blocking).
    pub fn is_alive(&self) -> bool {
        self.process
            .try_lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Shut down the sidecar process gracefully and clear all sessions.
    pub async fn shutdown(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut proc) = guard.take() {
            let _ = proc.child.kill().await;
            tracing::info!("Claude sidecar process shut down");
        }
        self.sessions.lock().await.clear();
    }

    /// End the session for a specific thread (e.g., on de-escalation).
    ///
    /// If this thread owns the active process session, the process is killed
    /// so it can be cleanly restarted for the next thread.
    pub async fn end_session(&self, thread_id: Uuid) {
        let mut sessions = self.sessions.lock().await;
        if sessions.remove(&thread_id).is_some() {
            tracing::info!(
                thread_id = %thread_id,
                "Sidecar session ended for thread"
            );
            // Kill the process since it held this thread's conversation state.
            // It will be re-spawned fresh for the next thread.
            let mut guard = self.process.lock().await;
            if let Some(mut proc) = guard.take() {
                let _ = proc.child.kill().await;
            }
        }
    }

    /// Check if a thread has an active session.
    pub async fn has_session(&self, thread_id: Uuid) -> bool {
        self.sessions.lock().await.contains_key(&thread_id)
    }

    /// Get the number of active sessions (for diagnostics).
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Ensure the managed process is running, spawning or re-spawning as needed.
    async fn ensure_process(&self) -> Result<(), LlmError> {
        let mut guard = self.process.lock().await;

        // Check if existing process is still alive.
        if let Some(ref mut proc) = *guard {
            match proc.child.try_wait() {
                Ok(Some(_exit)) => {
                    tracing::warn!("Claude sidecar exited, will re-spawn");
                    *guard = None;
                    // Invalidate all sessions since the process died.
                    self.sessions.lock().await.clear();
                }
                Ok(None) => return Ok(()), // still running
                Err(e) => {
                    tracing::warn!("Error checking sidecar status: {}", e);
                    *guard = None;
                    self.sessions.lock().await.clear();
                }
            }
        }

        // Backoff on restarts: 1s, 2s, 4s, max 10s.
        let count = self
            .restart_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if count > 0 {
            let delay = std::cmp::min(1u64 << count.min(4), 10);
            tracing::info!(
                restart_count = count,
                backoff_secs = delay,
                "Backing off before sidecar restart"
            );
            tokio::time::sleep(Duration::from_secs(delay)).await;
        }

        // Spawn new process.
        let proc = self.spawn_process().await?;
        *guard = Some(proc);
        self.restart_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Spawn the Claude CLI subprocess.
    async fn spawn_process(&self) -> Result<ManagedProcess, LlmError> {
        let mut cmd = Command::new(&self.config.claude_binary);
        cmd.arg("-p")
            .arg("--output-format")
            .arg("stream-json")
            .arg("--input-format")
            .arg("stream-json")
            .arg("--model")
            .arg(&self.config.model)
            .arg("--no-session-persistence");

        if self.config.skip_permissions {
            cmd.arg("--dangerously-skip-permissions");
        }

        if let Some(ref prompt) = self.config.system_prompt_append {
            cmd.arg("--append-system-prompt").arg(prompt);
        }
        if let Some(ref mcp_path) = self.config.mcp_config_path {
            cmd.arg("--mcp-config").arg(mcp_path);
        }
        if let Some(ref wd) = self.config.working_dir {
            cmd.current_dir(wd);
        }

        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: format!("Failed to spawn claude CLI: {}", e),
        })?;

        let stdin = child.stdin.take().ok_or_else(|| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: "Failed to capture stdin".to_string(),
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "claude_sidecar".to_string(),
                reason: "Failed to capture stdout".to_string(),
            })?;

        // Drain stderr to tracing in background.
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "claude_sidecar", "{}", line);
                }
            });
        }

        tracing::info!(
            model = %self.config.model,
            binary = %self.config.claude_binary,
            "Claude sidecar process spawned"
        );

        Ok(ManagedProcess {
            child,
            stdin,
            stdout_reader: BufReader::new(stdout),
            session_id: None,
        })
    }

    /// Perform a session-aware exchange with the sidecar.
    ///
    /// If `thread_id` is provided and the thread has an active session,
    /// only new messages (delta) are sent. Otherwise, full history is sent
    /// and a new session is created.
    ///
    /// Returns an `ExchangeResult` which may be a completion or an approval request.
    /// Used by the HTTP sidecar server for approval proxying.
    pub async fn session_exchange(
        &self,
        messages: &[ChatMessage],
        thread_id: Option<Uuid>,
    ) -> Result<ExchangeResult, LlmError> {
        let (prompt, is_continuation) = self
            .resolve_prompt(messages, thread_id)
            .await?;

        if is_continuation
            && let Some(tid) = thread_id
        {
            let prev_sent = {
                let sessions = self.sessions.lock().await;
                sessions.get(&tid).map(|s| s.messages_sent).unwrap_or(0)
            };
            let delta_count = messages.len().saturating_sub(prev_sent);
            tracing::debug!(
                thread_id = %tid,
                delta_messages = delta_count,
                "Sending delta messages to existing session"
            );
        }

        let result = self.exchange_or_approve(&prompt).await;

        // On success (completed exchange), update session tracking.
        if let Ok(ExchangeResult::Complete { .. }) = &result
            && let Some(tid) = thread_id
        {
            let mut sessions = self.sessions.lock().await;
            let cli_session_id = {
                let guard = self.process.lock().await;
                guard.as_ref().and_then(|p| p.session_id.clone())
            };

            sessions
                .entry(tid)
                .and_modify(|s| {
                    s.messages_sent = messages.len();
                    if s.cli_session_id.is_none() {
                        s.cli_session_id = cli_session_id.clone();
                    }
                })
                .or_insert_with(|| SidecarSession {
                    thread_id: tid,
                    cli_session_id,
                    messages_sent: messages.len(),
                });
        }

        result
    }

    /// Session-aware exchange that auto-approves permissions and returns a tuple.
    ///
    /// Used by the `LlmProvider` trait implementation.
    async fn session_exchange_auto_approve(
        &self,
        messages: &[ChatMessage],
        thread_id: Option<Uuid>,
    ) -> Result<(String, Vec<ToolCall>, u32, u32), LlmError> {
        let result = self.session_exchange(messages, thread_id).await?;
        match result {
            ExchangeResult::Complete {
                content,
                tool_calls,
                input_tokens,
                output_tokens,
            } => Ok((content, tool_calls, input_tokens, output_tokens)),
            ExchangeResult::NeedApproval(approval) => {
                // Auto-approve and continue.
                tracing::debug!(
                    tool = %approval.tool_name,
                    "Auto-approving tool (via LlmProvider interface)"
                );
                self.approve(&approval.request_id, true).await?;
                // Read the continuation.
                match self.continue_after_approval().await? {
                    ExchangeResult::Complete {
                        content,
                        tool_calls,
                        input_tokens,
                        output_tokens,
                    } => Ok((content, tool_calls, input_tokens, output_tokens)),
                    ExchangeResult::NeedApproval(_) => {
                        // Nested approval - for now, auto-approve via exchange()
                        // which has the loop.
                        Err(LlmError::RequestFailed {
                            provider: "claude_sidecar".to_string(),
                            reason: "Nested approval requests not supported in auto-approve mode"
                                .to_string(),
                        })
                    }
                }
            }
        }
    }

    /// Determine the prompt to send: full history (bootstrap) or delta (continuation).
    ///
    /// Returns `(prompt, is_continuation)`.
    async fn resolve_prompt(
        &self,
        messages: &[ChatMessage],
        thread_id: Option<Uuid>,
    ) -> Result<(String, bool), LlmError> {
        let Some(tid) = thread_id else {
            // No thread tracking: stateless mode, send full history.
            return Ok((messages_to_prompt(messages), false));
        };

        let sessions = self.sessions.lock().await;
        let Some(session) = sessions.get(&tid) else {
            // New session: send full history as bootstrap.
            drop(sessions);
            tracing::info!(
                thread_id = %tid,
                total_messages = messages.len(),
                "Bootstrapping new sidecar session with full history"
            );
            return Ok((messages_to_prompt(messages), false));
        };

        // Existing session: extract only the new messages.
        let sent = session.messages_sent;
        drop(sessions);

        if sent >= messages.len() {
            // No new messages -- this shouldn't normally happen, but handle gracefully.
            tracing::warn!(
                thread_id = %tid,
                sent,
                total = messages.len(),
                "No new messages to send (already sent all)"
            );
            return Ok((messages_to_prompt(messages), false));
        }

        let delta = &messages[sent..];
        tracing::debug!(
            thread_id = %tid,
            sent,
            new_messages = delta.len(),
            "Sending delta messages to existing session"
        );
        Ok((messages_to_prompt(delta), true))
    }

    /// Send a prompt and return either a completed response or an approval request.
    ///
    /// This is the core exchange method used by the HTTP server for proxied approvals.
    /// When an approval request is returned, the caller must call `approve()` and
    /// then call `continue_after_approval()` to get the final result.
    pub async fn exchange_or_approve(
        &self,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        self.ensure_process().await?;

        let timeout = Duration::from_secs(self.config.request_timeout_secs);
        let result = tokio::time::timeout(timeout, self.exchange_inner(prompt)).await;

        match result {
            Ok(inner) => inner,
            Err(_) => {
                // Timeout: kill the process so it restarts on next call.
                let mut guard = self.process.lock().await;
                if let Some(mut proc) = guard.take() {
                    let _ = proc.child.kill().await;
                }
                // Invalidate sessions since the process is dead.
                self.sessions.lock().await.clear();
                Err(LlmError::RequestFailed {
                    provider: "claude_sidecar".to_string(),
                    reason: format!(
                        "Request timed out after {}s",
                        self.config.request_timeout_secs
                    ),
                })
            }
        }
    }

    /// Send an approval decision for a pending tool permission request.
    pub async fn approve(&self, request_id: &str, approved: bool) -> Result<(), LlmError> {
        let mut guard = self.process.lock().await;
        let proc = guard.as_mut().ok_or_else(|| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: "Sidecar process not running".to_string(),
        })?;

        let response = serde_json::json!({
            "type": "tool_use_permission_response",
            "id": request_id,
            "approved": approved,
        });
        let mut line =
            serde_json::to_string(&response).map_err(|e| LlmError::RequestFailed {
                provider: "claude_sidecar".to_string(),
                reason: format!("Failed to serialize approval: {}", e),
            })?;
        line.push('\n');

        proc.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_sidecar".to_string(),
                reason: format!("Failed to write approval to sidecar stdin: {}", e),
            })?;
        let _ = proc.stdin.flush().await;

        tracing::debug!(
            request_id,
            approved,
            "Sent tool approval to sidecar"
        );
        Ok(())
    }

    /// Continue reading from the sidecar after an approval was sent.
    ///
    /// Call this after `approve()` to get the next result or approval request.
    pub async fn continue_after_approval(&self) -> Result<ExchangeResult, LlmError> {
        let timeout = Duration::from_secs(self.config.request_timeout_secs);
        let result = tokio::time::timeout(timeout, self.read_until_result_or_approval()).await;

        match result {
            Ok(inner) => inner,
            Err(_) => {
                let mut guard = self.process.lock().await;
                if let Some(mut proc) = guard.take() {
                    let _ = proc.child.kill().await;
                }
                self.sessions.lock().await.clear();
                Err(LlmError::RequestFailed {
                    provider: "claude_sidecar".to_string(),
                    reason: format!(
                        "Request timed out after {}s",
                        self.config.request_timeout_secs
                    ),
                })
            }
        }
    }

    /// Read stdout until we get either a result or an approval request.
    async fn read_until_result_or_approval(&self) -> Result<ExchangeResult, LlmError> {
        let mut guard = self.process.lock().await;
        let proc = guard.as_mut().ok_or_else(|| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: "Sidecar process not running".to_string(),
        })?;

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        let mut buf = String::new();
        loop {
            buf.clear();
            let bytes_read = match proc.stdout_reader.read_line(&mut buf).await {
                Ok(n) => n,
                Err(e) => {
                    drop(guard);
                    let mut g = self.process.lock().await;
                    *g = None;
                    self.sessions.lock().await.clear();
                    return Err(LlmError::RequestFailed {
                        provider: "claude_sidecar".to_string(),
                        reason: format!("Failed to read sidecar stdout: {}", e),
                    });
                }
            };

            if bytes_read == 0 {
                drop(guard);
                let mut g = self.process.lock().await;
                *g = None;
                self.sessions.lock().await.clear();
                return Err(LlmError::RequestFailed {
                    provider: "claude_sidecar".to_string(),
                    reason: "Sidecar process exited unexpectedly".to_string(),
                });
            }

            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }

            let msg: SidecarMessage = match serde_json::from_str(trimmed) {
                Ok(m) => m,
                Err(e) => {
                    tracing::debug!("Ignoring unparseable sidecar output: {} ({})", trimmed, e);
                    continue;
                }
            };

            match msg {
                SidecarMessage::System { session_id } => {
                    if let Some(sid) = session_id {
                        proc.session_id = Some(sid);
                    }
                }
                SidecarMessage::Assistant { message } => {
                    if let Some(content) = message.get("content") {
                        let text = extract_text(content);
                        if !text.is_empty() {
                            text_content.push_str(&text);
                        }
                        let calls = extract_tool_calls(content);
                        tool_calls.extend(calls);
                    }
                }
                SidecarMessage::Result {
                    result,
                    input_tokens: it,
                    output_tokens: ot,
                    ..
                } => {
                    if text_content.is_empty() {
                        text_content = result;
                    }
                    self.restart_count
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                    return Ok(ExchangeResult::Complete {
                        content: text_content,
                        tool_calls,
                        input_tokens: it.unwrap_or(0),
                        output_tokens: ot.unwrap_or(0),
                    });
                }
                SidecarMessage::ToolUsePermission {
                    id,
                    tool_name,
                    input,
                } => {
                    let request_id = id.unwrap_or_else(|| Uuid::new_v4().to_string());
                    let tool = tool_name.unwrap_or_else(|| "unknown".to_string());
                    let params = input.unwrap_or(serde_json::Value::Null);
                    tracing::info!(
                        tool = %tool,
                        request_id = %request_id,
                        "Tool permission requested by sidecar"
                    );
                    return Ok(ExchangeResult::NeedApproval(SidecarApproval {
                        request_id,
                        tool_name: tool,
                        parameters: params,
                    }));
                }
            }
        }
    }

    /// Inner exchange: write to stdin, read from stdout until result or approval.
    async fn exchange_inner(
        &self,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        let mut guard = self.process.lock().await;
        let proc = guard.as_mut().ok_or_else(|| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: "Sidecar process not running".to_string(),
        })?;

        // Write the user message as a JSON line to stdin.
        let input = serde_json::json!({
            "type": "user",
            "content": prompt,
        });
        let mut line = serde_json::to_string(&input).map_err(|e| LlmError::RequestFailed {
            provider: "claude_sidecar".to_string(),
            reason: format!("Failed to serialize input: {}", e),
        })?;
        line.push('\n');

        if let Err(e) = proc.stdin.write_all(line.as_bytes()).await {
            // Write failed: process likely crashed. Clear it for re-spawn.
            drop(guard);
            let mut g = self.process.lock().await;
            *g = None;
            self.sessions.lock().await.clear();
            return Err(LlmError::RequestFailed {
                provider: "claude_sidecar".to_string(),
                reason: format!("Failed to write to sidecar stdin: {}", e),
            });
        }
        let _ = proc.stdin.flush().await;

        // Drop guard temporarily so read_until_result_or_approval can acquire it.
        drop(guard);

        self.read_until_result_or_approval().await
    }
}

/// Extract the thread_id from request metadata, if present.
fn thread_id_from_metadata(metadata: &HashMap<String, String>) -> Option<Uuid> {
    metadata
        .get("thread_id")
        .and_then(|s| Uuid::parse_str(s).ok())
}

/// Build the text prompt from IronClaw messages.
///
/// The Claude CLI in print mode takes a single prompt string. We concatenate
/// messages with role prefixes to preserve conversation context.
fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                parts.push(format!("[System context]: {}", msg.content));
            }
            Role::User => parts.push(msg.content.clone()),
            Role::Assistant => parts.push(format!("[Previous assistant]: {}", msg.content)),
            Role::Tool => {
                let name = msg.name.as_deref().unwrap_or("tool");
                parts.push(format!("[Tool result from {}]: {}", name, msg.content));
            }
        }
    }
    parts.join("\n\n")
}

/// Extract text content from a Claude assistant message content value.
fn extract_text(content: &serde_json::Value) -> String {
    let Some(array) = content.as_array() else {
        return content.as_str().unwrap_or("").to_string();
    };
    array
        .iter()
        .filter_map(|block| {
            if block.get("type")?.as_str()? == "text" {
                block.get("text")?.as_str().map(String::from)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Extract tool calls from a Claude assistant message content array.
fn extract_tool_calls(content: &serde_json::Value) -> Vec<ToolCall> {
    let Some(array) = content.as_array() else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|block| {
            if block.get("type")?.as_str()? == "tool_use" {
                Some(ToolCall {
                    id: block.get("id")?.as_str()?.to_string(),
                    name: block.get("name")?.as_str()?.to_string(),
                    arguments: block.get("input")?.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

#[async_trait]
impl LlmProvider for ClaudeSidecarProvider {
    fn model_name(&self) -> &str {
        &self.model_label
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        // Map our model alias to a known model ID for cost lookup.
        let model_id = match self.config.model.as_str() {
            "sonnet" => "claude-sonnet-4-6",
            "opus" => "claude-opus-4-6",
            "haiku" => "claude-haiku-4-5",
            other => other,
        };
        costs::model_cost(model_id).unwrap_or_else(costs::default_cost)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let thread_id = thread_id_from_metadata(&request.metadata);
        let (content, _tool_calls, input_tokens, output_tokens) =
            self.session_exchange_auto_approve(&request.messages, thread_id)
                .await?;

        Ok(CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            finish_reason: FinishReason::Stop,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        })
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let thread_id = thread_id_from_metadata(&request.metadata);
        let (content, tool_calls, input_tokens, output_tokens) =
            self.session_exchange_auto_approve(&request.messages, thread_id)
                .await?;

        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolUse
        } else {
            FinishReason::Stop
        };

        Ok(ToolCompletionResponse {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        })
    }

    async fn model_metadata(&self) -> Result<ModelMetadata, LlmError> {
        Ok(ModelMetadata {
            id: self.model_label.clone(),
            context_length: Some(200_000),
        })
    }

    async fn end_session(&self, thread_id: Uuid) {
        self.end_session(thread_id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    fn test_config() -> SidecarConfig {
        SidecarConfig {
            model: "sonnet".to_string(),
            system_prompt_append: None,
            mcp_config_path: None,
            working_dir: None,
            claude_binary: "claude".to_string(),
            spawn_timeout_secs: 30,
            request_timeout_secs: 300,
            skip_permissions: true,
        }
    }

    // --- Unit tests: provider construction ---

    #[test]
    fn test_provider_model_name() {
        let provider = ClaudeSidecarProvider::new(test_config());
        assert_eq!(provider.model_name(), "claude-sidecar/sonnet");
    }

    #[test]
    fn test_provider_model_name_opus() {
        let mut config = test_config();
        config.model = "opus".to_string();
        let provider = ClaudeSidecarProvider::new(config);
        assert_eq!(provider.model_name(), "claude-sidecar/opus");
    }

    #[test]
    fn test_provider_cost_per_token_sonnet() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let (input, output) = provider.cost_per_token();
        assert!(input > Decimal::ZERO, "Sonnet input cost should be > 0");
        assert!(output > Decimal::ZERO, "Sonnet output cost should be > 0");
    }

    #[test]
    fn test_provider_cost_per_token_opus() {
        let mut config = test_config();
        config.model = "opus".to_string();
        let provider = ClaudeSidecarProvider::new(config);
        let (input, output) = provider.cost_per_token();
        assert!(input > Decimal::ZERO);
        assert!(output > Decimal::ZERO);
        // Opus should be more expensive than sonnet.
        let sonnet_provider = ClaudeSidecarProvider::new(test_config());
        let (sonnet_in, _) = sonnet_provider.cost_per_token();
        assert!(input > sonnet_in, "Opus should cost more than Sonnet");
    }

    #[test]
    fn test_provider_starts_without_process() {
        let provider = ClaudeSidecarProvider::new(test_config());
        assert!(!provider.is_alive());
    }

    // --- Unit tests: protocol parsing ---

    #[test]
    fn test_parse_result_message() {
        let line = r#"{"type":"result","result":"The answer is 4","session_id":"abc","cost_usd":0.003,"duration_ms":1200,"input_tokens":50,"output_tokens":10}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse result");
        match msg {
            SidecarMessage::Result {
                result,
                input_tokens,
                output_tokens,
                ..
            } => {
                assert_eq!(result, "The answer is 4");
                assert_eq!(input_tokens.unwrap_or(0), 50);
                assert_eq!(output_tokens.unwrap_or(0), 10);
            }
            other => panic!("Expected Result, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_system_init_message() {
        let line = r#"{"type":"system","subtype":"init","session_id":"sess-123"}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse system");
        match msg {
            SidecarMessage::System { session_id } => {
                assert_eq!(session_id, Some("sess-123".to_string()));
            }
            other => panic!("Expected System, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_assistant_text_message() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]}}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse assistant");
        match msg {
            SidecarMessage::Assistant { message } => {
                let content = message.get("content").expect("content field");
                let text = extract_text(content);
                assert_eq!(text, "Hello world");
            }
            other => panic!("Expected Assistant, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_assistant_message_with_tool_use() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"memory_search","input":{"query":"test"}}]}}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse tool_use");
        match msg {
            SidecarMessage::Assistant { message } => {
                let content = message.get("content").expect("content field");
                let calls = extract_tool_calls(content);
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "tu_1");
                assert_eq!(calls[0].name, "memory_search");
                assert_eq!(calls[0].arguments, serde_json::json!({"query": "test"}));
            }
            other => panic!("Expected Assistant, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_assistant_mixed_content() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Let me search. "},{"type":"tool_use","id":"tu_2","name":"search","input":{"q":"x"}}]}}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse mixed");
        match msg {
            SidecarMessage::Assistant { message } => {
                let content = message.get("content").expect("content field");
                let text = extract_text(content);
                assert_eq!(text, "Let me search. ");
                let calls = extract_tool_calls(content);
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
            }
            other => panic!("Expected Assistant, got: {:?}", other),
        }
    }

    #[test]
    fn test_extract_text_plain_string() {
        let val = serde_json::json!("plain text");
        assert_eq!(extract_text(&val), "plain text");
    }

    #[test]
    fn test_extract_text_empty_array() {
        let val = serde_json::json!([]);
        assert_eq!(extract_text(&val), "");
    }

    #[test]
    fn test_extract_tool_calls_no_tools() {
        let val = serde_json::json!([{"type": "text", "text": "hello"}]);
        let calls = extract_tool_calls(&val);
        assert!(calls.is_empty());
    }

    // --- Unit tests: approval parsing ---

    #[test]
    fn test_parse_tool_use_permission_message() {
        let line = r#"{"type":"tool_use_permission","id":"perm-1","tool_name":"shell","input":{"command":"ls"}}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse permission");
        match msg {
            SidecarMessage::ToolUsePermission {
                id,
                tool_name,
                input,
            } => {
                assert_eq!(id, Some("perm-1".to_string()));
                assert_eq!(tool_name, Some("shell".to_string()));
                assert_eq!(
                    input,
                    Some(serde_json::json!({"command": "ls"}))
                );
            }
            other => panic!("Expected ToolUsePermission, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_tool_use_permission_minimal() {
        let line = r#"{"type":"tool_use_permission"}"#;
        let msg: SidecarMessage = serde_json::from_str(line).expect("parse minimal permission");
        match msg {
            SidecarMessage::ToolUsePermission {
                id,
                tool_name,
                input,
            } => {
                assert!(id.is_none());
                assert!(tool_name.is_none());
                assert!(input.is_none());
            }
            other => panic!("Expected ToolUsePermission, got: {:?}", other),
        }
    }

    #[test]
    fn test_exchange_result_approval_construction() {
        let approval = SidecarApproval {
            request_id: "req-1".to_string(),
            tool_name: "shell".to_string(),
            parameters: serde_json::json!({"command": "ls"}),
        };
        let result = ExchangeResult::NeedApproval(approval);
        match result {
            ExchangeResult::NeedApproval(a) => {
                assert_eq!(a.request_id, "req-1");
                assert_eq!(a.tool_name, "shell");
            }
            _ => panic!("Expected NeedApproval"),
        }
    }

    #[test]
    fn test_skip_permissions_config_default() {
        let config = test_config();
        assert!(config.skip_permissions, "Default test config should skip permissions");
    }

    // --- Unit tests: message translation ---

    #[test]
    fn test_messages_to_prompt_user_only() {
        let messages = vec![ChatMessage::user("Hello")];
        let prompt = messages_to_prompt(&messages);
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn test_messages_to_prompt_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[System context]: You are helpful"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_messages_to_prompt_with_assistant() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("Thanks"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[Previous assistant]: Hi there"));
        assert!(prompt.contains("Thanks"));
    }

    #[test]
    fn test_messages_to_prompt_with_tool_result() {
        let messages = vec![
            ChatMessage::user("Search for X"),
            ChatMessage::tool_result("call_1", "search", "found 3 results"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[Tool result from search]: found 3 results"));
    }

    // --- Unit tests: session tracking ---

    #[tokio::test]
    async fn test_session_starts_empty() {
        let provider = ClaudeSidecarProvider::new(test_config());
        assert_eq!(provider.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_has_session_false_initially() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();
        assert!(!provider.has_session(thread_id).await);
    }

    #[tokio::test]
    async fn test_end_session_noop_when_no_session() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();
        // Should not panic or error.
        provider.end_session(thread_id).await;
        assert_eq!(provider.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_end_session_removes_tracked_session() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();

        // Manually insert a session for testing.
        {
            let mut sessions = provider.sessions.lock().await;
            sessions.insert(
                thread_id,
                SidecarSession {
                    thread_id,
                    cli_session_id: Some("sess-abc".to_string()),
                    messages_sent: 5,
                },
            );
        }
        assert!(provider.has_session(thread_id).await);
        assert_eq!(provider.session_count().await, 1);

        provider.end_session(thread_id).await;
        assert!(!provider.has_session(thread_id).await);
        assert_eq!(provider.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_shutdown_clears_all_sessions() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();

        {
            let mut sessions = provider.sessions.lock().await;
            sessions.insert(
                t1,
                SidecarSession {
                    thread_id: t1,
                    cli_session_id: None,
                    messages_sent: 3,
                },
            );
            sessions.insert(
                t2,
                SidecarSession {
                    thread_id: t2,
                    cli_session_id: None,
                    messages_sent: 1,
                },
            );
        }
        assert_eq!(provider.session_count().await, 2);

        provider.shutdown().await;
        assert_eq!(provider.session_count().await, 0);
    }

    // --- Unit tests: resolve_prompt (session-aware prompt building) ---

    #[tokio::test]
    async fn test_resolve_prompt_no_thread_id_sends_full() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
            ChatMessage::user("Follow-up"),
        ];

        let (prompt, is_continuation) = provider.resolve_prompt(&messages, None).await.unwrap();
        assert!(!is_continuation);
        assert!(prompt.contains("[System context]: sys"));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("Follow-up"));
    }

    #[tokio::test]
    async fn test_resolve_prompt_new_session_sends_full() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::user("World"),
        ];

        let (prompt, is_continuation) = provider
            .resolve_prompt(&messages, Some(thread_id))
            .await
            .unwrap();
        assert!(!is_continuation);
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("World"));
    }

    #[tokio::test]
    async fn test_resolve_prompt_existing_session_sends_delta() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();

        // Simulate an existing session that already sent 2 messages.
        {
            let mut sessions = provider.sessions.lock().await;
            sessions.insert(
                thread_id,
                SidecarSession {
                    thread_id,
                    cli_session_id: Some("sess-1".to_string()),
                    messages_sent: 2,
                },
            );
        }

        let messages = vec![
            ChatMessage::user("Hello"),       // already sent (index 0)
            ChatMessage::assistant("Hi"),      // already sent (index 1)
            ChatMessage::user("New question"), // delta (index 2)
        ];

        let (prompt, is_continuation) = provider
            .resolve_prompt(&messages, Some(thread_id))
            .await
            .unwrap();
        assert!(is_continuation);
        // Delta should only contain the new message.
        assert!(prompt.contains("New question"));
        assert!(!prompt.contains("Hello"), "Should not contain old messages");
        assert!(!prompt.contains("[Previous assistant]: Hi"), "Should not contain old assistant messages");
    }

    #[tokio::test]
    async fn test_resolve_prompt_no_new_messages_falls_back_to_full() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();

        {
            let mut sessions = provider.sessions.lock().await;
            sessions.insert(
                thread_id,
                SidecarSession {
                    thread_id,
                    cli_session_id: Some("sess-1".to_string()),
                    messages_sent: 3,
                },
            );
        }

        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
            ChatMessage::user("Done"),
        ];

        // All 3 messages already sent. Should fall back to full.
        let (prompt, is_continuation) = provider
            .resolve_prompt(&messages, Some(thread_id))
            .await
            .unwrap();
        assert!(!is_continuation);
        assert!(prompt.contains("Hello"));
    }

    // --- Unit tests: thread_id extraction from metadata ---

    #[test]
    fn test_thread_id_from_metadata_valid() {
        let id = Uuid::new_v4();
        let mut meta = HashMap::new();
        meta.insert("thread_id".to_string(), id.to_string());
        assert_eq!(thread_id_from_metadata(&meta), Some(id));
    }

    #[test]
    fn test_thread_id_from_metadata_missing() {
        let meta = HashMap::new();
        assert_eq!(thread_id_from_metadata(&meta), None);
    }

    #[test]
    fn test_thread_id_from_metadata_invalid_uuid() {
        let mut meta = HashMap::new();
        meta.insert("thread_id".to_string(), "not-a-uuid".to_string());
        assert_eq!(thread_id_from_metadata(&meta), None);
    }

    // --- Async tests: spawn failures ---

    #[tokio::test]
    async fn test_spawn_nonexistent_binary_fails() {
        let config = SidecarConfig {
            model: "sonnet".to_string(),
            system_prompt_append: None,
            mcp_config_path: None,
            working_dir: None,
            claude_binary: "this-binary-does-not-exist-ironclaw-test".to_string(),
            spawn_timeout_secs: 5,
            request_timeout_secs: 5,
            skip_permissions: true,
        };
        let provider = ClaudeSidecarProvider::new(config);
        let request = CompletionRequest::new(vec![ChatMessage::user("test")]);
        let result = provider.complete(request).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LlmError::RequestFailed { reason, .. } => {
                assert!(
                    reason.contains("spawn"),
                    "Error should mention spawn: {}",
                    reason
                );
            }
            other => panic!("Expected RequestFailed, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_eof_clears_for_respawn() {
        // Use `echo` which exits immediately after writing to stdout.
        let config = SidecarConfig {
            model: "sonnet".to_string(),
            system_prompt_append: None,
            mcp_config_path: None,
            working_dir: None,
            claude_binary: "echo".to_string(),
            spawn_timeout_secs: 5,
            request_timeout_secs: 5,
            skip_permissions: true,
        };
        let provider = ClaudeSidecarProvider::new(config);
        let request = CompletionRequest::new(vec![ChatMessage::user("test")]);
        let result = provider.complete(request).await;
        // echo doesn't speak JSON protocol, so we should get an error.
        assert!(result.is_err());
        // The process should be cleared (None) so next call re-spawns.
        // Give a moment for the cleanup.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!provider.is_alive());
    }

    #[tokio::test]
    async fn test_process_eof_clears_sessions() {
        let config = SidecarConfig {
            model: "sonnet".to_string(),
            system_prompt_append: None,
            mcp_config_path: None,
            working_dir: None,
            claude_binary: "echo".to_string(),
            spawn_timeout_secs: 5,
            request_timeout_secs: 5,
            skip_permissions: true,
        };
        let provider = ClaudeSidecarProvider::new(config);
        let thread_id = Uuid::new_v4();

        // Pre-populate a session.
        {
            let mut sessions = provider.sessions.lock().await;
            sessions.insert(
                thread_id,
                SidecarSession {
                    thread_id,
                    cli_session_id: Some("sess-old".to_string()),
                    messages_sent: 3,
                },
            );
        }
        assert_eq!(provider.session_count().await, 1);

        let request = CompletionRequest::new(vec![ChatMessage::user("test")]);
        let _ = provider.complete(request).await;

        // Sessions should be cleared after process crash.
        assert_eq!(provider.session_count().await, 0);
    }

    #[test]
    fn test_health_check_no_process() {
        let provider = ClaudeSidecarProvider::new(test_config());
        assert!(!provider.is_alive());
    }

    // --- Async tests: model metadata ---

    #[tokio::test]
    async fn test_model_metadata() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let meta = provider.model_metadata().await.expect("metadata");
        assert_eq!(meta.id, "claude-sidecar/sonnet");
        assert_eq!(meta.context_length, Some(200_000));
    }

    // --- Integration tests (require claude binary) ---

    /// Integration test: full completion with real Claude CLI.
    /// Requires `claude` binary installed. Marked #[ignore].
    #[tokio::test]
    #[ignore]
    async fn test_complete_with_real_claude() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let request = CompletionRequest::new(vec![ChatMessage::user(
            "What is 2 + 2? Reply with just the number.",
        )]);
        let response = provider.complete(request).await.expect("completion");
        assert!(
            response.content.contains('4'),
            "Response: {}",
            response.content
        );
        assert!(response.output_tokens > 0);
    }

    /// Integration test: complete_with_tools with real Claude CLI.
    /// Requires `claude` binary installed. Marked #[ignore].
    #[tokio::test]
    #[ignore]
    async fn test_complete_with_tools_real_claude() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let request = ToolCompletionRequest::new(
            vec![ChatMessage::user("What is the capital of France?")],
            vec![],
        );
        let response = provider
            .complete_with_tools(request)
            .await
            .expect("completion");
        assert!(
            response.content.is_some(),
            "Should have text content for a simple question"
        );
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    /// Integration test: session-based conversation with real Claude CLI.
    /// Sends two messages to the same thread, verifying the second uses delta.
    /// Requires `claude` binary installed. Marked #[ignore].
    #[tokio::test]
    #[ignore]
    async fn test_session_continuation_real_claude() {
        let provider = ClaudeSidecarProvider::new(test_config());
        let thread_id = Uuid::new_v4();

        // First message: bootstraps the session with full history.
        let mut request1 = CompletionRequest::new(vec![
            ChatMessage::system("You are a math tutor. Be concise."),
            ChatMessage::user("Remember this number: 42"),
        ]);
        request1
            .metadata
            .insert("thread_id".to_string(), thread_id.to_string());
        let response1 = provider.complete(request1).await.expect("first completion");
        assert!(!response1.content.is_empty());
        assert!(provider.has_session(thread_id).await);

        // Second message: should send only the delta (new user message).
        let mut request2 = CompletionRequest::new(vec![
            ChatMessage::system("You are a math tutor. Be concise."),
            ChatMessage::user("Remember this number: 42"),
            ChatMessage::assistant(&response1.content),
            ChatMessage::user("What number did I ask you to remember?"),
        ]);
        request2
            .metadata
            .insert("thread_id".to_string(), thread_id.to_string());
        let response2 = provider.complete(request2).await.expect("second completion");
        assert!(
            response2.content.contains("42"),
            "Should remember 42, got: {}",
            response2.content
        );
        // Second call should use fewer input tokens since only delta was sent.
        assert!(
            response2.input_tokens < response1.input_tokens + 1000,
            "Delta should use fewer tokens: first={}, second={}",
            response1.input_tokens,
            response2.input_tokens
        );
    }

    // --- Backend enum parsing ---

    #[test]
    fn test_claude_sidecar_backend_parsing() {
        use crate::config::LlmBackend;

        let backend: LlmBackend = "claude_sidecar".parse().expect("parse claude_sidecar");
        assert_eq!(backend, LlmBackend::ClaudeSidecar);

        let backend: LlmBackend = "claude-sidecar".parse().expect("parse claude-sidecar");
        assert_eq!(backend, LlmBackend::ClaudeSidecar);

        let backend: LlmBackend = "sidecar".parse().expect("parse sidecar");
        assert_eq!(backend, LlmBackend::ClaudeSidecar);

        assert_eq!(format!("{}", LlmBackend::ClaudeSidecar), "claude_sidecar");
    }

    #[test]
    fn test_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ClaudeSidecarProvider>();
    }
}
