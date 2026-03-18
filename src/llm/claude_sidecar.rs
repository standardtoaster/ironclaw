//! Claude Code CLI sidecar provider.
//!
//! Manages a warm `claude -p --output-format stream-json` subprocess and
//! translates IronClaw's LlmProvider calls to CLI stdin/stdout exchanges.
//! One instance per lens. Lazy spawn on first call, auto-restart on crash.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

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
}

/// Internal state of the managed subprocess.
struct ManagedProcess {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout_reader: BufReader<tokio::process::ChildStdout>,
    /// Captured from the CLI's `system/init` message. Will be used for
    /// `--continue --session-id` in a follow-up (session continuity).
    #[allow(dead_code)]
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
}

/// An LLM provider that manages a warm Claude Code CLI subprocess.
///
/// Lazy-spawns `claude -p --output-format stream-json` on first call.
/// One instance per lens. Restarts on crash with exponential backoff.
pub struct ClaudeSidecarProvider {
    config: SidecarConfig,
    process: Arc<Mutex<Option<ManagedProcess>>>,
    model_label: String,
    restart_count: Arc<std::sync::atomic::AtomicU32>,
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
        }
    }

    /// Check if the sidecar process is currently held (non-blocking).
    pub fn is_alive(&self) -> bool {
        self.process
            .try_lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Shut down the sidecar process gracefully.
    pub async fn shutdown(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut proc) = guard.take() {
            let _ = proc.child.kill().await;
            tracing::info!("Claude sidecar process shut down");
        }
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
                }
                Ok(None) => return Ok(()), // still running
                Err(e) => {
                    tracing::warn!("Error checking sidecar status: {}", e);
                    *guard = None;
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
            .arg("--dangerously-skip-permissions")
            .arg("--no-session-persistence");

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

    /// Send a prompt to the sidecar and collect the response.
    async fn exchange(
        &self,
        prompt: &str,
    ) -> Result<(String, Vec<ToolCall>, u32, u32), LlmError> {
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

    /// Inner exchange: write to stdin, read from stdout until result message.
    async fn exchange_inner(
        &self,
        prompt: &str,
    ) -> Result<(String, Vec<ToolCall>, u32, u32), LlmError> {
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
            return Err(LlmError::RequestFailed {
                provider: "claude_sidecar".to_string(),
                reason: format!("Failed to write to sidecar stdin: {}", e),
            });
        }
        let _ = proc.stdin.flush().await;

        // Read stdout lines until we get a "result" message.
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();
        #[allow(unused_assignments)]
        let mut input_tokens = 0u32;
        #[allow(unused_assignments)]
        let mut output_tokens = 0u32;

        let mut buf = String::new();
        loop {
            buf.clear();
            let bytes_read = match proc.stdout_reader.read_line(&mut buf).await {
                Ok(n) => n,
                Err(e) => {
                    // Read failed: process likely crashed. Clear for re-spawn.
                    drop(guard);
                    let mut g = self.process.lock().await;
                    *g = None;
                    return Err(LlmError::RequestFailed {
                        provider: "claude_sidecar".to_string(),
                        reason: format!("Failed to read sidecar stdout: {}", e),
                    });
                }
            };

            if bytes_read == 0 {
                // EOF: process exited. Clear for re-spawn.
                drop(guard);
                let mut g = self.process.lock().await;
                *g = None;
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
                    input_tokens = it.unwrap_or(0);
                    output_tokens = ot.unwrap_or(0);
                    break;
                }
            }
        }

        // Reset restart count on successful exchange.
        self.restart_count
            .store(0, std::sync::atomic::Ordering::Relaxed);

        Ok((text_content, tool_calls, input_tokens, output_tokens))
    }
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
        let prompt = messages_to_prompt(&request.messages);
        let (content, _tool_calls, input_tokens, output_tokens) = self.exchange(&prompt).await?;

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
        let prompt = messages_to_prompt(&request.messages);
        let (content, tool_calls, input_tokens, output_tokens) = self.exchange(&prompt).await?;

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
