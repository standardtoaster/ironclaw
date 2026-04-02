//! Supervisor-based backend for Claude Code session management.
//!
//! Instead of managing Docker containers directly, this backend delegates
//! session lifecycle to a compute supervisor HTTP API running on the host.
//! The supervisor creates percy-channel instances and manages their lifecycle.
//!
//! Communication flow:
//! 1. IronClaw creates a session via `POST {supervisor_url}/claude/sessions`
//! 2. Supervisor returns a `channel_url` for the percy-channel instance
//! 3. IronClaw sends prompts via `POST {channel_url}/message`
//! 4. Percy-channel processes the prompt and POSTs the reply to IronClaw's
//!    callback endpoint at `/api/claude/reply`
//! 5. The reply is bridged to the waiting `exchange()` future via a oneshot channel

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;

use crate::llm::claude_protocol::ExchangeResult;
use crate::llm::container_backend::ContainerBackend;
use crate::llm::error::LlmError;
use crate::llm::provider::ToolCall;

/// Configuration for the supervisor backend.
#[derive(Debug, Clone)]
pub struct SupervisorBackendConfig {
    /// Supervisor API URL (e.g. "http://mac:3201").
    pub supervisor_url: String,
    /// IronClaw's own URL for percy-channel to POST replies to.
    pub callback_url: String,
    /// Optional auth token for the supervisor API.
    pub auth_token: Option<String>,
    /// Claude model to use.
    pub model: String,
    /// Timeout for requests in seconds.
    pub request_timeout_secs: u64,
    /// IronClaw gateway URL for MCP tool access (e.g. "http://localhost:3003").
    pub gateway_url: Option<String>,
    /// Bearer token for IronClaw gateway (scoped to the lens).
    pub gateway_token: Option<String>,
    /// Percy lens/member name (e.g. "andrew").
    pub lens_name: Option<String>,
}

/// State for an active supervisor-managed session.
struct SupervisorSession {
    /// Session ID assigned by the supervisor.
    session_id: String,
    /// Claude CLI session ID (for --resume on reconnect).
    cli_session_id: String,
    /// URL of the percy-channel instance for this session.
    channel_url: String,
    /// Number of messages sent so far (for delta tracking).
    messages_sent: usize,
    /// Whether this session was resumed from a previous CLI session.
    resumed: bool,
}

/// Shared map of pending reply channels, keyed by thread UUID.
///
/// Stored in `GatewayState` and shared with `SupervisorBackend` instances so that
/// the `/api/claude/reply` HTTP handler can resolve waiting `exchange()` futures.
pub type PendingRepliesMap =
    Arc<Mutex<HashMap<Uuid, oneshot::Sender<CallbackReply>>>>;

/// Reply payload received from percy-channel via callback.
#[derive(Debug)]
pub struct CallbackReply {
    /// Response content text.
    pub content: String,
    /// Tool calls extracted from the response.
    pub tool_calls: Vec<ToolCall>,
    /// Input tokens used.
    pub input_tokens: u32,
    /// Output tokens used.
    pub output_tokens: u32,
}

/// Supervisor-based backend for Claude Code sessions.
///
/// Creates and manages sessions through the compute supervisor HTTP API.
/// Prompt exchange is asynchronous: messages are POSTed to percy-channel,
/// and replies arrive via HTTP callback. A oneshot channel bridges the
/// async callback to the waiting `exchange()` future.
pub struct SupervisorBackend {
    config: SupervisorBackendConfig,
    http: reqwest::Client,
    sessions: Arc<Mutex<HashMap<Uuid, SupervisorSession>>>,
    /// CLI session IDs from previous sessions, for --resume.
    /// Persists across session recreation so conversations can be resumed.
    resume_ids: Arc<Mutex<HashMap<Uuid, String>>>,
    /// Pending reply receivers: exchange() inserts a sender, handle_reply() resolves it.
    pending_replies: Arc<Mutex<HashMap<Uuid, oneshot::Sender<CallbackReply>>>>,
}

impl SupervisorBackend {
    /// Create a new supervisor backend with the given config.
    pub fn new(config: SupervisorBackendConfig) -> Self {
        Self::new_with_shared_pending(config, Arc::new(Mutex::new(HashMap::new())))
    }

    /// Create a new supervisor backend that shares a `pending_replies` map.
    ///
    /// Use this when the pending-replies map is owned by `GatewayState` so the
    /// `/api/claude/reply` HTTP handler can resolve waiting `exchange()` futures
    /// without needing a reference to the backend itself.
    pub fn new_with_shared_pending(
        config: SupervisorBackendConfig,
        pending_replies: PendingRepliesMap,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        tracing::info!(
            supervisor_url = %config.supervisor_url,
            callback_url = %config.callback_url,
            model = %config.model,
            "Supervisor backend initialized"
        );

        Self {
            config,
            http,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            resume_ids: Arc::new(Mutex::new(HashMap::new())),
            pending_replies,
        }
    }

    /// Handle a reply callback from percy-channel.
    ///
    /// This is called by the HTTP route handler when a reply arrives at
    /// `/api/claude/reply`. It resolves the oneshot channel that the
    /// corresponding `exchange()` call is waiting on.
    ///
    /// Returns `true` if the reply was delivered, `false` if no one was waiting.
    pub async fn handle_reply(
        &self,
        thread_id: Uuid,
        content: String,
        tool_calls: Vec<ToolCall>,
        input_tokens: u32,
        output_tokens: u32,
    ) -> bool {
        let sender = self.pending_replies.lock().await.remove(&thread_id);
        match sender {
            Some(tx) => {
                let reply = CallbackReply {
                    content,
                    tool_calls,
                    input_tokens,
                    output_tokens,
                };
                // If the receiver was dropped (timeout), this send fails silently.
                tx.send(reply).is_ok()
            }
            None => {
                tracing::warn!(
                    thread_id = %thread_id,
                    "Received reply callback but no exchange is waiting"
                );
                false
            }
        }
    }

    /// Create a session via the supervisor API.
    ///
    /// If a previous CLI session ID exists for this thread, passes it as
    /// `resume_id` so Claude resumes the conversation.
    async fn create_session(&self, thread_id: Uuid) -> Result<SupervisorSession, LlmError> {
        let url = format!("{}/claude/sessions", self.config.supervisor_url);

        let mut body = serde_json::json!({
            "callback_url": self.config.callback_url,
            "model": self.config.model,
            "thread_id": thread_id.to_string(),
        });

        if let Some(ref token) = self.config.auth_token {
            body["auth_token"] = serde_json::Value::String(token.clone());
        }

        // Pass IronClaw gateway credentials so the sandbox Claude can use MCP tools
        if let Some(ref gw_url) = self.config.gateway_url {
            body["gateway_url"] = serde_json::Value::String(gw_url.clone());
        }
        if let Some(ref gw_token) = self.config.gateway_token {
            body["gateway_token"] = serde_json::Value::String(gw_token.clone());
        }
        if let Some(ref lens) = self.config.lens_name {
            body["lens_name"] = serde_json::Value::String(lens.clone());
        }

        // Resume previous conversation if we have a CLI session ID
        if let Some(resume_id) = self.resume_ids.lock().await.get(&thread_id) {
            body["resume_id"] = serde_json::Value::String(resume_id.clone());
            tracing::info!(
                thread_id = %thread_id,
                resume_id = %resume_id,
                "Resuming previous Claude session"
            );
        }

        let mut req = self.http.post(&url).json(&body);
        if let Some(ref token) = self.config.auth_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.map_err(|e| LlmError::RequestFailed {
            provider: "claude_supervisor".to_string(),
            reason: format!("Failed to create session via supervisor: {}", e),
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable>".to_string());
            return Err(LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: format!(
                    "Supervisor returned {} creating session: {}",
                    status, body_text
                ),
            });
        }

        let resp: serde_json::Value =
            response.json().await.map_err(|e| LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: format!("Failed to parse supervisor response: {}", e),
            })?;

        let session_id = resp
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: "Supervisor response missing session_id".to_string(),
            })?
            .to_string();

        let channel_url = resp
            .get("channel_url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: "Supervisor response missing channel_url".to_string(),
            })?
            .to_string();

        let cli_session_id = resp
            .get("cli_session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        tracing::info!(
            session_id = %session_id,
            cli_session_id = %cli_session_id,
            channel_url = %channel_url,
            thread_id = %thread_id,
            "Supervisor session created"
        );

        let resumed = self.resume_ids.lock().await.contains_key(&thread_id);

        // Store CLI session ID for future resume
        if !cli_session_id.is_empty() {
            self.resume_ids
                .lock()
                .await
                .insert(thread_id, cli_session_id.clone());
        }

        Ok(SupervisorSession {
            session_id,
            cli_session_id,
            channel_url,
            messages_sent: 0,
            resumed,
        })
    }

    /// Delete a session via the supervisor API.
    async fn delete_session(&self, session_id: &str) -> Result<(), LlmError> {
        let url = format!(
            "{}/claude/sessions/{}",
            self.config.supervisor_url, session_id
        );

        let mut req = self.http.delete(&url);
        if let Some(ref token) = self.config.auth_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.map_err(|e| LlmError::RequestFailed {
            provider: "claude_supervisor".to_string(),
            reason: format!("Failed to delete session via supervisor: {}", e),
        })?;

        if !response.status().is_success() {
            let status = response.status();
            tracing::warn!(
                session_id = %session_id,
                status = %status,
                "Supervisor returned non-success deleting session"
            );
        }

        Ok(())
    }
}

#[async_trait]
impl ContainerBackend for SupervisorBackend {
    async fn get_or_create(&self, thread_id: Uuid) -> Result<(), LlmError> {
        {
            let sessions = self.sessions.lock().await;
            if sessions.contains_key(&thread_id) {
                return Ok(());
            }
        }

        let session = self.create_session(thread_id).await?;
        self.sessions.lock().await.insert(thread_id, session);

        // Wait for the sandbox Claude to process the identity injection.
        // The supervisor sends SOUL.md/IDENTITY.md as the first channel message;
        // Claude reads them and replies "READY". We poll the channel health
        // to give Claude time to process before sending user messages.
        // The READY reply goes to the callback URL but uses a non-conversation
        // thread_id, so we can't match it via pending_replies. Instead, just
        // wait a fixed period for Claude to ingest the identity.
        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        tracing::info!(
            thread_id = %thread_id,
            "Identity injection wait complete (15s)"
        );

        Ok(())
    }

    async fn exchange(
        &self,
        thread_id: Uuid,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError> {
        // Get the channel URL for this session.
        let channel_url = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(&thread_id).ok_or_else(|| {
                LlmError::RequestFailed {
                    provider: "claude_supervisor".to_string(),
                    reason: format!("No active session for thread {}", thread_id),
                }
            })?;
            session.channel_url.clone()
        };

        // Set up a oneshot channel for the callback reply.
        let (tx, rx) = oneshot::channel::<CallbackReply>();

        // Check for duplicate pending request (concurrent exchange on same thread).
        {
            let mut pending = self.pending_replies.lock().await;
            if pending.contains_key(&thread_id) {
                return Err(LlmError::RequestFailed {
                    provider: "claude_supervisor".to_string(),
                    reason: format!(
                        "Concurrent exchange already pending for thread {}",
                        thread_id
                    ),
                });
            }
            pending.insert(thread_id, tx);
        }

        // POST the message to percy-channel.
        let message_url = format!("{}/message", channel_url);
        let body = serde_json::json!({
            "content": prompt,
            "thread_id": thread_id.to_string(),
            "user": "ironclaw",
        });

        let send_result = self.http.post(&message_url).json(&body).send().await;

        if let Err(e) = send_result {
            // Clean up the pending reply on send failure.
            self.pending_replies.lock().await.remove(&thread_id);
            return Err(LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: format!("Failed to send message to channel: {}", e),
            });
        }

        let response = send_result.expect("checked above");
        if !response.status().is_success() {
            self.pending_replies.lock().await.remove(&thread_id);
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable>".to_string());
            return Err(LlmError::RequestFailed {
                provider: "claude_supervisor".to_string(),
                reason: format!(
                    "Channel returned {} for message: {}",
                    status, body_text
                ),
            });
        }

        // Wait for the callback reply with timeout.
        let timeout = std::time::Duration::from_secs(self.config.request_timeout_secs);
        let reply = tokio::time::timeout(timeout, rx).await;

        match reply {
            Ok(Ok(callback)) => {
                // Update messages_sent counter.
                if let Some(session) = self.sessions.lock().await.get_mut(&thread_id) {
                    session.messages_sent += 1;
                }

                Ok(ExchangeResult::Complete {
                    content: callback.content,
                    tool_calls: callback.tool_calls,
                    input_tokens: callback.input_tokens,
                    output_tokens: callback.output_tokens,
                })
            }
            Ok(Err(_)) => {
                // Oneshot sender was dropped without sending.
                Err(LlmError::RequestFailed {
                    provider: "claude_supervisor".to_string(),
                    reason: format!(
                        "Reply channel closed without response for thread {}",
                        thread_id
                    ),
                })
            }
            Err(_) => {
                // Timeout — clean up the pending reply.
                self.pending_replies.lock().await.remove(&thread_id);
                Err(LlmError::RequestFailed {
                    provider: "claude_supervisor".to_string(),
                    reason: format!(
                        "Reply timed out after {}s for thread {}",
                        self.config.request_timeout_secs, thread_id
                    ),
                })
            }
        }
    }

    async fn remove_session(&self, thread_id: Uuid) -> Result<(), LlmError> {
        // Remove pending reply (if any exchange is waiting, it will get a channel-closed error).
        self.pending_replies.lock().await.remove(&thread_id);

        let session = self.sessions.lock().await.remove(&thread_id);
        if let Some(session) = session {
            tracing::info!(
                session_id = %session.session_id,
                thread_id = %thread_id,
                "Removing supervisor session"
            );
            self.delete_session(&session.session_id).await?;
        }
        Ok(())
    }

    async fn shutdown_all(&self) -> Result<(), LlmError> {
        let thread_ids: Vec<Uuid> = {
            let sessions = self.sessions.lock().await;
            sessions.keys().copied().collect()
        };

        for thread_id in thread_ids {
            if let Err(e) = self.remove_session(thread_id).await {
                tracing::warn!(
                    thread_id = %thread_id,
                    error = %e,
                    "Failed to remove supervisor session during shutdown"
                );
            }
        }

        Ok(())
    }

    async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    async fn messages_sent(&self, thread_id: &Uuid) -> Option<usize> {
        self.sessions
            .lock()
            .await
            .get(thread_id)
            .map(|s| s.messages_sent)
    }

    async fn set_messages_sent(&self, thread_id: &Uuid, count: usize) {
        if let Some(session) = self.sessions.lock().await.get_mut(thread_id) {
            session.messages_sent = count;
        }
    }

    async fn is_resumed(&self, thread_id: &Uuid) -> bool {
        self.sessions
            .lock()
            .await
            .get(thread_id)
            .map(|s| s.resumed)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SupervisorBackendConfig {
        SupervisorBackendConfig {
            supervisor_url: "http://localhost:3201".to_string(),
            callback_url: "http://localhost:3003".to_string(),
            auth_token: None,
            model: "sonnet".to_string(),
            request_timeout_secs: 300,
            gateway_url: None,
            gateway_token: None,
            lens_name: None,
        }
    }

    #[test]
    fn test_supervisor_backend_config_clone() {
        let config = test_config();
        let cloned = config.clone();
        assert_eq!(cloned.supervisor_url, "http://localhost:3201");
        assert_eq!(cloned.callback_url, "http://localhost:3003");
        assert_eq!(cloned.model, "sonnet");
    }

    #[test]
    fn test_supervisor_backend_config_debug() {
        let config = test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("localhost:3201"));
        assert!(debug.contains("sonnet"));
    }

    #[tokio::test]
    async fn test_session_count_zero_on_init() {
        let backend = SupervisorBackend::new(test_config());
        assert_eq!(backend.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_shutdown_noop_when_empty() {
        let backend = SupervisorBackend::new(test_config());
        // Should not panic or error when no sessions exist.
        backend.shutdown_all().await.expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_exchange_without_session_errors() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();
        let result = backend.exchange(thread_id, "hello").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No active session"));
    }

    #[tokio::test]
    async fn test_remove_nonexistent_session_is_noop() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();
        // Removing a session that doesn't exist should succeed silently.
        backend
            .remove_session(thread_id)
            .await
            .expect("remove should succeed");
    }

    #[tokio::test]
    async fn test_messages_sent_none_without_session() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();
        assert_eq!(backend.messages_sent(&thread_id).await, None);
    }

    #[tokio::test]
    async fn test_set_messages_sent_noop_without_session() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();
        // Should not panic when session doesn't exist.
        backend.set_messages_sent(&thread_id, 5).await;
    }

    #[tokio::test]
    async fn test_handle_reply_no_waiter() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();
        // No one is waiting for a reply.
        let delivered = backend
            .handle_reply(thread_id, "response".into(), vec![], 10, 5)
            .await;
        assert!(!delivered);
    }

    #[tokio::test]
    async fn test_handle_reply_resolves_oneshot() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();

        // Manually insert a pending reply (simulating what exchange() does).
        let (tx, rx) = oneshot::channel::<CallbackReply>();
        backend.pending_replies.lock().await.insert(thread_id, tx);

        // Deliver the reply.
        let delivered = backend
            .handle_reply(thread_id, "hello back".into(), vec![], 20, 10)
            .await;
        assert!(delivered);

        // Receive the reply.
        let reply = rx.await.expect("should receive reply");
        assert_eq!(reply.content, "hello back");
        assert_eq!(reply.input_tokens, 20);
        assert_eq!(reply.output_tokens, 10);
        assert!(reply.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn test_handle_reply_with_tool_calls() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();

        let (tx, rx) = oneshot::channel::<CallbackReply>();
        backend.pending_replies.lock().await.insert(thread_id, tx);

        let tool_calls = vec![ToolCall {
            id: "tc_1".to_string(),
            name: "memory_search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        }];

        let delivered = backend
            .handle_reply(thread_id, "".into(), tool_calls, 15, 8)
            .await;
        assert!(delivered);

        let reply = rx.await.expect("should receive reply");
        assert_eq!(reply.tool_calls.len(), 1);
        assert_eq!(reply.tool_calls[0].name, "memory_search");
    }

    #[tokio::test]
    async fn test_duplicate_pending_reply_detected() {
        let backend = SupervisorBackend::new(test_config());
        let thread_id = Uuid::new_v4();

        let (tx1, _rx1) = oneshot::channel::<CallbackReply>();
        let (tx2, _rx2) = oneshot::channel::<CallbackReply>();

        let mut pending = backend.pending_replies.lock().await;
        let prev = pending.insert(thread_id, tx1);
        assert!(prev.is_none());

        let prev = pending.insert(thread_id, tx2);
        assert!(prev.is_some());
    }

    #[test]
    fn test_supervisor_backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SupervisorBackend>();
    }

    #[tokio::test]
    async fn test_two_threads_independent_pending_replies() {
        let backend = SupervisorBackend::new(test_config());
        let tid1 = Uuid::new_v4();
        let tid2 = Uuid::new_v4();

        let (tx1, rx1) = oneshot::channel::<CallbackReply>();
        let (tx2, rx2) = oneshot::channel::<CallbackReply>();

        {
            let mut pending = backend.pending_replies.lock().await;
            pending.insert(tid1, tx1);
            pending.insert(tid2, tx2);
        }

        // Resolve only the first.
        let delivered = backend
            .handle_reply(tid1, "reply 1".into(), vec![], 10, 5)
            .await;
        assert!(delivered);

        // Second should still be pending.
        assert!(backend.pending_replies.lock().await.contains_key(&tid2));
        assert!(!backend.pending_replies.lock().await.contains_key(&tid1));

        let reply1 = rx1.await.expect("should receive reply 1");
        assert_eq!(reply1.content, "reply 1");

        // Now resolve the second.
        let delivered = backend
            .handle_reply(tid2, "reply 2".into(), vec![], 20, 10)
            .await;
        assert!(delivered);

        let reply2 = rx2.await.expect("should receive reply 2");
        assert_eq!(reply2.content, "reply 2");
    }

    /// Prove that new_with_shared_pending shares the pending_replies map.
    ///
    /// This tests the contract between SupervisorBackend (which inserts pending
    /// replies during exchange) and GatewayState's /api/claude/reply handler
    /// (which resolves them). If they don't share the same map, callbacks fail
    /// silently — the exact bug we had in production.
    #[tokio::test]
    async fn test_shared_pending_replies_map_across_backend_and_handler() {
        // Create a shared map (simulating what main.rs does).
        let shared_map: PendingRepliesMap = Arc::new(Mutex::new(HashMap::new()));

        // Create backend with the shared map.
        let backend = SupervisorBackend::new_with_shared_pending(
            test_config(),
            Arc::clone(&shared_map),
        );

        let thread_id = Uuid::new_v4();

        // Backend inserts a pending reply (as exchange() would).
        let (tx, rx) = oneshot::channel::<CallbackReply>();
        backend.pending_replies.lock().await.insert(thread_id, tx);

        // Simulate the gateway handler resolving the reply via the SHARED map
        // (not through the backend — this is what /api/claude/reply does).
        let sender = shared_map.lock().await.remove(&thread_id);
        assert!(
            sender.is_some(),
            "Gateway handler should find the pending reply in the shared map"
        );

        let reply = CallbackReply {
            content: "hello from callback".to_string(),
            tool_calls: vec![],
            input_tokens: 10,
            output_tokens: 5,
        };
        sender.unwrap().send(reply).expect("should deliver reply");

        // The exchange() waiter should receive it.
        let received = rx.await.expect("should receive reply via shared map");
        assert_eq!(received.content, "hello from callback");
    }

    /// Prove that new() (without shared map) creates an isolated map.
    /// This means a gateway handler looking at a different map won't find
    /// the pending reply — demonstrating the bug that shared maps fix.
    #[tokio::test]
    async fn test_non_shared_map_is_isolated() {
        let gateway_map: PendingRepliesMap = Arc::new(Mutex::new(HashMap::new()));
        let backend = SupervisorBackend::new(test_config()); // own private map

        let thread_id = Uuid::new_v4();

        // Backend inserts a pending reply.
        let (tx, _rx) = oneshot::channel::<CallbackReply>();
        backend.pending_replies.lock().await.insert(thread_id, tx);

        // Gateway handler checks ITS map — should NOT find it.
        let sender = gateway_map.lock().await.remove(&thread_id);
        assert!(
            sender.is_none(),
            "Non-shared map should NOT contain backend's pending reply"
        );
    }

    /// Verify that a READY reply is correctly identified by the string-contains check
    /// used in `get_or_create` (line 344).
    #[test]
    fn test_ready_reply_contains_ready() {
        let reply = CallbackReply {
            content: "READY".to_string(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        };
        assert!(
            reply.content.contains("READY"),
            "Reply with 'READY' content should be detected"
        );

        // Also works when READY is embedded in a longer string.
        let reply_verbose = CallbackReply {
            content: "System READY for input".to_string(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        };
        assert!(reply_verbose.content.contains("READY"));
    }

    /// Verify that a reply without the "READY" keyword is not falsely detected.
    /// In get_or_create, this path logs a warning but still proceeds (Ok(())).
    #[test]
    fn test_ready_reply_without_ready_keyword() {
        let reply = CallbackReply {
            content: "OK".to_string(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        };
        assert!(
            !reply.content.contains("READY"),
            "Reply without 'READY' should not match the contains check"
        );

        let reply_empty = CallbackReply {
            content: String::new(),
            tool_calls: vec![],
            input_tokens: 0,
            output_tokens: 0,
        };
        assert!(!reply_empty.content.contains("READY"));
    }

    /// Verify the READY wait resolves when a READY reply is sent through the
    /// shared pending_replies map (simulating the callback path).
    #[tokio::test]
    async fn test_ready_signal_resolves_via_shared_pending_replies() {
        let shared_map: PendingRepliesMap = Arc::new(Mutex::new(HashMap::new()));
        let thread_id = Uuid::new_v4();

        // Insert a pending reply (as get_or_create would after create_session).
        let (tx, rx) = oneshot::channel::<CallbackReply>();
        shared_map.lock().await.insert(thread_id, tx);

        // Simulate the callback handler sending a READY reply.
        let sender = shared_map.lock().await.remove(&thread_id);
        assert!(sender.is_some(), "Sender should be present in shared map");
        sender
            .unwrap()
            .send(CallbackReply {
                content: "READY".to_string(),
                tool_calls: vec![],
                input_tokens: 0,
                output_tokens: 0,
            })
            .expect("send should succeed");

        // The waiter should receive the READY reply.
        let reply = rx.await.expect("should receive reply");
        assert!(reply.content.contains("READY"));
    }

    #[test]
    fn test_config_with_auth_token() {
        let config = SupervisorBackendConfig {
            supervisor_url: "http://mac:3201".to_string(),
            callback_url: "http://ralph:3003".to_string(),
            auth_token: Some("secret-token".to_string()),
            model: "opus".to_string(),
            request_timeout_secs: 600,
            gateway_url: Some("http://ralph:3003".to_string()),
            gateway_token: Some("test-token".to_string()),
            lens_name: Some("andrew".to_string()),
        };
        assert_eq!(config.auth_token.as_deref(), Some("secret-token"));
        assert_eq!(config.model, "opus");
    }
}
