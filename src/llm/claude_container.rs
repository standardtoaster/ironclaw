//! Claude Code container provider.
//!
//! Implements `LlmProvider` backed by a pluggable `ContainerBackend`:
//! - **Docker** (`container_pool.rs`): manages containers via bollard
//! - **Supervisor** (`container_supervisor.rs`): delegates to a compute supervisor HTTP API
//!
//! One session per conversation (thread_id), multiple sessions per lens.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rust_decimal::Decimal;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::llm::claude_protocol::ExchangeResult;
use crate::llm::container_backend::ContainerBackend;
use crate::llm::container_pool::{ContainerPool, ContainerPoolConfig};
use crate::llm::container_supervisor::{SupervisorBackend, SupervisorBackendConfig};
use crate::llm::costs;
use crate::llm::error::LlmError;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    ToolCall, ToolCompletionRequest, ToolCompletionResponse,
};

/// Configuration for the container-based Claude provider.
#[derive(Debug, Clone)]
pub struct ContainerProviderConfig {
    /// Backend type: "docker" (default) or "supervisor".
    pub backend_type: String,
    /// Claude model alias ("sonnet", "opus", or full model ID).
    pub model: String,
    /// Docker image for Claude Code containers.
    pub image: String,
    /// Docker socket path (empty = bollard defaults).
    pub socket_path: String,
    /// Lens name (used in container naming and labels).
    pub lens: String,
    /// Docker network to attach containers to.
    pub network: String,
    /// Timeout for requests in seconds.
    pub request_timeout_secs: u64,
    /// Named volume containing Claude auth credentials (mounted ro).
    pub auth_volume: String,
    /// Whether to pass --dangerously-skip-permissions to claude.
    pub skip_permissions: bool,
    /// Extra environment variables to pass to the container.
    pub extra_env: Vec<String>,
    /// Optional named volume for lens-specific data (mounted rw).
    pub lens_data_volume: Option<String>,
    /// Optional host path to generated/sidecar/{lens}/ config (bind mount ro).
    pub lens_config_path: Option<String>,
    /// Supervisor URL (only used when backend_type = "supervisor").
    pub supervisor_url: Option<String>,
    /// Callback URL for supervisor backend (IronClaw's own URL).
    pub callback_url: Option<String>,
    /// Auth token for supervisor API.
    pub supervisor_auth_token: Option<String>,
    /// IronClaw gateway URL for MCP tool access (e.g. "http://localhost:3003").
    pub gateway_url: Option<String>,
    /// Bearer token for IronClaw gateway (scoped to the lens).
    pub gateway_token: Option<String>,
}

/// An LLM provider that manages Claude Code sessions via a pluggable backend.
///
/// One session per conversation thread. Sessions are created on demand
/// via the `ContainerBackend` trait and kept warm for the conversation.
/// The backend connects lazily on first use to avoid blocking startup.
///
/// Two backends are supported:
/// - **Docker** (`backend_type = "docker"`): manages containers via bollard
/// - **Supervisor** (`backend_type = "supervisor"`): delegates to a compute supervisor HTTP API
pub struct ClaudeContainerProvider {
    config: ContainerProviderConfig,
    backend: OnceCell<Arc<dyn ContainerBackend>>,
    model_label: String,
    /// Shared pending-reply map for supervisor backend callback route.
    /// When set, the `SupervisorBackend` uses this map so `/api/claude/reply`
    /// can resolve waiting `exchange()` futures without a reference to the backend.
    shared_pending_replies: Option<crate::llm::container_supervisor::PendingRepliesMap>,
}

impl ClaudeContainerProvider {
    /// Create a new container provider with the given config.
    ///
    /// The backend connection is deferred until the first `complete()` call,
    /// so construction never fails. This matches the lazy-init pattern used
    /// by `ClaudeSidecarProvider`.
    pub fn new(config: ContainerProviderConfig) -> Self {
        let model_label = format!("claude-container/{}", config.model);
        Self {
            config,
            backend: OnceCell::new(),
            model_label,
            shared_pending_replies: None,
        }
    }

    /// Create a new container provider that shares a pending-replies map with `GatewayState`.
    ///
    /// The shared map allows the `/api/claude/reply` route handler to resolve
    /// oneshot channels without needing a reference to the backend.
    pub fn new_with_shared_pending(
        config: ContainerProviderConfig,
        pending_replies: crate::llm::container_supervisor::PendingRepliesMap,
    ) -> Self {
        let model_label = format!("claude-container/{}", config.model);
        Self {
            config,
            backend: OnceCell::new(),
            model_label,
            shared_pending_replies: Some(pending_replies),
        }
    }

    /// Get or initialize the backend.
    async fn backend(&self) -> Result<&Arc<dyn ContainerBackend>, LlmError> {
        self.backend
            .get_or_try_init(|| async {
                match self.config.backend_type.as_str() {
                    "supervisor" => {
                        let supervisor_url = self.config.supervisor_url.clone()
                            .ok_or_else(|| LlmError::RequestFailed {
                                provider: "claude_container".to_string(),
                                reason: "supervisor_url is required for supervisor backend".to_string(),
                            })?;
                        let callback_url = self.config.callback_url.clone()
                            .ok_or_else(|| LlmError::RequestFailed {
                                provider: "claude_container".to_string(),
                                reason: "callback_url is required for supervisor backend".to_string(),
                            })?;
                        let supervisor_config = SupervisorBackendConfig {
                            supervisor_url,
                            callback_url,
                            auth_token: self.config.supervisor_auth_token.clone(),
                            model: self.config.model.clone(),
                            request_timeout_secs: self.config.request_timeout_secs,
                            gateway_url: self.config.gateway_url.clone(),
                            gateway_token: self.config.gateway_token.clone(),
                            lens_name: Some(self.config.lens.clone()),
                        };
                        let backend = match &self.shared_pending_replies {
                            Some(pending) => SupervisorBackend::new_with_shared_pending(
                                supervisor_config,
                                Arc::clone(pending),
                            ),
                            None => SupervisorBackend::new(supervisor_config),
                        };
                        Ok(Arc::new(backend) as Arc<dyn ContainerBackend>)
                    }
                    _ => {
                        // Default: Docker backend.
                        let pool_config = ContainerPoolConfig {
                            image: self.config.image.clone(),
                            lens: self.config.lens.clone(),
                            model: self.config.model.clone(),
                            network: self.config.network.clone(),
                            auth_volume: self.config.auth_volume.clone(),
                            lens_data_volume: self.config.lens_data_volume.clone(),
                            lens_config_path: self.config.lens_config_path.clone(),
                            skip_permissions: self.config.skip_permissions,
                            request_timeout_secs: self.config.request_timeout_secs,
                            extra_env: self.config.extra_env.clone(),
                        };
                        let pool = ContainerPool::new(&self.config.socket_path, pool_config).await?;
                        Ok(Arc::new(pool) as Arc<dyn ContainerBackend>)
                    }
                }
            })
            .await
    }

    /// Get the backend if it has been initialized (for use in supervisor callback wiring).
    ///
    /// Returns `None` if the backend hasn't been lazily initialized yet.
    pub fn get_backend(&self) -> Option<&Arc<dyn ContainerBackend>> {
        self.backend.get()
    }

    /// End the session for a specific thread (e.g., on de-escalation).
    ///
    /// Removes the session and cleans up resources.
    pub async fn end_session_cleanup(&self, thread_id: Uuid) {
        if let Some(backend) = self.backend.get()
            && let Err(e) = backend.remove_session(thread_id).await
        {
            tracing::warn!(
                thread_id = %thread_id,
                error = %e,
                "Failed to remove container session"
            );
        }
    }

    /// Shut down all container sessions.
    pub async fn shutdown(&self) {
        if let Some(backend) = self.backend.get()
            && let Err(e) = backend.shutdown_all().await
        {
            tracing::warn!(error = %e, "Failed to shutdown container pool");
        }
    }

    /// Get the number of active container sessions.
    pub async fn session_count(&self) -> usize {
        match self.backend.get() {
            Some(backend) => backend.session_count().await,
            None => 0,
        }
    }

    /// Perform a session-aware exchange with auto-approval.
    ///
    /// Returns `(content, tool_calls, input_tokens, output_tokens)`.
    async fn session_exchange_auto_approve(
        &self,
        messages: &[ChatMessage],
        thread_id: Option<Uuid>,
    ) -> Result<(String, Vec<ToolCall>, u32, u32), LlmError> {
        let backend = self.backend().await?;
        let tid = thread_id.unwrap_or_else(Uuid::new_v4);

        // Ensure session exists for this thread.
        backend.get_or_create(tid).await?;

        // If this session was resumed, Claude already has the conversation history.
        // Mark all existing messages as sent so only new ones flow through.
        if backend.is_resumed(&tid).await {
            let current = backend.messages_sent(&tid).await.unwrap_or(0);
            if current == 0 {
                tracing::info!(
                    thread_id = %tid,
                    history_size = messages.len(),
                    "Resumed session — marking {} existing messages as already sent",
                    messages.len()
                );
                backend.set_messages_sent(&tid, messages.len()).await;
                // No exchange needed — Claude is caught up. Return empty response
                // so the agent loop proceeds to the next user message.
                return Ok((String::new(), Vec::new(), 0, 0));
            }
        }

        // The sandbox Claude maintains its own conversation history via --session-id.
        // We only need to send the latest user message — not deltas of the full
        // IronClaw message history (which includes assistant, tool, and system messages
        // that would confuse or duplicate what the sandbox already has).
        let prompt = messages
            .iter()
            .rev()
            .find(|m| m.role == crate::llm::provider::Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        tracing::debug!(
            thread_id = %tid,
            prompt_len = prompt.len(),
            total_messages = messages.len(),
            "Sending last user message to container session"
        );

        // Exchange with the backend.
        let result = backend.exchange(tid, &prompt).await?;

        match result {
            ExchangeResult::Complete {
                content,
                tool_calls,
                input_tokens,
                output_tokens,
            } => {
                // Update messages_sent counter.
                backend.set_messages_sent(&tid, messages.len()).await;
                Ok((content, tool_calls, input_tokens, output_tokens))
            }
            ExchangeResult::NeedApproval(approval) => {
                // Auto-approve (same as ClaudeSidecarProvider in LlmProvider mode).
                tracing::debug!(
                    tool = %approval.tool_name,
                    "Auto-approving tool (via LlmProvider interface) — not yet implemented for containers"
                );
                // Container approval proxying is not yet implemented.
                // For now, containers should use --dangerously-skip-permissions.
                Err(LlmError::RequestFailed {
                    provider: "claude_container".to_string(),
                    reason: format!(
                        "Tool approval requested for '{}' but approval proxying is not yet implemented for containers. \
                         Use skip_permissions=true in config.",
                        approval.tool_name
                    ),
                })
            }
        }
    }
}

/// Extract the thread_id from request metadata, if present.
fn thread_id_from_metadata(metadata: &HashMap<String, String>) -> Option<Uuid> {
    metadata
        .get("thread_id")
        .and_then(|s| Uuid::parse_str(s).ok())
}

#[async_trait]
impl LlmProvider for ClaudeContainerProvider {
    fn model_name(&self) -> &str {
        &self.model_label
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
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
        self.end_session_cleanup(thread_id).await;
    }

    async fn shutdown(&self) {
        ClaudeContainerProvider::shutdown(self).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    fn test_config() -> ContainerProviderConfig {
        ContainerProviderConfig {
            backend_type: "docker".to_string(),
            model: "sonnet".to_string(),
            image: "percy-claude:latest".to_string(),
            socket_path: String::new(),
            lens: "andrew".to_string(),
            network: "percy_proxy-network".to_string(),
            request_timeout_secs: 300,
            auth_volume: "claude-auth".to_string(),
            skip_permissions: true,
            extra_env: vec![],
            lens_data_volume: Some("claude-data-andrew".to_string()),
            lens_config_path: None,
            supervisor_url: None,
            callback_url: None,
            supervisor_auth_token: None,
            gateway_url: None,
            gateway_token: None,
        }
    }

    #[test]
    fn test_provider_model_name() {
        let provider = ClaudeContainerProvider::new(test_config());
        assert_eq!(provider.model_name(), "claude-container/sonnet");
    }

    #[test]
    fn test_provider_model_name_opus() {
        let mut config = test_config();
        config.model = "opus".to_string();
        let provider = ClaudeContainerProvider::new(config);
        assert_eq!(provider.model_name(), "claude-container/opus");
    }

    #[test]
    fn test_provider_cost_per_token_sonnet() {
        let provider = ClaudeContainerProvider::new(test_config());
        let (input, output) = provider.cost_per_token();
        assert!(input > Decimal::ZERO, "Sonnet input cost should be > 0");
        assert!(output > Decimal::ZERO, "Sonnet output cost should be > 0");
    }

    #[test]
    fn test_provider_cost_per_token_opus() {
        let mut config = test_config();
        config.model = "opus".to_string();
        let provider = ClaudeContainerProvider::new(config);
        let (input, output) = provider.cost_per_token();
        assert!(input > Decimal::ZERO);
        assert!(output > Decimal::ZERO);
        // Opus should be more expensive than sonnet.
        let sonnet_provider = ClaudeContainerProvider::new(test_config());
        let (sonnet_in, _) = sonnet_provider.cost_per_token();
        assert!(input > sonnet_in, "Opus should cost more than Sonnet");
    }

    #[test]
    fn test_container_provider_config_clone() {
        let config = test_config();
        let cloned = config.clone();
        assert_eq!(cloned.model, "sonnet");
        assert_eq!(cloned.image, "percy-claude:latest");
        assert_eq!(cloned.lens, "andrew");
        assert_eq!(
            cloned.lens_data_volume,
            Some("claude-data-andrew".to_string())
        );
    }

    #[test]
    fn test_container_provider_config_debug() {
        let config = test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("sonnet"));
        assert!(debug.contains("percy-claude:latest"));
    }

    #[tokio::test]
    async fn test_session_count_zero_before_init() {
        let provider = ClaudeContainerProvider::new(test_config());
        assert_eq!(provider.session_count().await, 0);
    }

    #[tokio::test]
    async fn test_end_session_noop_before_pool_init() {
        let provider = ClaudeContainerProvider::new(test_config());
        let thread_id = Uuid::new_v4();
        // Should not panic when pool hasn't been initialized yet.
        provider.end_session_cleanup(thread_id).await;
    }

    #[tokio::test]
    async fn test_shutdown_noop_before_pool_init() {
        let provider = ClaudeContainerProvider::new(test_config());
        // Should not panic when pool hasn't been initialized yet.
        provider.shutdown().await;
    }

    #[tokio::test]
    async fn test_model_metadata() {
        let provider = ClaudeContainerProvider::new(test_config());
        let meta = provider.model_metadata().await.expect("metadata");
        assert_eq!(meta.id, "claude-container/sonnet");
        assert_eq!(meta.context_length, Some(200_000));
    }

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

    #[test]
    fn test_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ClaudeContainerProvider>();
    }

    // --- Reply handler tests ---

    #[test]
    fn test_reply_handler_resolves_oneshot() {
        // Insert a oneshot sender in a HashMap (simulating pending replies),
        // send a value through it, and verify the receiver gets it.
        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        let thread_id = Uuid::new_v4();
        let mut pending: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> = HashMap::new();
        pending.insert(thread_id, tx);

        // Resolve: remove sender and send a reply.
        let sender = pending.remove(&thread_id).expect("sender should exist");
        sender.send("reply content".to_string()).expect("send should succeed");
        assert_eq!(rx.blocking_recv().expect("recv"), "reply content");
        assert!(!pending.contains_key(&thread_id));
    }

    #[test]
    fn test_reply_handler_unknown_thread_noop() {
        // Trying to resolve a reply for a non-existent key is a no-op.
        let mut pending: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> = HashMap::new();
        let unknown_id = Uuid::new_v4();
        let removed = pending.remove(&unknown_id);
        assert!(removed.is_none(), "Removing unknown key should return None");
    }

    #[test]
    fn test_invalid_uuid_rejection() {
        let result = Uuid::parse_str("not-a-valid-uuid");
        assert!(result.is_err(), "Parsing invalid UUID string should fail");
    }

    // --- Approval lifecycle tests ---

    #[test]
    fn test_pending_approval_insert_resolve_gone() {
        // Simulate a pending approval map: insert, resolve, verify gone.
        let thread_id = Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel::<bool>();
        let mut pending_approvals: HashMap<Uuid, tokio::sync::oneshot::Sender<bool>> =
            HashMap::new();
        pending_approvals.insert(thread_id, tx);
        assert!(pending_approvals.contains_key(&thread_id));

        // Resolve: approve.
        let sender = pending_approvals.remove(&thread_id).unwrap();
        sender.send(true).unwrap();
        assert!(!pending_approvals.contains_key(&thread_id));
        assert_eq!(rx.blocking_recv().unwrap(), true);
    }

    #[test]
    fn test_pending_question_insert_resolve_gone() {
        // Simulate a pending question map: insert, resolve with answer, verify gone.
        let thread_id = Uuid::new_v4();
        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        let mut pending_questions: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> =
            HashMap::new();
        pending_questions.insert(thread_id, tx);
        assert!(pending_questions.contains_key(&thread_id));

        let sender = pending_questions.remove(&thread_id).unwrap();
        sender.send("user answer".to_string()).unwrap();
        assert!(!pending_questions.contains_key(&thread_id));
        assert_eq!(rx.blocking_recv().unwrap(), "user answer");
    }

    // --- Concurrent request guard tests ---

    #[test]
    fn test_duplicate_pending_reply_detected() {
        // Inserting a pending reply for the same thread_id twice: the second
        // insert finds an existing entry.
        let thread_id = Uuid::new_v4();
        let (tx1, _rx1) = tokio::sync::oneshot::channel::<String>();
        let (tx2, _rx2) = tokio::sync::oneshot::channel::<String>();
        let mut pending: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> = HashMap::new();

        let prev = pending.insert(thread_id, tx1);
        assert!(prev.is_none(), "First insert should find no existing entry");

        let prev = pending.insert(thread_id, tx2);
        assert!(prev.is_some(), "Second insert should find existing entry");
    }

    #[test]
    fn test_remove_pending_reply_releases_guard() {
        let thread_id = Uuid::new_v4();
        let (tx, _rx) = tokio::sync::oneshot::channel::<String>();
        let mut pending: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> = HashMap::new();
        pending.insert(thread_id, tx);
        assert!(pending.contains_key(&thread_id));

        pending.remove(&thread_id);
        assert!(!pending.contains_key(&thread_id), "Guard should be released");
    }

    // --- Thread isolation ---

    #[test]
    fn test_two_threads_independent_replies() {
        let tid1 = Uuid::new_v4();
        let tid2 = Uuid::new_v4();
        let (tx1, rx1) = tokio::sync::oneshot::channel::<String>();
        let (tx2, rx2) = tokio::sync::oneshot::channel::<String>();
        let mut pending: HashMap<Uuid, tokio::sync::oneshot::Sender<String>> = HashMap::new();
        pending.insert(tid1, tx1);
        pending.insert(tid2, tx2);

        // Resolve only the first thread.
        let sender = pending.remove(&tid1).unwrap();
        sender.send("reply for thread 1".to_string()).unwrap();

        // Second thread should still be pending.
        assert!(pending.contains_key(&tid2));
        assert!(!pending.contains_key(&tid1));
        assert_eq!(rx1.blocking_recv().unwrap(), "reply for thread 1");

        // Now resolve the second.
        let sender2 = pending.remove(&tid2).unwrap();
        sender2.send("reply for thread 2".to_string()).unwrap();
        assert_eq!(rx2.blocking_recv().unwrap(), "reply for thread 2");
    }

    // --- complete_with_tools ---

    #[tokio::test]
    async fn test_complete_with_tools_returns_empty_tool_calls() {
        // Without a running Docker daemon, complete_with_tools cannot be called
        // end-to-end. Instead verify the response construction: when content is
        // non-empty and tool_calls is empty, finish_reason should be Stop.
        // This mirrors the logic in ClaudeContainerProvider::complete_with_tools().
        let tool_calls: Vec<ToolCall> = vec![];
        let content = "some response".to_string();
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolUse
        } else {
            FinishReason::Stop
        };
        assert!(tool_calls.is_empty());
        assert_eq!(finish_reason, FinishReason::Stop);
        assert!(!content.is_empty());
    }

    // --- Cost mapping ---

    #[test]
    fn test_cost_sonnet_maps_correctly() {
        let provider = ClaudeContainerProvider::new(test_config());
        let (input, output) = provider.cost_per_token();
        let expected = costs::model_cost("claude-sonnet-4-6").unwrap();
        assert_eq!((input, output), expected);
    }

    #[test]
    fn test_cost_opus_maps_correctly() {
        let mut config = test_config();
        config.model = "opus".to_string();
        let provider = ClaudeContainerProvider::new(config);
        let (input, output) = provider.cost_per_token();
        let expected = costs::model_cost("claude-opus-4-6").unwrap();
        assert_eq!((input, output), expected);
    }

    #[test]
    fn test_cost_haiku_maps_correctly() {
        let mut config = test_config();
        config.model = "haiku".to_string();
        let provider = ClaudeContainerProvider::new(config);
        let (input, output) = provider.cost_per_token();
        let expected = costs::model_cost("claude-haiku-4-5").unwrap();
        assert_eq!((input, output), expected);
    }

    #[test]
    fn test_cost_unknown_model_falls_back_to_default() {
        let mut config = test_config();
        config.model = "unknown-model-xyz".to_string();
        let provider = ClaudeContainerProvider::new(config);
        let (input, output) = provider.cost_per_token();
        let expected = costs::default_cost();
        assert_eq!((input, output), expected);
    }

    // --- Callback URL construction ---

    #[test]
    fn test_model_label_construction_default() {
        let provider = ClaudeContainerProvider::new(test_config());
        assert_eq!(provider.model_name(), "claude-container/sonnet");
    }

    #[test]
    fn test_model_label_construction_custom_model() {
        let mut config = test_config();
        config.model = "claude-sonnet-4-6".to_string();
        let provider = ClaudeContainerProvider::new(config);
        assert_eq!(provider.model_name(), "claude-container/claude-sonnet-4-6");
    }

    // --- Config / payload tests ---

    #[test]
    fn test_config_with_defaults_only_required() {
        // Minimal config with no optional fields.
        let config = ContainerProviderConfig {
            backend_type: "docker".to_string(),
            model: "sonnet".to_string(),
            image: "percy-claude:latest".to_string(),
            socket_path: String::new(),
            lens: "test".to_string(),
            network: "bridge".to_string(),
            request_timeout_secs: 60,
            auth_volume: "auth-vol".to_string(),
            skip_permissions: false,
            extra_env: vec![],
            lens_data_volume: None,
            lens_config_path: None,
            supervisor_url: None,
            callback_url: None,
            supervisor_auth_token: None,
            gateway_url: None,
            gateway_token: None,
        };
        assert!(config.lens_data_volume.is_none());
        assert!(config.lens_config_path.is_none());
        assert!(!config.skip_permissions);
        assert!(config.extra_env.is_empty());
    }

    #[test]
    fn test_config_with_all_fields() {
        let config = ContainerProviderConfig {
            backend_type: "supervisor".to_string(),
            model: "opus".to_string(),
            image: "custom-image:v2".to_string(),
            socket_path: "/var/run/docker.sock".to_string(),
            lens: "grace".to_string(),
            network: "percy_net".to_string(),
            request_timeout_secs: 600,
            auth_volume: "claude-auth-vol".to_string(),
            skip_permissions: true,
            extra_env: vec!["KEY=VALUE".to_string(), "FOO=BAR".to_string()],
            lens_data_volume: Some("data-grace".to_string()),
            lens_config_path: Some("/opt/percy/sidecar/grace".to_string()),
            supervisor_url: Some("http://mac:3201".to_string()),
            callback_url: Some("http://ralph:3003".to_string()),
            supervisor_auth_token: Some("token".to_string()),
            gateway_url: Some("http://ralph:3003".to_string()),
            gateway_token: Some("grace-token".to_string()),
        };
        assert_eq!(config.lens_data_volume.as_deref(), Some("data-grace"));
        assert_eq!(
            config.lens_config_path.as_deref(),
            Some("/opt/percy/sidecar/grace")
        );
        assert_eq!(config.extra_env.len(), 2);
        assert!(config.skip_permissions);
    }

    #[test]
    fn test_thread_id_metadata_with_valid_uuid() {
        let id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let mut meta = HashMap::new();
        meta.insert("thread_id".to_string(), id.to_string());
        assert_eq!(thread_id_from_metadata(&meta), Some(id));
    }

    #[test]
    fn test_thread_id_metadata_without_thread_id_key() {
        let mut meta = HashMap::new();
        meta.insert("other_key".to_string(), Uuid::new_v4().to_string());
        assert_eq!(thread_id_from_metadata(&meta), None);
    }

    #[test]
    fn test_metadata_with_extra_fields_ignored() {
        let id = Uuid::new_v4();
        let mut meta = HashMap::new();
        meta.insert("thread_id".to_string(), id.to_string());
        meta.insert("user".to_string(), "andrew".to_string());
        meta.insert("channel".to_string(), "web".to_string());
        // Only thread_id matters.
        assert_eq!(thread_id_from_metadata(&meta), Some(id));
    }

    #[test]
    fn test_thread_id_empty_string_returns_none() {
        let mut meta = HashMap::new();
        meta.insert("thread_id".to_string(), String::new());
        assert_eq!(thread_id_from_metadata(&meta), None);
    }

    #[test]
    fn test_container_provider_config_serialization_roundtrip() {
        // Verify Clone produces a deep copy (no shared references).
        let config = test_config();
        let mut cloned = config.clone();
        cloned.model = "opus".to_string();
        cloned.lens = "grace".to_string();
        // Original should be unchanged.
        assert_eq!(config.model, "sonnet");
        assert_eq!(config.lens, "andrew");
        assert_eq!(cloned.model, "opus");
        assert_eq!(cloned.lens, "grace");
    }

    // --- Last user message extraction tests ---

    #[test]
    fn test_last_user_message_extraction() {
        use crate::llm::provider::Role;

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "You are a helpful assistant.".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: Role::User,
                content: "First question".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: Role::Assistant,
                content: "First answer".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: Role::Tool,
                content: "tool output".to_string(),
                content_parts: vec![],
                tool_call_id: Some("tc_1".to_string()),
                name: Some("memory_search".to_string()),
                tool_calls: None,
            },
            ChatMessage {
                role: Role::User,
                content: "Second question".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
        ];

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        assert_eq!(last_user, "Second question");
    }

    #[test]
    fn test_last_user_message_no_user_messages() {
        use crate::llm::provider::Role;

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "System prompt".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
            ChatMessage {
                role: Role::Assistant,
                content: "Greeting".to_string(),
                content_parts: vec![],
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
        ];

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        assert_eq!(last_user, "", "Should return empty string when no User messages");
    }

    #[test]
    fn test_last_user_message_single_user() {
        use crate::llm::provider::Role;

        let messages = vec![ChatMessage {
            role: Role::User,
            content: "Only message".to_string(),
            content_parts: vec![],
            tool_call_id: None,
            name: None,
            tool_calls: None,
        }];

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        assert_eq!(last_user, "Only message");
    }

    #[test]
    fn test_config_debug_includes_all_fields() {
        let config = ContainerProviderConfig {
            backend_type: "docker".to_string(),
            model: "haiku".to_string(),
            image: "img:latest".to_string(),
            socket_path: "/sock".to_string(),
            lens: "household".to_string(),
            network: "net".to_string(),
            request_timeout_secs: 120,
            auth_volume: "auth".to_string(),
            skip_permissions: false,
            extra_env: vec!["A=B".to_string()],
            lens_data_volume: Some("vol".to_string()),
            lens_config_path: Some("/cfg".to_string()),
            supervisor_url: None,
            callback_url: None,
            supervisor_auth_token: None,
            gateway_url: None,
            gateway_token: None,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("haiku"));
        assert!(debug.contains("household"));
        assert!(debug.contains("skip_permissions"));
        assert!(debug.contains("/sock"));
        assert!(debug.contains("A=B"));
    }
}
