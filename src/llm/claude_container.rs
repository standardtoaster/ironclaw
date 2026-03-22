//! Claude Code container provider.
//!
//! Implements `LlmProvider` backed by the `ContainerPool` from `container_pool.rs`.
//! One container per conversation (thread_id), multiple containers per lens.
//! Containers run Claude Code CLI in interactive streaming mode and communicate
//! via NDJSON over stdin/stdout.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rust_decimal::Decimal;
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::llm::claude_protocol::{self, ExchangeResult};
use crate::llm::container_pool::{ContainerPool, ContainerPoolConfig};
use crate::llm::costs;
use crate::llm::error::LlmError;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    ToolCall, ToolCompletionRequest, ToolCompletionResponse,
};

/// Configuration for the container-based Claude provider.
#[derive(Debug, Clone)]
pub struct ContainerProviderConfig {
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
}

/// An LLM provider that manages Claude Code containers via bollard.
///
/// One container per conversation thread. Containers are created on demand
/// via `ContainerPool` and kept warm for the duration of the conversation.
/// The pool connects lazily on first use to avoid blocking startup when
/// Docker isn't available.
pub struct ClaudeContainerProvider {
    config: ContainerProviderConfig,
    pool: OnceCell<Arc<ContainerPool>>,
    model_label: String,
}

impl ClaudeContainerProvider {
    /// Create a new container provider with the given config.
    ///
    /// The Docker connection is deferred until the first `complete()` call,
    /// so construction never fails. This matches the lazy-init pattern used
    /// by `ClaudeSidecarProvider`.
    pub fn new(config: ContainerProviderConfig) -> Self {
        let model_label = format!("claude-container/{}", config.model);
        Self {
            config,
            pool: OnceCell::new(),
            model_label,
        }
    }

    /// Get or initialize the container pool.
    async fn pool(&self) -> Result<&Arc<ContainerPool>, LlmError> {
        self.pool
            .get_or_try_init(|| async {
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
                let pool =
                    ContainerPool::new(&self.config.socket_path, pool_config).await?;
                Ok(Arc::new(pool))
            })
            .await
    }

    /// End the session for a specific thread (e.g., on de-escalation).
    ///
    /// Removes the container and cleans up resources.
    pub async fn end_session_cleanup(&self, thread_id: Uuid) {
        if let Some(pool) = self.pool.get()
            && let Err(e) = pool.remove_session(thread_id).await
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
        if let Some(pool) = self.pool.get()
            && let Err(e) = pool.shutdown_all().await
        {
            tracing::warn!(error = %e, "Failed to shutdown container pool");
        }
    }

    /// Get the number of active container sessions.
    pub async fn session_count(&self) -> usize {
        match self.pool.get() {
            Some(pool) => pool.session_count().await,
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
        let pool = self.pool().await?;
        let tid = thread_id.unwrap_or_else(Uuid::new_v4);

        // Ensure container exists for this thread.
        pool.get_or_create(tid).await?;

        // Compute delta prompt.
        let messages_sent = pool.messages_sent(&tid).await.unwrap_or(0);
        let (prompt, is_continuation) = claude_protocol::resolve_delta(messages, messages_sent);

        if is_continuation {
            let delta_count = messages.len().saturating_sub(messages_sent);
            tracing::debug!(
                thread_id = %tid,
                delta_messages = delta_count,
                "Sending delta messages to container session"
            );
        } else {
            tracing::debug!(
                thread_id = %tid,
                total_messages = messages.len(),
                "Sending full history to container session"
            );
        }

        // Exchange with the container.
        let result = pool.exchange(tid, &prompt).await?;

        match result {
            ExchangeResult::Complete {
                content,
                tool_calls,
                input_tokens,
                output_tokens,
            } => {
                // Update messages_sent counter.
                pool.set_messages_sent(&tid, messages.len()).await;
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

}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    fn test_config() -> ContainerProviderConfig {
        ContainerProviderConfig {
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

}
