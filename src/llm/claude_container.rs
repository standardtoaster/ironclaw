//! Claude Code container provider.
//!
//! Implements `LlmProvider` backed by the `ContainerPool` from `container_pool.rs`.
//! One container per conversation (thread_id), multiple containers per lens.
//! Containers run Claude Code CLI with a channel MCP server and communicate
//! via HTTP callbacks.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use rust_decimal::Decimal;
use serde::Deserialize;
use tokio::sync::{OnceCell, oneshot};
use uuid::Uuid;

use crate::channels::web::sse::SseManager;
use crate::channels::web::types::SseEvent;
use crate::llm::container_pool::{
    ChannelReply, ContainerPool, ContainerPoolConfig, PendingApproval, PendingQuestion,
};
use crate::llm::costs;
use crate::llm::error::LlmError;
use crate::llm::provider::{
    CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    ToolCompletionRequest, ToolCompletionResponse,
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
    /// Host/IP for the callback URL that containers POST results back to.
    pub callback_host: Option<String>,
    /// Port for the callback URL.
    pub callback_port: Option<u16>,
    /// Auth token for callback authentication.
    pub auth_token: Option<String>,
}

/// An LLM provider that manages Claude Code containers via bollard.
///
/// One container per conversation thread. Containers are created on demand
/// via `ContainerPool` and kept warm for the duration of the conversation.
/// The pool connects lazily on first use to avoid blocking startup when
/// Docker isn't available.
///
/// Communication is asynchronous: messages are sent to containers via HTTP,
/// and replies are received via HTTP callbacks to IronClaw's web gateway.
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

    /// Create a new container provider with a pre-initialized pool.
    ///
    /// Use this when the pool needs to be shared (e.g., with callback routes
    /// registered on the web gateway).
    pub fn new_with_pool(config: ContainerProviderConfig, pool: Arc<ContainerPool>) -> Self {
        let model_label = format!("claude-container/{}", config.model);
        let cell = OnceCell::new();
        // OnceCell::set cannot fail here since the cell is freshly created.
        let _ = cell.set(pool);
        Self {
            config,
            pool: cell,
            model_label,
        }
    }

    /// Get the pool if it has been initialized.
    ///
    /// Returns `None` if no request has been made yet (lazy init not triggered).
    /// Use this to obtain the pool for building callback routes.
    pub fn initialized_pool(&self) -> Option<&Arc<ContainerPool>> {
        self.pool.get()
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
                    callback_host: self.config.callback_host.clone(),
                    callback_port: self.config.callback_port,
                    auth_token: self.config.auth_token.clone(),
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

    /// Build callback routes for the Claude container provider.
    ///
    /// These routes are registered with the web gateway to receive HTTP callbacks
    /// from containers (reply, approval, user-input).
    pub fn callback_routes(pool: Arc<ContainerPool>, sse: Arc<SseManager>) -> Router {
        let state = ChannelCallbackState { pool, sse };
        Router::new()
            .route("/api/claude/reply", post(handle_claude_reply))
            .route("/api/claude/approval", post(handle_claude_approval))
            .route("/api/claude/user-input", post(handle_claude_user_input))
            .route(
                "/api/claude/approval-response",
                post(handle_claude_approval_response),
            )
            .route(
                "/api/claude/user-input-response",
                post(handle_claude_user_input_response),
            )
            .with_state(state)
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
        let thread_id = thread_id_from_metadata(&request.metadata).ok_or_else(|| {
            LlmError::RequestFailed {
                provider: "claude_container".into(),
                reason: "missing thread_id in metadata".into(),
            }
        })?;

        let pool = self.pool().await?;
        let session = pool.get_or_create(thread_id).await?;

        // Extract latest user message.
        let content = request
            .messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Guard against concurrent calls for the same thread.
        if pool.pending_replies.contains_key(&thread_id) {
            return Err(LlmError::RequestFailed {
                provider: "claude_container".into(),
                reason: "concurrent call in flight for this thread".into(),
            });
        }

        // Set up oneshot for reply reception.
        let (tx, rx) = oneshot::channel::<ChannelReply>();
        pool.pending_replies.insert(thread_id, tx);

        // Send message to container.
        tracing::info!(thread_id = %thread_id, ip = %session.container_ip, content_len = content.len(), "Sending message to container");
        if let Err(e) = pool.send_message(&session, &thread_id, &content).await {
            pool.pending_replies.remove(&thread_id);
            return Err(e);
        }

        pool.update_messages_sent(&thread_id).await;

        // Wait for reply with timeout.
        let timeout = Duration::from_secs(self.config.request_timeout_secs);
        tracing::info!(thread_id = %thread_id, timeout_secs = self.config.request_timeout_secs, "Awaiting oneshot reply");
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(reply)) => {
                tracing::info!(thread_id = %thread_id, content_len = reply.content.len(), "Oneshot resolved with reply");
                if let Some(sid) = reply.session_id {
                    pool.update_session_id(&thread_id, sid).await;
                }
                Ok(CompletionResponse {
                    content: reply.content,
                    input_tokens: reply.input_tokens,
                    output_tokens: reply.output_tokens,
                    finish_reason: FinishReason::Stop,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                })
            }
            Ok(Err(_)) => {
                tracing::warn!(thread_id = %thread_id, "Oneshot sender dropped");
                pool.pending_replies.remove(&thread_id);
                Err(LlmError::RequestFailed {
                    provider: "claude_container".into(),
                    reason: "reply channel closed unexpectedly".into(),
                })
            }
            Err(_) => {
                tracing::warn!(thread_id = %thread_id, timeout_secs = self.config.request_timeout_secs, "Oneshot timed out");
                pool.pending_replies.remove(&thread_id);
                Err(LlmError::RequestFailed {
                    provider: "claude_container".into(),
                    reason: format!(
                        "no reply within {}s",
                        self.config.request_timeout_secs
                    ),
                })
            }
        }
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        // Build a CompletionRequest from the tool request and delegate.
        let completion_request = CompletionRequest {
            messages: request.messages,
            metadata: request.metadata,
            model: request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop_sequences: None,
        };

        let response = self.complete(completion_request).await?;

        Ok(ToolCompletionResponse {
            content: Some(response.content),
            tool_calls: vec![],
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
            finish_reason: response.finish_reason,
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
}

// ---------------------------------------------------------------------------
// Callback route types and handlers
// ---------------------------------------------------------------------------

/// Shared state for callback route handlers.
#[derive(Clone)]
pub struct ChannelCallbackState {
    pub pool: Arc<ContainerPool>,
    pub sse: Arc<SseManager>,
}

/// Payload received when a container finishes processing a message.
#[derive(Deserialize)]
struct ReplyPayload {
    thread_id: String,
    content: String,
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    session_id: Option<String>,
}

/// Payload received when a container needs tool approval.
#[derive(Deserialize)]
struct ApprovalPayload {
    request_id: String,
    tool_name: String,
    tool_input: serde_json::Value,
    thread_id: Option<String>,
}

/// Payload received when a container's `percy_ask_user` tool fires.
#[derive(Deserialize)]
struct UserInputPayload {
    request_id: String,
    question: String,
    options: Option<Vec<String>>,
    thread_id: Option<String>,
}

/// Payload received when the UI responds to an approval request.
#[derive(Deserialize)]
struct ApprovalResponsePayload {
    request_id: String,
    decision: String,
}

/// Payload received when the UI responds to a user-input request.
#[derive(Deserialize)]
struct UserInputResponsePayload {
    request_id: String,
    answer: String,
}

/// Handle a reply callback from a container.
///
/// Looks up the pending oneshot channel by thread_id UUID and sends the reply.
async fn handle_claude_reply(
    State(state): State<ChannelCallbackState>,
    Json(payload): Json<ReplyPayload>,
) -> axum::http::StatusCode {
    let thread_id = match Uuid::parse_str(&payload.thread_id) {
        Ok(id) => id,
        Err(_) => {
            tracing::warn!(
                thread_id = %payload.thread_id,
                "Invalid thread_id in reply callback"
            );
            return axum::http::StatusCode::BAD_REQUEST;
        }
    };

    let reply = ChannelReply {
        content: payload.content,
        input_tokens: payload.input_tokens,
        output_tokens: payload.output_tokens,
        session_id: payload.session_id,
    };

    if let Some((_, tx)) = state.pool.pending_replies.remove(&thread_id) {
        if tx.send(reply).is_err() {
            tracing::warn!(
                thread_id = %thread_id,
                "Reply receiver dropped before send"
            );
        }
        axum::http::StatusCode::OK
    } else {
        tracing::warn!(
            thread_id = %thread_id,
            "No pending reply channel for thread"
        );
        axum::http::StatusCode::NOT_FOUND
    }
}

/// Handle an approval request callback from a container.
///
/// Stores the pending approval and broadcasts an SSE event to the UI.
async fn handle_claude_approval(
    State(state): State<ChannelCallbackState>,
    Json(payload): Json<ApprovalPayload>,
) -> axum::http::StatusCode {
    let thread_id = payload.thread_id.as_deref().and_then(|s| Uuid::parse_str(s).ok());

    // Look up the container session to record IP/port for forwarding the response.
    let (container_ip, container_port) = if let Some(tid) = thread_id {
        match state.pool.session_by_thread(&tid).await {
            Some(session) => (session.container_ip, session.channel_port),
            None => {
                tracing::warn!(
                    request_id = %payload.request_id,
                    "No session found for approval request"
                );
                return axum::http::StatusCode::NOT_FOUND;
            }
        }
    } else {
        tracing::warn!(
            request_id = %payload.request_id,
            "Approval request missing thread_id"
        );
        return axum::http::StatusCode::BAD_REQUEST;
    };

    // Store pending approval for later resolution.
    state.pool.pending_approvals.insert(
        payload.request_id.clone(),
        PendingApproval {
            container_ip,
            container_port,
            thread_id: thread_id.unwrap_or_default(),
        },
    );

    // Broadcast SSE event to the UI.
    let description = format!("Tool '{}' requires approval", payload.tool_name);
    let parameters = serde_json::to_string(&payload.tool_input).unwrap_or_default();
    state.sse.broadcast(SseEvent::ApprovalNeeded {
        request_id: payload.request_id,
        tool_name: payload.tool_name,
        description,
        parameters,
        thread_id: payload.thread_id,
        allow_always: true,
    });

    axum::http::StatusCode::OK
}

/// Handle a user-input request callback from a container.
///
/// Stores the pending question and broadcasts an SSE event to the UI.
async fn handle_claude_user_input(
    State(state): State<ChannelCallbackState>,
    Json(payload): Json<UserInputPayload>,
) -> axum::http::StatusCode {
    let thread_id = payload.thread_id.as_deref().and_then(|s| Uuid::parse_str(s).ok());

    let (container_ip, container_port) = if let Some(tid) = thread_id {
        match state.pool.session_by_thread(&tid).await {
            Some(session) => (session.container_ip, session.channel_port),
            None => {
                tracing::warn!(
                    request_id = %payload.request_id,
                    "No session found for user-input request"
                );
                return axum::http::StatusCode::NOT_FOUND;
            }
        }
    } else {
        tracing::warn!(
            request_id = %payload.request_id,
            "User-input request missing thread_id"
        );
        return axum::http::StatusCode::BAD_REQUEST;
    };

    state.pool.pending_questions.insert(
        payload.request_id.clone(),
        PendingQuestion {
            container_ip,
            container_port,
            thread_id: thread_id.unwrap_or_default(),
        },
    );

    state.sse.broadcast(SseEvent::UserInputNeeded {
        request_id: payload.request_id,
        question: payload.question,
        options: payload.options,
        metadata: None,
        thread_id: payload.thread_id,
    });

    axum::http::StatusCode::OK
}

/// Forward an approval decision from the UI to the container.
async fn handle_claude_approval_response(
    State(state): State<ChannelCallbackState>,
    Json(payload): Json<ApprovalResponsePayload>,
) -> axum::http::StatusCode {
    let approval = match state.pool.pending_approvals.remove(&payload.request_id) {
        Some((_, a)) => a,
        None => {
            tracing::warn!(
                request_id = %payload.request_id,
                "No pending approval for request"
            );
            return axum::http::StatusCode::NOT_FOUND;
        }
    };

    // Forward the decision to the container's approval-response endpoint.
    let url = format!(
        "http://{}:{}/approval-response",
        approval.container_ip, approval.container_port
    );

    let body = serde_json::json!({
        "request_id": payload.request_id,
        "decision": payload.decision,
    });

    let client = reqwest::Client::new();
    match client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => axum::http::StatusCode::OK,
        Ok(resp) => {
            tracing::warn!(
                request_id = %payload.request_id,
                status = %resp.status(),
                "Container approval-response returned error"
            );
            axum::http::StatusCode::BAD_GATEWAY
        }
        Err(e) => {
            tracing::warn!(
                request_id = %payload.request_id,
                error = %e,
                "Failed to forward approval response to container"
            );
            axum::http::StatusCode::BAD_GATEWAY
        }
    }
}

/// Forward a user-input answer from the UI to the container.
async fn handle_claude_user_input_response(
    State(state): State<ChannelCallbackState>,
    Json(payload): Json<UserInputResponsePayload>,
) -> axum::http::StatusCode {
    let question = match state.pool.pending_questions.remove(&payload.request_id) {
        Some((_, q)) => q,
        None => {
            tracing::warn!(
                request_id = %payload.request_id,
                "No pending question for request"
            );
            return axum::http::StatusCode::NOT_FOUND;
        }
    };

    // Forward the answer to the container's ask-response endpoint.
    let url = format!(
        "http://{}:{}/ask-response",
        question.container_ip, question.container_port
    );

    let body = serde_json::json!({
        "request_id": payload.request_id,
        "answer": payload.answer,
    });

    let client = reqwest::Client::new();
    match client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => axum::http::StatusCode::OK,
        Ok(resp) => {
            tracing::warn!(
                request_id = %payload.request_id,
                status = %resp.status(),
                "Container ask-response returned error"
            );
            axum::http::StatusCode::BAD_GATEWAY
        }
        Err(e) => {
            tracing::warn!(
                request_id = %payload.request_id,
                error = %e,
                "Failed to forward user-input response to container"
            );
            axum::http::StatusCode::BAD_GATEWAY
        }
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
            callback_host: Some("192.168.1.100".to_string()),
            callback_port: Some(3001),
            auth_token: Some("test-token".to_string()),
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
        assert_eq!(
            cloned.callback_host,
            Some("192.168.1.100".to_string())
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

    #[test]
    fn test_claude_container_backend_parsing() {
        use crate::config::LlmBackend;

        let backend: LlmBackend = "claude_container".parse().expect("parse claude_container");
        assert_eq!(backend, LlmBackend::ClaudeContainer);

        let backend: LlmBackend = "claude-container".parse().expect("parse claude-container");
        assert_eq!(backend, LlmBackend::ClaudeContainer);

        let backend: LlmBackend = "container".parse().expect("parse container");
        assert_eq!(backend, LlmBackend::ClaudeContainer);
    }

    #[test]
    fn test_complete_requires_thread_id() {
        // Verify that thread_id_from_metadata returns None for empty metadata,
        // which would cause complete() to return an error.
        let meta = HashMap::new();
        assert!(thread_id_from_metadata(&meta).is_none());
    }
}
