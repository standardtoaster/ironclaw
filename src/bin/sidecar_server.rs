//! HTTP wrapper for the Claude sidecar process.
//!
//! A thin HTTP service that multiplexes sessions on a warm `claude` CLI process.
//! Designed to be run alongside IronClaw as a sidecar service.
//!
//! ## Endpoints
//!
//! - `POST /session/create` — Bootstrap a new session with history + system prompt
//! - `POST /session/{id}/message` — Send a new message to an existing session
//! - `POST /session/{id}/tool-result` — Send a tool result to an existing session
//! - `POST /session/{id}/approve` — Approve or reject a pending tool permission
//! - `DELETE /session/{id}` — End a session
//! - `GET /health` — Health check
//!
//! ## Approval flow
//!
//! When Claude requests tool approval (non-`--dangerously-skip-permissions` mode),
//! the server returns `{ "status": "approval_needed", ... }` instead of a normal
//! response. The caller must then POST to `/session/{id}/approve` with the decision,
//! and the server returns Claude's continuation response.
//!
//! ## Usage
//!
//! ```bash
//! SIDECAR_PORT=3200 SIDECAR_MODEL=sonnet cargo run --bin sidecar_server
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use ironclaw::llm::claude_sidecar::{
    ClaudeSidecarProvider, ExchangeResult, SidecarApproval, SidecarConfig,
};
use ironclaw::llm::ChatMessage;

/// Application state shared across handlers.
struct AppState {
    /// The sidecar provider managing the warm Claude process.
    provider: ClaudeSidecarProvider,
    /// Session metadata: session_id -> conversation state.
    sessions: Mutex<HashMap<Uuid, SessionState>>,
}

/// Per-session state.
struct SessionState {
    /// Accumulated messages (full conversation history).
    messages: Vec<ChatMessage>,
    /// The thread_id used for provider session tracking (same as session_id).
    thread_id: Uuid,
    /// Pending approval request, if Claude is waiting for one.
    pending_approval: Option<SidecarApproval>,
}

// --- Request/Response types ---

#[derive(Deserialize)]
struct CreateSessionRequest {
    /// Initial conversation history.
    messages: Vec<MessageInput>,
    /// Optional system prompt.
    #[serde(default)]
    system: Option<String>,
}

#[derive(Deserialize)]
struct MessageInput {
    role: String,
    content: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct CreateSessionResponse {
    session_id: Uuid,
}

#[derive(Deserialize)]
struct SendMessageRequest {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ToolResultRequest {
    tool_call_id: String,
    content: String,
}

#[derive(Deserialize)]
struct ApprovalRequest {
    request_id: String,
    approved: bool,
}

/// Unified response that can represent either a completion or an approval request.
#[derive(Serialize)]
struct SessionResponse {
    /// "complete" or "approval_needed"
    status: String,
    /// Assistant's response (when status = "complete")
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    /// Text content (when status = "complete")
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Tool calls (when status = "complete")
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCallOutput>,
    /// Token usage
    #[serde(skip_serializing_if = "Option::is_none")]
    input_tokens: Option<u32>,
    /// Token usage
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u32>,
    /// Tool requesting approval (when status = "approval_needed")
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    /// Tool parameters (when status = "approval_needed")
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
    /// Approval request ID (when status = "approval_needed")
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
}

#[derive(Serialize)]
struct ToolCallOutput {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// --- Helper to convert ExchangeResult to SessionResponse ---

fn exchange_to_response(result: ExchangeResult) -> SessionResponse {
    match result {
        ExchangeResult::Complete {
            content,
            tool_calls,
            input_tokens,
            output_tokens,
        } => SessionResponse {
            status: "complete".to_string(),
            role: Some("assistant".to_string()),
            content: Some(content),
            tool_calls: tool_calls
                .into_iter()
                .map(|tc| ToolCallOutput {
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments,
                })
                .collect(),
            input_tokens: Some(input_tokens),
            output_tokens: Some(output_tokens),
            tool_name: None,
            parameters: None,
            request_id: None,
        },
        ExchangeResult::NeedApproval(approval) => SessionResponse {
            status: "approval_needed".to_string(),
            role: None,
            content: None,
            tool_calls: Vec::new(),
            input_tokens: None,
            output_tokens: None,
            tool_name: Some(approval.tool_name),
            parameters: Some(approval.parameters),
            request_id: Some(approval.request_id),
        },
    }
}

// --- Handlers ---

async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateSessionRequest>,
) -> Result<(StatusCode, Json<CreateSessionResponse>), (StatusCode, Json<ErrorResponse>)> {
    let session_id = Uuid::new_v4();

    // Build the message list from input.
    let mut messages = Vec::new();
    if let Some(system) = body.system {
        messages.push(ChatMessage::system(system));
    }
    for msg in &body.messages {
        messages.push(input_to_chat_message(msg));
    }

    // Store session state before the LLM call.
    {
        let mut sessions = state.sessions.lock().await;
        sessions.insert(
            session_id,
            SessionState {
                messages: messages.clone(),
                thread_id: session_id,
                pending_approval: None,
            },
        );
    }

    Ok((
        StatusCode::CREATED,
        Json(CreateSessionResponse { session_id }),
    ))
}

async fn send_message(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<Uuid>,
    Json(body): Json<SendMessageRequest>,
) -> Result<Json<SessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Append new message and snapshot.
    let (messages_snapshot, thread_id) = {
        let mut sessions = state.sessions.lock().await;
        let session = sessions.get_mut(&session_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Session {} not found", session_id),
                }),
            )
        })?;

        if session.pending_approval.is_some() {
            return Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "Session has a pending approval. Call /session/{id}/approve first."
                        .to_string(),
                }),
            ));
        }

        let new_msg = match body.role.as_str() {
            "user" => ChatMessage::user(&body.content),
            "assistant" => ChatMessage::assistant(&body.content),
            _ => ChatMessage::user(&body.content),
        };
        session.messages.push(new_msg);
        (session.messages.clone(), session.thread_id)
    };

    // Exchange via the provider's session-aware path.
    let result = state
        .provider
        .session_exchange(&messages_snapshot, Some(thread_id))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("LLM error: {}", e),
                }),
            )
        })?;

    let response = exchange_to_response(result);

    // Update session state based on the result.
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&session_id) {
        if response.status == "complete" {
            if let Some(ref content) = response.content {
                session.messages.push(ChatMessage::assistant(content));
            }
        } else if response.status == "approval_needed" {
            session.pending_approval = Some(SidecarApproval {
                request_id: response.request_id.clone().unwrap_or_default(),
                tool_name: response.tool_name.clone().unwrap_or_default(),
                parameters: response.parameters.clone().unwrap_or(serde_json::Value::Null),
            });
        }
    }

    Ok(Json(response))
}

async fn send_tool_result(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<Uuid>,
    Json(body): Json<ToolResultRequest>,
) -> Result<Json<SessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (messages_snapshot, thread_id) = {
        let mut sessions = state.sessions.lock().await;
        let session = sessions.get_mut(&session_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Session {} not found", session_id),
                }),
            )
        })?;

        session.messages.push(ChatMessage::tool_result(
            &body.tool_call_id,
            "tool",
            &body.content,
        ));
        (session.messages.clone(), session.thread_id)
    };

    let result = state
        .provider
        .session_exchange(&messages_snapshot, Some(thread_id))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("LLM error: {}", e),
                }),
            )
        })?;

    let response = exchange_to_response(result);

    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&session_id) {
        if response.status == "complete" {
            if let Some(ref content) = response.content {
                session.messages.push(ChatMessage::assistant(content));
            }
        } else if response.status == "approval_needed" {
            session.pending_approval = Some(SidecarApproval {
                request_id: response.request_id.clone().unwrap_or_default(),
                tool_name: response.tool_name.clone().unwrap_or_default(),
                parameters: response.parameters.clone().unwrap_or(serde_json::Value::Null),
            });
        }
    }

    Ok(Json(response))
}

async fn approve_tool(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<Uuid>,
    Json(body): Json<ApprovalRequest>,
) -> Result<Json<SessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Check that there's a pending approval.
    {
        let mut sessions = state.sessions.lock().await;
        let session = sessions.get_mut(&session_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Session {} not found", session_id),
                }),
            )
        })?;

        let pending = session.pending_approval.take().ok_or_else(|| {
            (
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "No pending approval for this session".to_string(),
                }),
            )
        })?;

        if pending.request_id != body.request_id {
            // Put it back if the request_id doesn't match.
            session.pending_approval = Some(pending);
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "request_id does not match pending approval".to_string(),
                }),
            ));
        }
    }

    // Send the approval decision to Claude.
    state
        .provider
        .approve(&body.request_id, body.approved)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Failed to send approval: {}", e),
                }),
            )
        })?;

    // Continue reading Claude's response.
    let result = state
        .provider
        .continue_after_approval()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("LLM error after approval: {}", e),
                }),
            )
        })?;

    let response = exchange_to_response(result);

    // Update session state.
    let mut sessions = state.sessions.lock().await;
    if let Some(session) = sessions.get_mut(&session_id) {
        if response.status == "complete" {
            if let Some(ref content) = response.content {
                session.messages.push(ChatMessage::assistant(content));
            }
        } else if response.status == "approval_needed" {
            session.pending_approval = Some(SidecarApproval {
                request_id: response.request_id.clone().unwrap_or_default(),
                tool_name: response.tool_name.clone().unwrap_or_default(),
                parameters: response.parameters.clone().unwrap_or(serde_json::Value::Null),
            });
        }
    }

    Ok(Json(response))
}

async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<Uuid>,
) -> StatusCode {
    let mut sessions = state.sessions.lock().await;
    if sessions.remove(&session_id).is_some() {
        // Clean up the provider's session tracking.
        state.provider.end_session(session_id).await;
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "sidecar-server",
    }))
}

fn input_to_chat_message(input: &MessageInput) -> ChatMessage {
    match input.role.as_str() {
        "system" => ChatMessage::system(&input.content),
        "user" => ChatMessage::user(&input.content),
        "assistant" => ChatMessage::assistant(&input.content),
        "tool" => ChatMessage::tool_result(
            input.tool_call_id.as_deref().unwrap_or("unknown"),
            input.name.as_deref().unwrap_or("tool"),
            &input.content,
        ),
        _ => ChatMessage::user(&input.content),
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let port: u16 = std::env::var("SIDECAR_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3200);

    let model = std::env::var("SIDECAR_MODEL").unwrap_or_else(|_| "sonnet".to_string());
    let binary = std::env::var("CLAUDE_BINARY").unwrap_or_else(|_| "claude".to_string());
    let skip_perms = std::env::var("SIDECAR_SKIP_PERMISSIONS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let config = SidecarConfig {
        model: model.clone(),
        system_prompt_append: std::env::var("SIDECAR_SYSTEM_PROMPT").ok(),
        mcp_config_path: std::env::var("SIDECAR_MCP_CONFIG").ok(),
        working_dir: std::env::var("SIDECAR_WORKING_DIR").ok(),
        claude_binary: binary,
        spawn_timeout_secs: 30,
        request_timeout_secs: std::env::var("SIDECAR_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
        skip_permissions: skip_perms,
    };

    let provider = ClaudeSidecarProvider::new(config);

    let state = Arc::new(AppState {
        provider,
        sessions: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/session/create", post(create_session))
        .route("/session/{id}/message", post(send_message))
        .route("/session/{id}/tool-result", post(send_tool_result))
        .route("/session/{id}/approve", post(approve_tool))
        .route("/session/{id}", delete(delete_session))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    tracing::info!(
        port,
        model = %model,
        skip_permissions = skip_perms,
        "Sidecar HTTP server starting"
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to {}: {}", addr, e));
    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| panic!("Server error: {}", e));
}
