//! Handler for bulk conversation ingestion.
//!
//! POST /api/conversations/ingest — accepts a list of messages and stores them
//! as a conversation in the `conversation_messages` table. Designed for passive
//! ingestion (e.g., a WhatsApp bridge listener storing overheard conversations).

use std::sync::Arc;

use axum::{Json, extract::State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::channels::web::server::GatewayState;

#[derive(Debug, Deserialize)]
pub struct IngestMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct IngestConversationRequest {
    /// Optional title for the conversation.
    pub title: Option<String>,
    /// Messages to ingest.
    pub messages: Vec<IngestMessage>,
}

#[derive(Debug, Serialize)]
pub struct IngestConversationResponse {
    pub conversation_id: String,
    pub message_count: usize,
}

/// POST /api/conversations/ingest
///
/// Creates a new conversation and adds all provided messages to it.
/// Uses the gateway's configured user ID for scoping.
pub async fn ingest_conversation_handler(
    State(state): State<Arc<GatewayState>>,
    Json(req): Json<IngestConversationRequest>,
) -> Result<Json<IngestConversationResponse>, (StatusCode, String)> {
    let db = state.store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Database not available".to_string(),
    ))?;

    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "messages array must not be empty".to_string(),
        ));
    }

    // Create the conversation. Use "ingest" as the channel to distinguish
    // passive ingestion from interactive chat.
    let thread_id = req.title.as_deref();
    let conversation_id = db
        .create_conversation("ingest", &state.default_user_id, thread_id)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create conversation: {e}"),
            )
        })?;

    // Add each message to the conversation.
    let mut count = 0usize;
    for msg in &req.messages {
        db.add_conversation_message(conversation_id, &msg.role, &msg.content)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to add message {count}: {e}"),
                )
            })?;
        count += 1;
    }

    Ok(Json(IngestConversationResponse {
        conversation_id: conversation_id.to_string(),
        message_count: count,
    }))
}
