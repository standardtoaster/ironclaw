//! Handler for bulk conversation ingestion.
//!
//! POST /api/conversations/ingest — accepts a list of messages and stores them
//! as a conversation in the `conversation_messages` table. Designed for passive
//! ingestion (e.g., a WhatsApp bridge listener storing overheard conversations).

use std::sync::Arc;

use axum::{Json, extract::State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::channels::web::auth::AuthenticatedUser;
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
    AuthenticatedUser(user): AuthenticatedUser,
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
        .create_conversation("ingest", &user.user_id, thread_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_request_deserializes_with_title() {
        let json = r#"{"title":"Team standup","messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}"#;
        let req: IngestConversationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.title.as_deref(), Some("Team standup"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hello");
        assert_eq!(req.messages[1].role, "assistant");
    }

    #[test]
    fn test_ingest_request_deserializes_without_title() {
        let json = r#"{"messages":[{"role":"user","content":"test"}]}"#;
        let req: IngestConversationRequest = serde_json::from_str(json).unwrap();
        assert!(req.title.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_ingest_request_deserializes_empty_messages() {
        let json = r#"{"messages":[]}"#;
        let req: IngestConversationRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
    }

    #[test]
    fn test_ingest_request_rejects_missing_messages() {
        let json = r#"{"title":"oops"}"#;
        let result: Result<IngestConversationRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_request_rejects_missing_role() {
        let json = r#"{"messages":[{"content":"no role"}]}"#;
        let result: Result<IngestConversationRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_request_rejects_missing_content() {
        let json = r#"{"messages":[{"role":"user"}]}"#;
        let result: Result<IngestConversationRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_response_serializes() {
        let resp = IngestConversationResponse {
            conversation_id: "abc-123".to_string(),
            message_count: 5,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["conversation_id"], "abc-123");
        assert_eq!(json["message_count"], 5);
    }

    #[test]
    fn test_ingest_message_accepts_any_role() {
        // The handler doesn't validate role values — any string is accepted.
        // This is intentional: passive ingestion stores raw conversation data.
        for role in &["user", "assistant", "system", "narrator", "bot", ""] {
            let json = format!(r#"{{"role":"{}","content":"test"}}"#, role);
            let msg: IngestMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg.role, *role);
        }
    }

    #[test]
    fn test_ingest_request_preserves_unicode_content() {
        let json = r#"{"messages":[{"role":"user","content":"こんにちは 🎉 café"}]}"#;
        let req: IngestConversationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages[0].content, "こんにちは 🎉 café");
    }

    #[test]
    fn test_ingest_request_many_messages() {
        let msgs: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"role":"user","content":"msg {}"}}"#, i))
            .collect();
        let json = format!(r#"{{"messages":[{}]}}"#, msgs.join(","));
        let req: IngestConversationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req.messages.len(), 100);
        assert_eq!(req.messages[99].content, "msg 99");
    }
}
