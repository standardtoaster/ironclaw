//! Event ingest handler for structured collections.
//!
//! Accepts external events (webhooks, automations, etc.) and inserts them
//! as records into structured collections with provenance tracking via `_lineage`.

use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::channels::web::auth::AuthenticatedUser;
use crate::channels::web::server::GatewayState;

/// Request body for the event ingest endpoint.
#[derive(Debug, Deserialize)]
pub struct EventIngestRequest {
    /// Source of the event (e.g. "webhook", "home_assistant", "whatsapp").
    pub source: String,
    /// Target collection name.
    pub collection: String,
    /// Record data (validated against collection schema).
    pub data: serde_json::Value,
    /// Optional ISO 8601 timestamp. Defaults to server time if omitted.
    pub timestamp: Option<String>,
    /// Optional human-readable origin context (e.g. "WhatsApp message from Andrew").
    pub context: Option<String>,
}

/// Response body for the event ingest endpoint.
#[derive(Debug, Serialize)]
pub struct EventIngestResponse {
    pub status: String,
    pub record_id: String,
    pub collection: String,
}

/// POST /api/events/ingest
///
/// Ingest an external event into a structured collection. Injects `_lineage`
/// system field for provenance tracking.
pub async fn events_ingest_handler(
    State(state): State<Arc<GatewayState>>,
    AuthenticatedUser(user): AuthenticatedUser,
    Json(req): Json<EventIngestRequest>,
) -> impl IntoResponse {
    let db = match &state.store {
        Some(db) => Arc::clone(db),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "database not available"})),
            )
                .into_response();
        }
    };

    let event_id = Uuid::new_v4();
    let timestamp = req
        .timestamp
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

    // Inject _lineage system field into the data.
    let mut data = req.data.clone();
    match data.as_object_mut() {
        Some(obj) => {
            obj.insert(
                "_lineage".to_string(),
                serde_json::json!({
                    "source": req.source,
                    "source_id": event_id.to_string(),
                    "created_by": req.source,
                    "context": req.context,
                    "timestamp": timestamp
                }),
            );
        }
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "event data must be a JSON object"})),
            )
                .into_response();
        }
    }

    // Clone data before insert (insert_record takes ownership).
    let data_for_event = data.clone();

    match db
        .insert_record(&user.user_id, &req.collection, data)
        .await
    {
        Ok(id) => {
            // Fire collection write triggers
            if let Some(tx) = &state.collection_write_tx {
                let _ = tx.send(crate::agent::collection_events::CollectionWriteEvent {
                    user_id: user.user_id.clone(),
                    collection: req.collection.clone(),
                    record_id: id,
                    data: data_for_event,
                });
            }

            (
                StatusCode::CREATED,
                Json(serde_json::json!(EventIngestResponse {
                    status: "created".to_string(),
                    record_id: id.to_string(),
                    collection: req.collection,
                })),
            )
                .into_response()
        }
        Err(e) => {
            let status = if e.to_string().contains("NotFound") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, Json(serde_json::json!({"error": e.to_string()}))).into_response()
        }
    }
}
