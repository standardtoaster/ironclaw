//! Workspace API handlers.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use crate::channels::web::server::GatewayState;

#[derive(Debug, Deserialize)]
pub struct WorkspaceListParams {
    pub status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorkspaceInfoResponse {
    pub id: String,
    pub topic: String,
    pub summary: Option<String>,
    pub status: String,
    pub turn_count: i32,
    pub last_accessed: String,
    pub conversation_id: String,
}

#[derive(Debug, Serialize)]
pub struct WorkspaceListResponse {
    pub workspaces: Vec<WorkspaceInfoResponse>,
}

pub async fn workspaces_list_handler(
    State(state): State<Arc<GatewayState>>,
    Query(params): Query<WorkspaceListParams>,
) -> Result<Json<WorkspaceListResponse>, (StatusCode, String)> {
    let db = state.store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Database not available".to_string(),
    ))?;

    let workspaces = db
        .list_agent_workspaces(&state.user_id, params.status.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<WorkspaceInfoResponse> = workspaces
        .into_iter()
        .map(|ws| WorkspaceInfoResponse {
            id: ws.id.to_string(),
            topic: ws.topic,
            summary: ws.summary,
            status: ws.status,
            turn_count: ws.turn_count,
            last_accessed: ws.last_accessed.to_rfc3339(),
            conversation_id: ws.conversation_id.to_string(),
        })
        .collect();

    Ok(Json(WorkspaceListResponse { workspaces: items }))
}

pub async fn workspaces_detail_handler(
    State(state): State<Arc<GatewayState>>,
    Path(id): Path<String>,
) -> Result<Json<WorkspaceInfoResponse>, (StatusCode, String)> {
    let _ = &state;
    let db = state.store.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Database not available".to_string(),
    ))?;

    let ws_id: uuid::Uuid = id.parse().map_err(|_| {
        (StatusCode::BAD_REQUEST, "Invalid workspace ID".to_string())
    })?;

    let ws = db
        .get_agent_workspace(ws_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Workspace not found".to_string()))?;

    Ok(Json(WorkspaceInfoResponse {
        id: ws.id.to_string(),
        topic: ws.topic,
        summary: ws.summary,
        status: ws.status,
        turn_count: ws.turn_count,
        last_accessed: ws.last_accessed.to_rfc3339(),
        conversation_id: ws.conversation_id.to_string(),
    }))
}
