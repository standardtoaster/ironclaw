//! Set-workspace-topic tool.
//!
//! Lets the workspace LLM label its workspace with a short topic string
//! and a corresponding embedding for future routing.

use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};
use crate::workspace::EmbeddingProvider;

/// Tool for setting or updating a workspace's topic.
///
/// Reads the `workspace_id` from job metadata (set by `delegate_to_workspace`
/// when dispatching), embeds the topic text, and persists both the topic
/// string and its embedding for future similarity-based routing.
pub struct SetWorkspaceTopicTool {
    db: Arc<dyn Database>,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl SetWorkspaceTopicTool {
    pub fn new(db: Arc<dyn Database>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { db, embedder }
    }
}

#[async_trait]
impl Tool for SetWorkspaceTopicTool {
    fn name(&self) -> &str {
        "set_workspace_topic"
    }

    fn description(&self) -> &str {
        "Set or update this workspace's topic. Call this after the first turn to label \
         what this workspace is about. Keep it concise (2-5 words)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "A short topic label for this workspace (2-5 words)"
                }
            },
            "required": ["topic"]
        })
    }

    fn requires_sanitization(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let topic = require_str(&params, "topic")?;

        // Read workspace_id from job metadata (set by delegate_to_workspace)
        let ws_id_str = ctx
            .metadata
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::ExecutionFailed("not in a workspace context".to_string())
            })?;

        let ws_id = Uuid::parse_str(ws_id_str).map_err(|e| {
            ToolError::ExecutionFailed(format!("invalid workspace_id in metadata: {e}"))
        })?;

        // Embed the topic text
        let embedding = self.embedder.embed(topic).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to embed topic: {e}"))
        })?;

        // Persist topic + embedding
        self.db
            .update_agent_workspace_topic(ws_id, topic, &embedding)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to update workspace topic: {e}"))
            })?;

        Ok(ToolOutput::text(
            format!("Workspace topic set to: {topic}"),
            start.elapsed(),
        ))
    }
}
