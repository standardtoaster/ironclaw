//! Lightweight workspace creation tool.
//!
//! Creates a new workspace with an embedded topic, WITHOUT spawning a worker
//! sub-agent. The main agent continues handling the conversation directly.
//! Future messages about the same topic auto-route via embedding similarity.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::agent::workspace_router::WorkspaceRouter;
use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

/// Tool for creating a new workspace without delegation.
///
/// The LLM calls this when it recognizes a new topic that will accumulate
/// context. It creates the workspace + conversation + embeds the topic,
/// then returns immediately. The main agent continues responding in the
/// current conversation. Future messages about this topic will auto-route
/// to the new workspace via embedding similarity.
pub struct CreateWorkspaceTool {
    router: Arc<WorkspaceRouter>,
    db: Arc<dyn Database>,
}

impl CreateWorkspaceTool {
    pub fn new(router: Arc<WorkspaceRouter>, db: Arc<dyn Database>) -> Self {
        Self { router, db }
    }
}

#[async_trait]
impl Tool for CreateWorkspaceTool {
    fn name(&self) -> &str {
        "create_workspace"
    }

    fn description(&self) -> &str {
        "Create a new topic workspace for organizing conversation context. Use this when \
         you recognize a substantive topic that will accumulate context over multiple \
         conversations (e.g. home automation setup, financial planning, vacation planning). \
         The workspace will automatically capture future messages about this topic. \
         Do NOT create workspaces for one-off questions."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Short topic label for the workspace (e.g. 'home assistant setup', 'US expat taxes', 'nanny schedule')"
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer description of what this workspace covers"
                }
            },
            "required": ["topic"]
        })
    }

    fn execution_timeout(&self) -> Duration {
        Duration::from_secs(30)
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
        let description = params.get("description").and_then(|v| v.as_str());
        let user_id = &ctx.user_id;

        // Check if a workspace with a similar topic already exists
        let topic_for_embed = if let Some(desc) = description {
            format!("{topic}: {desc}")
        } else {
            topic.to_string()
        };
        let topic_for_embed = &topic_for_embed[..topic_for_embed.len().min(500)];

        if let Ok(Some(existing)) = self
            .router
            .route_with_hint(user_id, topic_for_embed, Some(topic))
            .await
        {
            return Ok(ToolOutput::text(
                format!(
                    "Workspace already exists for this topic (id: {}, topic: '{}').",
                    existing.id, existing.topic
                ),
                start.elapsed(),
            ));
        }

        // Create conversation + workspace
        let conversation_id = self
            .db
            .create_conversation("workspace", user_id, None)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to create conversation: {e}"))
            })?;

        let ws = self
            .db
            .create_agent_workspace(user_id, conversation_id)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to create workspace: {e}"))
            })?;

        // Embed and store the topic
        let embedding = self.router.embed(topic_for_embed).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to embed topic: {e}"))
        })?;

        self.db
            .update_agent_workspace_topic(ws.id, topic_for_embed, &embedding)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to set workspace topic: {e}"))
            })?;

        tracing::info!(
            workspace_id = %ws.id,
            topic = %topic,
            "Created new workspace"
        );

        Ok(ToolOutput::text(
            format!(
                "Created workspace '{}' (id: {}). Future messages about this topic will \
                 automatically use this workspace's context.",
                topic, ws.id
            ),
            start.elapsed(),
        ))
    }
}
