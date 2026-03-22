//! Workspace summary tool — returns the last N turns of a workspace conversation.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

pub struct WorkspaceSummaryTool {
    db: Arc<dyn Database>,
}

impl WorkspaceSummaryTool {
    pub fn new(db: Arc<dyn Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for WorkspaceSummaryTool {
    fn name(&self) -> &str {
        "workspace_summary"
    }

    fn description(&self) -> &str {
        "Get a summary of a workspace's conversation history. Returns the most \
         recent messages from the workspace, useful for understanding what was \
         discussed or decided."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "The workspace ID to summarize"
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum number of message turns to include (default 20)"
                }
            },
            "required": ["workspace_id"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();
        let ws_id_str = require_str(&params, "workspace_id")?;
        let ws_id = uuid::Uuid::parse_str(ws_id_str).map_err(|e| {
            ToolError::InvalidParameters(format!("invalid workspace_id: {e}"))
        })?;

        let max_turns = params
            .get("max_turns")
            .and_then(|v| v.as_i64())
            .unwrap_or(20)
            .min(100) as usize;

        let ws = self
            .db
            .get_agent_workspace(ws_id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("db error: {e}")))?
            .ok_or_else(|| {
                ToolError::ExecutionFailed(format!("workspace {ws_id} not found"))
            })?;

        let messages = self
            .db
            .list_conversation_messages(ws.conversation_id)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to load messages: {e}"))
            })?;

        // Filter to user/assistant messages and take the last N
        let relevant: Vec<_> = messages
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .collect();
        let start_idx = relevant.len().saturating_sub(max_turns);
        let recent = &relevant[start_idx..];

        if recent.is_empty() {
            return Ok(ToolOutput::text(
                format!(
                    "Workspace \"{}\" (ID: {}) has no conversation history yet.",
                    ws.topic, ws.id
                ),
                start.elapsed(),
            ));
        }

        let topic = if ws.topic.is_empty() {
            "untitled"
        } else {
            &ws.topic
        };

        let mut output = format!(
            "## Workspace: {} ({})\n\nStatus: {} | Turns: {} | Last accessed: {}\n\n---\n\n",
            topic,
            ws.id,
            ws.status,
            ws.turn_count,
            ws.last_accessed.format("%Y-%m-%d %H:%M UTC"),
        );

        for msg in recent {
            let role_label = if msg.role == "user" { "User" } else { "Assistant" };
            // Truncate very long messages
            let content = if msg.content.len() > 1000 {
                format!("{}...", &msg.content[..1000])
            } else {
                msg.content.clone()
            };
            output.push_str(&format!("**{}:** {}\n\n", role_label, content));
        }

        Ok(ToolOutput::text(output.trim(), start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}
