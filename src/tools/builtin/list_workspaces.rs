//! List-workspaces tool.
//!
//! Lets the LLM (or user) discover existing workspace topics and IDs.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput};

/// Tool for listing a user's workspaces.
pub struct ListWorkspacesTool {
    db: Arc<dyn Database>,
}

impl ListWorkspacesTool {
    pub fn new(db: Arc<dyn Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for ListWorkspacesTool {
    fn name(&self) -> &str {
        "list_workspaces"
    }

    fn description(&self) -> &str {
        "List active workspaces. Use this to discover existing workspace topics before \
         delegating, or to find a specific workspace ID."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "watching", "archived"],
                    "description": "Filter workspaces by status. Defaults to showing all."
                }
            }
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

        let status = params.get("status").and_then(|v| v.as_str());
        let user_id = &ctx.user_id;

        let workspaces = self
            .db
            .list_agent_workspaces(user_id, status)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to list workspaces: {e}"))
            })?;

        if workspaces.is_empty() {
            return Ok(ToolOutput::text("No workspaces found.", start.elapsed()));
        }

        let mut lines = Vec::with_capacity(workspaces.len() + 1);
        lines.push(format!("Found {} workspace(s):\n", workspaces.len()));

        for ws in &workspaces {
            lines.push(format!(
                "- **{}** (id: {})\n  status: {}, turns: {}, last accessed: {}",
                ws.topic,
                ws.id,
                ws.status,
                ws.turn_count,
                ws.last_accessed.format("%Y-%m-%d %H:%M UTC"),
            ));
        }

        Ok(ToolOutput::text(lines.join("\n"), start.elapsed()))
    }
}
