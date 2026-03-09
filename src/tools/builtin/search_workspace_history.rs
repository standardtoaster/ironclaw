//! Cross-workspace conversation search tool.
//!
//! Searches conversation messages across all workspace histories for the
//! current user, enabling cross-workspace intelligence.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::db::AgentWorkspaceStore;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

pub struct SearchWorkspaceHistoryTool {
    db: Arc<dyn AgentWorkspaceStore>,
}

impl SearchWorkspaceHistoryTool {
    pub fn new(db: Arc<dyn AgentWorkspaceStore>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for SearchWorkspaceHistoryTool {
    fn name(&self) -> &str {
        "search_workspace_history"
    }

    fn description(&self) -> &str {
        "Search across all workspace conversation histories. Use this to find \
         information from past workspace conversations, including delegated tasks \
         and their results."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for in conversation messages"
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Optional workspace ID to limit search to a specific workspace"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();
        let query = require_str(&params, "query")?;

        let workspace_id = params
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());

        let limit = params
            .get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10)
            .min(50);

        let results = self
            .db
            .search_workspace_messages(&ctx.user_id, query, workspace_id, limit)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("search failed: {e}"))
            })?;

        if results.is_empty() {
            return Ok(ToolOutput::text(
                "No matching messages found across workspace histories.",
                start.elapsed(),
            ));
        }

        let mut output = String::new();
        for r in &results {
            let age = chrono::Utc::now()
                .signed_duration_since(r.created_at)
                .num_days();
            let age_str = if age == 0 {
                "today".to_string()
            } else if age == 1 {
                "1 day ago".to_string()
            } else {
                format!("{age} days ago")
            };
            let topic = if r.topic.is_empty() {
                "untitled"
            } else {
                &r.topic
            };
            output.push_str(&format!("[Workspace: {} ({})]\n", topic, age_str));
            let role_label = if r.role == "user" { "User" } else { "Assistant" };
            // Truncate long messages
            let content = if r.content.len() > 500 {
                format!("{}...", &r.content[..500])
            } else {
                r.content.clone()
            };
            output.push_str(&format!("{}: {}\n\n", role_label, content));
        }

        Ok(ToolOutput::text(output.trim(), start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}
