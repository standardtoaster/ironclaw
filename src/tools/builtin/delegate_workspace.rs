//! Delegate-to-workspace tool.
//!
//! Lets the LLM delegate a task to a persistent workspace. The tool routes
//! the request to an existing workspace (or creates a new one), dispatches
//! a job to the workspace's conversation, waits for completion, and returns
//! the result.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use crate::agent::workspace_router::WorkspaceRouter;
use crate::context::JobContext;
use crate::db::Database;
use crate::tools::builtin::job::SchedulerSlot;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

/// Tool for delegating work to a persistent workspace.
///
/// Routes the prompt to an existing workspace by embedding similarity (or
/// explicit ID), dispatches a job to the workspace's conversation, and
/// blocks until the job completes (up to 5 minutes).
pub struct DelegateToWorkspaceTool {
    router: Arc<WorkspaceRouter>,
    scheduler: SchedulerSlot,
    db: Arc<dyn Database>,
}

impl DelegateToWorkspaceTool {
    pub fn new(
        router: Arc<WorkspaceRouter>,
        scheduler: SchedulerSlot,
        db: Arc<dyn Database>,
    ) -> Self {
        Self {
            router,
            scheduler,
            db,
        }
    }
}

#[async_trait]
impl Tool for DelegateToWorkspaceTool {
    fn name(&self) -> &str {
        "delegate_to_workspace"
    }

    fn description(&self) -> &str {
        "Delegate a task or question to a persistent workspace. Use this for complex, \
         domain-specific, or multi-step work that benefits from dedicated context and \
         conversation history. Simple factual questions should be answered directly \
         without delegation. The workspace retains memory across calls, so follow-up \
         tasks on the same topic will automatically resume in the same workspace."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task or question to delegate to the workspace"
                },
                "workspace_hint": {
                    "type": "string",
                    "description": "Optional topic hint for routing (e.g. 'grocery list', 'nanny schedule'). Helps match the right workspace when multiple exist."
                },
                "workspace_id": {
                    "type": "string",
                    "description": "Explicit workspace ID to target. Bypasses automatic routing. Use only when you know the exact workspace."
                }
            },
            "required": ["prompt"]
        })
    }

    fn execution_timeout(&self) -> Duration {
        Duration::from_secs(600)
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

        let prompt = require_str(&params, "prompt")?;
        let workspace_hint = params.get("workspace_hint").and_then(|v| v.as_str());
        let workspace_id_str = params.get("workspace_id").and_then(|v| v.as_str());

        let user_id = &ctx.user_id;

        // 1. Resolve the target workspace
        let workspace = if let Some(id_str) = workspace_id_str {
            // Explicit workspace ID provided — look it up directly
            let ws_id = Uuid::parse_str(id_str).map_err(|e| {
                ToolError::InvalidParameters(format!("invalid workspace_id: {e}"))
            })?;
            self.db
                .get_agent_workspace(ws_id)
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("failed to get workspace: {e}")))?
                .ok_or_else(|| {
                    ToolError::ExecutionFailed(format!("workspace {ws_id} not found"))
                })?
        } else {
            // Route by embedding similarity
            match self
                .router
                .route_with_hint(user_id, prompt, workspace_hint)
                .await
            {
                Ok(Some(ws)) => ws,
                Ok(None) => {
                    // No matching workspace — create a new conversation and workspace
                    let conversation_id = self
                        .db
                        .create_conversation("workspace", user_id, None)
                        .await
                        .map_err(|e| {
                            ToolError::ExecutionFailed(format!(
                                "failed to create conversation: {e}"
                            ))
                        })?;

                    self.db
                        .create_agent_workspace(user_id, conversation_id)
                        .await
                        .map_err(|e| {
                            ToolError::ExecutionFailed(format!(
                                "failed to create workspace: {e}"
                            ))
                        })?
                }
                Err(e) => {
                    return Err(ToolError::ExecutionFailed(format!(
                        "workspace routing failed: {e}"
                    )));
                }
            }
        };

        // 2. Get the scheduler from the slot
        let scheduler = {
            let guard = self.scheduler.read().await;
            guard.clone().ok_or_else(|| {
                ToolError::ExecutionFailed(
                    "scheduler not available (agent not fully initialized)".to_string(),
                )
            })?
        };

        // 3. Dispatch a job to the workspace's conversation
        let title = workspace_hint
            .map(|h| format!("Workspace: {h}"))
            .unwrap_or_else(|| "Workspace task".to_string());

        let metadata = serde_json::json!({
            "workspace_id": workspace.id.to_string(),
        });

        let job_id = scheduler
            .dispatch_job_to_conversation(
                user_id,
                workspace.conversation_id,
                &title,
                prompt,
                Some(metadata),
            )
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to dispatch job: {e}")))?;

        // 4. Wait for completion (up to 5 minutes)
        let result = scheduler
            .await_job(job_id, Duration::from_secs(300))
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("workspace job failed or timed out: {e}"))
            })?;

        // 5. Touch the workspace to update last_accessed
        if let Err(e) = self.db.touch_agent_workspace(workspace.id).await {
            tracing::warn!(
                workspace_id = %workspace.id,
                "failed to touch workspace: {e}"
            );
        }

        Ok(ToolOutput::text(result, start.elapsed()))
    }
}
