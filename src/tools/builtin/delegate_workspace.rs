//! Delegate-to-workspace tool.
//!
//! Lets the LLM delegate a task to a persistent workspace. The tool routes
//! the request to an existing workspace (or creates a new one), dispatches
//! a job to the workspace's conversation, waits for completion, and returns
//! the result.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::agent::workspace_queue::{MessagePriority, WorkspaceMessage, WorkspaceQueueManager};
use crate::agent::workspace_router::WorkspaceRouter;
use crate::context::JobContext;
use crate::db::Database;
use crate::tools::builtin::job::SchedulerSlot;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

/// Tool for delegating work to a persistent workspace.
///
/// Routes the prompt to an existing workspace by embedding similarity (or
/// explicit ID), enqueues the work through the `WorkspaceQueueManager` to
/// enforce single-writer access, and blocks until the job completes.
pub struct DelegateToWorkspaceTool {
    router: Arc<WorkspaceRouter>,
    scheduler: SchedulerSlot,
    db: Arc<dyn Database>,
    queue: Arc<WorkspaceQueueManager>,
}

impl DelegateToWorkspaceTool {
    pub fn new(
        router: Arc<WorkspaceRouter>,
        scheduler: SchedulerSlot,
        db: Arc<dyn Database>,
        queue: Arc<WorkspaceQueueManager>,
    ) -> Self {
        Self {
            router,
            scheduler,
            db,
            queue,
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

        // 0. Check delegation depth limit (max 3 levels)
        const MAX_DELEGATION_DEPTH: u64 = 3;
        let delegation_depth = ctx
            .metadata
            .get("delegation_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if delegation_depth >= MAX_DELEGATION_DEPTH {
            return Err(ToolError::ExecutionFailed(format!(
                "delegation depth limit exceeded (max {MAX_DELEGATION_DEPTH})"
            )));
        }

        // 1. Resolve the target workspace
        let workspace = if let Some(id_str) = workspace_id_str {
            // Explicit workspace ID provided — look it up directly
            let ws_id = Uuid::parse_str(id_str).map_err(|e| {
                ToolError::InvalidParameters(format!("invalid workspace_id: {e}"))
            })?;
            let ws = self
                .db
                .get_agent_workspace(ws_id)
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("failed to get workspace: {e}")))?
                .ok_or_else(|| {
                    ToolError::ExecutionFailed(format!("workspace {ws_id} not found"))
                })?;

            // Verify the workspace belongs to the calling user
            if ws.user_id != *user_id {
                return Err(ToolError::ExecutionFailed(
                    "workspace belongs to a different user".to_string(),
                ));
            }

            ws
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

        // 3. Determine priority: depth > 0 means workspace-to-workspace delegation
        let priority = if delegation_depth > 0 {
            MessagePriority::Delegated
        } else {
            MessagePriority::User
        };

        // 4. Enqueue the message through the queue manager
        let title = workspace_hint
            .map(|h| format!("Workspace: {h}"))
            .unwrap_or_else(|| "Workspace task".to_string());

        let job_metadata = serde_json::json!({
            "workspace_id": workspace.id.to_string(),
            "delegation_depth": delegation_depth + 1,
        });

        let (response_tx, response_rx) = oneshot::channel();

        let msg = WorkspaceMessage {
            prompt: prompt.to_string(),
            priority,
            response_tx,
            ttl: Some(Duration::from_secs(600)),
            enqueued_at: std::time::Instant::now(),
            metadata: Some(serde_json::json!({
                "user_id": user_id,
                "title": title,
                "job_metadata": job_metadata,
            })),
        };

        self.queue
            .enqueue(workspace.id, msg)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("queue error: {e}")))?;

        // 5. Kick off processing if the workspace is idle
        self.queue
            .start_processing(
                workspace.id,
                workspace.conversation_id,
                scheduler,
                Arc::clone(&self.db),
            )
            .await;

        // 6. Wait for the result via the response channel
        let result = tokio::time::timeout(Duration::from_secs(600), response_rx)
            .await
            .map_err(|_| ToolError::ExecutionFailed("workspace job timed out".to_string()))?
            .map_err(|_| {
                ToolError::ExecutionFailed("workspace job was cancelled".to_string())
            })?
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("workspace job failed: {e}"))
            })?;

        Ok(ToolOutput::text(result, start.elapsed()))
    }
}
