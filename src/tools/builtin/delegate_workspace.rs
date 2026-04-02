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
        "Delegate a task to a workspace sub-agent for autonomous execution. Use this \
         only for complex, multi-step work that needs a dedicated worker (e.g. long \
         research tasks, code generation). For simply creating a new topic workspace, \
         use create_workspace instead. The workspace retains memory across calls, so \
         follow-up tasks on the same topic will automatically resume in the same workspace."
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
                    "description": "Optional topic hint for routing (e.g. 'grocery list', 'project tracker'). Helps match the right workspace when multiple exist."
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

                    let mut ws = self.db
                        .create_agent_workspace(user_id, conversation_id)
                        .await
                        .map_err(|e| {
                            ToolError::ExecutionFailed(format!(
                                "failed to create workspace: {e}"
                            ))
                        })?;

                    // Auto-set topic from prompt + hint so the workspace is routable
                    let topic_text = if let Some(hint) = workspace_hint {
                        format!("{hint}: {prompt}")
                    } else {
                        prompt.to_string()
                    };
                    let topic_for_embed = &topic_text[..topic_text.len().min(500)];
                    if let Ok(embedding) = self.router.embed(topic_for_embed).await {
                        if let Err(e) = self
                            .db
                            .update_agent_workspace_topic(ws.id, topic_for_embed, &embedding)
                            .await
                        {
                            tracing::warn!(
                                "Failed to auto-set workspace topic: {}",
                                e
                            );
                        } else {
                            // Update in-memory struct so job metadata reflects the topic
                            ws.topic = topic_for_embed.to_string();
                        }
                    }

                    ws
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
            "workspace_topic": workspace.topic,
            "workspace_turn_count": workspace.turn_count,
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

        // 6. Wait for the result via the response channel.
        // The tool-level execution_timeout (600s) provides the outer timeout;
        // no need for a redundant inner timeout here.
        let result = response_rx
            .await
            .map_err(|_| {
                ToolError::ExecutionFailed("workspace job was cancelled".to_string())
            })?
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("workspace job failed: {e}"))
            })?;

        Ok(ToolOutput::text(result, start.elapsed()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use chrono::{DateTime, Utc};
    use rust_decimal::Decimal;
    use uuid::Uuid;

    use crate::agent::routine::{Routine, RoutineRun, RunStatus};
    use crate::agent::BrokenTool;
    use crate::context::{ActionRecord, JobState};
    use crate::db::structured::{
        Aggregation, CollectionSchema, Filter, Record,
    };
    use crate::db::{
        AgentWorkspace, AgentWorkspaceStore, ConversationStore, Database, JobStore, RoutineStore,
        SandboxStore, SettingsStore, ToolFailureStore, WorkspaceStore,
    };
    use crate::error::{DatabaseError, WorkspaceError};
    use crate::history::{
        AgentJobRecord, AgentJobSummary, ConversationMessage, ConversationSummary, JobEventRecord,
        LlmCallRecord, SandboxJobRecord, SandboxJobSummary, SettingRow,
    };
    use crate::workspace::{
        EmbeddingError, EmbeddingProvider, MemoryChunk, MemoryDocument, SearchConfig, SearchResult,
        WorkspaceEntry,
    };

    // ── Stub Database ──────────────────────────────────────────────────
    //
    // Implements the full Database super-trait. Only the methods actually
    // exercised by the tests have real logic; everything else panics with
    // a clear message.

    struct StubDb {
        workspaces: Mutex<Vec<AgentWorkspace>>,
    }

    impl StubDb {
        fn new() -> Self {
            Self {
                workspaces: Mutex::new(Vec::new()),
            }
        }

        fn insert_workspace(&self, ws: AgentWorkspace) {
            self.workspaces.lock().unwrap().push(ws);
        }
    }

    // -- ConversationStore (all stubs) --

    #[async_trait]
    impl ConversationStore for StubDb {
        async fn create_conversation(
            &self,
            _channel: &str,
            _user_id: &str,
            _thread_id: Option<&str>,
        ) -> Result<Uuid, DatabaseError> {
            Ok(Uuid::new_v4())
        }
        async fn touch_conversation(&self, _id: Uuid) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn add_conversation_message(
            &self,
            _cid: Uuid,
            _role: &str,
            _content: &str,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn ensure_conversation(
            &self,
            _id: Uuid,
            _channel: &str,
            _user_id: &str,
            _thread_id: Option<&str>,
        ) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
        async fn list_conversations_with_preview(
            &self,
            _user_id: &str,
            _channel: &str,
            _limit: i64,
        ) -> Result<Vec<ConversationSummary>, DatabaseError> {
            unimplemented!()
        }
        async fn get_or_create_assistant_conversation(
            &self,
            _user_id: &str,
            _channel: &str,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn create_conversation_with_metadata(
            &self,
            _channel: &str,
            _user_id: &str,
            _metadata: &serde_json::Value,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn list_conversation_messages_paginated(
            &self,
            _cid: Uuid,
            _before: Option<DateTime<Utc>>,
            _limit: i64,
        ) -> Result<(Vec<ConversationMessage>, bool), DatabaseError> {
            unimplemented!()
        }
        async fn update_conversation_metadata_field(
            &self,
            _id: Uuid,
            _key: &str,
            _value: &serde_json::Value,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_conversation_metadata(
            &self,
            _id: Uuid,
        ) -> Result<Option<serde_json::Value>, DatabaseError> {
            unimplemented!()
        }
        async fn list_conversation_messages(
            &self,
            _cid: Uuid,
        ) -> Result<Vec<ConversationMessage>, DatabaseError> {
            unimplemented!()
        }
        async fn conversation_belongs_to_user(
            &self,
            _cid: Uuid,
            _user_id: &str,
        ) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
        async fn list_conversations_all_channels(
            &self,
            _user_id: &str,
            _limit: i64,
        ) -> Result<Vec<ConversationSummary>, DatabaseError> {
            Ok(vec![])
        }
        async fn get_or_create_routine_conversation(
            &self,
            _routine_id: Uuid,
            _routine_name: &str,
            _user_id: &str,
        ) -> Result<Uuid, DatabaseError> {
            Ok(Uuid::new_v4())
        }
        async fn get_or_create_heartbeat_conversation(
            &self,
            _user_id: &str,
        ) -> Result<Uuid, DatabaseError> {
            Ok(Uuid::new_v4())
        }
    }

    // -- JobStore (all stubs) --

    #[async_trait]
    impl JobStore for StubDb {
        async fn save_job(&self, _ctx: &JobContext) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_job(&self, _id: Uuid) -> Result<Option<JobContext>, DatabaseError> {
            unimplemented!()
        }
        async fn update_job_status(
            &self,
            _id: Uuid,
            _status: JobState,
            _reason: Option<&str>,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn mark_job_stuck(&self, _id: Uuid) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_stuck_jobs(&self) -> Result<Vec<Uuid>, DatabaseError> {
            unimplemented!()
        }
        async fn list_agent_jobs(&self) -> Result<Vec<AgentJobRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn list_agent_jobs_for_user(
            &self,
            _user_id: &str,
        ) -> Result<Vec<AgentJobRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn agent_job_summary(&self) -> Result<AgentJobSummary, DatabaseError> {
            unimplemented!()
        }
        async fn agent_job_summary_for_user(
            &self,
            _user_id: &str,
        ) -> Result<AgentJobSummary, DatabaseError> {
            unimplemented!()
        }
        async fn get_agent_job_failure_reason(
            &self,
            _id: Uuid,
        ) -> Result<Option<String>, DatabaseError> {
            Ok(None)
        }
        async fn save_action(
            &self,
            _job_id: Uuid,
            _action: &ActionRecord,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_job_actions(
            &self,
            _job_id: Uuid,
        ) -> Result<Vec<ActionRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn record_llm_call(
            &self,
            _record: &LlmCallRecord<'_>,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn save_estimation_snapshot(
            &self,
            _job_id: Uuid,
            _category: &str,
            _tool_names: &[String],
            _estimated_cost: Decimal,
            _estimated_time_secs: i32,
            _estimated_value: Decimal,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn update_estimation_actuals(
            &self,
            _id: Uuid,
            _actual_cost: Decimal,
            _actual_time_secs: i32,
            _actual_value: Option<Decimal>,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
    }

    // -- SandboxStore (all stubs) --

    #[async_trait]
    impl SandboxStore for StubDb {
        async fn save_sandbox_job(
            &self,
            _job: &SandboxJobRecord,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_sandbox_job(
            &self,
            _id: Uuid,
        ) -> Result<Option<SandboxJobRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn list_sandbox_jobs(&self) -> Result<Vec<SandboxJobRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn update_sandbox_job_status(
            &self,
            _id: Uuid,
            _status: &str,
            _success: Option<bool>,
            _message: Option<&str>,
            _started_at: Option<DateTime<Utc>>,
            _completed_at: Option<DateTime<Utc>>,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn cleanup_stale_sandbox_jobs(&self) -> Result<u64, DatabaseError> {
            unimplemented!()
        }
        async fn sandbox_job_summary(&self) -> Result<SandboxJobSummary, DatabaseError> {
            unimplemented!()
        }
        async fn list_sandbox_jobs_for_user(
            &self,
            _user_id: &str,
        ) -> Result<Vec<SandboxJobRecord>, DatabaseError> {
            unimplemented!()
        }
        async fn sandbox_job_summary_for_user(
            &self,
            _user_id: &str,
        ) -> Result<SandboxJobSummary, DatabaseError> {
            unimplemented!()
        }
        async fn sandbox_job_belongs_to_user(
            &self,
            _job_id: Uuid,
            _user_id: &str,
        ) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
        async fn update_sandbox_job_mode(
            &self,
            _id: Uuid,
            _mode: &str,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_sandbox_job_mode(
            &self,
            _id: Uuid,
        ) -> Result<Option<String>, DatabaseError> {
            unimplemented!()
        }
        async fn save_job_event(
            &self,
            _job_id: Uuid,
            _event_type: &str,
            _data: &serde_json::Value,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn list_job_events(
            &self,
            _job_id: Uuid,
            _limit: Option<i64>,
        ) -> Result<Vec<JobEventRecord>, DatabaseError> {
            unimplemented!()
        }
    }

    // -- RoutineStore (all stubs) --

    #[async_trait]
    impl RoutineStore for StubDb {
        async fn create_routine(&self, _routine: &Routine) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_routine(&self, _id: Uuid) -> Result<Option<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn get_routine_by_name(
            &self,
            _user_id: &str,
            _name: &str,
        ) -> Result<Option<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn list_routines(
            &self,
            _user_id: &str,
        ) -> Result<Vec<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn list_all_routines(&self) -> Result<Vec<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn list_event_routines(&self) -> Result<Vec<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn list_due_cron_routines(&self) -> Result<Vec<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn update_routine(&self, _routine: &Routine) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn update_routine_runtime(
            &self,
            _id: Uuid,
            _last_run_at: DateTime<Utc>,
            _next_fire_at: Option<DateTime<Utc>>,
            _run_count: u64,
            _consecutive_failures: u32,
            _state: &serde_json::Value,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn delete_routine(&self, _id: Uuid) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
        async fn create_routine_run(&self, _run: &RoutineRun) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn complete_routine_run(
            &self,
            _id: Uuid,
            _status: RunStatus,
            _result_summary: Option<&str>,
            _tokens_used: Option<i32>,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn list_routine_runs(
            &self,
            _routine_id: Uuid,
            _limit: i64,
        ) -> Result<Vec<RoutineRun>, DatabaseError> {
            unimplemented!()
        }
        async fn count_running_routine_runs(
            &self,
            _routine_id: Uuid,
        ) -> Result<i64, DatabaseError> {
            unimplemented!()
        }
        async fn count_running_routine_runs_batch(
            &self,
            _routine_ids: &[Uuid],
        ) -> Result<std::collections::HashMap<Uuid, i64>, DatabaseError> {
            unimplemented!()
        }
        async fn batch_get_last_run_status(
            &self,
            _routine_ids: &[Uuid],
        ) -> Result<std::collections::HashMap<Uuid, RunStatus>, DatabaseError> {
            unimplemented!()
        }
        async fn link_routine_run_to_job(
            &self,
            _run_id: Uuid,
            _job_id: Uuid,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_webhook_routine_by_path(
            &self,
            _path: &str,
        ) -> Result<Option<Routine>, DatabaseError> {
            unimplemented!()
        }
        async fn list_dispatched_routine_runs(&self) -> Result<Vec<RoutineRun>, DatabaseError> {
            unimplemented!()
        }
        async fn batch_get_last_run_status(
            &self,
            _routine_ids: &[Uuid],
        ) -> Result<std::collections::HashMap<Uuid, RunStatus>, DatabaseError> {
            unimplemented!()
        }
    }

    // -- ToolFailureStore (all stubs) --

    #[async_trait]
    impl ToolFailureStore for StubDb {
        async fn record_tool_failure(
            &self,
            _tool_name: &str,
            _error_message: &str,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_broken_tools(
            &self,
            _threshold: i32,
        ) -> Result<Vec<BrokenTool>, DatabaseError> {
            unimplemented!()
        }
        async fn mark_tool_repaired(&self, _tool_name: &str) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn increment_repair_attempts(&self, _tool_name: &str) -> Result<(), DatabaseError> {
            unimplemented!()
        }
    }

    // -- SettingsStore (all stubs) --

    #[async_trait]
    impl SettingsStore for StubDb {
        async fn get_setting(
            &self,
            _user_id: &str,
            _key: &str,
        ) -> Result<Option<serde_json::Value>, DatabaseError> {
            unimplemented!()
        }
        async fn get_setting_full(
            &self,
            _user_id: &str,
            _key: &str,
        ) -> Result<Option<SettingRow>, DatabaseError> {
            unimplemented!()
        }
        async fn set_setting(
            &self,
            _user_id: &str,
            _key: &str,
            _value: &serde_json::Value,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn delete_setting(
            &self,
            _user_id: &str,
            _key: &str,
        ) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
        async fn list_settings(
            &self,
            _user_id: &str,
        ) -> Result<Vec<SettingRow>, DatabaseError> {
            unimplemented!()
        }
        async fn get_all_settings(
            &self,
            _user_id: &str,
        ) -> Result<HashMap<String, serde_json::Value>, DatabaseError> {
            unimplemented!()
        }
        async fn set_all_settings(
            &self,
            _user_id: &str,
            _settings: &HashMap<String, serde_json::Value>,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn has_settings(&self, _user_id: &str) -> Result<bool, DatabaseError> {
            unimplemented!()
        }
    }

    // -- WorkspaceStore (all stubs) --

    #[async_trait]
    impl WorkspaceStore for StubDb {
        async fn get_document_by_path(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _path: &str,
        ) -> Result<MemoryDocument, WorkspaceError> {
            unimplemented!()
        }
        async fn get_document_by_id(&self, _id: Uuid) -> Result<MemoryDocument, WorkspaceError> {
            unimplemented!()
        }
        async fn get_or_create_document_by_path(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _path: &str,
        ) -> Result<MemoryDocument, WorkspaceError> {
            unimplemented!()
        }
        async fn update_document(&self, _id: Uuid, _content: &str) -> Result<(), WorkspaceError> {
            unimplemented!()
        }
        async fn delete_document_by_path(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _path: &str,
        ) -> Result<(), WorkspaceError> {
            unimplemented!()
        }
        async fn list_directory(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _directory: &str,
        ) -> Result<Vec<WorkspaceEntry>, WorkspaceError> {
            unimplemented!()
        }
        async fn list_all_paths(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
        ) -> Result<Vec<String>, WorkspaceError> {
            unimplemented!()
        }
        async fn list_documents(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
        ) -> Result<Vec<MemoryDocument>, WorkspaceError> {
            unimplemented!()
        }
        async fn delete_chunks(&self, _document_id: Uuid) -> Result<(), WorkspaceError> {
            unimplemented!()
        }
        async fn insert_chunk(
            &self,
            _document_id: Uuid,
            _chunk_index: i32,
            _content: &str,
            _embedding: Option<&[f32]>,
        ) -> Result<Uuid, WorkspaceError> {
            unimplemented!()
        }
        async fn update_chunk_embedding(
            &self,
            _chunk_id: Uuid,
            _embedding: &[f32],
        ) -> Result<(), WorkspaceError> {
            unimplemented!()
        }
        async fn get_chunks_without_embeddings(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _limit: usize,
        ) -> Result<Vec<MemoryChunk>, WorkspaceError> {
            unimplemented!()
        }
        async fn hybrid_search(
            &self,
            _user_id: &str,
            _agent_id: Option<Uuid>,
            _query: &str,
            _embedding: Option<&[f32]>,
            _config: &SearchConfig,
        ) -> Result<Vec<SearchResult>, WorkspaceError> {
            unimplemented!()
        }
        async fn search_conversation_messages(
            &self,
            _user_id: &str,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>, WorkspaceError> {
            unimplemented!()
        }
    }

    // -- AgentWorkspaceStore --

    #[async_trait]
    impl AgentWorkspaceStore for StubDb {
        async fn create_agent_workspace(
            &self,
            user_id: &str,
            conversation_id: Uuid,
        ) -> Result<AgentWorkspace, DatabaseError> {
            let ws = AgentWorkspace {
                id: Uuid::new_v4(),
                user_id: user_id.to_string(),
                topic: String::new(),
                conversation_id,
                status: "active".to_string(),
                last_accessed: Utc::now(),
                turn_count: 0,
                created_at: Utc::now(),
                summary: None,
            };
            self.workspaces.lock().unwrap().push(ws.clone());
            Ok(ws)
        }

        async fn update_agent_workspace_topic(
            &self,
            _id: Uuid,
            _topic: &str,
            _embedding: &[f32],
        ) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn find_matching_workspace(
            &self,
            _user_id: &str,
            _embedding: &[f32],
            _threshold: f64,
        ) -> Result<Option<AgentWorkspace>, DatabaseError> {
            Ok(None)
        }

        async fn find_top_matching_workspaces(
            &self,
            _user_id: &str,
            _embedding: &[f32],
            _limit: i64,
        ) -> Result<Vec<(AgentWorkspace, f64)>, DatabaseError> {
            Ok(vec![])
        }

        async fn get_agent_workspace(
            &self,
            id: Uuid,
        ) -> Result<Option<AgentWorkspace>, DatabaseError> {
            let guard = self.workspaces.lock().unwrap();
            Ok(guard.iter().find(|ws| ws.id == id).cloned())
        }

        async fn get_agent_workspace_by_conversation(
            &self,
            conversation_id: Uuid,
        ) -> Result<Option<AgentWorkspace>, DatabaseError> {
            let guard = self.workspaces.lock().unwrap();
            Ok(guard.iter().find(|ws| ws.conversation_id == conversation_id).cloned())
        }

        async fn list_agent_workspaces(
            &self,
            _user_id: &str,
            _status: Option<&str>,
        ) -> Result<Vec<AgentWorkspace>, DatabaseError> {
            unimplemented!()
        }

        async fn touch_agent_workspace(&self, _id: Uuid) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn update_agent_workspace_status(
            &self,
            _id: Uuid,
            _status: &str,
        ) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn archive_stale_workspaces(
            &self,
            _user_id: &str,
            _stale_days: i64,
        ) -> Result<u64, DatabaseError> {
            Ok(0)
        }

        async fn update_agent_workspace_summary(
            &self,
            _id: Uuid,
            _summary: &str,
        ) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn get_workspace_embedding(
            &self,
            _id: Uuid,
        ) -> Result<Option<Vec<f32>>, DatabaseError> {
            Ok(None)
        }

        async fn search_workspace_messages(
            &self,
            _user_id: &str,
            _query: &str,
            _workspace_id: Option<Uuid>,
            _limit: i64,
        ) -> Result<Vec<crate::db::WorkspaceMessageResult>, DatabaseError> {
            Ok(Vec::new())
        }
    }

    // -- StructuredStore (all stubs) --

    #[async_trait]
    impl crate::db::structured::StructuredStore for StubDb {
        async fn register_collection(
            &self,
            _user_id: &str,
            _schema: &CollectionSchema,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn get_collection_schema(
            &self,
            _user_id: &str,
            _collection: &str,
        ) -> Result<CollectionSchema, DatabaseError> {
            unimplemented!()
        }
        async fn list_collections(
            &self,
            _user_id: &str,
        ) -> Result<Vec<CollectionSchema>, DatabaseError> {
            unimplemented!()
        }
        async fn drop_collection(
            &self,
            _user_id: &str,
            _collection: &str,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn insert_record(
            &self,
            _user_id: &str,
            _collection: &str,
            _data: serde_json::Value,
        ) -> Result<Uuid, DatabaseError> {
            unimplemented!()
        }
        async fn get_record(
            &self,
            _user_id: &str,
            _record_id: Uuid,
        ) -> Result<Record, DatabaseError> {
            unimplemented!()
        }
        async fn update_record(
            &self,
            _user_id: &str,
            _record_id: Uuid,
            _updates: serde_json::Value,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn delete_record(
            &self,
            _user_id: &str,
            _record_id: Uuid,
        ) -> Result<(), DatabaseError> {
            unimplemented!()
        }
        async fn query_records(
            &self,
            _user_id: &str,
            _collection: &str,
            _filters: &[Filter],
            _order_by: Option<&str>,
            _limit: usize,
        ) -> Result<Vec<Record>, DatabaseError> {
            unimplemented!()
        }
        async fn aggregate(
            &self,
            _user_id: &str,
            _collection: &str,
            _aggregation: &Aggregation,
        ) -> Result<serde_json::Value, DatabaseError> {
            unimplemented!()
        }
    }

    // -- Database (supertrait) --

    #[async_trait]
    impl Database for StubDb {
        async fn run_migrations(&self) -> Result<(), DatabaseError> {
            Ok(())
        }
    }

    // ── Fake embedder (minimal, for WorkspaceRouter) ───────────────────

    struct FakeEmbedder;

    #[async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        fn dimension(&self) -> usize {
            4
        }
        fn model_name(&self) -> &str {
            "fake"
        }
        fn max_input_length(&self) -> usize {
            10_000
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbeddingError> {
            // Always returns the same vector — tests that use routing will get
            // `None` from `find_matching_workspace` (StubDb returns None).
            Ok(vec![1.0, 0.0, 0.0, 0.0])
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    fn make_tool(db: Arc<dyn Database>) -> DelegateToWorkspaceTool {
        let embedder: Arc<dyn EmbeddingProvider> = Arc::new(FakeEmbedder);
        let store = Arc::clone(&db) as Arc<dyn AgentWorkspaceStore>;
        let router = Arc::new(WorkspaceRouter::new(store, embedder, 0.8));
        let scheduler: SchedulerSlot = Arc::new(tokio::sync::RwLock::new(None));
        let queue = Arc::new(WorkspaceQueueManager::new(10));
        DelegateToWorkspaceTool::new(router, scheduler, db, queue)
    }

    fn make_ctx(user_id: &str, metadata: serde_json::Value) -> JobContext {
        let mut ctx = JobContext::with_user(user_id, "test", "test");
        ctx.metadata = metadata;
        ctx
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_depth_limit_rejects_at_max() {
        let db: Arc<dyn Database> = Arc::new(StubDb::new());
        let tool = make_tool(db);
        let ctx = make_ctx("user-a", serde_json::json!({"delegation_depth": 3}));

        let params = serde_json::json!({"prompt": "do something"});
        let err = tool.execute(params, &ctx).await.unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("delegation depth limit exceeded"),
            "expected depth limit error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_depth_limit_allows_below_max() {
        let db: Arc<dyn Database> = Arc::new(StubDb::new());
        let tool = make_tool(db);
        let ctx = make_ctx("user-a", serde_json::json!({"delegation_depth": 2}));

        let params = serde_json::json!({"prompt": "do something"});
        let err = tool.execute(params, &ctx).await.unwrap_err();

        // Should NOT be a depth-limit error — it will fail later (no scheduler),
        // but the depth check itself should pass.
        let msg = err.to_string();
        assert!(
            !msg.contains("delegation depth limit exceeded"),
            "depth 2 should not trigger the limit (max 3), got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_ownership_check_rejects_wrong_user() {
        let db = Arc::new(StubDb::new());

        // Insert a workspace owned by "user-a"
        let ws = AgentWorkspace {
            id: Uuid::new_v4(),
            user_id: "user-a".to_string(),
            topic: String::new(),
            conversation_id: Uuid::new_v4(),
            status: "active".to_string(),
            last_accessed: Utc::now(),
            turn_count: 0,
            created_at: Utc::now(),
            summary: None,
        };
        let ws_id = ws.id;
        db.insert_workspace(ws);

        let db: Arc<dyn Database> = db;
        let tool = make_tool(db);

        // Call as "user-b" targeting user-a's workspace
        let ctx = make_ctx("user-b", serde_json::json!({}));
        let params = serde_json::json!({
            "prompt": "do something",
            "workspace_id": ws_id.to_string()
        });
        let err = tool.execute(params, &ctx).await.unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("belongs to a different user"),
            "expected ownership error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_invalid_workspace_id_format() {
        let db: Arc<dyn Database> = Arc::new(StubDb::new());
        let tool = make_tool(db);
        let ctx = make_ctx("user-a", serde_json::json!({}));

        let params = serde_json::json!({
            "prompt": "do something",
            "workspace_id": "not-a-uuid"
        });
        let err = tool.execute(params, &ctx).await.unwrap_err();

        // Should be InvalidParameters
        match &err {
            ToolError::InvalidParameters(msg) => {
                assert!(
                    msg.contains("invalid workspace_id"),
                    "expected invalid workspace_id message, got: {msg}"
                );
            }
            other => panic!("expected InvalidParameters, got: {other:?}"),
        }
    }
}
