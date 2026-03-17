//! Job scheduler for parallel execution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::agent::task::{Task, TaskContext, TaskOutput};
use crate::agent::worker::{Worker, WorkerDeps};
use crate::config::AgentConfig;
use crate::context::{ContextManager, JobContext, JobState};
use crate::db::Database;
use crate::error::{Error, JobError};
use crate::hooks::HookRegistry;
use crate::llm::LlmProvider;
use crate::safety::SafetyLayer;
use crate::tools::{ApprovalContext, ToolRegistry};

/// Message to send to a worker.
#[derive(Debug)]
pub enum WorkerMessage {
    /// Start working on the job.
    Start,
    /// Stop the job.
    Stop,
    /// Check health.
    Ping,
    /// Inject a follow-up user message into the worker's reasoning context.
    UserMessage(String),
}

/// Status of a scheduled job.
#[derive(Debug)]
pub struct ScheduledJob {
    pub handle: JoinHandle<()>,
    pub tx: mpsc::Sender<WorkerMessage>,
}

/// Status of a scheduled sub-task.
struct ScheduledSubtask {
    handle: JoinHandle<Result<TaskOutput, Error>>,
}

/// Schedules and manages parallel job execution.
pub struct Scheduler {
    config: AgentConfig,
    context_manager: Arc<ContextManager>,
    llm: Arc<dyn LlmProvider>,
    safety: Arc<SafetyLayer>,
    tools: Arc<ToolRegistry>,
    store: Option<Arc<dyn Database>>,
    hooks: Arc<HookRegistry>,
    /// SSE broadcast manager for live job event streaming.
    sse_tx: Option<Arc<crate::channels::web::sse::SseManager>>,
    /// HTTP interceptor for trace recording/replay (propagated to workers).
    http_interceptor: Option<Arc<dyn crate::llm::recording::HttpInterceptor>>,
    /// Core tool names for worker filtering.
    core_tools: Vec<String>,
    /// Running jobs (main LLM-driven jobs).
    jobs: Arc<RwLock<HashMap<Uuid, ScheduledJob>>>,
    /// Running sub-tasks (tool executions, background tasks).
    subtasks: Arc<RwLock<HashMap<Uuid, ScheduledSubtask>>>,
    /// Waiters for job completion. When a job's worker handle finishes,
    /// all registered oneshot senders are fired.
    completion_waiters: Arc<RwLock<HashMap<Uuid, Vec<oneshot::Sender<()>>>>>,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(
        config: AgentConfig,
        context_manager: Arc<ContextManager>,
        llm: Arc<dyn LlmProvider>,
        safety: Arc<SafetyLayer>,
        tools: Arc<ToolRegistry>,
        store: Option<Arc<dyn Database>>,
        hooks: Arc<HookRegistry>,
    ) -> Self {
        Self {
            config,
            context_manager,
            llm,
            safety,
            tools,
            store,
            hooks,
            sse_tx: None,
            http_interceptor: None,
            core_tools: Vec::new(),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            subtasks: Arc::new(RwLock::new(HashMap::new())),
            completion_waiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the SSE broadcast manager for live job event streaming.
    pub fn set_sse_sender(&mut self, sse: Arc<crate::channels::web::sse::SseManager>) {
        self.sse_tx = Some(sse);
    }

    /// Set core tool names for worker filtering.
    pub fn set_core_tools(&mut self, tools: Vec<String>) {
        self.core_tools = tools;
    }

    /// Set the HTTP interceptor for trace recording/replay.
    pub fn set_http_interceptor(
        &mut self,
        interceptor: Arc<dyn crate::llm::recording::HttpInterceptor>,
    ) {
        self.http_interceptor = Some(interceptor);
    }

    /// Create, persist, and schedule a job in one shot.
    ///
    /// This is the preferred entry point for dispatching new jobs. It:
    /// 1. Creates the job context via `ContextManager`
    /// 2. Optionally applies metadata (e.g. `max_iterations`)
    /// 3. Persists the job to the database (so FK references from
    ///    `job_actions` / `llm_calls` work immediately)
    /// 4. Schedules the job for worker execution
    ///
    /// Returns the new job ID.
    pub async fn dispatch_job(
        &self,
        user_id: &str,
        title: &str,
        description: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid, JobError> {
        self.dispatch_job_inner(user_id, title, description, metadata, None)
            .await
    }

    /// Dispatch a job with an explicit approval context for autonomous execution.
    ///
    /// Same as `dispatch_job`, but the worker will use the given `ApprovalContext`
    /// to determine which tools are pre-approved (instead of blocking all non-`Never` tools).
    pub async fn dispatch_job_with_context(
        &self,
        user_id: &str,
        title: &str,
        description: &str,
        metadata: Option<serde_json::Value>,
        approval_context: ApprovalContext,
    ) -> Result<Uuid, JobError> {
        self.dispatch_job_inner(
            user_id,
            title,
            description,
            metadata,
            Some(approval_context),
        )
        .await
    }

    /// Shared implementation for `dispatch_job` and `dispatch_job_with_context`.
    async fn dispatch_job_inner(
        &self,
        user_id: &str,
        title: &str,
        description: &str,
        metadata: Option<serde_json::Value>,
        approval_context: Option<ApprovalContext>,
    ) -> Result<Uuid, JobError> {
        let job_id = self
            .context_manager
            .create_job_for_user(user_id, title, description)
            .await?;

        // Apply metadata if provided
        if let Some(meta) = metadata {
            self.context_manager
                .update_context(job_id, |ctx| {
                    ctx.metadata = meta;
                })
                .await?;
        }

        // Persist to DB before scheduling so the worker's FK references are valid
        if let Some(ref store) = self.store {
            let ctx = self.context_manager.get_context(job_id).await?;
            store.save_job(&ctx).await.map_err(|e| JobError::Failed {
                id: job_id,
                reason: format!("failed to persist job: {e}"),
            })?;
        }

        self.schedule_with_context(job_id, approval_context).await?;
        Ok(job_id)
    }

    /// Create, persist, and schedule a job attached to an existing conversation.
    ///
    /// Like `dispatch_job`, but instead of creating a new conversation the job
    /// references an existing `conversation_id`. The `prompt` is appended as a
    /// new user message to that conversation before the worker starts, so the
    /// worker will see the full history when it hydrates the thread.
    ///
    /// Returns the new job ID.
    pub async fn dispatch_job_to_conversation(
        &self,
        user_id: &str,
        conversation_id: Uuid,
        title: &str,
        prompt: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<Uuid, JobError> {
        // 1. Create job context linked to the existing conversation
        let job_id = self
            .context_manager
            .create_job_for_conversation(user_id, conversation_id, title, prompt)
            .await?;

        // 2. Apply metadata if provided
        if let Some(meta) = metadata {
            self.context_manager
                .update_context(job_id, |ctx| {
                    ctx.metadata = meta;
                })
                .await?;
        }

        // 3. Append the prompt as a user message to the existing conversation
        if let Some(ref store) = self.store {
            store
                .add_conversation_message(conversation_id, "user", prompt)
                .await
                .map_err(|e| JobError::Failed {
                    id: job_id,
                    reason: format!("failed to append message to conversation: {e}"),
                })?;
        }

        // 4. Persist job to DB before scheduling so FK references are valid
        if let Some(ref store) = self.store {
            let ctx = self.context_manager.get_context(job_id).await?;
            store.save_job(&ctx).await.map_err(|e| JobError::Failed {
                id: job_id,
                reason: format!("failed to persist job: {e}"),
            })?;
        }

        // 5. Schedule the worker
        self.schedule(job_id).await?;
        Ok(job_id)
    }

    /// Block until a dispatched job completes and return its final response text.
    ///
    /// Registers a completion waiter for the given job, then blocks until either
    /// the worker finishes or the timeout elapses. On completion, reads the job
    /// context to determine success/failure and extracts the last assistant
    /// message from the conversation as the response text.
    pub async fn await_job(
        &self,
        job_id: Uuid,
        timeout: Duration,
    ) -> Result<String, JobError> {
        // Register a waiter unconditionally, then check if the job is already
        // gone. This closes the race where the cleanup task fires between
        // checking `jobs` and registering the waiter -- if that happens the
        // waiter would never be notified and we'd hang until timeout.
        let (tx, rx) = oneshot::channel();
        self.completion_waiters
            .write()
            .await
            .entry(job_id)
            .or_default()
            .push(tx);

        // Re-check: if the job is no longer in `jobs`, the cleanup task
        // already fired (and already notified all waiters that were registered
        // at that point). Our waiter might or might not have been notified
        // depending on timing, so just skip straight to reading the result.
        let already_done = !self.jobs.read().await.contains_key(&job_id);

        if !already_done {
            // Wait for completion (with timeout)
            tokio::time::timeout(timeout, rx)
                .await
                .map_err(|_| JobError::Stuck {
                    id: job_id,
                    duration: timeout,
                })?
                .map_err(|_| JobError::Failed {
                    id: job_id,
                    reason: "Job completion channel dropped unexpectedly".to_string(),
                })?;
        }

        // Read final job state
        let job_ctx = self.context_manager.get_context(job_id).await?;

        match job_ctx.state {
            JobState::Completed => {}
            JobState::Failed => {
                let reason = job_ctx
                    .transitions
                    .last()
                    .and_then(|t| t.reason.clone())
                    .unwrap_or_else(|| "unknown failure".to_string());
                return Err(JobError::Failed {
                    id: job_id,
                    reason,
                });
            }
            JobState::Stuck => {
                return Err(JobError::Stuck {
                    id: job_id,
                    duration: Duration::from_secs(0),
                });
            }
            JobState::Cancelled => {
                return Err(JobError::Failed {
                    id: job_id,
                    reason: "Job was cancelled".to_string(),
                });
            }
            other => {
                return Err(JobError::Failed {
                    id: job_id,
                    reason: format!("Job ended in unexpected state: {other}"),
                });
            }
        }

        // Extract the last assistant message from the conversation
        if let (Some(store), Some(conversation_id)) = (&self.store, job_ctx.conversation_id) {
            let messages = store
                .list_conversation_messages(conversation_id)
                .await
                .map_err(|e| JobError::Failed {
                    id: job_id,
                    reason: format!("Failed to read conversation messages: {e}"),
                })?;

            // Find the last assistant message
            if let Some(msg) = messages.iter().rev().find(|m| m.role == "assistant") {
                return Ok(msg.content.clone());
            }
        }

        // Fallback: no conversation or no assistant messages found
        Ok("Job completed successfully.".to_string())
    }

    /// Schedule a job for execution.
    pub async fn schedule(&self, job_id: Uuid) -> Result<(), JobError> {
        self.schedule_with_context(job_id, None).await
    }

    /// Schedule a job with an optional approval context.
    async fn schedule_with_context(
        &self,
        job_id: Uuid,
        approval_context: Option<ApprovalContext>,
    ) -> Result<(), JobError> {
        // Hold write lock for the entire check-insert sequence to prevent
        // TOCTOU races where two concurrent calls both pass the checks.
        {
            let mut jobs = self.jobs.write().await;

            if jobs.contains_key(&job_id) {
                return Ok(());
            }

            if jobs.len() >= self.config.max_parallel_jobs {
                return Err(JobError::MaxJobsExceeded {
                    max: self.config.max_parallel_jobs,
                });
            }

            // Transition job to in_progress
            self.context_manager
                .update_context(job_id, |ctx| {
                    ctx.transition_to(
                        JobState::InProgress,
                        Some("Scheduled for execution".to_string()),
                    )
                })
                .await?
                .map_err(|s| JobError::ContextError {
                    id: job_id,
                    reason: s,
                })?;

            // Create worker channel
            let (tx, rx) = mpsc::channel(16);

            // Create worker with shared dependencies
            let deps = WorkerDeps {
                context_manager: self.context_manager.clone(),
                llm: self.llm.clone(),
                safety: self.safety.clone(),
                tools: self.tools.clone(),
                store: self.store.clone(),
                hooks: self.hooks.clone(),
                timeout: self.config.job_timeout,
                use_planning: self.config.use_planning,
                sse_tx: self.sse_tx.clone(),
                approval_context,
                http_interceptor: self.http_interceptor.clone(),
                core_tools: self.core_tools.clone(),
            };
            let worker = Worker::new(job_id, deps);

            // Spawn worker task
            let handle = tokio::spawn(async move {
                if let Err(e) = worker.run(rx).await {
                    tracing::error!("Worker for job {} failed: {}", job_id, e);
                }
            });

            // Start the worker
            if tx.send(WorkerMessage::Start).await.is_err() {
                tracing::error!(job_id = %job_id, "Worker died before receiving Start message");
            }

            // Insert while still holding the write lock
            jobs.insert(job_id, ScheduledJob { handle, tx });
        }

        // Cleanup task for this job to avoid capacity leaks
        let jobs = Arc::clone(&self.jobs);
        let completion_waiters = Arc::clone(&self.completion_waiters);
        tokio::spawn(async move {
            loop {
                let finished = {
                    let jobs_read = jobs.read().await;
                    match jobs_read.get(&job_id) {
                        Some(scheduled) => scheduled.handle.is_finished(),
                        None => true,
                    }
                };

                if finished {
                    jobs.write().await.remove(&job_id);
                    // Notify any waiters that the job has finished
                    if let Some(waiters) = completion_waiters.write().await.remove(&job_id) {
                        for tx in waiters {
                            let _ = tx.send(());
                        }
                    }
                    break;
                }

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        tracing::info!("Scheduled job {} for execution", job_id);
        Ok(())
    }

    /// Schedule a sub-task from within a worker.
    ///
    /// Sub-tasks are lightweight tasks that don't go through the full job lifecycle.
    /// They're used for parallel tool execution and background computations.
    ///
    /// Returns a oneshot receiver to get the result.
    pub async fn spawn_subtask(
        &self,
        parent_id: Uuid,
        task: Task,
    ) -> Result<oneshot::Receiver<Result<TaskOutput, Error>>, JobError> {
        let task_id = Uuid::new_v4();
        let (result_tx, result_rx) = oneshot::channel();

        let handle = match task {
            Task::Job { .. } => {
                // Jobs should go through schedule(), not spawn_subtask
                return Err(JobError::ContextError {
                    id: parent_id,
                    reason: "Use schedule() for Job tasks, not spawn_subtask()".to_string(),
                });
            }

            Task::ToolExec {
                parent_id: tool_parent_id,
                tool_name,
                params,
            } => {
                let tools = self.tools.clone();
                let context_manager = self.context_manager.clone();
                let safety = self.safety.clone();

                // TODO: propagate parent job's ApprovalContext here when subtasks
                // are used in autonomous/routine paths (currently only used in tests).
                tokio::spawn(async move {
                    let result = Self::execute_tool_task(
                        tools,
                        context_manager,
                        safety,
                        None,
                        tool_parent_id,
                        &tool_name,
                        params,
                    )
                    .await;

                    // Send result (ignore if receiver dropped)
                    let _ = result_tx.send(result);
                })
            }

            Task::Background { id: _, handler } => {
                let ctx = TaskContext::new(task_id).with_parent(parent_id);

                tokio::spawn(async move {
                    let result = handler.run(ctx).await;
                    let _ = result_tx.send(result);
                })
            }
        };

        // Track the subtask
        self.subtasks.write().await.insert(
            task_id,
            ScheduledSubtask {
                handle: tokio::spawn(async move {
                    // Wrap the handle to get its result
                    match handle.await {
                        Ok(()) => Err(Error::Job(JobError::ContextError {
                            id: task_id,
                            reason: "Subtask completed but result not captured".to_string(),
                        })),
                        Err(e) => Err(Error::Job(JobError::ContextError {
                            id: task_id,
                            reason: format!("Subtask panicked: {}", e),
                        })),
                    }
                }),
            },
        );

        // Cleanup task for subtask tracking
        let subtasks = Arc::clone(&self.subtasks);
        tokio::spawn(async move {
            loop {
                let finished = {
                    let subtasks_read = subtasks.read().await;
                    match subtasks_read.get(&task_id) {
                        Some(scheduled) => scheduled.handle.is_finished(),
                        None => true,
                    }
                };

                if finished {
                    subtasks.write().await.remove(&task_id);
                    break;
                }

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        tracing::debug!(
            parent_id = %parent_id,
            task_id = %task_id,
            "Spawned subtask"
        );

        Ok(result_rx)
    }

    /// Schedule multiple tasks in parallel and wait for all to complete.
    ///
    /// Returns results in the same order as the input tasks.
    pub async fn spawn_batch(
        &self,
        parent_id: Uuid,
        tasks: Vec<Task>,
    ) -> Vec<Result<TaskOutput, Error>> {
        if tasks.is_empty() {
            return Vec::new();
        }

        let mut receivers = Vec::with_capacity(tasks.len());

        // Spawn all tasks
        for task in tasks {
            match self.spawn_subtask(parent_id, task).await {
                Ok(rx) => receivers.push(Some(rx)),
                Err(e) => {
                    // Store the error directly
                    receivers.push(None);
                    tracing::warn!(
                        parent_id = %parent_id,
                        error = %e,
                        "Failed to spawn subtask in batch"
                    );
                }
            }
        }

        // Collect results
        let mut results = Vec::with_capacity(receivers.len());
        for rx in receivers {
            let result = match rx {
                Some(receiver) => match receiver.await {
                    Ok(task_result) => task_result,
                    Err(_) => Err(Error::Job(JobError::ContextError {
                        id: parent_id,
                        reason: "Subtask channel closed unexpectedly".to_string(),
                    })),
                },
                None => Err(Error::Job(JobError::ContextError {
                    id: parent_id,
                    reason: "Subtask failed to spawn".to_string(),
                })),
            };
            results.push(result);
        }

        results
    }

    /// Execute a single tool as a subtask.
    async fn execute_tool_task(
        tools: Arc<ToolRegistry>,
        context_manager: Arc<ContextManager>,
        safety: Arc<SafetyLayer>,
        approval_context: Option<ApprovalContext>,
        job_id: Uuid,
        tool_name: &str,
        params: serde_json::Value,
    ) -> Result<TaskOutput, Error> {
        let start = std::time::Instant::now();

        // Get the tool
        let tool = tools.get(tool_name).await.ok_or_else(|| {
            Error::Tool(crate::error::ToolError::NotFound {
                name: tool_name.to_string(),
            })
        })?;

        // Get job context
        let job_ctx: JobContext = context_manager.get_context(job_id).await?;
        if job_ctx.state == JobState::Cancelled {
            return Err(crate::error::ToolError::ExecutionFailed {
                name: tool_name.to_string(),
                reason: "Job is cancelled".to_string(),
            }
            .into());
        }

        let requirement = tool.requires_approval(&params);
        let blocked =
            ApprovalContext::is_blocked_or_default(&approval_context, tool_name, requirement);
        if blocked {
            return Err(crate::error::ToolError::AuthRequired {
                name: tool_name.to_string(),
            }
            .into());
        }

        // Validate tool parameters
        let validation = safety.validator().validate_tool_params(&params);
        if !validation.is_valid {
            let details = validation
                .errors
                .iter()
                .map(|e| format!("{}: {}", e.field, e.message))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(crate::error::ToolError::InvalidParameters {
                name: tool_name.to_string(),
                reason: format!("Invalid tool parameters: {}", details),
            }
            .into());
        }

        // Execute with per-tool timeout
        let tool_timeout = tool.execution_timeout();
        let result =
            tokio::time::timeout(tool_timeout, async { tool.execute(params, &job_ctx).await })
                .await
                .map_err(|_| {
                    Error::Tool(crate::error::ToolError::Timeout {
                        name: tool_name.to_string(),
                        timeout: tool_timeout,
                    })
                })?
                .map_err(|e| {
                    Error::Tool(crate::error::ToolError::ExecutionFailed {
                        name: tool_name.to_string(),
                        reason: e.to_string(),
                    })
                })?;

        Ok(TaskOutput::new(result.result, start.elapsed()))
    }

    /// Stop a running job.
    pub async fn stop(&self, job_id: Uuid) -> Result<(), JobError> {
        let mut jobs = self.jobs.write().await;

        if let Some(scheduled) = jobs.remove(&job_id) {
            // Send stop signal
            let _ = scheduled.tx.send(WorkerMessage::Stop).await;

            // Give it a moment to clean up
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            // Abort if still running
            if !scheduled.handle.is_finished() {
                scheduled.handle.abort();
            }

            // Update job state
            self.context_manager
                .update_context(job_id, |ctx| {
                    if let Err(e) = ctx.transition_to(
                        JobState::Cancelled,
                        Some("Stopped by scheduler".to_string()),
                    ) {
                        tracing::warn!(
                            job_id = %job_id,
                            error = %e,
                            "Failed to transition job to Cancelled state"
                        );
                    }
                })
                .await?;

            // Persist cancellation (fire-and-forget)
            if let Some(ref store) = self.store {
                let store = store.clone();
                tokio::spawn(async move {
                    if let Err(e) = store
                        .update_job_status(
                            job_id,
                            JobState::Cancelled,
                            Some("Stopped by scheduler"),
                        )
                        .await
                    {
                        tracing::warn!("Failed to persist cancellation for job {}: {}", job_id, e);
                    }
                });
            }

            tracing::info!("Stopped job {}", job_id);
        }

        Ok(())
    }

    /// Send a follow-up user message to a running job.
    ///
    /// Returns `Ok(())` if the message was queued, `Err` if the job is not running.
    pub async fn send_message(&self, job_id: Uuid, content: String) -> Result<(), JobError> {
        // Clone the sender while holding the lock, then release before the
        // async send to avoid blocking scheduler writes during backpressure.
        let tx = {
            let jobs = self.jobs.read().await;
            let scheduled = jobs.get(&job_id).ok_or(JobError::NotFound { id: job_id })?;
            scheduled.tx.clone()
        };
        tx.send(WorkerMessage::UserMessage(content))
            .await
            .map_err(|_| JobError::Failed {
                id: job_id,
                reason: "Worker channel closed".to_string(),
            })?;
        Ok(())
    }

    /// Check if a job is running.
    pub async fn is_running(&self, job_id: Uuid) -> bool {
        self.jobs.read().await.contains_key(&job_id)
    }

    /// Get count of running jobs.
    pub async fn running_count(&self) -> usize {
        self.jobs.read().await.len()
    }

    /// Get count of running subtasks.
    pub async fn subtask_count(&self) -> usize {
        self.subtasks.read().await.len()
    }

    /// Get all running job IDs.
    pub async fn running_jobs(&self) -> Vec<Uuid> {
        self.jobs.read().await.keys().cloned().collect()
    }

    /// Clean up finished jobs and subtasks.
    pub async fn cleanup_finished(&self) {
        // Clean up jobs
        {
            let mut jobs = self.jobs.write().await;
            let mut finished = Vec::new();

            for (id, scheduled) in jobs.iter() {
                if scheduled.handle.is_finished() {
                    finished.push(*id);
                }
            }

            for id in finished {
                jobs.remove(&id);
                tracing::debug!("Cleaned up finished job {}", id);
            }
        }

        // Clean up subtasks
        {
            let mut subtasks = self.subtasks.write().await;
            let mut finished = Vec::new();

            for (id, scheduled) in subtasks.iter() {
                if scheduled.handle.is_finished() {
                    finished.push(*id);
                }
            }

            for id in finished {
                subtasks.remove(&id);
                tracing::trace!("Cleaned up finished subtask {}", id);
            }
        }
    }

    /// Stop all jobs.
    pub async fn stop_all(&self) {
        let job_ids: Vec<Uuid> = self.jobs.read().await.keys().cloned().collect();

        for job_id in job_ids {
            let _ = self.stop(job_id).await;
        }

        // Abort all subtasks
        let mut subtasks = self.subtasks.write().await;
        for (_, scheduled) in subtasks.drain() {
            scheduled.handle.abort();
        }
    }

    /// Get access to the tools registry.
    pub fn tools(&self) -> &Arc<ToolRegistry> {
        &self.tools
    }

    /// Get access to the context manager.
    pub fn context_manager(&self) -> &Arc<ContextManager> {
        &self.context_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use rust_decimal::Decimal;
    use tokio::sync::oneshot;

    use crate::config::AgentConfig;
    use crate::config::SafetyConfig;
    use crate::context::ContextManager;
    use crate::error::{JobError, LlmError};
    use crate::hooks::HookRegistry;
    use crate::llm::{
        CompletionRequest, CompletionResponse, LlmProvider,
        ToolCompletionRequest, ToolCompletionResponse,
    };
    use crate::safety::SafetyLayer;
    use crate::tools::{ApprovalRequirement, Tool, ToolError, ToolOutput, ToolRegistry};

    /// Stub LLM provider that always returns an error. Tests that exercise
    /// `await_job` / waiter mechanics never reach the LLM, so this is fine.
    struct StubLlm;

    #[async_trait]
    impl LlmProvider for StubLlm {
        fn model_name(&self) -> &str {
            "stub"
        }

        fn cost_per_token(&self) -> (Decimal, Decimal) {
            (Decimal::ZERO, Decimal::ZERO)
        }

        async fn complete(&self, _req: CompletionRequest) -> Result<CompletionResponse, LlmError> {
            Err(LlmError::RequestFailed {
                provider: "stub".to_string(),
                reason: "stub provider".to_string(),
            })
        }

        async fn complete_with_tools(
            &self,
            _req: ToolCompletionRequest,
        ) -> Result<ToolCompletionResponse, LlmError> {
            Err(LlmError::RequestFailed {
                provider: "stub".to_string(),
                reason: "stub provider".to_string(),
            })
        }
    }

    /// Build a test `AgentConfig` with sensible defaults.
    fn test_agent_config() -> AgentConfig {
        AgentConfig {
            name: "test".to_string(),
            max_parallel_jobs: 10,
            job_timeout: Duration::from_secs(30),
            stuck_threshold: Duration::from_secs(60),
            repair_check_interval: Duration::from_secs(60),
            max_repair_attempts: 3,
            use_planning: false,
            session_idle_timeout: Duration::from_secs(600),
            allow_local_tools: false,
            max_cost_per_day_cents: None,
            max_actions_per_hour: None,
            max_tool_iterations: 50,
            auto_approve_tools: false,
            default_timezone: "UTC".to_string(),
        }
    }

    /// Build a minimal `Scheduler` suitable for testing waiter/await mechanics.
    /// The LLM is a stub so worker-driven jobs will fail, but direct
    /// manipulation of `jobs` and `completion_waiters` works.
    fn test_scheduler() -> Scheduler {
        let ctx_mgr = Arc::new(ContextManager::new(10));
        let safety = Arc::new(SafetyLayer::new(&SafetyConfig {
            max_output_length: 100_000,
            injection_check_enabled: false,
        }));
        Scheduler::new(
            test_agent_config(),
            ctx_mgr,
            Arc::new(StubLlm),
            safety,
            Arc::new(ToolRegistry::new()),
            None,
            Arc::new(HookRegistry::new()),
        )
    }

    // ---------------------------------------------------------------
    // Waiter mechanism: register, notify, receive
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_waiter_notified_on_job_removal() {
        // Simulates the cleanup task: insert a fake job, register a waiter,
        // remove the job and fire waiters, verify the waiter receives the signal.
        let scheduler = test_scheduler();
        let job_id = Uuid::new_v4();

        // Insert a fake entry into `jobs` so the scheduler thinks it's running.
        let (dummy_tx, _dummy_rx) = mpsc::channel(1);
        let handle = tokio::spawn(async {});
        scheduler
            .jobs
            .write()
            .await
            .insert(job_id, ScheduledJob { handle, tx: dummy_tx });

        // Register a waiter.
        let (tx, rx) = oneshot::channel();
        scheduler
            .completion_waiters
            .write()
            .await
            .entry(job_id)
            .or_default()
            .push(tx);

        // Simulate cleanup: remove from jobs, notify waiters.
        scheduler.jobs.write().await.remove(&job_id);
        if let Some(waiters) = scheduler.completion_waiters.write().await.remove(&job_id) {
            for w in waiters {
                let _ = w.send(());
            }
        }

        // The waiter should have been notified.
        let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
        assert!(result.is_ok(), "waiter should have been notified");
        assert!(result.unwrap().is_ok(), "oneshot should receive ()");
    }

    #[tokio::test]
    async fn test_multiple_waiters_all_notified() {
        let scheduler = test_scheduler();
        let job_id = Uuid::new_v4();

        // Insert a fake job.
        let (dummy_tx, _dummy_rx) = mpsc::channel(1);
        let handle = tokio::spawn(async {});
        scheduler
            .jobs
            .write()
            .await
            .insert(job_id, ScheduledJob { handle, tx: dummy_tx });

        // Register 5 waiters.
        let mut receivers = Vec::new();
        for _ in 0..5 {
            let (tx, rx) = oneshot::channel();
            scheduler
                .completion_waiters
                .write()
                .await
                .entry(job_id)
                .or_default()
                .push(tx);
            receivers.push(rx);
        }

        // Notify all.
        if let Some(waiters) = scheduler.completion_waiters.write().await.remove(&job_id) {
            for w in waiters {
                let _ = w.send(());
            }
        }

        for (i, rx) in receivers.into_iter().enumerate() {
            let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
            assert!(result.is_ok(), "waiter {i} should have been notified");
        }
    }

    // ---------------------------------------------------------------
    // await_job: already-done fast path (the race condition fix)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_await_job_already_done_returns_immediately() {
        // The key race fix: if a job is already gone from `jobs` by the time
        // we check, `await_job` should skip waiting and go straight to reading
        // the context. We create a completed job in ContextManager (but do NOT
        // insert it into `jobs`) so the fast path fires.
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Already done", "desc")
            .await
            .unwrap();

        // Transition to InProgress then Completed so state machine is happy.
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(
                    crate::context::JobState::InProgress,
                    Some("started".to_string()),
                )
            })
            .await
            .unwrap()
            .unwrap();

        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(
                    crate::context::JobState::Completed,
                    Some("done".to_string()),
                )
            })
            .await
            .unwrap()
            .unwrap();

        // Job is NOT in `scheduler.jobs` -- simulates cleanup already ran.
        // `await_job` should return the fallback message (no DB, so no
        // conversation messages).
        let result = scheduler
            .await_job(job_id, Duration::from_millis(100))
            .await;
        assert!(result.is_ok(), "should succeed: {result:?}");
        assert_eq!(result.unwrap(), "Job completed successfully.");
    }

    // ---------------------------------------------------------------
    // await_job: timeout
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_await_job_timeout() {
        // Insert a fake job that never finishes — await_job should hit timeout.
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Stuck job", "desc")
            .await
            .unwrap();

        // Insert a fake running job (a future that sleeps forever).
        let (dummy_tx, _dummy_rx) = mpsc::channel(1);
        let handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
        });
        scheduler
            .jobs
            .write()
            .await
            .insert(job_id, ScheduledJob { handle, tx: dummy_tx });

        let result = scheduler
            .await_job(job_id, Duration::from_millis(50))
            .await;

        assert!(result.is_err(), "should timeout");
        match result.unwrap_err() {
            JobError::Stuck { id, .. } => assert_eq!(id, job_id),
            other => panic!("expected Stuck error, got: {other}"),
        }

        // Clean up the spawned task.
        if let Some(job) = scheduler.jobs.write().await.remove(&job_id) {
            job.handle.abort();
        }
    }

    // ---------------------------------------------------------------
    // await_job: failed / stuck / cancelled states
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_await_job_failed_state() {
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Failing job", "desc")
            .await
            .unwrap();

        // Transition: Pending -> InProgress -> Failed
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::InProgress, None)
            })
            .await
            .unwrap()
            .unwrap();
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(
                    crate::context::JobState::Failed,
                    Some("LLM refused".to_string()),
                )
            })
            .await
            .unwrap()
            .unwrap();

        // Job is not in `jobs` (already cleaned up).
        let result = scheduler
            .await_job(job_id, Duration::from_millis(100))
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            JobError::Failed { id, reason } => {
                assert_eq!(id, job_id);
                assert!(reason.contains("LLM refused"), "reason: {reason}");
            }
            other => panic!("expected Failed, got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_await_job_cancelled_state() {
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Cancelled job", "desc")
            .await
            .unwrap();

        // Transition: Pending -> InProgress -> Cancelled
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::InProgress, None)
            })
            .await
            .unwrap()
            .unwrap();
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(
                    crate::context::JobState::Cancelled,
                    Some("user cancelled".to_string()),
                )
            })
            .await
            .unwrap()
            .unwrap();

        let result = scheduler
            .await_job(job_id, Duration::from_millis(100))
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            JobError::Failed { reason, .. } => {
                assert!(reason.contains("cancelled"), "reason: {reason}");
            }
            other => panic!("expected Failed (cancelled), got: {other}"),
        }
    }

    #[tokio::test]
    async fn test_await_job_stuck_state() {
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Stuck job", "desc")
            .await
            .unwrap();

        // Transition: Pending -> InProgress -> Stuck
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::InProgress, None)
            })
            .await
            .unwrap()
            .unwrap();
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::Stuck, None)
            })
            .await
            .unwrap()
            .unwrap();

        let result = scheduler
            .await_job(job_id, Duration::from_millis(100))
            .await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), JobError::Stuck { .. }));
    }

    // ---------------------------------------------------------------
    // Race condition: waiter registered just as cleanup fires
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_await_job_race_cleanup_fires_between_register_and_check() {
        // Reproduce the race scenario that was fixed:
        // 1. Job exists in `jobs` when await_job starts
        // 2. Between registering the waiter and checking `already_done`,
        //    the cleanup fires (removes from `jobs`, notifies existing waiters)
        // 3. Our waiter WAS registered before cleanup, so it gets notified
        //
        // We simulate this by:
        // - Inserting a job that finishes almost instantly
        // - Calling await_job which registers the waiter
        // - The cleanup loop (spawned by `schedule`) would fire, but here we
        //   simulate it manually to control timing.
        let scheduler = test_scheduler();

        let job_id = scheduler
            .context_manager
            .create_job_for_user("test-user", "Race test", "desc")
            .await
            .unwrap();

        // Transition to Completed so await_job can read the final state.
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::InProgress, None)
            })
            .await
            .unwrap()
            .unwrap();
        scheduler
            .context_manager
            .update_context(job_id, |ctx| {
                ctx.transition_to(crate::context::JobState::Completed, None)
            })
            .await
            .unwrap()
            .unwrap();

        // Insert a fake job that is already finished (handle completes immediately).
        let (dummy_tx, _dummy_rx) = mpsc::channel(1);
        let handle = tokio::spawn(async {}); // finishes right away
        // Let the handle finish.
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(handle.is_finished());

        scheduler
            .jobs
            .write()
            .await
            .insert(job_id, ScheduledJob { handle, tx: dummy_tx });

        // Spawn a task that simulates cleanup after a small delay.
        let jobs = Arc::clone(&scheduler.jobs);
        let waiters = Arc::clone(&scheduler.completion_waiters);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            jobs.write().await.remove(&job_id);
            if let Some(ws) = waiters.write().await.remove(&job_id) {
                for w in ws {
                    let _ = w.send(());
                }
            }
        });

        // `await_job` registers a waiter, then the cleanup fires, notifying it.
        let result = scheduler
            .await_job(job_id, Duration::from_secs(2))
            .await;
        assert!(result.is_ok(), "should succeed via waiter notification: {result:?}");
    }

    // ---------------------------------------------------------------
    // await_job for nonexistent job
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_await_job_nonexistent_job() {
        let scheduler = test_scheduler();
        let fake_id = Uuid::new_v4();

        // Job doesn't exist in `jobs` (already_done path), but also not in
        // ContextManager → should return NotFound.
        let result = scheduler
            .await_job(fake_id, Duration::from_millis(100))
            .await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), JobError::NotFound { .. }),
            "expected NotFound for nonexistent job"
        );
    }

    // ---------------------------------------------------------------
    // dispatch_job creates a running job
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_dispatch_job_creates_running_job() {
        let scheduler = test_scheduler();

        let job_id = scheduler
            .dispatch_job("test-user", "Test job", "description", None)
            .await
            .unwrap();

        // The job should be tracked as running (at least briefly, before the
        // stub LLM makes the worker fail and cleanup removes it).
        // Check the context was created.
        let ctx = scheduler.context_manager.get_context(job_id).await.unwrap();
        assert_eq!(ctx.title, "Test job");
        assert_eq!(ctx.user_id, "test-user");
    }

    // ---------------------------------------------------------------
    // dispatch_job_to_conversation sets conversation_id
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_dispatch_job_to_conversation_sets_conversation_id() {
        let scheduler = test_scheduler();
        let conversation_id = Uuid::new_v4();

        // No DB, so `add_conversation_message` is skipped (store is None).
        let job_id = scheduler
            .dispatch_job_to_conversation(
                "test-user",
                conversation_id,
                "Conv job",
                "hello",
                None,
            )
            .await
            .unwrap();

        let ctx = scheduler.context_manager.get_context(job_id).await.unwrap();
        assert_eq!(ctx.conversation_id, Some(conversation_id));
        assert_eq!(ctx.title, "Conv job");
    }

    // ---------------------------------------------------------------
    // dispatch_job with metadata
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_dispatch_job_with_metadata() {
        let scheduler = test_scheduler();
        let meta = serde_json::json!({"max_iterations": 5});

        let job_id = scheduler
            .dispatch_job("test-user", "Meta job", "desc", Some(meta.clone()))
            .await
            .unwrap();

        let ctx = scheduler.context_manager.get_context(job_id).await.unwrap();
        assert_eq!(ctx.metadata["max_iterations"], 5);
    }

    /// A tool that returns `UnlessAutoApproved`.
    struct SoftApprovalTool;

    #[async_trait::async_trait]
    impl Tool for SoftApprovalTool {
        fn name(&self) -> &str {
            "soft_gate"
        }
        fn description(&self) -> &str {
            "needs soft approval"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: &JobContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(
                "soft_ok",
                std::time::Instant::now().elapsed(),
            ))
        }
        fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
            ApprovalRequirement::UnlessAutoApproved
        }
        fn requires_sanitization(&self) -> bool {
            false
        }
    }

    /// A tool that returns `Always`.
    struct HardApprovalTool;

    #[async_trait::async_trait]
    impl Tool for HardApprovalTool {
        fn name(&self) -> &str {
            "hard_gate"
        }
        fn description(&self) -> &str {
            "needs hard approval"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: &JobContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(
                "hard_ok",
                std::time::Instant::now().elapsed(),
            ))
        }
        fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
            ApprovalRequirement::Always
        }
        fn requires_sanitization(&self) -> bool {
            false
        }
    }

    async fn setup_tools_and_job() -> (
        Arc<ToolRegistry>,
        Arc<ContextManager>,
        Arc<SafetyLayer>,
        Uuid,
    ) {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(SoftApprovalTool)).await;
        registry.register(Arc::new(HardApprovalTool)).await;

        let cm = Arc::new(ContextManager::new(5));
        let job_id = cm.create_job("test", "approval test").await.unwrap();
        cm.update_context(job_id, |ctx| ctx.transition_to(JobState::InProgress, None))
            .await
            .unwrap()
            .unwrap();

        let safety = Arc::new(SafetyLayer::new(&SafetyConfig {
            max_output_length: 100_000,
            injection_check_enabled: false,
        }));

        (Arc::new(registry), cm, safety, job_id)
    }

    #[tokio::test]
    async fn test_execute_tool_task_blocks_without_context() {
        let (tools, cm, safety, job_id) = setup_tools_and_job().await;

        // Without approval context, UnlessAutoApproved is blocked
        let result = Scheduler::execute_tool_task(
            tools.clone(),
            cm.clone(),
            safety.clone(),
            None,
            job_id,
            "soft_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(
            result.is_err(),
            "soft_gate should be blocked without context"
        );

        // Always is also blocked
        let result = Scheduler::execute_tool_task(
            tools,
            cm,
            safety,
            None,
            job_id,
            "hard_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(
            result.is_err(),
            "hard_gate should be blocked without context"
        );
    }

    #[tokio::test]
    async fn test_execute_tool_task_autonomous_unblocks_soft() {
        let (tools, cm, safety, job_id) = setup_tools_and_job().await;

        // Autonomous context auto-approves UnlessAutoApproved
        let result = Scheduler::execute_tool_task(
            tools.clone(),
            cm.clone(),
            safety.clone(),
            Some(ApprovalContext::autonomous()),
            job_id,
            "soft_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(
            result.is_ok(),
            "soft_gate should pass with autonomous context"
        );

        // But still blocks Always
        let result = Scheduler::execute_tool_task(
            tools,
            cm,
            safety,
            Some(ApprovalContext::autonomous()),
            job_id,
            "hard_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(
            result.is_err(),
            "hard_gate should still be blocked without explicit permission"
        );
    }

    #[tokio::test]
    async fn test_execute_tool_task_autonomous_with_permissions() {
        let (tools, cm, safety, job_id) = setup_tools_and_job().await;

        // Autonomous context with explicit permission for hard_gate
        let ctx = ApprovalContext::autonomous_with_tools(["hard_gate".to_string()]);

        let result = Scheduler::execute_tool_task(
            tools.clone(),
            cm.clone(),
            safety.clone(),
            Some(ctx.clone()),
            job_id,
            "soft_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(result.is_ok(), "soft_gate should pass");

        let result = Scheduler::execute_tool_task(
            tools,
            cm,
            safety,
            Some(ctx),
            job_id,
            "hard_gate",
            serde_json::json!({}),
        )
        .await;
        assert!(
            result.is_ok(),
            "hard_gate should pass with explicit permission"
        );
    }
}
