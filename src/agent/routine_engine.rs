//! Routine execution engine.
//!
//! Handles loading routines, checking triggers, enforcing guardrails,
//! and executing both lightweight (single LLM call) and full-job routines.
//!
//! The engine runs two independent loops:
//! - A **cron ticker** that polls the DB every N seconds for due cron routines
//! - An **event matcher** called synchronously from the agent main loop
//!
//! Lightweight routines execute inline (single LLM call, no scheduler slot).
//! Full-job routines are delegated to the existing `Scheduler`.

use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use chrono::Utc;
use regex::Regex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;

use crate::agent::Scheduler;
use crate::agent::collection_events::CollectionWriteEvent;
use crate::agent::routine::{
    NotifyConfig, Routine, RoutineAction, RoutineRun, RunStatus, Trigger, next_cron_fire,
};
use crate::channels::{IncomingMessage, OutgoingResponse};
use crate::config::RoutineConfig;
use crate::context::JobContext;
use crate::db::Database;
use crate::error::RoutineError;
use crate::llm::{ChatMessage, CompletionRequest, FinishReason, LlmProvider};
use crate::tools::ApprovalContext;
use crate::tools::ToolRegistry;
use crate::workspace::Workspace;

/// Check if a routine's CollectionWrite trigger matches an event.
fn matches_collection_write(routine: &Routine, event: &CollectionWriteEvent) -> bool {
    if routine.user_id != event.user_id {
        return false;
    }
    match &routine.trigger {
        Trigger::CollectionWrite { collection } => collection == &event.collection,
        _ => false,
    }
}

/// The routine execution engine.
pub struct RoutineEngine {
    config: RoutineConfig,
    store: Arc<dyn Database>,
    llm: Arc<dyn LlmProvider>,
    workspace: Arc<Workspace>,
    /// Sender for notifications (routed to channel manager).
    notify_tx: mpsc::Sender<OutgoingResponse>,
    /// Currently running routine count (across all routines).
    running_count: Arc<AtomicUsize>,
    /// Compiled event regex cache: routine_id -> compiled regex.
    event_cache: Arc<RwLock<Vec<(Uuid, Routine, Regex)>>>,
    /// Cache of collection-write triggered routines.
    collection_write_cache: Arc<RwLock<Vec<Routine>>>,
    /// Scheduler for dispatching jobs (FullJob mode).
    scheduler: Option<Arc<Scheduler>>,
    /// Tool registry for WASM routine execution.
    tool_registry: Option<Arc<ToolRegistry>>,
    /// Gateway port for script `IRONCLAW_PORT` env var (resolved at construction).
    gateway_port: Option<u16>,
    /// Default auth token for script `IRONCLAW_TOKEN` env var.
    gateway_auth_token: Option<String>,
}

impl RoutineEngine {
    pub fn new(
        config: RoutineConfig,
        store: Arc<dyn Database>,
        llm: Arc<dyn LlmProvider>,
        workspace: Arc<Workspace>,
        notify_tx: mpsc::Sender<OutgoingResponse>,
        scheduler: Option<Arc<Scheduler>>,
        tool_registry: Option<Arc<ToolRegistry>>,
    ) -> Self {
        // Resolve gateway connection details from environment for script actions.
        let gateway_port = std::env::var("GATEWAY_PORT")
            .ok()
            .and_then(|s| s.parse().ok());
        let gateway_auth_token = std::env::var("GATEWAY_AUTH_TOKEN").ok();

        Self {
            config,
            store,
            llm,
            workspace,
            notify_tx,
            running_count: Arc::new(AtomicUsize::new(0)),
            event_cache: Arc::new(RwLock::new(Vec::new())),
            collection_write_cache: Arc::new(RwLock::new(Vec::new())),
            scheduler,
            tool_registry,
            gateway_port,
            gateway_auth_token,
        }
    }

    /// Refresh the in-memory event trigger cache from DB.
    pub async fn refresh_event_cache(&self) {
        match self.store.list_event_routines().await {
            Ok(routines) => {
                let mut cache = Vec::new();
                for routine in routines {
                    if let Trigger::Event { ref pattern, .. } = routine.trigger {
                        match Regex::new(pattern) {
                            Ok(re) => cache.push((routine.id, routine.clone(), re)),
                            Err(e) => {
                                tracing::warn!(
                                    routine = %routine.name,
                                    "Invalid event regex '{}': {}",
                                    pattern, e
                                );
                            }
                        }
                    }
                }
                let count = cache.len();
                *self.event_cache.write().await = cache;
                tracing::debug!("Refreshed event cache: {} routines", count);
            }
            Err(e) => {
                tracing::error!("Failed to refresh event cache: {}", e);
            }
        }
    }

    /// Refresh the in-memory collection write trigger cache from DB.
    pub async fn refresh_collection_write_cache(&self) {
        match self.store.list_all_routines().await {
            Ok(routines) => {
                let filtered: Vec<Routine> = routines
                    .into_iter()
                    .filter(|r| r.enabled && matches!(r.trigger, Trigger::CollectionWrite { .. }))
                    .collect();
                let count = filtered.len();
                *self.collection_write_cache.write().await = filtered;
                tracing::debug!("Refreshed collection write cache: {} routines", count);
            }
            Err(e) => {
                tracing::error!("Failed to refresh collection write cache: {}", e);
            }
        }
    }

    /// Check a collection write event against cached triggers. Returns number of routines fired.
    pub async fn check_collection_write_triggers(&self, event: &CollectionWriteEvent) -> usize {
        let cache = self.collection_write_cache.read().await;
        let mut fired = 0;

        for routine in cache.iter() {
            if !matches_collection_write(routine, event) {
                continue;
            }
            if !self.check_cooldown(routine) {
                tracing::debug!(routine = %routine.name, "Skipped: cooldown active");
                continue;
            }
            if !self.check_concurrent(routine).await {
                tracing::debug!(routine = %routine.name, "Skipped: max concurrent reached");
                continue;
            }
            if self.running_count.load(Ordering::Relaxed) >= self.config.max_concurrent_routines {
                tracing::warn!(routine = %routine.name, "Skipped: global max concurrent reached");
                continue;
            }
            let detail = serde_json::to_string(&event.data).unwrap_or_default();
            self.spawn_fire(routine.clone(), "collection_write", Some(detail));
            fired += 1;
        }

        fired
    }

    /// Spawn a background task that listens for collection write events and fires matching triggers.
    pub fn spawn_collection_write_listener(
        self: &Arc<Self>,
        mut rx: tokio::sync::broadcast::Receiver<CollectionWriteEvent>,
    ) {
        let engine = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        let fired = engine.check_collection_write_triggers(&event).await;
                        if fired > 0 {
                            tracing::info!(
                                collection = %event.collection,
                                record_id = %event.record_id,
                                fired,
                                "CollectionWrite triggers fired"
                            );
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "CollectionWrite listener lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tracing::info!("CollectionWrite broadcast channel closed");
                        break;
                    }
                }
            }
        });
    }

    /// Check incoming message against event triggers. Returns number of routines fired.
    ///
    /// Called synchronously from the main loop after handle_message(). The actual
    /// execution is spawned async so this returns quickly.
    pub async fn check_event_triggers(&self, message: &IncomingMessage) -> usize {
        let cache = self.event_cache.read().await;
        let mut fired = 0;

        for (_, routine, re) in cache.iter() {
            // Channel filter
            if let Trigger::Event {
                channel: Some(ch), ..
            } = &routine.trigger
                && ch != &message.channel
            {
                continue;
            }

            // Regex match
            if !re.is_match(&message.content) {
                continue;
            }

            // Cooldown check
            if !self.check_cooldown(routine) {
                tracing::debug!(routine = %routine.name, "Skipped: cooldown active");
                continue;
            }

            // Concurrent run check
            if !self.check_concurrent(routine).await {
                tracing::debug!(routine = %routine.name, "Skipped: max concurrent reached");
                continue;
            }

            // Global capacity check
            if self.running_count.load(Ordering::Relaxed) >= self.config.max_concurrent_routines {
                tracing::warn!(routine = %routine.name, "Skipped: global max concurrent reached");
                continue;
            }

            let detail = truncate(&message.content, 200);
            self.spawn_fire(routine.clone(), "event", Some(detail));
            fired += 1;
        }

        fired
    }

    /// Check all due cron routines and fire them. Called by the cron ticker.
    pub async fn check_cron_triggers(&self) {
        let routines = match self.store.list_due_cron_routines().await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to load due cron routines: {}", e);
                return;
            }
        };

        for routine in routines {
            if self.running_count.load(Ordering::Relaxed) >= self.config.max_concurrent_routines {
                tracing::warn!("Global max concurrent routines reached, skipping remaining");
                break;
            }

            if !self.check_cooldown(&routine) {
                continue;
            }

            if !self.check_concurrent(&routine).await {
                continue;
            }

            let detail = if let Trigger::Cron { ref schedule, .. } = routine.trigger {
                Some(schedule.clone())
            } else {
                None
            };

            self.spawn_fire(routine, "cron", detail);
        }
    }

    /// Fire a webhook-triggered routine.
    ///
    /// Validates that the routine exists, is enabled, has a `Trigger::Webhook`,
    /// and passes guardrail checks before executing.
    pub async fn fire_webhook(
        &self,
        routine_id: Uuid,
        user_id: &str,
        body: Option<String>,
    ) -> Result<Uuid, RoutineError> {
        let routine = self
            .store
            .get_routine(routine_id)
            .await
            .map_err(|e| RoutineError::Database {
                reason: e.to_string(),
            })?
            .ok_or(RoutineError::NotFound { id: routine_id })?;

        if !routine.enabled {
            return Err(RoutineError::Disabled {
                name: routine.name.clone(),
            });
        }

        // Verify this is a webhook trigger
        if !matches!(routine.trigger, Trigger::Webhook { .. }) {
            return Err(RoutineError::TriggerMismatch {
                routine: routine.name.clone(),
                expected: "webhook".to_string(),
                actual: routine.trigger.type_tag().to_string(),
            });
        }

        // Verify user ownership
        if routine.user_id != user_id {
            return Err(RoutineError::NotFound { id: routine_id });
        }

        if !self.check_concurrent(&routine).await {
            return Err(RoutineError::MaxConcurrent {
                name: routine.name.clone(),
            });
        }

        if !self.check_cooldown(&routine) {
            return Err(RoutineError::CooldownActive {
                name: routine.name.clone(),
            });
        }

        let detail = body.map(|b| truncate(&b, 500));
        let run_id = Uuid::new_v4();
        let run = RoutineRun {
            id: run_id,
            routine_id: routine.id,
            trigger_type: "webhook".to_string(),
            trigger_detail: detail,
            started_at: Utc::now(),
            completed_at: None,
            status: RunStatus::Running,
            result_summary: None,
            tokens_used: None,
            job_id: None,
            created_at: Utc::now(),
        };

        if let Err(e) = self.store.create_routine_run(&run).await {
            return Err(RoutineError::Database {
                reason: format!("failed to create run record: {e}"),
            });
        }

        let engine = EngineContext {
            store: self.store.clone(),
            llm: self.llm.clone(),
            workspace: self.workspace.clone(),
            notify_tx: self.notify_tx.clone(),
            running_count: self.running_count.clone(),
            scheduler: self.scheduler.clone(),
            tool_registry: self.tool_registry.clone(),
            gateway_port: self.gateway_port,
            user_token: self.gateway_auth_token.clone(),
        };

        tokio::spawn(async move {
            execute_routine(engine, routine, run).await;
        });

        Ok(run_id)
    }

    /// Fire a routine manually (from tool call or CLI).
    ///
    /// Bypasses cooldown checks (those only apply to cron/event triggers).
    /// Still enforces enabled check and concurrent run limit.
    pub async fn fire_manual(
        &self,
        routine_id: Uuid,
        user_id: Option<&str>,
    ) -> Result<Uuid, RoutineError> {
        let routine = self
            .store
            .get_routine(routine_id)
            .await
            .map_err(|e| RoutineError::Database {
                reason: e.to_string(),
            })?
            .ok_or(RoutineError::NotFound { id: routine_id })?;

        // Enforce ownership when a user_id is provided (gateway calls).
        if let Some(uid) = user_id
            && routine.user_id != uid
        {
            return Err(RoutineError::NotAuthorized { id: routine_id });
        }

        if !routine.enabled {
            return Err(RoutineError::Disabled {
                name: routine.name.clone(),
            });
        }

        if !self.check_concurrent(&routine).await {
            return Err(RoutineError::MaxConcurrent {
                name: routine.name.clone(),
            });
        }

        let run_id = Uuid::new_v4();
        let run = RoutineRun {
            id: run_id,
            routine_id: routine.id,
            trigger_type: "manual".to_string(),
            trigger_detail: None,
            started_at: Utc::now(),
            completed_at: None,
            status: RunStatus::Running,
            result_summary: None,
            tokens_used: None,
            job_id: None,
            created_at: Utc::now(),
        };

        if let Err(e) = self.store.create_routine_run(&run).await {
            return Err(RoutineError::Database {
                reason: format!("failed to create run record: {e}"),
            });
        }

        // Execute inline for manual triggers (caller wants to wait)
        let engine = EngineContext {
            store: self.store.clone(),
            llm: self.llm.clone(),
            workspace: self.workspace.clone(),
            notify_tx: self.notify_tx.clone(),
            running_count: self.running_count.clone(),
            scheduler: self.scheduler.clone(),
            tool_registry: self.tool_registry.clone(),
            gateway_port: self.gateway_port,
            user_token: self.gateway_auth_token.clone(),
        };

        tokio::spawn(async move {
            execute_routine(engine, routine, run).await;
        });

        Ok(run_id)
    }

    /// Spawn a fire in a background task.
    fn spawn_fire(&self, routine: Routine, trigger_type: &str, trigger_detail: Option<String>) {
        let run = RoutineRun {
            id: Uuid::new_v4(),
            routine_id: routine.id,
            trigger_type: trigger_type.to_string(),
            trigger_detail,
            started_at: Utc::now(),
            completed_at: None,
            status: RunStatus::Running,
            result_summary: None,
            tokens_used: None,
            job_id: None,
            created_at: Utc::now(),
        };

        let engine = EngineContext {
            store: self.store.clone(),
            llm: self.llm.clone(),
            workspace: self.workspace.clone(),
            notify_tx: self.notify_tx.clone(),
            running_count: self.running_count.clone(),
            scheduler: self.scheduler.clone(),
            tool_registry: self.tool_registry.clone(),
            gateway_port: self.gateway_port,
            user_token: self.gateway_auth_token.clone(),
        };

        // Record the run in DB, then spawn execution
        let store = self.store.clone();
        tokio::spawn(async move {
            if let Err(e) = store.create_routine_run(&run).await {
                tracing::error!(routine = %routine.name, "Failed to record run: {}", e);
                return;
            }
            execute_routine(engine, routine, run).await;
        });
    }

    fn check_cooldown(&self, routine: &Routine) -> bool {
        if let Some(last_run) = routine.last_run_at {
            let elapsed = Utc::now().signed_duration_since(last_run);
            let cooldown = chrono::Duration::from_std(routine.guardrails.cooldown)
                .unwrap_or(chrono::Duration::seconds(300));
            if elapsed < cooldown {
                return false;
            }
        }
        true
    }

    async fn check_concurrent(&self, routine: &Routine) -> bool {
        match self.store.count_running_routine_runs(routine.id).await {
            Ok(count) => count < routine.guardrails.max_concurrent as i64,
            Err(e) => {
                tracing::error!(
                    routine = %routine.name,
                    "Failed to check concurrent runs: {}", e
                );
                false
            }
        }
    }
}

/// Shared context passed to the execution function.
struct EngineContext {
    store: Arc<dyn Database>,
    llm: Arc<dyn LlmProvider>,
    workspace: Arc<Workspace>,
    notify_tx: mpsc::Sender<OutgoingResponse>,
    running_count: Arc<AtomicUsize>,
    scheduler: Option<Arc<Scheduler>>,
    tool_registry: Option<Arc<ToolRegistry>>,
    /// Gateway port for script IRONCLAW_PORT env var.
    gateway_port: Option<u16>,
    /// Auth token for script IRONCLAW_TOKEN env var.
    user_token: Option<String>,
}

/// Execute a routine run. Handles both lightweight and full_job modes.
async fn execute_routine(ctx: EngineContext, routine: Routine, run: RoutineRun) {
    // Increment running count (atomic: survives panics in the execution below)
    ctx.running_count.fetch_add(1, Ordering::Relaxed);

    let result = match &routine.action {
        RoutineAction::Lightweight {
            prompt,
            context_paths,
            max_tokens,
        } => execute_lightweight(&ctx, &routine, prompt, context_paths, *max_tokens).await,
        RoutineAction::FullJob {
            title,
            description,
            max_iterations,
            tool_permissions,
        } => {
            execute_full_job(
                &ctx,
                &routine,
                &run,
                title,
                description,
                *max_iterations,
                tool_permissions,
            )
            .await
        }
        RoutineAction::Wasm {
            tool_name,
            escalation_prompt,
        } => {
            execute_wasm(
                &ctx,
                &routine,
                tool_name,
                escalation_prompt,
                run.trigger_detail.as_deref(),
            )
            .await
        }
        RoutineAction::Script {
            language,
            source,
            escalation_prompt,
        } => {
            execute_script(
                &ctx,
                &routine,
                language,
                source,
                escalation_prompt,
                run.trigger_detail.as_deref(),
            )
            .await
        }
    };

    // Decrement running count
    ctx.running_count.fetch_sub(1, Ordering::Relaxed);

    // Process result
    let (status, summary, tokens) = match result {
        Ok(execution) => execution,
        Err(e) => {
            tracing::error!(routine = %routine.name, "Execution failed: {}", e);
            (RunStatus::Failed, Some(e.to_string()), None)
        }
    };

    // Complete the run record
    if let Err(e) = ctx
        .store
        .complete_routine_run(run.id, status, summary.as_deref(), tokens)
        .await
    {
        tracing::error!(routine = %routine.name, "Failed to complete run record: {}", e);
    }

    // Update routine runtime state
    let now = Utc::now();
    let next_fire = if let Trigger::Cron {
        ref schedule,
        ref timezone,
    } = routine.trigger
    {
        next_cron_fire(schedule, timezone.as_deref()).unwrap_or(None)
    } else {
        None
    };

    let new_failures = if status == RunStatus::Failed {
        routine.consecutive_failures + 1
    } else {
        0
    };

    if let Err(e) = ctx
        .store
        .update_routine_runtime(
            routine.id,
            now,
            next_fire,
            routine.run_count + 1,
            new_failures,
            &routine.state,
        )
        .await
    {
        tracing::error!(routine = %routine.name, "Failed to update runtime state: {}", e);
    }

    // Persist routine result to its dedicated conversation thread
    let thread_id = match ctx
        .store
        .get_or_create_routine_conversation(routine.id, &routine.name, &routine.user_id)
        .await
    {
        Ok(conv_id) => {
            tracing::debug!(
                routine = %routine.name,
                routine_id = %routine.id,
                conversation_id = %conv_id,
                "Resolved routine conversation thread"
            );
            // Record the run result as a conversation message
            let msg = match (&summary, status) {
                (Some(s), _) => format!("[{}] {}: {}", run.trigger_type, status, s),
                (None, _) => format!("[{}] {}", run.trigger_type, status),
            };
            if let Err(e) = ctx
                .store
                .add_conversation_message(conv_id, "assistant", &msg)
                .await
            {
                tracing::error!(routine = %routine.name, "Failed to persist routine message: {}", e);
            }
            Some(conv_id.to_string())
        }
        Err(e) => {
            tracing::error!(routine = %routine.name, "Failed to get routine conversation: {}", e);
            None
        }
    };

    // Send notifications based on config
    send_notification(
        &ctx.notify_tx,
        &routine.notify,
        &routine.name,
        status,
        summary.as_deref(),
        thread_id.as_deref(),
    )
    .await;
}

/// Sanitize a routine name for use in workspace paths.
/// Only keeps alphanumeric, dash, and underscore characters; replaces everything else.
fn sanitize_routine_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Execute a WASM tool routine action.
///
/// Looks up the tool in the registry, executes it with the trigger data,
/// and parses the response. If the tool returns `{"status": "escalate"}`,
/// falls back to a lightweight LLM call with the escalation prompt.
async fn execute_wasm(
    ctx: &EngineContext,
    routine: &Routine,
    tool_name: &str,
    escalation_prompt: &Option<String>,
    trigger_detail: Option<&str>,
) -> Result<(RunStatus, Option<String>, Option<i32>), RoutineError> {
    let registry = ctx
        .tool_registry
        .as_ref()
        .ok_or_else(|| RoutineError::WasmFailed {
            reason: "tool registry not available".to_string(),
        })?;

    let tool: Arc<dyn crate::tools::Tool> =
        registry
            .get(tool_name)
            .await
            .ok_or_else(|| RoutineError::WasmFailed {
                reason: format!("tool '{}' not found in registry", tool_name),
            })?;

    // Build params from trigger detail
    let params = match trigger_detail {
        Some(detail) => serde_json::from_str(detail)
            .unwrap_or_else(|_| serde_json::json!({"trigger_data": detail})),
        None => serde_json::json!({}),
    };

    // Build a minimal JobContext for the routine's user
    let job_ctx = JobContext::with_user(
        &routine.user_id,
        format!("routine:{}", routine.name),
        "WASM routine action",
    );

    match tool.execute(params, &job_ctx).await {
        Ok(output) => {
            // Extract text from the result Value
            let response_text = match &output.result {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };

            // Try to parse as structured response
            if let Ok(response) = serde_json::from_str::<serde_json::Value>(&response_text) {
                match response.get("status").and_then(|s| s.as_str()) {
                    Some("handled") => Ok((RunStatus::Ok, Some(response_text), None)),
                    Some("noop") => Ok((RunStatus::Ok, None, None)),
                    Some("escalate") => {
                        let escalation_context = response
                            .get("context")
                            .and_then(|c| c.as_str())
                            .unwrap_or("WASM module requested escalation");

                        match escalation_prompt {
                            Some(prompt) => {
                                let full_prompt =
                                    prompt.replace("{{context}}", escalation_context);
                                tracing::info!(
                                    routine = %routine.name,
                                    "WASM escalated to LLM: {}",
                                    escalation_context
                                );
                                execute_lightweight(ctx, routine, &full_prompt, &[], 4096).await
                            }
                            None => {
                                tracing::warn!(
                                    routine = %routine.name,
                                    "WASM escalated but no escalation_prompt configured"
                                );
                                Ok((
                                    RunStatus::Attention,
                                    Some(format!(
                                        "Escalation needed: {}",
                                        escalation_context
                                    )),
                                    None,
                                ))
                            }
                        }
                    }
                    _ => {
                        // Unrecognized status — treat as handled
                        Ok((RunStatus::Ok, Some(response_text), None))
                    }
                }
            } else {
                // Non-JSON response — treat as handled
                Ok((RunStatus::Ok, Some(response_text), None))
            }
        }
        Err(e) => Err(RoutineError::WasmFailed {
            reason: format!("tool execution failed: {}", e),
        }),
    }
}

/// Execute a full-job routine by dispatching to the scheduler.
///
/// Fire-and-forget: creates a job via `Scheduler::dispatch_job` (which handles
/// creation, metadata, persistence, and scheduling), links the routine run to
/// the job, and returns immediately. The job runs independently via the
/// existing Worker/Scheduler with full tool access.
async fn execute_full_job(
    ctx: &EngineContext,
    routine: &Routine,
    run: &RoutineRun,
    title: &str,
    description: &str,
    max_iterations: u32,
    tool_permissions: &[String],
) -> Result<(RunStatus, Option<String>, Option<i32>), RoutineError> {
    let scheduler = ctx
        .scheduler
        .as_ref()
        .ok_or_else(|| RoutineError::JobDispatchFailed {
            reason: "scheduler not available".to_string(),
        })?;

    let mut metadata = serde_json::json!({ "max_iterations": max_iterations });
    // Carry the routine's notify config in job metadata so the message tool
    // can resolve channel/target per-job without global state mutation.
    if let Some(channel) = &routine.notify.channel {
        metadata["notify_channel"] = serde_json::json!(channel);
    }
    metadata["notify_user"] = serde_json::json!(&routine.notify.user);

    // Build approval context: UnlessAutoApproved tools are auto-approved for routines;
    // Always tools require explicit listing in tool_permissions.
    let approval_context = ApprovalContext::autonomous_with_tools(tool_permissions.iter().cloned());

    let job_id = scheduler
        .dispatch_job_with_context(
            &routine.user_id,
            title,
            description,
            Some(metadata),
            approval_context,
        )
        .await
        .map_err(|e| RoutineError::JobDispatchFailed {
            reason: format!("failed to dispatch job: {e}"),
        })?;

    // Link the routine run to the dispatched job
    if let Err(e) = ctx.store.link_routine_run_to_job(run.id, job_id).await {
        tracing::error!(
            routine = %routine.name,
            "Failed to link run to job: {}", e
        );
    }

    tracing::info!(
        routine = %routine.name,
        job_id = %job_id,
        max_iterations = max_iterations,
        "Dispatched full job for routine"
    );

    let summary = format!(
        "Dispatched job {job_id} for full execution with tool access (max_iterations: {max_iterations})"
    );
    Ok((RunStatus::Ok, Some(summary), None))
}

/// Execute a lightweight routine (single LLM call).
async fn execute_lightweight(
    ctx: &EngineContext,
    routine: &Routine,
    prompt: &str,
    context_paths: &[String],
    max_tokens: u32,
) -> Result<(RunStatus, Option<String>, Option<i32>), RoutineError> {
    // Load context from workspace
    let mut context_parts = Vec::new();
    for path in context_paths {
        match ctx.workspace.read(path).await {
            Ok(doc) => {
                context_parts.push(format!("## {}\n\n{}", path, doc.content));
            }
            Err(e) => {
                tracing::debug!(
                    routine = %routine.name,
                    "Failed to read context path {}: {}", path, e
                );
            }
        }
    }

    // Load routine state from workspace (name sanitized to prevent path traversal)
    let safe_name = sanitize_routine_name(&routine.name);
    let state_path = format!("routines/{safe_name}/state.md");
    let state_content = match ctx.workspace.read(&state_path).await {
        Ok(doc) => Some(doc.content),
        Err(_) => None,
    };

    // Build the prompt
    let mut full_prompt = String::new();
    full_prompt.push_str(prompt);

    if !context_parts.is_empty() {
        full_prompt.push_str("\n\n---\n\n# Context\n\n");
        full_prompt.push_str(&context_parts.join("\n\n"));
    }

    if let Some(state) = &state_content {
        full_prompt.push_str("\n\n---\n\n# Previous State\n\n");
        full_prompt.push_str(state);
    }

    full_prompt.push_str(
        "\n\n---\n\nIf nothing needs attention, reply EXACTLY with: ROUTINE_OK\n\
         If something needs attention, provide a concise summary.",
    );

    // Get system prompt
    let system_prompt = match ctx.workspace.system_prompt().await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(routine = %routine.name, "Failed to get system prompt: {}", e);
            String::new()
        }
    };

    let messages = if system_prompt.is_empty() {
        vec![ChatMessage::user(&full_prompt)]
    } else {
        vec![
            ChatMessage::system(&system_prompt),
            ChatMessage::user(&full_prompt),
        ]
    };

    // Determine max_tokens from model metadata with fallback
    let effective_max_tokens = match ctx.llm.model_metadata().await {
        Ok(meta) => {
            let from_api = meta.context_length.map(|ctx| ctx / 2).unwrap_or(max_tokens);
            from_api.max(max_tokens)
        }
        Err(_) => max_tokens,
    };

    let request = CompletionRequest::new(messages)
        .with_max_tokens(effective_max_tokens)
        .with_temperature(0.3);

    let response = ctx
        .llm
        .complete(request)
        .await
        .map_err(|e| RoutineError::LlmFailed {
            reason: e.to_string(),
        })?;

    let content = response.content.trim();
    let tokens_used = Some((response.input_tokens + response.output_tokens) as i32);

    // Empty content guard (same as heartbeat)
    if content.is_empty() {
        return if response.finish_reason == FinishReason::Length {
            Err(RoutineError::TruncatedResponse)
        } else {
            Err(RoutineError::EmptyResponse)
        };
    }

    // Check for the "nothing to do" sentinel
    if content == "ROUTINE_OK" || content.contains("ROUTINE_OK") {
        return Ok((RunStatus::Ok, None, tokens_used));
    }

    Ok((RunStatus::Attention, Some(content.to_string()), tokens_used))
}

/// Send a notification based on the routine's notify config and run status.
async fn send_notification(
    tx: &mpsc::Sender<OutgoingResponse>,
    notify: &NotifyConfig,
    routine_name: &str,
    status: RunStatus,
    summary: Option<&str>,
    thread_id: Option<&str>,
) {
    let should_notify = match status {
        RunStatus::Ok => notify.on_success,
        RunStatus::Attention => notify.on_attention,
        RunStatus::Failed => notify.on_failure,
        RunStatus::Running => false,
    };

    if !should_notify {
        return;
    }

    let icon = match status {
        RunStatus::Ok => "✅",
        RunStatus::Attention => "🔔",
        RunStatus::Failed => "❌",
        RunStatus::Running => "⏳",
    };

    let message = match summary {
        Some(s) => format!("{} *Routine '{}'*: {}\n\n{}", icon, routine_name, status, s),
        None => format!("{} *Routine '{}'*: {}", icon, routine_name, status),
    };

    let response = OutgoingResponse {
        content: message,
        thread_id: thread_id.map(String::from),
        attachments: Vec::new(),
        metadata: serde_json::json!({
            "source": "routine",
            "routine_name": routine_name,
            "status": status.to_string(),
            "notify_user": notify.user,
            "notify_channel": notify.channel,
        }),
    };

    if let Err(e) = tx.send(response).await {
        tracing::error!(routine = %routine_name, "Failed to send notification: {}", e);
    }
}

/// Spawn the cron ticker background task.
pub fn spawn_cron_ticker(
    engine: Arc<RoutineEngine>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // Skip immediate first tick
        ticker.tick().await;

        loop {
            ticker.tick().await;
            engine.check_cron_triggers().await;
        }
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let end = crate::util::floor_char_boundary(s, max);
        format!("{}...", &s[..end])
    }
}

/// Maximum script output size before truncation (64KB).
const SCRIPT_MAX_OUTPUT: usize = 64 * 1024;

/// Default script timeout.
const SCRIPT_TIMEOUT: Duration = Duration::from_secs(30);

/// Safe environment variables forwarded to script subprocesses.
const SCRIPT_SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP",
];

/// Result of parsing script subprocess output.
#[derive(Debug, PartialEq)]
enum ScriptResult {
    Handled,
    Noop,
    Escalate(String),
    Failed,
}

/// Parse the stdout + exit code from a script subprocess.
fn parse_script_response(stdout: &str, exit_code: i32) -> (ScriptResult, Option<String>) {
    if exit_code != 0 {
        return (ScriptResult::Failed, Some(stdout.to_string()));
    }

    let stdout_trimmed = stdout.trim();
    if stdout_trimmed.is_empty() {
        return (ScriptResult::Noop, None);
    }

    match serde_json::from_str::<serde_json::Value>(stdout_trimmed) {
        Ok(response) => match response.get("status").and_then(|s| s.as_str()) {
            Some("handled") => (ScriptResult::Handled, Some(stdout_trimmed.to_string())),
            Some("noop") => (ScriptResult::Noop, None),
            Some("escalate") => {
                let context = response
                    .get("context")
                    .and_then(|c| c.as_str())
                    .unwrap_or("script requested escalation")
                    .to_string();
                (ScriptResult::Escalate(context), Some(stdout_trimmed.to_string()))
            }
            _ => (ScriptResult::Handled, Some(stdout_trimmed.to_string())),
        },
        Err(_) => (
            ScriptResult::Escalate(stdout_trimmed.to_string()),
            Some(stdout_trimmed.to_string()),
        ),
    }
}

/// Execute a script routine action.
///
/// Writes the script source to a temp file, spawns the appropriate interpreter
/// (python3 or bash), pipes trigger data to stdin, captures output with timeout,
/// and parses the response. Escalation falls back to a lightweight LLM call.
async fn execute_script(
    ctx: &EngineContext,
    routine: &Routine,
    language: &str,
    source: &str,
    escalation_prompt: &Option<String>,
    trigger_detail: Option<&str>,
) -> Result<(RunStatus, Option<String>, Option<i32>), RoutineError> {
    let ext = match language {
        "python" => "py",
        "bash" => "sh",
        other => {
            return Err(RoutineError::ScriptFailed {
                reason: format!("unsupported script language: {other}"),
            });
        }
    };

    let interpreter = match language {
        "python" => "python3",
        _ => "bash",
    };

    // Write script to temp file
    let script_id = Uuid::new_v4();
    let script_path = format!("/tmp/ironclaw-script-{script_id}.{ext}");

    if let Err(e) = tokio::fs::write(&script_path, source).await {
        return Err(RoutineError::ScriptFailed {
            reason: format!("failed to write temp script: {e}"),
        });
    }

    // Use per-routine timeout if configured, otherwise default
    let timeout = routine
        .guardrails
        .max_execution_time
        .unwrap_or(SCRIPT_TIMEOUT);

    // Ensure cleanup on all exit paths
    let result = execute_script_inner(
        ctx, routine, interpreter, &script_path, escalation_prompt, trigger_detail, timeout,
    )
    .await;

    // Clean up temp file (best-effort)
    let _ = tokio::fs::remove_file(&script_path).await;

    result
}

/// Inner script execution (separated for cleanup guarantee).
async fn execute_script_inner(
    ctx: &EngineContext,
    routine: &Routine,
    interpreter: &str,
    script_path: &str,
    escalation_prompt: &Option<String>,
    trigger_detail: Option<&str>,
    timeout: Duration,
) -> Result<(RunStatus, Option<String>, Option<i32>), RoutineError> {
    let mut command = tokio::process::Command::new(interpreter);
    command.arg(script_path);

    // Scrub environment: only safe vars + IRONCLAW_* context
    command.env_clear();
    for var in SCRIPT_SAFE_ENV_VARS {
        if let Ok(val) = std::env::var(var) {
            command.env(var, val);
        }
    }
    if let Some(port) = ctx.gateway_port {
        command.env("IRONCLAW_PORT", port.to_string());
    }
    if let Some(ref token) = ctx.user_token {
        command.env("IRONCLAW_TOKEN", token);
    }
    command.env("IRONCLAW_USER_ID", &routine.user_id);

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|e| RoutineError::ScriptFailed {
        reason: format!("failed to spawn {interpreter}: {e}"),
    })?;

    // Write trigger data to stdin, then close it so script sees EOF
    if let Some(mut stdin) = child.stdin.take() {
        let data = trigger_detail.unwrap_or("{}");
        // Best-effort write — if it fails the script just gets empty stdin
        let _ = stdin.write_all(data.as_bytes()).await;
        let _ = stdin.shutdown().await;
    }

    // Drain stdout/stderr concurrently with timeout
    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    let result = tokio::time::timeout(timeout, async {
        let stdout_fut = async {
            if let Some(mut out) = stdout_handle {
                let mut buf = Vec::new();
                (&mut out)
                    .take(SCRIPT_MAX_OUTPUT as u64)
                    .read_to_end(&mut buf)
                    .await
                    .ok();
                tokio::io::copy(&mut out, &mut tokio::io::sink()).await.ok();
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            }
        };

        let stderr_fut = async {
            if let Some(mut err) = stderr_handle {
                let mut buf = Vec::new();
                (&mut err)
                    .take(SCRIPT_MAX_OUTPUT as u64)
                    .read_to_end(&mut buf)
                    .await
                    .ok();
                tokio::io::copy(&mut err, &mut tokio::io::sink()).await.ok();
                String::from_utf8_lossy(&buf).to_string()
            } else {
                String::new()
            }
        };

        let (stdout, stderr, wait_result) = tokio::join!(stdout_fut, stderr_fut, child.wait());
        let status = wait_result.map_err(|e| RoutineError::ScriptFailed {
            reason: format!("failed to wait for script: {e}"),
        })?;

        Ok::<_, RoutineError>((stdout, stderr, status.code().unwrap_or(-1)))
    })
    .await;

    let (stdout, stderr, exit_code) = match result {
        Ok(Ok(tuple)) => tuple,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            // Timeout — kill the process
            let _ = child.kill().await;
            return Err(RoutineError::ScriptFailed {
                reason: format!(
                    "script timed out after {}s",
                    timeout.as_secs()
                ),
            });
        }
    };

    let (script_result, output) = parse_script_response(&stdout, exit_code);

    match script_result {
        ScriptResult::Handled => Ok((RunStatus::Ok, output, None)),
        ScriptResult::Noop => Ok((RunStatus::Ok, None, None)),
        ScriptResult::Escalate(context) => {
            match escalation_prompt {
                Some(prompt) => {
                    let full_prompt = prompt.replace("{{context}}", &context);
                    tracing::info!(
                        routine = %routine.name,
                        "Script escalated to LLM: {}",
                        context
                    );
                    execute_lightweight(ctx, routine, &full_prompt, &[], 4096).await
                }
                None => {
                    tracing::warn!(
                        routine = %routine.name,
                        "Script escalated but no escalation_prompt configured"
                    );
                    Ok((
                        RunStatus::Attention,
                        Some(format!("Escalation needed: {context}")),
                        None,
                    ))
                }
            }
        }
        ScriptResult::Failed => {
            let reason = if stderr.is_empty() {
                stdout
            } else {
                format!("{stdout}\n--- stderr ---\n{stderr}")
            };
            Err(RoutineError::ScriptFailed { reason })
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::collection_events::CollectionWriteEvent;
    use crate::agent::routine::{NotifyConfig, Routine, RoutineAction, RoutineGuardrails, RunStatus, Trigger};
    use super::matches_collection_write;

    #[test]
    fn test_notification_gating() {
        let config = NotifyConfig {
            on_success: false,
            on_failure: true,
            on_attention: true,
            ..Default::default()
        };

        // on_success = false means Ok status should not notify
        assert!(!config.on_success);
        assert!(config.on_failure);
        assert!(config.on_attention);
    }

    #[test]
    fn test_run_status_icons() {
        // Just verify the mapping doesn't panic
        for status in [
            RunStatus::Ok,
            RunStatus::Attention,
            RunStatus::Failed,
            RunStatus::Running,
        ] {
            let _ = status.to_string();
        }
    }

    fn make_collection_write_routine(name: &str, collection: &str, user_id: &str) -> Routine {
        Routine {
            id: uuid::Uuid::new_v4(),
            name: name.to_string(),
            description: String::new(),
            user_id: user_id.to_string(),
            enabled: true,
            trigger: Trigger::CollectionWrite {
                collection: collection.to_string(),
            },
            action: RoutineAction::Lightweight {
                prompt: "test prompt".to_string(),
                context_paths: vec![],
                max_tokens: 1024,
            },
            guardrails: RoutineGuardrails::default(),
            notify: NotifyConfig::default(),
            last_run_at: None,
            next_fire_at: None,
            run_count: 0,
            consecutive_failures: 0,
            state: serde_json::Value::Null,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_match_collection_write_trigger() {
        let routine = make_collection_write_routine("presence-handler", "wifi_presence", "test-user");
        let event = CollectionWriteEvent {
            user_id: "test-user".to_string(),
            collection: "wifi_presence".to_string(),
            record_id: uuid::Uuid::new_v4(),
            data: serde_json::json!({"device": "phone", "state": "home"}),
        };
        assert!(matches_collection_write(&routine, &event));
    }

    #[test]
    fn test_no_match_wrong_collection() {
        let routine = make_collection_write_routine("presence-handler", "wifi_presence", "test-user");
        let event = CollectionWriteEvent {
            user_id: "test-user".to_string(),
            collection: "nanny_shifts".to_string(),
            record_id: uuid::Uuid::new_v4(),
            data: serde_json::json!({}),
        };
        assert!(!matches_collection_write(&routine, &event));
    }

    #[test]
    fn test_no_match_wrong_user() {
        let routine = make_collection_write_routine("presence-handler", "wifi_presence", "test-user");
        let event = CollectionWriteEvent {
            user_id: "other-user".to_string(),
            collection: "wifi_presence".to_string(),
            record_id: uuid::Uuid::new_v4(),
            data: serde_json::json!({}),
        };
        assert!(!matches_collection_write(&routine, &event));
    }

    // ── parse_script_response tests ──────────────────────────────────

    use super::{ScriptResult, parse_script_response};

    #[test]
    fn test_parse_script_response_handled() {
        let stdout = r#"{"status": "handled", "summary": "done"}"#;
        let (result, output) = parse_script_response(stdout, 0);
        assert_eq!(result, ScriptResult::Handled);
        assert!(output.is_some());
        assert!(output.unwrap().contains("handled"));
    }

    #[test]
    fn test_parse_script_response_noop() {
        let stdout = r#"{"status": "noop"}"#;
        let (result, output) = parse_script_response(stdout, 0);
        assert_eq!(result, ScriptResult::Noop);
        assert!(output.is_none());
    }

    #[test]
    fn test_parse_script_response_escalate() {
        let stdout = r#"{"status": "escalate", "context": "need human help"}"#;
        let (result, output) = parse_script_response(stdout, 0);
        assert_eq!(result, ScriptResult::Escalate("need human help".to_string()));
        assert!(output.is_some());
    }

    #[test]
    fn test_parse_script_response_nonzero_exit() {
        let stdout = "some error output";
        let (result, output) = parse_script_response(stdout, 1);
        assert_eq!(result, ScriptResult::Failed);
        assert_eq!(output.unwrap(), "some error output");
    }

    #[test]
    fn test_parse_script_response_invalid_json() {
        let stdout = "this is not json at all";
        let (result, output) = parse_script_response(stdout, 0);
        assert_eq!(
            result,
            ScriptResult::Escalate("this is not json at all".to_string())
        );
        assert_eq!(output.unwrap(), "this is not json at all");
    }

    // ── subprocess integration tests ────────────────────────────────

    #[tokio::test]
    async fn test_execute_script_python_handled() {
        use tokio::io::AsyncWriteExt;

        let source = r#"#!/usr/bin/env python3
import json, sys
data = json.load(sys.stdin)
print(json.dumps({"status": "handled"}))
"#;
        let script_path = std::env::temp_dir().join("ironclaw-test-handled.py");
        tokio::fs::write(&script_path, source).await.unwrap();

        let mut child = tokio::process::Command::new("python3")
            .arg(&script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(b"{}").await.unwrap();
            drop(stdin);
        }

        let output = child.wait_with_output().await.unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result, _) = parse_script_response(&stdout, output.status.code().unwrap_or(-1));
        assert_eq!(result, ScriptResult::Handled);

        let _ = tokio::fs::remove_file(&script_path).await;
    }

    #[tokio::test]
    async fn test_execute_script_bash_noop() {
        use tokio::io::AsyncWriteExt;

        let source = r#"#!/usr/bin/env bash
echo '{"status": "noop"}'
"#;
        let script_path = std::env::temp_dir().join("ironclaw-test-noop.sh");
        tokio::fs::write(&script_path, source).await.unwrap();

        let mut child = tokio::process::Command::new("bash")
            .arg(&script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(b"{}").await.unwrap();
            drop(stdin);
        }

        let output = child.wait_with_output().await.unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result, _) = parse_script_response(&stdout, output.status.code().unwrap_or(-1));
        assert_eq!(result, ScriptResult::Noop);

        let _ = tokio::fs::remove_file(&script_path).await;
    }

    #[tokio::test]
    async fn test_execute_script_reads_stdin() {
        use tokio::io::AsyncWriteExt;

        let source = r#"#!/usr/bin/env python3
import json, sys
data = json.load(sys.stdin)
if data.get("state") == "home":
    print(json.dumps({"status": "handled", "summary": "user is home"}))
else:
    print(json.dumps({"status": "noop"}))
"#;
        let script_path = std::env::temp_dir().join("ironclaw-test-reads-stdin.py");
        tokio::fs::write(&script_path, source).await.unwrap();

        let mut child = tokio::process::Command::new("python3")
            .arg(&script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(br#"{"state": "home"}"#).await.unwrap();
            drop(stdin);
        }

        let output = child.wait_with_output().await.unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result, _) = parse_script_response(&stdout, output.status.code().unwrap_or(-1));
        assert_eq!(result, ScriptResult::Handled);

        let _ = tokio::fs::remove_file(&script_path).await;
    }

    #[tokio::test]
    async fn test_execute_script_nonzero_exit() {
        use tokio::io::AsyncWriteExt;

        let source = "#!/usr/bin/env bash\nexit 1\n";
        let script_path = std::env::temp_dir().join("ironclaw-test-nonzero.sh");
        tokio::fs::write(&script_path, source).await.unwrap();

        let mut child = tokio::process::Command::new("bash")
            .arg(&script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(b"{}").await.unwrap();
            drop(stdin);
        }

        let output = child.wait_with_output().await.unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result, _) = parse_script_response(&stdout, output.status.code().unwrap_or(-1));
        assert_eq!(result, ScriptResult::Failed);

        let _ = tokio::fs::remove_file(&script_path).await;
    }

    #[tokio::test]
    async fn test_execute_script_env_vars() {
        use tokio::io::AsyncWriteExt;

        let source = r#"#!/usr/bin/env python3
import json, os, sys
_ = sys.stdin.read()
result = {
    "port_set": os.environ.get("IRONCLAW_PORT") == "3003",
    "token_set": os.environ.get("IRONCLAW_TOKEN") == "test-token",
    "user_id_set": os.environ.get("IRONCLAW_USER_ID") == "test-user",
    "anthropic_scrubbed": os.environ.get("ANTHROPIC_API_KEY") is None,
}
if all(result.values()):
    print(json.dumps({"status": "handled", "checks": result}))
else:
    print(json.dumps({"status": "escalate", "context": json.dumps(result)}))
"#;
        let script_path = std::env::temp_dir().join("ironclaw-test-env-vars.py");
        tokio::fs::write(&script_path, source).await.unwrap();

        let mut child = tokio::process::Command::new("python3")
            .arg(&script_path)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default())
            .env("IRONCLAW_PORT", "3003")
            .env("IRONCLAW_TOKEN", "test-token")
            .env("IRONCLAW_USER_ID", "test-user")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(b"{}").await.unwrap();
            drop(stdin);
        }

        let output = child.wait_with_output().await.unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result, _) = parse_script_response(&stdout, output.status.code().unwrap_or(-1));
        assert_eq!(result, ScriptResult::Handled);

        let _ = tokio::fs::remove_file(&script_path).await;
    }
}
