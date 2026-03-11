//! Main agent loop.
//!
//! Contains the `Agent` struct, `AgentDeps`, and the core event loop (`run`).
//! The heavy lifting is delegated to sibling modules:
//!
//! - `dispatcher` - Tool dispatch (agentic loop, tool execution)
//! - `commands` - System commands and job handlers
//! - `thread_ops` - Thread/session operations (user input, undo, approval, persistence)

use std::sync::Arc;

use futures::StreamExt;

use crate::agent::context_monitor::ContextMonitor;
use crate::agent::heartbeat::spawn_heartbeat;
use crate::agent::organizer_runner::{OrganizerConfig, spawn_organizer};
use crate::agent::routine_engine::{RoutineEngine, spawn_cron_ticker};
use crate::agent::self_repair::{DefaultSelfRepair, RepairResult, SelfRepair};
use crate::agent::session_manager::SessionManager;
use crate::agent::submission::{Submission, SubmissionParser, SubmissionResult};
use crate::agent::{HeartbeatConfig as AgentHeartbeatConfig, Router, Scheduler};
use crate::channels::{ChannelManager, IncomingMessage, OutgoingResponse, StatusUpdate};
use crate::config::{AgentConfig, HeartbeatConfig, RoutineConfig, SkillsConfig};
use crate::context::ContextManager;
use crate::db::Database;
use crate::error::Error;
use crate::extensions::ExtensionManager;
use crate::hooks::HookRegistry;
use crate::llm::LlmProvider;
use crate::safety::SafetyLayer;
use crate::skills::SkillRegistry;
use crate::tools::ToolRegistry;
use uuid::Uuid;

use crate::workspace::Workspace;

/// Collapse a tool output string into a single-line preview for display.
pub(crate) fn truncate_for_preview(output: &str, max_chars: usize) -> String {
    let collapsed: String = output
        .chars()
        .take(max_chars + 50)
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    // char_indices gives us byte offsets at char boundaries, so the slice is always valid UTF-8.
    if collapsed.chars().count() > max_chars {
        let byte_offset = collapsed
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(collapsed.len());
        format!("{}...", &collapsed[..byte_offset])
    } else {
        collapsed
    }
}

/// Core dependencies for the agent.
///
/// Bundles the shared components to reduce argument count.
pub struct AgentDeps {
    pub store: Option<Arc<dyn Database>>,
    pub llm: Arc<dyn LlmProvider>,
    /// Cheap/fast LLM for lightweight tasks (heartbeat, routing, evaluation).
    /// Falls back to the main `llm` if None.
    pub cheap_llm: Option<Arc<dyn LlmProvider>>,
    pub safety: Arc<SafetyLayer>,
    pub tools: Arc<ToolRegistry>,
    pub workspace: Option<Arc<Workspace>>,
    pub extension_manager: Option<Arc<ExtensionManager>>,
    pub skill_registry: Option<Arc<std::sync::RwLock<SkillRegistry>>>,
    pub skill_catalog: Option<Arc<crate::skills::catalog::SkillCatalog>>,
    pub skills_config: SkillsConfig,
    pub hooks: Arc<HookRegistry>,
    /// Cost enforcement guardrails (daily budget, hourly rate limits).
    pub cost_guard: Arc<crate::agent::cost_guard::CostGuard>,
    /// SSE broadcast manager for live job event streaming to the web gateway.
    pub sse_tx: Option<Arc<crate::channels::web::sse::SseManager>>,
    /// HTTP interceptor for trace recording/replay.
    pub http_interceptor: Option<Arc<dyn crate::llm::recording::HttpInterceptor>>,
    /// Audio transcription middleware for voice messages.
    pub transcription: Option<Arc<crate::transcription::TranscriptionMiddleware>>,
    /// Document text extraction middleware for PDF, DOCX, PPTX, etc.
    pub document_extraction: Option<Arc<crate::document_extraction::DocumentExtractionMiddleware>>,
    /// Workspace router for automatic topic-based context switching.
    pub workspace_router: Option<Arc<crate::agent::workspace_router::WorkspaceRouter>>,
    /// Pluggable thread resolver for message-to-thread routing and context injection.
    /// When set, takes priority over `workspace_router` for routing decisions.
    pub thread_resolver: Option<Arc<dyn crate::agent::thread_resolver::ThreadResolver>>,
    /// Tool names that are always included in LLM context.
    /// When non-empty, only core + discovered tools are sent to the LLM.
    /// Empty = backward compatible (all tools sent).
    pub core_tools: Vec<String>,
    /// Receiver for organizer signals (passed to the organizer runner on spawn).
    /// Created in main.rs alongside the resolver; consumed once by spawn_organizer.
    pub organize_rx: Option<tokio::sync::mpsc::Receiver<crate::agent::organizer_runner::OrganizerSignal>>,
}

/// The main agent that coordinates all components.
pub struct Agent {
    pub(super) config: AgentConfig,
    pub(super) deps: AgentDeps,
    pub(super) channels: Arc<ChannelManager>,
    pub(super) context_manager: Arc<ContextManager>,
    pub(super) scheduler: Arc<Scheduler>,
    pub(super) router: Router,
    pub(super) session_manager: Arc<SessionManager>,
    pub(super) context_monitor: ContextMonitor,
    pub(super) heartbeat_config: Option<HeartbeatConfig>,
    pub(super) hygiene_config: Option<crate::config::HygieneConfig>,
    pub(super) routine_config: Option<RoutineConfig>,
    /// Optional slot to expose the routine engine to the gateway for manual triggering.
    pub(super) routine_engine_slot:
        Option<Arc<tokio::sync::RwLock<Option<Arc<crate::agent::routine_engine::RoutineEngine>>>>>,
    /// Broadcast sender for collection write events (shared with gateway).
    pub(super) collection_write_tx:
        Option<tokio::sync::broadcast::Sender<crate::agent::collection_events::CollectionWriteEvent>>,
    /// Gateway state for injecting routine engine (late init).
    pub(super) gateway_state:
        Option<Arc<crate::channels::web::server::GatewayState>>,
}

impl Agent {
    /// Create a new agent.
    ///
    /// Optionally accepts pre-created `ContextManager` and `SessionManager` for sharing
    /// with external components (job tools, web gateway). Creates new ones if not provided.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AgentConfig,
        deps: AgentDeps,
        channels: Arc<ChannelManager>,
        heartbeat_config: Option<HeartbeatConfig>,
        hygiene_config: Option<crate::config::HygieneConfig>,
        routine_config: Option<RoutineConfig>,
        context_manager: Option<Arc<ContextManager>>,
        session_manager: Option<Arc<SessionManager>>,
        collection_write_tx: Option<tokio::sync::broadcast::Sender<crate::agent::collection_events::CollectionWriteEvent>>,
    ) -> Self {
        let context_manager = context_manager
            .unwrap_or_else(|| Arc::new(ContextManager::new(config.max_parallel_jobs)));

        let session_manager = session_manager.unwrap_or_else(|| Arc::new(SessionManager::new()));

        let mut scheduler = Scheduler::new(
            config.clone(),
            context_manager.clone(),
            deps.llm.clone(),
            deps.safety.clone(),
            deps.tools.clone(),
            deps.store.clone(),
            deps.hooks.clone(),
        );
        if let Some(ref sse) = deps.sse_tx {
            scheduler.set_sse_sender(Arc::clone(sse));
        }
        if !deps.core_tools.is_empty() {
            scheduler.set_core_tools(deps.core_tools.clone());
        }
        if let Some(ref interceptor) = deps.http_interceptor {
            scheduler.set_http_interceptor(Arc::clone(interceptor));
        }
        let scheduler = Arc::new(scheduler);

        Self {
            config,
            deps,
            channels,
            context_manager,
            scheduler,
            router: Router::new(),
            session_manager,
            context_monitor: ContextMonitor::new(),
            heartbeat_config,
            hygiene_config,
            routine_config,
            routine_engine_slot: None,
            collection_write_tx,
            gateway_state: None,
        }
    }

    /// Set the routine engine slot for exposing the engine to the gateway.
    pub fn set_routine_engine_slot(
        &mut self,
        slot: Arc<tokio::sync::RwLock<Option<Arc<crate::agent::routine_engine::RoutineEngine>>>>,
    ) {
        self.routine_engine_slot = Some(slot);
    }

    /// Inject the gateway state for late-init fields (e.g., routine engine).
    pub fn with_gateway_state(
        mut self,
        state: Arc<crate::channels::web::server::GatewayState>,
    ) -> Self {
        self.gateway_state = Some(state);
        self
    }

    // Convenience accessors

    /// Get the scheduler (for external wiring, e.g. CreateJobTool).
    pub fn scheduler(&self) -> Arc<Scheduler> {
        Arc::clone(&self.scheduler)
    }

    pub(super) fn store(&self) -> Option<&Arc<dyn Database>> {
        self.deps.store.as_ref()
    }

    /// Get or create the default "general" workspace for a user.
    ///
    /// Returns the existing workspace if one with topic "general" already exists,
    /// otherwise creates a new one with an embedding for general conversation.
    async fn get_or_create_default_workspace(
        &self,
        user_id: &str,
        router: &crate::agent::workspace_router::WorkspaceRouter,
        store: &Arc<dyn Database>,
    ) -> Result<Option<crate::db::AgentWorkspace>, Error> {
        // Check for existing default workspace
        let workspaces = store.list_agent_workspaces(user_id, Some("active")).await?;
        if let Some(existing) = workspaces.iter().find(|ws| ws.topic == "general") {
            return Ok(Some(existing.clone()));
        }

        // Create new default workspace
        let conversation_id = store
            .create_conversation("workspace", user_id, None)
            .await?;
        let ws = store
            .create_agent_workspace(user_id, conversation_id)
            .await?;

        let embedding = router
            .embed("general conversation and miscellaneous topics")
            .await
            .map_err(|e| {
                crate::error::DatabaseError::Query(format!("embed failed: {e}"))
            })?;
        store
            .update_agent_workspace_topic(ws.id, "general", &embedding)
            .await?;

        tracing::info!(workspace_id = %ws.id, "Created default 'general' workspace");

        // Re-fetch the workspace to get the updated topic
        let workspaces = store.list_agent_workspaces(user_id, Some("active")).await?;
        Ok(workspaces.into_iter().find(|w| w.id == ws.id))
    }

    pub(super) fn llm(&self) -> &Arc<dyn LlmProvider> {
        &self.deps.llm
    }

    /// Get the cheap/fast LLM provider, falling back to the main one.
    pub(super) fn cheap_llm(&self) -> &Arc<dyn LlmProvider> {
        self.deps.cheap_llm.as_ref().unwrap_or(&self.deps.llm)
    }

    pub(super) fn safety(&self) -> &Arc<SafetyLayer> {
        &self.deps.safety
    }

    pub(super) fn tools(&self) -> &Arc<ToolRegistry> {
        &self.deps.tools
    }

    pub(super) fn workspace(&self) -> Option<&Arc<Workspace>> {
        self.deps.workspace.as_ref()
    }

    pub(super) fn hooks(&self) -> &Arc<HookRegistry> {
        &self.deps.hooks
    }

    pub(super) fn cost_guard(&self) -> &Arc<crate::agent::cost_guard::CostGuard> {
        &self.deps.cost_guard
    }

    pub(super) fn skill_registry(&self) -> Option<&Arc<std::sync::RwLock<SkillRegistry>>> {
        self.deps.skill_registry.as_ref()
    }

    pub(super) fn skill_catalog(&self) -> Option<&Arc<crate::skills::catalog::SkillCatalog>> {
        self.deps.skill_catalog.as_ref()
    }

    /// Select active skills for a message using deterministic prefiltering.
    pub(super) fn select_active_skills(
        &self,
        message_content: &str,
    ) -> Vec<crate::skills::LoadedSkill> {
        if let Some(registry) = self.skill_registry() {
            let guard = match registry.read() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!("Skill registry lock poisoned: {}", e);
                    return vec![];
                }
            };
            let available = guard.skills();
            let skills_cfg = &self.deps.skills_config;
            let selected = crate::skills::prefilter_skills(
                message_content,
                available,
                skills_cfg.max_active_skills,
                skills_cfg.max_context_tokens,
            );

            if !selected.is_empty() {
                tracing::debug!(
                    "Selected {} skill(s) for message: {}",
                    selected.len(),
                    selected
                        .iter()
                        .map(|s| s.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            selected.into_iter().cloned().collect()
        } else {
            vec![]
        }
    }

    /// Run the agent main loop.
    pub async fn run(mut self) -> Result<(), Error> {
        // Start channels
        let mut message_stream = self.channels.start_all().await?;

        // Start self-repair task with notification forwarding
        let repair = Arc::new(DefaultSelfRepair::new(
            self.context_manager.clone(),
            self.config.stuck_threshold,
            self.config.max_repair_attempts,
        ));
        let repair_interval = self.config.repair_check_interval;
        let repair_channels = self.channels.clone();
        let repair_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(repair_interval).await;

                // Check stuck jobs
                let stuck_jobs = repair.detect_stuck_jobs().await;
                for job in stuck_jobs {
                    tracing::info!("Attempting to repair stuck job {}", job.job_id);
                    let result = repair.repair_stuck_job(&job).await;
                    let notification = match &result {
                        Ok(RepairResult::Success { message }) => {
                            tracing::info!("Repair succeeded: {}", message);
                            Some(format!(
                                "Job {} was stuck for {}s, recovery succeeded: {}",
                                job.job_id,
                                job.stuck_duration.as_secs(),
                                message
                            ))
                        }
                        Ok(RepairResult::Failed { message }) => {
                            tracing::error!("Repair failed: {}", message);
                            Some(format!(
                                "Job {} was stuck for {}s, recovery failed permanently: {}",
                                job.job_id,
                                job.stuck_duration.as_secs(),
                                message
                            ))
                        }
                        Ok(RepairResult::ManualRequired { message }) => {
                            tracing::warn!("Manual intervention needed: {}", message);
                            Some(format!(
                                "Job {} needs manual intervention: {}",
                                job.job_id, message
                            ))
                        }
                        Ok(RepairResult::Retry { message }) => {
                            tracing::warn!("Repair needs retry: {}", message);
                            None // Don't spam the user on retries
                        }
                        Err(e) => {
                            tracing::error!("Repair error: {}", e);
                            None
                        }
                    };

                    if let Some(msg) = notification {
                        let response = OutgoingResponse::text(format!("Self-Repair: {}", msg));
                        let _ = repair_channels.broadcast_all("default", response).await;
                    }
                }

                // Check broken tools
                let broken_tools = repair.detect_broken_tools().await;
                for tool in broken_tools {
                    tracing::info!("Attempting to repair broken tool: {}", tool.name);
                    match repair.repair_broken_tool(&tool).await {
                        Ok(RepairResult::Success { message }) => {
                            let response = OutgoingResponse::text(format!(
                                "Self-Repair: Tool '{}' repaired: {}",
                                tool.name, message
                            ));
                            let _ = repair_channels.broadcast_all("default", response).await;
                        }
                        Ok(result) => {
                            tracing::info!("Tool repair result: {:?}", result);
                        }
                        Err(e) => {
                            tracing::error!("Tool repair error: {}", e);
                        }
                    }
                }
            }
        });

        // Spawn session pruning task
        let session_mgr = self.session_manager.clone();
        let session_idle_timeout = self.config.session_idle_timeout;
        let pruning_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(600)); // Every 10 min
            interval.tick().await; // Skip immediate first tick
            loop {
                interval.tick().await;
                session_mgr.prune_stale_sessions(session_idle_timeout).await;
            }
        });

        // Spawn heartbeat if enabled
        let heartbeat_handle = if let Some(ref hb_config) = self.heartbeat_config {
            if hb_config.enabled {
                if let Some(workspace) = self.workspace() {
                    let mut config = AgentHeartbeatConfig::default()
                        .with_interval(std::time::Duration::from_secs(hb_config.interval_secs));
                    config.quiet_hours_start = hb_config.quiet_hours_start;
                    config.quiet_hours_end = hb_config.quiet_hours_end;
                    config.timezone = hb_config
                        .timezone
                        .clone()
                        .or_else(|| Some(self.config.default_timezone.clone()));
                    if let (Some(user), Some(channel)) =
                        (&hb_config.notify_user, &hb_config.notify_channel)
                    {
                        config = config.with_notify(user, channel);
                    }

                    // Set up notification channel
                    let (notify_tx, mut notify_rx) =
                        tokio::sync::mpsc::channel::<OutgoingResponse>(16);

                    // Spawn notification forwarder that routes through channel manager
                    let notify_channel = hb_config.notify_channel.clone();
                    let notify_user = hb_config.notify_user.clone();
                    let channels = self.channels.clone();
                    tokio::spawn(async move {
                        while let Some(response) = notify_rx.recv().await {
                            let user = notify_user.as_deref().unwrap_or("default");

                            // Try the configured channel first, fall back to
                            // broadcasting on all channels.
                            let targeted_ok = if let Some(ref channel) = notify_channel {
                                channels
                                    .broadcast(channel, user, response.clone())
                                    .await
                                    .is_ok()
                            } else {
                                false
                            };

                            if !targeted_ok {
                                let results = channels.broadcast_all(user, response).await;
                                for (ch, result) in results {
                                    if let Err(e) = result {
                                        tracing::warn!(
                                            "Failed to broadcast heartbeat to {}: {}",
                                            ch,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    });

                    let hygiene = self
                        .hygiene_config
                        .as_ref()
                        .map(|h| h.to_workspace_config())
                        .unwrap_or_default();

                    Some(spawn_heartbeat(
                        config,
                        hygiene,
                        workspace.clone(),
                        self.cheap_llm().clone(),
                        Some(notify_tx),
                        self.store().map(Arc::clone),
                    ))
                } else {
                    tracing::warn!("Heartbeat enabled but no workspace available");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Spawn organizer if enabled
        let organizer_config = OrganizerConfig::from_env();
        let organizer_handle = if organizer_config.enabled {
            if let Some(ref resolver) = self.deps.thread_resolver {
                let rx = self.deps.organize_rx.take().unwrap_or_else(|| {
                    // No signal channel configured; create a dummy one.
                    // The organizer will still run on ceiling ticks.
                    let (_tx, rx) = tokio::sync::mpsc::channel(1);
                    rx
                });
                Some(spawn_organizer(organizer_config, Arc::clone(resolver), rx))
            } else {
                tracing::warn!("Organizer enabled but no thread resolver available");
                None
            }
        } else {
            None
        };

        // Spawn routine engine if enabled
        let routine_handle = if let Some(ref rt_config) = self.routine_config {
            if rt_config.enabled {
                if let (Some(store), Some(workspace)) = (self.store(), self.workspace()) {
                    // Set up notification channel (same pattern as heartbeat)
                    let (notify_tx, mut notify_rx) =
                        tokio::sync::mpsc::channel::<OutgoingResponse>(32);

                    let engine = Arc::new(RoutineEngine::new(
                        rt_config.clone(),
                        Arc::clone(store),
                        self.llm().clone(),
                        Arc::clone(workspace),
                        notify_tx,
                        Some(self.scheduler.clone()),
                        Some(Arc::clone(&self.deps.tools)),
                    ));

                    // Register routine tools
                    self.deps
                        .tools
                        .register_routine_tools(Arc::clone(store), Arc::clone(&engine));

                    // Load initial caches
                    engine.refresh_event_cache().await;
                    engine.refresh_collection_write_cache().await;

                    // Spawn collection write listener if broadcast channel is available
                    if let Some(ref tx) = self.collection_write_tx {
                        let rx = tx.subscribe();
                        engine.spawn_collection_write_listener(rx);
                        tracing::info!("CollectionWrite listener spawned");
                    }

                    // Spawn notification forwarder (mirrors heartbeat pattern)
                    let channels = self.channels.clone();
                    tokio::spawn(async move {
                        while let Some(response) = notify_rx.recv().await {
                            let user = response
                                .metadata
                                .get("notify_user")
                                .and_then(|v| v.as_str())
                                .unwrap_or("default")
                                .to_string();
                            let notify_channel = response
                                .metadata
                                .get("notify_channel")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());

                            // Try the configured channel first, fall back to
                            // broadcasting on all channels.
                            let targeted_ok = if let Some(ref channel) = notify_channel {
                                channels
                                    .broadcast(channel, &user, response.clone())
                                    .await
                                    .is_ok()
                            } else {
                                false
                            };

                            if !targeted_ok {
                                let results = channels.broadcast_all(&user, response).await;
                                for (ch, result) in results {
                                    if let Err(e) = result {
                                        tracing::warn!(
                                            "Failed to broadcast routine notification to {}: {}",
                                            ch,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    });

                    // Spawn cron ticker
                    let cron_interval =
                        std::time::Duration::from_secs(rt_config.cron_check_interval_secs);
                    let cron_handle = spawn_cron_ticker(Arc::clone(&engine), cron_interval);

                    // Store engine reference for event trigger checking
                    // Safety: we're in run() which takes self, no other reference exists
                    let engine_ref = Arc::clone(&engine);
                    // SAFETY: self is consumed by run(), we can smuggle the engine in
                    // via a local to use in the message loop below.

                    // Expose engine to gateway for manual triggering
                    if let Some(ref slot) = self.routine_engine_slot {
                        *slot.write().await = Some(Arc::clone(&engine));
                    }

                    tracing::info!(
                        "Routines enabled: cron ticker every {}s, max {} concurrent",
                        rt_config.cron_check_interval_secs,
                        rt_config.max_concurrent_routines
                    );

                    Some((cron_handle, engine_ref))
                } else {
                    tracing::warn!("Routines enabled but store/workspace not available");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Extract engine ref for use in message loop
        let routine_engine_for_loop = routine_handle.as_ref().map(|(_, e)| Arc::clone(e));

        // Main message loop
        tracing::info!("Agent {} ready and listening", self.config.name);

        loop {
            let message = tokio::select! {
                biased;
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Ctrl+C received, shutting down...");
                    break;
                }
                msg = message_stream.next() => {
                    match msg {
                        Some(m) => m,
                        None => {
                            tracing::info!("All channel streams ended, shutting down...");
                            break;
                        }
                    }
                }
            };

            // Apply transcription middleware to audio attachments
            let mut message = message;
            if let Some(ref transcription) = self.deps.transcription {
                transcription.process(&mut message).await;
            }

            // Apply document extraction middleware to document attachments
            if let Some(ref doc_extraction) = self.deps.document_extraction {
                doc_extraction.process(&mut message).await;
            }

            // Store successfully extracted document text in workspace for indexing
            self.store_extracted_documents(&message).await;

            // Check if the caller requested response suppression (passive ingestion).
            let suppress = message
                .metadata
                .get("suppress_response")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            match self.handle_message(&mut message).await {
                Ok(Some(response)) if !response.is_empty() && !suppress => {
                    // Hook: BeforeOutbound — allow hooks to modify or suppress outbound
                    let event = crate::hooks::HookEvent::Outbound {
                        user_id: message.user_id.clone(),
                        channel: message.channel.clone(),
                        content: response.clone(),
                        thread_id: message.thread_id.clone(),
                    };
                    match self.hooks().run(&event).await {
                        Err(err) => {
                            tracing::warn!("BeforeOutbound hook blocked response: {}", err);
                        }
                        Ok(crate::hooks::HookOutcome::Continue {
                            modified: Some(new_content),
                        }) => {
                            if let Err(e) = self
                                .channels
                                .respond(&message, OutgoingResponse::text(new_content))
                                .await
                            {
                                tracing::error!(
                                    channel = %message.channel,
                                    error = %e,
                                    "Failed to send response to channel"
                                );
                            }
                        }
                        _ => {
                            if let Err(e) = self
                                .channels
                                .respond(&message, OutgoingResponse::text(response))
                                .await
                            {
                                tracing::error!(
                                    channel = %message.channel,
                                    error = %e,
                                    "Failed to send response to channel"
                                );
                            }
                        }
                    }
                }
                Ok(Some(response)) if suppress && !response.is_empty() => {
                    tracing::debug!(
                        user = %message.user_id,
                        response_len = response.len(),
                        "suppress_response=true, skipping outbound response"
                    );
                }
                Ok(Some(empty)) => {
                    // Empty response, nothing to send (e.g. approval handled via send_status)
                    tracing::debug!(
                        channel = %message.channel,
                        user = %message.user_id,
                        empty_len = empty.len(),
                        "Suppressed empty response (not sent to channel)"
                    );
                }
                Ok(None) => {
                    // Shutdown signal received (/quit, /exit, /shutdown)
                    tracing::info!("Shutdown command received, exiting...");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error handling message: {}", e);
                    if let Err(send_err) = self
                        .channels
                        .respond(&message, OutgoingResponse::text(format!("Error: {}", e)))
                        .await
                    {
                        tracing::error!(
                            channel = %message.channel,
                            error = %send_err,
                            "Failed to send error response to channel"
                        );
                    }
                }
            }

            // Check event triggers (cheap in-memory regex, fires async if matched)
            if let Some(ref engine) = routine_engine_for_loop {
                let fired = engine.check_event_triggers(&message).await;
                if fired > 0 {
                    tracing::debug!("Fired {} event-triggered routines", fired);
                }
            }
        }

        // Cleanup
        tracing::info!("Agent shutting down...");
        repair_handle.abort();
        pruning_handle.abort();
        if let Some(handle) = heartbeat_handle {
            handle.abort();
        }
        if let Some((cron_handle, _)) = routine_handle {
            cron_handle.abort();
        }
        if let Some(handle) = organizer_handle {
            handle.abort();
        }
        self.scheduler.stop_all().await;
        self.channels.shutdown_all().await?;

        Ok(())
    }

    /// Store extracted document text in workspace memory for future search/recall.
    async fn store_extracted_documents(&self, message: &IncomingMessage) {
        let workspace = match self.workspace() {
            Some(ws) => ws,
            None => return,
        };

        for attachment in &message.attachments {
            if attachment.kind != crate::channels::AttachmentKind::Document {
                continue;
            }
            let text = match &attachment.extracted_text {
                Some(t) if !t.starts_with('[') => t, // skip error messages like "[Failed to..."
                _ => continue,
            };

            // Sanitize filename: strip path separators to prevent directory traversal
            let raw_name = attachment.filename.as_deref().unwrap_or("unnamed_document");
            let filename: String = raw_name
                .chars()
                .map(|c| {
                    if c == '/' || c == '\\' || c == '\0' {
                        '_'
                    } else {
                        c
                    }
                })
                .collect();
            let filename = filename.trim_start_matches('.');
            let filename = if filename.is_empty() {
                "unnamed_document"
            } else {
                filename
            };
            let date = chrono::Utc::now().format("%Y-%m-%d");
            let path = format!("documents/{date}/{filename}");

            let header = format!(
                "# {filename}\n\n\
                 > Uploaded by **{}** via **{}** on {date}\n\
                 > MIME: {} | Size: {} bytes\n\n---\n\n",
                message.user_id,
                message.channel,
                attachment.mime_type,
                attachment.size_bytes.unwrap_or(0),
            );
            let content = format!("{header}{text}");

            match workspace.write(&path, &content).await {
                Ok(_) => {
                    tracing::info!(
                        path = %path,
                        text_len = text.len(),
                        "Stored extracted document in workspace memory"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path,
                        error = %e,
                        "Failed to store extracted document in workspace"
                    );
                }
            }
        }
    }

    async fn handle_message(&self, message: &mut IncomingMessage) -> Result<Option<String>, Error> {
        // Set message tool context for this turn (current channel and target)
        // For Signal, use signal_target from metadata (group:ID or phone number),
        // otherwise fall back to user_id
        let target = message
            .metadata
            .get("signal_target")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| message.user_id.clone());
        self.tools()
            .set_message_tool_context(Some(message.channel.clone()), Some(target))
            .await;

        // Parse submission type first
        let mut submission = SubmissionParser::parse(&message.content);
        tracing::debug!(
            "[agent_loop] Parsed submission: {:?}",
            std::any::type_name_of_val(&submission)
        );

        // Hook: BeforeInbound — allow hooks to modify or reject user input
        if let Submission::UserInput { ref content } = submission {
            let event = crate::hooks::HookEvent::Inbound {
                user_id: message.user_id.clone(),
                channel: message.channel.clone(),
                content: content.clone(),
                thread_id: message.thread_id.clone(),
            };
            match self.hooks().run(&event).await {
                Err(crate::hooks::HookError::Rejected { reason }) => {
                    return Ok(Some(format!("[Message rejected: {}]", reason)));
                }
                Err(err) => {
                    return Ok(Some(format!("[Message blocked by hook policy: {}]", err)));
                }
                Ok(crate::hooks::HookOutcome::Continue {
                    modified: Some(new_content),
                }) => {
                    submission = Submission::UserInput {
                        content: new_content,
                    };
                }
                _ => {} // Continue, fail-open errors already logged in registry
            }
        }

        // Thread routing: if no explicit thread_id, resolve which thread to use.
        // Priority: thread_resolver (pluggable) > workspace_router (legacy).
        let mut routed_thread_id = message.thread_id.clone();
        let mut resolver_context: Option<String> = None;
        if routed_thread_id.is_none()
            && let Submission::UserInput { ref content } = submission
        {
            if let Some(ref resolver) = self.deps.thread_resolver {
                // Use the pluggable thread resolver
                let default_thread_id = Uuid::new_v4(); // placeholder; resolver may ignore it
                match resolver
                    .resolve(&message.user_id, content, default_thread_id)
                    .await
                {
                    Ok(resolution) => {
                        if let Some(tid) = resolution.thread_id {
                            tracing::info!(
                                "ThreadResolver: routed to thread {}",
                                tid
                            );
                            routed_thread_id = Some(tid.to_string());
                        }
                        resolver_context = resolution.context;
                        // Broadcast routing metadata via SSE
                        if !resolution.metadata.is_empty()
                            && let Some(ref sse) = self.deps.sse_tx
                            && let (Some(ws_id), Some(topic)) = (
                                resolution.metadata.get("workspace_id"),
                                resolution.metadata.get("topic"),
                            )
                        {
                            sse.broadcast_for_user(
                                &message.user_id,
                                crate::channels::web::types::SseEvent::WorkspaceRouted {
                                    workspace_id: ws_id.clone(),
                                    topic: topic.clone(),
                                    is_new: resolution.metadata.get("is_new")
                                        .is_some_and(|v| v == "true"),
                                    thread_id: routed_thread_id.clone(),
                                },
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("ThreadResolver failed: {}, falling back", e);
                    }
                }
            } else if let Some(ref router) = self.deps.workspace_router {
                // Legacy workspace router fallback
                match router.route(&message.user_id, content).await {
                    Ok(Some(ws)) if ws.topic != "general" => {
                        tracing::info!(
                            "Workspace routing: matched workspace {} (topic: {})",
                            ws.id, ws.topic
                        );
                        routed_thread_id = Some(ws.conversation_id.to_string());
                        if let Some(ref sse) = self.deps.sse_tx {
                            sse.broadcast_for_user(&message.user_id, crate::channels::web::types::SseEvent::WorkspaceRouted {
                                workspace_id: ws.id.to_string(),
                                topic: ws.topic.clone(),
                                is_new: false,
                                thread_id: Some(ws.conversation_id.to_string()),
                            });
                        }
                        if let Some(ref store) = self.deps.store {
                            let _ = store.touch_agent_workspace(ws.id).await;
                        }
                    }
                    Ok(Some(_)) | Ok(None) => {
                        if let Some(ref store) = self.deps.store {
                            match self
                                .get_or_create_default_workspace(
                                    &message.user_id,
                                    router,
                                    store,
                                )
                                .await
                            {
                                Ok(Some(ws)) => {
                                    tracing::info!(
                                        "Workspace routing: default workspace {} (topic: {})",
                                        ws.id, ws.topic
                                    );
                                    routed_thread_id = Some(ws.conversation_id.to_string());
                                    if let Some(ref sse) = self.deps.sse_tx {
                                        sse.broadcast_for_user(
                                            &message.user_id,
                                            crate::channels::web::types::SseEvent::WorkspaceRouted {
                                                workspace_id: ws.id.to_string(),
                                                topic: ws.topic.clone(),
                                                is_new: false,
                                                thread_id: Some(ws.conversation_id.to_string()),
                                            },
                                        );
                                    }
                                }
                                Ok(None) => {
                                    tracing::info!(
                                        "Workspace routing: no default workspace, using ephemeral thread"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to get/create default workspace: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Workspace routing failed: {}", e);
                    }
                }
            }
        }

        // Inject resolver context into message metadata so the dispatcher can use it
        if let Some(ref ctx) = resolver_context
            && let Some(obj) = message.metadata.as_object_mut()
        {
            obj.insert("__resolver_context".to_string(), serde_json::json!(ctx));
        }

        // Hydrate thread from DB if it's a historical thread not in memory
        let effective_thread_id = routed_thread_id.as_ref().or(message.thread_id.as_ref());
        if let Some(external_thread_id) = effective_thread_id {
            self.maybe_hydrate_thread(message, external_thread_id).await;
        }

        // Resolve session and thread — use routed thread_id if workspace routing matched
        let (session, thread_id) = self
            .session_manager
            .resolve_thread(
                &message.user_id,
                &message.channel,
                routed_thread_id.as_deref().or(message.thread_id.as_deref()),
            )
            .await;

        // Propagate the resolved thread_id back to the message so the gateway
        // can tag SSE responses with the correct thread. Without this, messages
        // arriving without an explicit thread_id would produce responses that
        // the gateway drops ("no thread_id — skipping").
        if message.thread_id.is_none() {
            message.thread_id = routed_thread_id.or_else(|| Some(thread_id.to_string()));
        }

        // Auth mode interception: if the thread is awaiting a token, route
        // the message directly to the credential store. Nothing touches
        // logs, turns, history, or compaction.
        let pending_auth = {
            let sess = session.lock().await;
            sess.threads
                .get(&thread_id)
                .and_then(|t| t.pending_auth.clone())
        };

        if let Some(pending) = pending_auth {
            match &submission {
                Submission::UserInput { content } => {
                    return self
                        .process_auth_token(message, &pending, content, session, thread_id)
                        .await;
                }
                _ => {
                    // Any control submission (interrupt, undo, etc.) cancels auth mode
                    let mut sess = session.lock().await;
                    if let Some(thread) = sess.threads.get_mut(&thread_id) {
                        thread.pending_auth = None;
                    }
                    // Fall through to normal handling
                }
            }
        }

        tracing::debug!(
            "Received message from {} on {} ({} chars)",
            message.user_id,
            message.channel,
            message.content.len()
        );

        // Process based on submission type
        let result = match submission {
            Submission::UserInput { content } => {
                self.process_user_input(message, session, thread_id, &content)
                    .await
            }
            Submission::SystemCommand { command, args } => {
                tracing::debug!(
                    "[agent_loop] SystemCommand: command={}, channel={}",
                    command,
                    message.channel
                );
                // Authorization checks (including restart channel check) are enforced in handle_system_command
                self.handle_system_command(&command, &args, &message.channel)
                    .await
            }
            Submission::Undo => self.process_undo(session, thread_id).await,
            Submission::Redo => self.process_redo(session, thread_id).await,
            Submission::Interrupt => self.process_interrupt(session, thread_id).await,
            Submission::Compact => self.process_compact(session, thread_id).await,
            Submission::Clear => self.process_clear(session, thread_id).await,
            Submission::NewThread => self.process_new_thread(message).await,
            Submission::Heartbeat => self.process_heartbeat().await,
            Submission::Summarize => self.process_summarize(session, thread_id).await,
            Submission::Suggest => self.process_suggest(session, thread_id).await,
            Submission::Organize => self.process_organize(&message.user_id).await,
            Submission::JobStatus { job_id } => {
                self.process_job_status(&message.user_id, job_id.as_deref())
                    .await
            }
            Submission::JobCancel { job_id } => {
                self.process_job_cancel(&message.user_id, &job_id).await
            }
            Submission::Quit => return Ok(None),
            Submission::SwitchThread { thread_id: target } => {
                self.process_switch_thread(message, target).await
            }
            Submission::Resume { checkpoint_id } => {
                self.process_resume(session, thread_id, checkpoint_id).await
            }
            Submission::ExecApproval {
                request_id,
                approved,
                always,
            } => {
                self.process_approval(
                    message,
                    session,
                    thread_id,
                    Some(request_id),
                    approved,
                    always,
                )
                .await
            }
            Submission::ApprovalResponse { approved, always } => {
                self.process_approval(message, session, thread_id, None, approved, always)
                    .await
            }
        };

        // Notify the resolver that a message was processed in this thread.
        // This allows stickiness tracking and other post-routing bookkeeping.
        if let Some(ref resolver) = self.deps.thread_resolver
            && result.is_ok()
        {
            resolver.notify_routed(&message.user_id, thread_id).await;
        }

        // Convert SubmissionResult to response string
        match result? {
            SubmissionResult::Response { content } => {
                // Suppress silent replies (e.g. from group chat "nothing to say" responses)
                if crate::llm::is_silent_reply(&content) {
                    tracing::debug!("Suppressing silent reply token");
                    Ok(None)
                } else {
                    Ok(Some(content))
                }
            }
            SubmissionResult::Ok { message } => Ok(message),
            SubmissionResult::Error { message } => Ok(Some(format!("Error: {}", message))),
            SubmissionResult::Interrupted => Ok(Some("Interrupted.".into())),
            SubmissionResult::NeedApproval {
                request_id,
                tool_name,
                description,
                parameters,
            } => {
                // Each channel renders the approval prompt via send_status.
                // Web gateway shows an inline card, REPL prints a formatted prompt, etc.
                let _ = self
                    .channels
                    .send_status(
                        &message.channel,
                        StatusUpdate::ApprovalNeeded {
                            request_id: request_id.to_string(),
                            tool_name,
                            description,
                            parameters,
                        },
                        &message.metadata,
                    )
                    .await;

                // Empty string signals the caller to skip respond() (no duplicate text)
                Ok(Some(String::new()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_for_preview;

    #[test]
    fn test_truncate_short_input() {
        assert_eq!(truncate_for_preview("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_empty_input() {
        assert_eq!(truncate_for_preview("", 10), "");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate_for_preview("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_over_limit() {
        let result = truncate_for_preview("hello world, this is long", 10);
        assert!(result.ends_with("..."));
        // "hello worl" = 10 chars + "..."
        assert_eq!(result, "hello worl...");
    }

    #[test]
    fn test_truncate_collapses_newlines() {
        let result = truncate_for_preview("line1\nline2\nline3", 100);
        assert!(!result.contains('\n'));
        assert_eq!(result, "line1 line2 line3");
    }

    #[test]
    fn test_truncate_collapses_whitespace() {
        let result = truncate_for_preview("hello   world", 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_truncate_multibyte_utf8() {
        // Each emoji is 4 bytes. Truncating at char boundary must not panic.
        let input = "😀😁😂🤣😃😄😅😆😉😊";
        let result = truncate_for_preview(input, 5);
        assert!(result.ends_with("..."));
        // First 5 chars = 5 emoji
        assert_eq!(result, "😀😁😂🤣😃...");
    }

    #[test]
    fn test_truncate_cjk_characters() {
        // CJK chars are 3 bytes each in UTF-8.
        let input = "你好世界测试数据很长的字符串";
        let result = truncate_for_preview(input, 4);
        assert_eq!(result, "你好世界...");
    }

    #[test]
    fn test_truncate_mixed_multibyte_and_ascii() {
        let input = "hello 世界 foo";
        let result = truncate_for_preview(input, 8);
        // 'h','e','l','l','o',' ','世','界' = 8 chars
        assert_eq!(result, "hello 世界...");
    }
}
