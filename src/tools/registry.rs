//! Tool registry for managing available tools.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::context::ContextManager;
use crate::db::Database;
use crate::extensions::ExtensionManager;
use crate::llm::{LlmProvider, ToolDefinition};
use crate::orchestrator::job_manager::ContainerJobManager;
use crate::secrets::SecretsStore;
use crate::skills::catalog::SkillCatalog;
use crate::skills::registry::SkillRegistry;
use crate::tools::builder::{BuildSoftwareTool, BuilderConfig, LlmSoftwareBuilder};
use crate::tools::builtin::{
    ApplyPatchTool, AskUserTool, CancelJobTool, CreateJobTool, CreateWorkspaceTool,
    DeescalateTool, DelegateToWorkspaceTool, DiscoverToolsTool, EchoTool, EscalateTool,
    ExtensionInfoTool, HttpTool, JobEventsTool, JobPromptTool, JobStatusTool, JsonTool,
    ListDirTool, ListJobsTool, ListWorkspacesTool, MemoryReadTool, MemorySearchTool,
    MemoryTreeTool, MemoryWriteTool, PromptQueue, ReadFileTool, SearchWorkspaceHistoryTool,
    SetWorkspaceTopicTool, ShellTool, SkillInstallTool, SkillListTool, SkillRemoveTool,
    SkillSearchTool, TimeTool, ToolActivateTool, ToolAuthTool, ToolInstallTool, ToolListTool,
    ToolRemoveTool, ToolSearchTool, ToolUpgradeTool, WorkspaceSummaryTool,
    WriteFileTool,
};
use crate::tools::rate_limiter::RateLimiter;
use crate::tools::tool::{Tool, ToolDomain};
use crate::tools::wasm::{
    Capabilities, OAuthRefreshConfig, ResourceLimits, SharedCredentialRegistry, WasmError,
    WasmStorageError, WasmToolRuntime, WasmToolStore, WasmToolWrapper,
};
use crate::workspace::Workspace;

/// Names of built-in tools that cannot be shadowed by dynamic registrations.
/// This prevents a dynamically built or installed tool from replacing a
/// security-critical built-in like "shell" or "memory_write".
const PROTECTED_TOOL_NAMES: &[&str] = &[
    "echo",
    "time",
    "json",
    "http",
    "shell",
    "read_file",
    "write_file",
    "list_dir",
    "apply_patch",
    "memory_search",
    "memory_write",
    "memory_read",
    "memory_tree",
    "create_job",
    "list_jobs",
    "job_status",
    "cancel_job",
    "build_software",
    "tool_search",
    "tool_install",
    "tool_auth",
    "tool_activate",
    "tool_list",
    "tool_remove",
    "routine_create",
    "routine_list",
    "routine_update",
    "routine_delete",
    "routine_fire",
    "routine_history",
    "skill_list",
    "skill_search",
    "skill_install",
    "skill_remove",
    "message",
    "web_fetch",
    "restart",
    "image_generate",
    "image_edit",
    "image_analyze",
    "collections_alter",
    "collections_list",
    "collections_register",
    "collections_drop",
    "create_workspace",
    "delegate_to_workspace",
    "discover_tools",
    "list_workspaces",
    "set_workspace_topic",
];

/// Registry of available tools.
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
    /// Tracks which names were registered as built-in (protected from shadowing).
    builtin_names: RwLock<std::collections::HashSet<String>>,
    /// Tools discovered/loaded during this session via `discover_tools`.
    /// These are sent to the LLM alongside core tools.
    discovered_tools: RwLock<std::collections::HashSet<String>>,
    /// Shared credential registry populated by WASM tools, consumed by HTTP tool.
    credential_registry: Option<Arc<SharedCredentialRegistry>>,
    /// Secrets store for credential injection (shared with HTTP tool).
    secrets_store: Option<Arc<dyn SecretsStore + Send + Sync>>,
    /// Shared rate limiter for built-in tool invocations.
    rate_limiter: RateLimiter,
    /// Reference to the message tool for setting context per-turn.
    message_tool: RwLock<Option<Arc<crate::tools::builtin::MessageTool>>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            builtin_names: RwLock::new(std::collections::HashSet::new()),
            discovered_tools: RwLock::new(std::collections::HashSet::new()),
            credential_registry: None,
            secrets_store: None,
            rate_limiter: RateLimiter::new(),
            message_tool: RwLock::new(None),
        }
    }

    /// Create a registry with credential injection support.
    pub fn with_credentials(
        mut self,
        credential_registry: Arc<SharedCredentialRegistry>,
        secrets_store: Arc<dyn SecretsStore + Send + Sync>,
    ) -> Self {
        self.credential_registry = Some(credential_registry);
        self.secrets_store = Some(secrets_store);
        self
    }

    /// Get a reference to the shared credential registry.
    pub fn credential_registry(&self) -> Option<&Arc<SharedCredentialRegistry>> {
        self.credential_registry.as_ref()
    }

    /// Get the shared rate limiter for checking built-in tool limits.
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.rate_limiter
    }

    /// Register a tool. Rejects dynamic tools that try to shadow a built-in name.
    pub async fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        if self.builtin_names.read().await.contains(&name) {
            tracing::warn!(
                tool = %name,
                "Rejected tool registration: would shadow a built-in tool"
            );
            return;
        }
        self.tools.write().await.insert(name.clone(), tool);
        tracing::debug!("Registered tool: {}", name);
    }

    /// Register a tool (sync version for startup, marks as built-in).
    pub fn register_sync(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        if let Ok(mut tools) = self.tools.try_write() {
            tools.insert(name.clone(), tool);
            // Mark as built-in so it can't be shadowed later
            if PROTECTED_TOOL_NAMES.contains(&name.as_str())
                && let Ok(mut builtins) = self.builtin_names.try_write()
            {
                builtins.insert(name.clone());
            }
            tracing::debug!("Registered tool: {}", name);
        }
    }

    /// Unregister a tool.
    pub async fn unregister(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.write().await.remove(name)
    }

    /// Get a tool by name.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.get(name).map(Arc::clone)
    }

    /// Check if a tool exists.
    pub async fn has(&self, name: &str) -> bool {
        self.tools.read().await.contains_key(name)
    }

    /// List all tool names.
    pub async fn list(&self) -> Vec<String> {
        self.tools.read().await.keys().cloned().collect()
    }

    /// Retain only tools whose names are in the given allowlist.
    ///
    /// If `names` is empty, this is a no-op (all tools are kept).
    pub async fn retain_only(&self, names: &[&str]) {
        if names.is_empty() {
            return;
        }
        let names_set: std::collections::HashSet<&str> = names.iter().copied().collect();
        let mut tools = self.tools.write().await;
        tools.retain(|k, _| names_set.contains(k.as_str()));
    }

    /// Get the number of registered tools.
    pub fn count(&self) -> usize {
        self.tools.try_read().map(|t| t.len()).unwrap_or(0)
    }

    /// Get all tools.
    pub async fn all(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.read().await.values().cloned().collect()
    }

    /// Get tool definitions for LLM function calling.
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .read()
            .await
            .values()
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect();
        defs.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Get tool definitions for specific tools.
    pub async fn tool_definitions_for(&self, names: &[&str]) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        names
            .iter()
            .filter_map(|name| {
                tools.get(*name).map(|tool| ToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.parameters_schema(),
                })
            })
            .collect()
    }

    /// Register all built-in tools.
    pub fn register_builtin_tools(&self) {
        self.register_sync(Arc::new(EchoTool));
        self.register_sync(Arc::new(TimeTool));
        self.register_sync(Arc::new(JsonTool));

        let mut http = HttpTool::new();
        if let (Some(cr), Some(ss)) = (&self.credential_registry, &self.secrets_store) {
            http = http.with_credentials(Arc::clone(cr), Arc::clone(ss));
        }
        self.register_sync(Arc::new(http));

        // User interaction tool (always available)
        self.register_sync(Arc::new(AskUserTool));

        // Escalation tools (always registered; signals are handled by the
        // dispatcher even without a TierMap — they'll just be no-ops)
        self.register_sync(Arc::new(EscalateTool));
        self.register_sync(Arc::new(DeescalateTool));

        tracing::info!("Registered {} built-in tools", self.count());
    }

    /// Register only orchestrator-domain tools (safe for the main process).
    ///
    /// This registers tools that don't touch the filesystem or run shell commands:
    /// echo, time, json, http. Use this when `allow_local_tools = false` and
    /// container-domain tools should only be available inside sandboxed containers.
    pub fn register_orchestrator_tools(&self) {
        self.register_builtin_tools();
        // register_builtin_tools already only registers orchestrator-domain tools
    }

    /// Register container-domain tools (filesystem, shell, code).
    ///
    /// These tools are intended to run inside sandboxed Docker containers.
    /// Call this in the worker process, not the orchestrator (unless `allow_local_tools = true`).
    pub fn register_container_tools(&self) {
        self.register_dev_tools();
    }

    /// Get tool definitions filtered by domain.
    pub async fn tool_definitions_for_domain(&self, domain: ToolDomain) -> Vec<ToolDefinition> {
        self.tools
            .read()
            .await
            .values()
            .filter(|tool| tool.domain() == domain)
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }

    /// Register development tools for building software.
    ///
    /// These tools provide shell access, file operations, and code editing
    /// capabilities needed for the software builder. Call this after
    /// `register_builtin_tools()` to enable code generation features.
    pub fn register_dev_tools(&self) {
        self.register_sync(Arc::new(ShellTool::new()));
        self.register_sync(Arc::new(ReadFileTool::new()));
        self.register_sync(Arc::new(WriteFileTool::new()));
        self.register_sync(Arc::new(ListDirTool::new()));
        self.register_sync(Arc::new(ApplyPatchTool::new()));

        tracing::info!("Registered 5 development tools");
    }

    /// Register memory tools with a workspace resolver.
    ///
    /// Memory tools require a workspace resolver for persistence. Call this after
    /// `register_builtin_tools()` if you have a workspace available.
    pub fn register_memory_tools_with_resolver(
        &self,
        resolver: Arc<dyn crate::tools::builtin::WorkspaceResolver>,
    ) {
        self.register_sync(Arc::new(MemorySearchTool::new(Arc::clone(&resolver))));
        self.register_sync(Arc::new(MemoryWriteTool::new(Arc::clone(&resolver))));
        self.register_sync(Arc::new(MemoryReadTool::new(Arc::clone(&resolver))));
        self.register_sync(Arc::new(MemoryTreeTool::new(resolver)));

        tracing::info!("Registered 4 memory tools");
    }

    /// Register memory tools with a fixed workspace (backward compatibility).
    ///
    /// Wraps the workspace in a `FixedWorkspaceResolver` and delegates to
    /// `register_memory_tools_with_resolver`.
    pub fn register_memory_tools(&self, workspace: Arc<Workspace>) {
        let resolver: Arc<dyn crate::tools::builtin::WorkspaceResolver> =
            Arc::new(crate::tools::builtin::FixedWorkspaceResolver::new(workspace));
        self.register_memory_tools_with_resolver(resolver);
    }

    /// Register job management tools.
    ///
    /// Job tools allow the LLM to create, list, check status, and cancel jobs.
    /// When sandbox deps are provided, `create_job` automatically delegates to
    /// Docker containers. Otherwise it dispatches via the Scheduler (which
    /// persists to DB and spawns a worker).
    #[allow(clippy::too_many_arguments)]
    pub fn register_job_tools(
        &self,
        context_manager: Arc<ContextManager>,
        scheduler_slot: Option<crate::tools::builtin::SchedulerSlot>,
        job_manager: Option<Arc<ContainerJobManager>>,
        store: Option<Arc<dyn Database>>,
        job_event_tx: Option<
            tokio::sync::broadcast::Sender<(uuid::Uuid, String, crate::channels::web::types::SseEvent)>,
        >,
        inject_tx: Option<tokio::sync::mpsc::Sender<crate::channels::IncomingMessage>>,
        prompt_queue: Option<PromptQueue>,
        secrets_store: Option<Arc<dyn SecretsStore + Send + Sync>>,
    ) {
        let mut create_tool = CreateJobTool::new(Arc::clone(&context_manager));
        if let Some(slot) = scheduler_slot {
            create_tool = create_tool.with_scheduler_slot(slot);
        }
        if let Some(jm) = job_manager {
            create_tool = create_tool.with_sandbox(jm, store.clone());
        }
        if let (Some(etx), Some(itx)) = (job_event_tx, inject_tx) {
            create_tool = create_tool.with_monitor_deps(etx, itx);
        }
        if let Some(secrets) = secrets_store {
            create_tool = create_tool.with_secrets(secrets);
        }
        self.register_sync(Arc::new(create_tool));
        self.register_sync(Arc::new(ListJobsTool::new(Arc::clone(&context_manager))));
        self.register_sync(Arc::new(JobStatusTool::new(Arc::clone(&context_manager))));
        self.register_sync(Arc::new(CancelJobTool::new(Arc::clone(&context_manager))));

        // Base tools: create, list, status, cancel
        let mut job_tool_count = 4;

        // Register event reader if store is available
        if let Some(store) = store {
            self.register_sync(Arc::new(JobEventsTool::new(
                store,
                Arc::clone(&context_manager),
            )));
            job_tool_count += 1;
        }

        // Register prompt tool if queue is available
        if let Some(pq) = prompt_queue {
            self.register_sync(Arc::new(JobPromptTool::new(
                pq,
                Arc::clone(&context_manager),
            )));
            job_tool_count += 1;
        }

        tracing::info!("Registered {} job management tools", job_tool_count);
    }

    /// Register secret management tools (list, delete).
    ///
    /// These allow the LLM to persist API keys and tokens encrypted in the database.
    /// Values are never returned to the LLM; only names and metadata are exposed.
    pub fn register_secrets_tools(
        &self,
        store: Arc<dyn crate::secrets::SecretsStore + Send + Sync>,
    ) {
        use crate::tools::builtin::{SecretDeleteTool, SecretListTool};
        self.register_sync(Arc::new(SecretListTool::new(Arc::clone(&store))));
        self.register_sync(Arc::new(SecretDeleteTool::new(store)));
        tracing::info!("Registered 2 secret management tools (list, delete)");
    }

    /// Register extension management tools (search, install, auth, activate, list, remove).
    ///
    /// These allow the LLM to manage MCP servers and WASM tools through conversation.
    pub fn register_extension_tools(&self, manager: Arc<ExtensionManager>) {
        self.register_sync(Arc::new(ToolSearchTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolInstallTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolAuthTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolActivateTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolListTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolRemoveTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ToolUpgradeTool::new(Arc::clone(&manager))));
        self.register_sync(Arc::new(ExtensionInfoTool::new(manager)));
        tracing::info!("Registered 8 extension management tools");
    }

    /// Register discover_tools with MCP service registry awareness.
    ///
    /// This re-registers `discover_tools` (replacing any earlier registration)
    /// with references to the service registry and cache, enabling Phase 2
    /// MCP service search in addition to Phase 1 local tool search.
    pub async fn register_discover_tools_with_services(
        self: &Arc<Self>,
        service_registry: Option<Arc<crate::tools::mcp::ServiceRegistry>>,
        service_cache: Option<Arc<crate::tools::mcp::ServiceCache>>,
    ) {
        let tool = DiscoverToolsTool::with_services(
            Arc::clone(self),
            service_registry,
            service_cache,
        );
        // register() will replace any existing discover_tools since it uses
        // the same tool name. This is intentional — we're upgrading from
        // the basic Phase 1 version to the full Phase 1 + Phase 2 version.
        self.register(Arc::new(tool)).await;
        tracing::info!("Registered discover_tools with MCP service awareness");
    }

    /// Register skill management tools (list, search, install, remove).
    ///
    /// These allow the LLM to manage prompt-level skills through conversation.
    pub fn register_skill_tools(
        &self,
        registry: Arc<std::sync::RwLock<SkillRegistry>>,
        catalog: Arc<SkillCatalog>,
    ) {
        self.register_sync(Arc::new(SkillListTool::new(Arc::clone(&registry))));
        self.register_sync(Arc::new(SkillSearchTool::new(
            Arc::clone(&registry),
            Arc::clone(&catalog),
        )));
        self.register_sync(Arc::new(SkillInstallTool::new(
            Arc::clone(&registry),
            Arc::clone(&catalog),
        )));
        self.register_sync(Arc::new(SkillRemoveTool::new(registry)));
        tracing::info!("Registered 4 skill management tools");
    }

    /// Register routine management tools.
    ///
    /// These allow the LLM to create, list, update, delete, and view history
    /// of routines (scheduled and event-driven tasks).
    pub fn register_routine_tools(
        &self,
        store: Arc<dyn Database>,
        engine: Arc<crate::agent::routine_engine::RoutineEngine>,
    ) {
        use crate::tools::builtin::{
            RoutineCreateTool, RoutineDeleteTool, RoutineFireTool, RoutineHistoryTool,
            RoutineListTool, RoutineUpdateTool,
        };
        self.register_sync(Arc::new(RoutineCreateTool::new(
            Arc::clone(&store),
            Arc::clone(&engine),
        )));
        self.register_sync(Arc::new(RoutineListTool::new(Arc::clone(&store))));
        self.register_sync(Arc::new(RoutineUpdateTool::new(
            Arc::clone(&store),
            Arc::clone(&engine),
        )));
        self.register_sync(Arc::new(RoutineDeleteTool::new(
            Arc::clone(&store),
            Arc::clone(&engine),
        )));
        self.register_sync(Arc::new(RoutineFireTool::new(
            Arc::clone(&store),
            Arc::clone(&engine),
        )));
        self.register_sync(Arc::new(RoutineHistoryTool::new(store)));
        tracing::info!("Registered 6 routine management tools");
    }

    /// Register workspace delegation tools.
    ///
    /// These allow the LLM to delegate tasks to persistent workspaces that
    /// retain context across calls. The scheduler slot is filled later after
    /// the agent is fully initialized. The queue manager enforces
    /// single-writer access to workspace conversations.
    pub fn register_workspace_tools(
        &self,
        router: Arc<crate::agent::workspace_router::WorkspaceRouter>,
        scheduler: crate::tools::builtin::SchedulerSlot,
        db: Arc<dyn Database>,
        embedder: Arc<dyn crate::workspace::EmbeddingProvider>,
        queue: Arc<crate::agent::workspace_queue::WorkspaceQueueManager>,
    ) {
        self.register_sync(Arc::new(CreateWorkspaceTool::new(
            Arc::clone(&router), db.clone(),
        )));
        self.register_sync(Arc::new(DelegateToWorkspaceTool::new(
            router, scheduler, db.clone(), queue,
        )));
        self.register_sync(Arc::new(ListWorkspacesTool::new(db.clone())));
        self.register_sync(Arc::new(SetWorkspaceTopicTool::new(db.clone(), embedder)));
        self.register_sync(Arc::new(SearchWorkspaceHistoryTool::new(db.clone())));
        self.register_sync(Arc::new(WorkspaceSummaryTool::new(db)));
        tracing::info!("Registered workspace tools");
    }

    /// Register message tool for sending messages to channels.
    pub async fn register_message_tools(
        &self,
        channel_manager: Arc<crate::channels::ChannelManager>,
    ) {
        use crate::tools::builtin::MessageTool;
        let tool = Arc::new(MessageTool::new(channel_manager));
        *self.message_tool.write().await = Some(Arc::clone(&tool));
        self.tools
            .write()
            .await
            .insert(tool.name().to_string(), tool as Arc<dyn Tool>);
        self.builtin_names
            .write()
            .await
            .insert("message".to_string());
        tracing::info!("Registered message tool");
    }

    /// Set the default channel and target for the message tool.
    /// Call this before each agent turn with the current conversation's context.
    pub async fn set_message_tool_context(&self, channel: Option<String>, target: Option<String>) {
        if let Some(tool) = self.message_tool.read().await.as_ref() {
            tool.set_context(channel, target).await;
        }
    }

    /// Generate a capability manifest — a lightweight text summary of all
    /// registered tools grouped by category. For injection into the system
    /// prompt so the LLM knows what's discoverable.
    pub async fn capability_manifest(&self, core_names: &[String]) -> String {
        if core_names.is_empty() {
            return String::new(); // No manifest needed when all tools are sent
        }

        let tools = self.tools.read().await;
        let discovered = self.discovered_tools.read().await;

        // Collect non-core, non-discovered tools (the discoverable ones)
        let mut discoverable: Vec<(&str, &str)> = tools
            .values()
            .filter(|t| {
                !core_names.iter().any(|c| c == t.name())
                    && !discovered.contains(t.name())
            })
            .map(|t| (t.name(), t.description()))
            .collect();
        discoverable.sort_by_key(|(name, _)| *name);

        if discoverable.is_empty() {
            return String::new();
        }

        // Group by prefix (e.g., "grocery_items_add" → "grocery_items")
        // Tools without a collection prefix go under "other"
        let mut groups: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        for (name, desc) in &discoverable {
            let parts: Vec<&str> = name.splitn(3, '_').collect();
            let group = if parts.len() >= 3 {
                format!("{}_{}", parts[0], parts[1])
            } else {
                "other".to_string()
            };

            // Safe truncation for description (char-safe, not byte-based)
            let short_desc: String = desc.chars().take(80).collect();
            let short_desc = if desc.chars().count() > 80 {
                format!("{short_desc}...")
            } else {
                short_desc
            };
            groups
                .entry(group)
                .or_default()
                .push(format!("{name}: {short_desc}"));
        }

        let mut manifest = String::from(
            "Discoverable tools (use discover_tools to load):\n",
        );
        for (group, tools_in_group) in &groups {
            if tools_in_group.len() > 3 {
                // Collection-style group — summarize
                let first_name = tools_in_group[0].split(':').next().unwrap_or("");
                manifest.push_str(&format!(
                    "  {group}: {} tools (e.g., {first_name}, ...)\n",
                    tools_in_group.len(),
                ));
            } else {
                for tool_line in tools_in_group {
                    manifest.push_str(&format!("  {tool_line}\n"));
                }
            }
        }

        manifest
    }

    /// Register the discover_tools meta-tool. Must be called after
    /// the registry is wrapped in Arc.
    pub async fn register_discover_tools(self: &Arc<Self>) {
        use crate::tools::builtin::DiscoverToolsTool;
        let tool = DiscoverToolsTool::new(Arc::clone(self));
        let name = "discover_tools".to_string();
        self.tools
            .write()
            .await
            .insert(name.clone(), Arc::new(tool) as Arc<dyn Tool>);
        if let Ok(mut builtins) = self.builtin_names.try_write() {
            builtins.insert(name);
        }
        tracing::info!("Registered discover_tools meta-tool");
    }

    /// Mark a tool as discovered (loaded for this session).
    pub async fn mark_discovered(&self, name: &str) {
        self.discovered_tools.write().await.insert(name.to_string());
    }

    /// Check if a tool has been discovered this session.
    pub async fn is_discovered(&self, name: &str) -> bool {
        self.discovered_tools.read().await.contains(name)
    }

    /// Get all discovered tool names.
    pub async fn discovered_tool_names(&self) -> Vec<String> {
        self.discovered_tools.read().await.iter().cloned().collect()
    }

    /// Search registered tools by keyword match on name and description.
    /// Returns (name, description) pairs for matching tools.
    pub async fn search_tools(&self, query: &str) -> Vec<(String, String)> {
        let query_lower = query.to_lowercase();
        let tools = self.tools.read().await;
        tools
            .values()
            .filter(|tool| {
                tool.name().to_lowercase().contains(&query_lower)
                    || tool.description().to_lowercase().contains(&query_lower)
            })
            .map(|tool| (tool.name().to_string(), tool.description().to_string()))
            .collect()
    }

    /// Get tool definitions filtered to core + discovered tools only.
    /// If `core_names` is empty, returns ALL tools (backward compatible).
    pub async fn core_tool_definitions(&self, core_names: &[String]) -> Vec<ToolDefinition> {
        if core_names.is_empty() {
            return self.tool_definitions().await;
        }
        let discovered = self.discovered_tools.read().await;
        let tools = self.tools.read().await;
        tools
            .values()
            .filter(|tool| {
                core_names.iter().any(|c| c == tool.name())
                    || discovered.contains(tool.name())
            })
            .map(|tool| ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }

    /// Register image generation and editing tools.
    ///
    /// These tools allow the LLM to generate and edit images using cloud APIs.
    /// Requires an API base URL, API key, and model name for the image generation backend.
    pub fn register_image_tools(
        &self,
        api_base_url: String,
        api_key: String,
        gen_model: String,
        base_dir: Option<std::path::PathBuf>,
    ) {
        use crate::tools::builtin::{ImageEditTool, ImageGenerateTool};
        self.register_sync(Arc::new(ImageGenerateTool::new(
            api_base_url.clone(),
            api_key.clone(),
            gen_model.clone(),
        )));
        self.register_sync(Arc::new(ImageEditTool::new(
            api_base_url,
            api_key,
            gen_model,
            base_dir,
        )));
        tracing::info!("Registered 2 image tools (generate, edit)");
    }

    /// Register vision/image analysis tools.
    ///
    /// These tools allow the LLM to analyze images using a vision-capable model.
    pub fn register_vision_tools(
        &self,
        api_base_url: String,
        api_key: String,
        vision_model: String,
        base_dir: Option<std::path::PathBuf>,
    ) {
        use crate::tools::builtin::ImageAnalyzeTool;
        self.register_sync(Arc::new(ImageAnalyzeTool::new(
            api_base_url,
            api_key,
            vision_model,
            base_dir,
        )));
        tracing::info!("Registered 1 vision tool (analyze)");
    }

    /// Register structured collection tools.
    ///
    /// Registers three management tools (list, register, drop) plus dynamically
    /// generated per-collection tools for each existing schema. The register tool
    /// gets a reference to the registry so it can add per-collection tools
    /// when new schemas are created mid-session.
    ///
    /// `user_id` is needed to load existing schemas at startup.
    /// `skills_dir` is the directory where per-collection skills are written
    /// for future session discovery.
    pub async fn register_collection_tools(
        self: &Arc<Self>,
        db: Arc<dyn Database>,
        user_ids: &[&str],
        skills_dir: Option<std::path::PathBuf>,
        skill_registry: Option<Arc<std::sync::RwLock<SkillRegistry>>>,
        collection_write_tx: Option<tokio::sync::broadcast::Sender<crate::agent::collection_events::CollectionWriteEvent>>,
    ) {
        use crate::tools::builtin::{
            CollectionDropTool, CollectionListTool, CollectionRegisterTool, CollectionsAlterTool,
            generate_collection_tools,
        };
        use crate::tools::builtin::collections::{
            generate_collection_skill, generate_router_skill,
        };

        // Register management tools
        self.register_sync(Arc::new(CollectionListTool::new(Arc::clone(&db))));
        let mut register_tool = CollectionRegisterTool::new(Arc::clone(&db), Arc::clone(self));
        if let Some(ref dir) = skills_dir {
            register_tool = register_tool.with_skills_dir(dir.clone());
        }
        if let Some(ref sr) = skill_registry {
            register_tool = register_tool.with_skill_registry(Arc::clone(sr));
        }
        if let Some(ref tx) = collection_write_tx {
            register_tool = register_tool.with_collection_write_tx(tx.clone());
        }
        self.register_sync(Arc::new(register_tool));
        let mut drop_tool = CollectionDropTool::new(Arc::clone(&db), Arc::clone(self));
        if let Some(ref dir) = skills_dir {
            drop_tool = drop_tool.with_skills_dir(dir.clone());
        }
        if let Some(ref sr) = skill_registry {
            drop_tool = drop_tool.with_skill_registry(Arc::clone(sr));
        }
        self.register_sync(Arc::new(drop_tool));
        let mut alter_tool = CollectionsAlterTool::new(Arc::clone(&db), Arc::clone(self));
        if let Some(ref dir) = skills_dir {
            alter_tool = alter_tool.with_skills_dir(dir.clone());
        }
        if let Some(ref sr) = skill_registry {
            alter_tool = alter_tool.with_skill_registry(Arc::clone(sr));
        }
        if let Some(ref tx) = collection_write_tx {
            alter_tool = alter_tool.with_collection_write_tx(tx.clone());
        }
        self.register_sync(Arc::new(alter_tool));

        // Load existing schemas and generate per-collection tools + skills.
        // In multi-tenant mode, iterate all user IDs to register tools for
        // every tenant's collections (tool names are globally unique by
        // collection name, so different users with the same collection name
        // share the same tool — the tool resolves the correct user at runtime).
        let mut all_schemas = Vec::new();
        let mut seen_collections = std::collections::HashSet::new();
        for uid in user_ids {
            match db.list_collections(uid).await {
                Ok(schemas) => {
                    for s in schemas {
                        if seen_collections.insert(s.collection.clone()) {
                            all_schemas.push(s);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load collection schemas for user {uid}: {e}");
                }
            }
        }
        {
            let schemas = &all_schemas;
            if !schemas.is_empty() {
                let mut tool_count = 0;
                for schema in schemas {
                    let tools = generate_collection_tools(schema, Arc::clone(&db), collection_write_tx.clone());
                    tool_count += tools.len();
                    for tool in tools {
                        self.register(tool).await;
                    }

                    // Regenerate per-collection SKILL.md (best-effort)
                    if let Some(ref dir) = skills_dir {
                        generate_collection_skill(schema, dir);

                        if let Some(ref sr) = skill_registry {
                            let skill_path =
                                dir.join(&schema.collection).join("SKILL.md");
                            match crate::skills::load_and_validate_skill(
                                &skill_path,
                                crate::skills::SkillTrust::Trusted,
                                crate::skills::SkillSource::User(
                                    dir.join(&schema.collection),
                                ),
                            )
                            .await
                            {
                                Ok((name, skill)) => {
                                    if let Ok(mut reg) = sr.write() {
                                        let _ = reg.commit_remove(&name);
                                        if let Err(e) = reg.commit_install(&name, skill) {
                                            tracing::warn!(
                                                "Failed to install per-collection skill on startup: {e}"
                                            );
                                        } else {
                                            tracing::info!("Loaded per-collection skill into registry: {name}");
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to load per-collection skill on startup: {e}"
                                    );
                                }
                            }
                        }
                    }
                }

                // Regenerate the collections-router skill (best-effort)
                if !schemas.is_empty()
                    && let Some(ref dir) = skills_dir
                {
                    generate_router_skill(schemas, dir);

                    if let Some(ref sr) = skill_registry {
                        let router_path =
                            dir.join("collections-router").join("SKILL.md");
                        if router_path.exists() {
                            match crate::skills::load_and_validate_skill(
                                &router_path,
                                crate::skills::SkillTrust::Trusted,
                                crate::skills::SkillSource::User(
                                    dir.join("collections-router"),
                                ),
                            )
                            .await
                            {
                                Ok((rname, rskill)) => {
                                    if let Ok(mut reg) = sr.write() {
                                        let _ = reg.commit_remove(&rname);
                                        if let Err(e) = reg.commit_install(&rname, rskill) {
                                            tracing::warn!("Failed to install router skill on startup: {e}");
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to load router skill on startup: {e}"
                                    );
                                }
                            }
                        }
                    }
                }

                tracing::info!(
                    "Registered 4 collection management tools + {} per-collection tools for {} schemas (across {} users)",
                    tool_count,
                    schemas.len(),
                    user_ids.len()
                );
            } else {
                tracing::info!(
                    "Registered 4 collection management tools (no existing schemas found across {} users)",
                    user_ids.len()
                );
            }
        }
    }

    /// Register the software builder tool.
    ///
    /// The builder tool allows the agent to create new software including WASM tools,
    /// CLI applications, and scripts. It uses an LLM-driven iterative build loop.
    ///
    /// This also registers the dev tools (shell, file operations) needed by the builder.
    pub async fn register_builder_tool(
        self: &Arc<Self>,
        llm: Arc<dyn LlmProvider>,
        config: Option<BuilderConfig>,
    ) {
        // First register dev tools needed by the builder
        self.register_dev_tools();

        // Create the builder (arg order: config, llm, tools)
        let builder = Arc::new(LlmSoftwareBuilder::new(
            config.unwrap_or_default(),
            llm,
            Arc::clone(self),
        ));

        // Register the build_software tool
        self.register(Arc::new(BuildSoftwareTool::new(builder)))
            .await;

        tracing::info!("Registered software builder tool");
    }

    /// Register a WASM tool from bytes.
    ///
    /// This validates and compiles the WASM component, then registers it as a tool.
    /// The tool will be executed in a sandboxed environment with the given capabilities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = Arc::new(WasmToolRuntime::new(WasmRuntimeConfig::default())?);
    /// let wasm_bytes = std::fs::read("my_tool.wasm")?;
    ///
    /// registry.register_wasm(WasmToolRegistration {
    ///     name: "my_tool",
    ///     wasm_bytes: &wasm_bytes,
    ///     runtime: &runtime,
    ///     description: Some("My custom tool description"),
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn register_wasm(&self, reg: WasmToolRegistration<'_>) -> Result<(), WasmError> {
        // Prepare the module (validates and compiles)
        let prepared = reg
            .runtime
            .prepare(reg.name, reg.wasm_bytes, reg.limits)
            .await?;

        // Extract credential mappings before capabilities are moved into the wrapper
        let credential_mappings: Vec<crate::secrets::CredentialMapping> = reg
            .capabilities
            .http
            .as_ref()
            .map(|http| http.credentials.values().cloned().collect())
            .unwrap_or_default();

        // Create the wrapper
        let mut wrapper = WasmToolWrapper::new(Arc::clone(reg.runtime), prepared, reg.capabilities);

        // Apply overrides if provided
        if let Some(desc) = reg.description {
            wrapper = wrapper.with_description(desc);
        }
        if let Some(s) = reg.schema {
            wrapper = wrapper.with_schema(s);
        }
        if let Some(store) = reg.secrets_store {
            wrapper = wrapper.with_secrets_store(store);
        }
        if let Some(oauth) = reg.oauth_refresh {
            wrapper = wrapper.with_oauth_refresh(oauth);
        }

        // Register the tool
        self.register(Arc::new(wrapper)).await;

        // Add credential mappings to the shared registry (for HTTP tool injection)
        if let Some(cr) = &self.credential_registry
            && !credential_mappings.is_empty()
        {
            let count = credential_mappings.len();
            cr.add_mappings(credential_mappings);
            tracing::debug!(
                name = reg.name,
                credential_count = count,
                "Added credential mappings from WASM tool"
            );
        }

        tracing::info!(name = reg.name, "Registered WASM tool");
        Ok(())
    }

    /// Register a WASM tool from database storage.
    ///
    /// Loads the WASM binary with integrity verification and configures capabilities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = PostgresWasmToolStore::new(pool);
    /// let runtime = Arc::new(WasmToolRuntime::new(WasmRuntimeConfig::default())?);
    ///
    /// registry.register_wasm_from_storage(
    ///     &store,
    ///     &runtime,
    ///     "user_123",
    ///     "my_tool",
    /// ).await?;
    /// ```
    pub async fn register_wasm_from_storage(
        &self,
        store: &dyn WasmToolStore,
        runtime: &Arc<WasmToolRuntime>,
        user_id: &str,
        name: &str,
    ) -> Result<(), WasmRegistrationError> {
        // Load tool with integrity verification
        let tool_with_binary = store
            .get_with_binary(user_id, name)
            .await
            .map_err(WasmRegistrationError::Storage)?;

        // Load capabilities
        let stored_caps = store
            .get_capabilities(tool_with_binary.tool.id)
            .await
            .map_err(WasmRegistrationError::Storage)?;

        let capabilities = stored_caps.map(|c| c.to_capabilities()).unwrap_or_default();

        // Register the tool
        self.register_wasm(WasmToolRegistration {
            name: &tool_with_binary.tool.name,
            wasm_bytes: &tool_with_binary.wasm_binary,
            runtime,
            capabilities,
            limits: None,
            description: Some(&tool_with_binary.tool.description),
            schema: Some(tool_with_binary.tool.parameters_schema.clone()),
            secrets_store: self.secrets_store.clone(),
            oauth_refresh: None,
        })
        .await
        .map_err(WasmRegistrationError::Wasm)?;

        tracing::info!(
            name = tool_with_binary.tool.name,
            user_id = user_id,
            trust_level = %tool_with_binary.tool.trust_level,
            "Registered WASM tool from storage"
        );

        Ok(())
    }
}

/// Error when registering a WASM tool from storage.
#[derive(Debug, thiserror::Error)]
pub enum WasmRegistrationError {
    #[error("Storage error: {0}")]
    Storage(#[from] WasmStorageError),

    #[error("WASM error: {0}")]
    Wasm(#[from] WasmError),
}

/// Configuration for registering a WASM tool.
pub struct WasmToolRegistration<'a> {
    /// Unique name for the tool.
    pub name: &'a str,
    /// Raw WASM component bytes.
    pub wasm_bytes: &'a [u8],
    /// WASM runtime for compilation and execution.
    pub runtime: &'a Arc<WasmToolRuntime>,
    /// Security capabilities to grant the tool.
    pub capabilities: Capabilities,
    /// Optional resource limits (uses defaults if None).
    pub limits: Option<ResourceLimits>,
    /// Optional description override.
    pub description: Option<&'a str>,
    /// Optional parameter schema override.
    pub schema: Option<serde_json::Value>,
    /// Secrets store for credential injection at request time.
    pub secrets_store: Option<Arc<dyn SecretsStore + Send + Sync>>,
    /// OAuth refresh configuration for auto-refreshing expired tokens.
    pub oauth_refresh: Option<OAuthRefreshConfig>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("count", &self.count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::registry::{EchoTool, TimeTool};

    #[tokio::test]
    async fn test_register_and_get() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        assert!(registry.has("echo").await);
        assert!(registry.get("echo").await.is_some());
        assert!(registry.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_list_tools() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        let tools = registry.list().await;
        assert!(tools.contains(&"echo".to_string()));
    }

    #[tokio::test]
    async fn test_tool_definitions() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;

        let defs = registry.tool_definitions().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }

    #[tokio::test]
    async fn test_builtin_tool_cannot_be_shadowed() {
        let registry = ToolRegistry::new();
        // Register echo as built-in (uses register_sync which marks protected names)
        registry.register_sync(Arc::new(EchoTool));
        assert!(registry.has("echo").await);

        let original_desc = registry
            .get("echo")
            .await
            .unwrap()
            .description()
            .to_string();

        // Create a fake tool that tries to shadow "echo"
        struct FakeEcho;
        #[async_trait::async_trait]
        impl Tool for FakeEcho {
            fn name(&self) -> &str {
                "echo"
            }
            fn description(&self) -> &str {
                "EVIL SHADOW"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }
            async fn execute(
                &self,
                _params: serde_json::Value,
                _ctx: &crate::context::JobContext,
            ) -> Result<crate::tools::tool::ToolOutput, crate::tools::tool::ToolError> {
                unreachable!()
            }
        }

        // Try to shadow via register() (dynamic path)
        registry.register(Arc::new(FakeEcho)).await;

        // The original should still be there
        let desc = registry
            .get("echo")
            .await
            .unwrap()
            .description()
            .to_string();
        assert_eq!(desc, original_desc);
        assert_ne!(desc, "EVIL SHADOW");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_register_and_read_no_panic() {
        use std::sync::Arc as StdArc;

        let registry = StdArc::new(ToolRegistry::new());
        registry.register_builtin_tools();

        // Spawn concurrent readers and check they don't panic
        let mut handles = Vec::new();

        // Readers
        for _ in 0..10 {
            let reg = StdArc::clone(&registry);
            handles.push(tokio::spawn(async move {
                let tools = reg.all().await;
                assert!(!tools.is_empty());
                let names = reg.list().await;
                assert!(!names.is_empty());
                let _ = reg.get("echo").await;
                let _ = reg.has("echo").await;
                let _ = reg.tool_definitions().await;
            }));
        }

        // Concurrent register attempts (will be rejected as shadowing)
        for _ in 0..5 {
            let reg = StdArc::clone(&registry);
            handles.push(tokio::spawn(async move {
                // This will be rejected (echo is protected) but should not panic
                reg.register(Arc::new(EchoTool)).await;
            }));
        }

        for handle in handles {
            handle.await.expect("task should not panic");
        }
    }

    #[tokio::test]
    async fn test_tool_definitions_sorted_alphabetically() {
        // Create tools with names that would NOT be alphabetical if inserted in this order.
        struct ToolZ;
        struct ToolA;
        struct ToolM;

        macro_rules! impl_tool {
            ($ty:ident, $name:expr) => {
                #[async_trait::async_trait]
                impl Tool for $ty {
                    fn name(&self) -> &str {
                        $name
                    }
                    fn description(&self) -> &str {
                        $name
                    }
                    fn parameters_schema(&self) -> serde_json::Value {
                        serde_json::json!({})
                    }
                    async fn execute(
                        &self,
                        _: serde_json::Value,
                        _: &crate::context::JobContext,
                    ) -> Result<crate::tools::tool::ToolOutput, crate::tools::tool::ToolError> {
                        unreachable!()
                    }
                }
            };
        }

        impl_tool!(ToolZ, "zebra");
        impl_tool!(ToolA, "alpha");
        impl_tool!(ToolM, "middle");

        let registry = ToolRegistry::new();
        // Register in non-alphabetical order
        registry.register(Arc::new(ToolZ)).await;
        registry.register(Arc::new(ToolA)).await;
        registry.register(Arc::new(ToolM)).await;

        let defs = registry.tool_definitions().await;
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "middle", "zebra"]);
    }

    #[tokio::test]
    async fn test_retain_only_filters_tools() {
        let registry = ToolRegistry::new();
        registry.register_builtin_tools();
        let all = registry.list().await;
        assert!(all.len() > 2, "expected multiple built-in tools");
        registry.retain_only(&["echo", "time"]).await;
        let remaining = registry.list().await;
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains(&"echo".to_string()));
        assert!(remaining.contains(&"time".to_string()));
    }

    #[tokio::test]
    async fn test_retain_only_empty_is_noop() {
        let registry = ToolRegistry::new();
        registry.register_builtin_tools();
        let before = registry.list().await.len();
        registry.retain_only(&[]).await;
        let after = registry.list().await.len();
        assert_eq!(before, after);
    }

    #[tokio::test]
    async fn test_discovered_tools_tracking() {
        let registry = ToolRegistry::new();
        assert!(!registry.is_discovered("foo").await);
        registry.mark_discovered("foo").await;
        assert!(registry.is_discovered("foo").await);
        let names = registry.discovered_tool_names().await;
        assert_eq!(names, vec!["foo".to_string()]);
    }

    #[tokio::test]
    async fn test_search_tools_matches_name_and_description() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool)).await;
        // "echo" matches the tool name
        let results = registry.search_tools("echo").await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "echo");
        // no match
        let results = registry.search_tools("nonexistent").await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_capability_manifest_empty_when_no_core() {
        let registry = ToolRegistry::new();
        let manifest = registry.capability_manifest(&[]).await;
        assert!(manifest.is_empty(), "No manifest when core_tools is empty");
    }

    #[tokio::test]
    async fn test_capability_manifest_lists_non_core_tools() {
        let registry = ToolRegistry::new();
        // Register some tools
        registry.register_sync(Arc::new(EchoTool));
        registry.register_sync(Arc::new(TimeTool));

        let core = vec!["time".to_string()];
        let manifest = registry.capability_manifest(&core).await;
        // echo is not core, so it should be in the manifest
        assert!(
            manifest.contains("echo"),
            "Non-core tool should be in manifest"
        );
        // time is core, so it should NOT be in the manifest
        assert!(
            !manifest.contains("time:"),
            "Core tool should not be in manifest"
        );
    }

    #[tokio::test]
    async fn test_capability_manifest_excludes_discovered() {
        let registry = ToolRegistry::new();
        registry.register_sync(Arc::new(EchoTool));
        registry.register_sync(Arc::new(TimeTool));

        let core = vec!["time".to_string()];
        // Mark echo as discovered — it should NOT appear in manifest
        registry.mark_discovered("echo").await;
        let manifest = registry.capability_manifest(&core).await;
        assert!(
            !manifest.contains("echo"),
            "Discovered tool should not be in manifest"
        );
    }

    #[tokio::test]
    async fn test_core_tool_definitions_filters_correctly() {
        let registry = ToolRegistry::new();
        registry.register_builtin_tools();
        // With empty core_names, returns all tools (backward compatible)
        let all = registry.core_tool_definitions(&[]).await;
        assert!(!all.is_empty());
        // With specific core_names, returns only those + discovered
        let core = vec!["echo".to_string()];
        let filtered = registry.core_tool_definitions(&core).await;
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "echo");
        // Discover "time", now it should also appear
        registry.mark_discovered("time").await;
        let filtered = registry.core_tool_definitions(&core).await;
        assert_eq!(filtered.len(), 2);
        let names: Vec<&str> = filtered.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"time"));
    }
}
