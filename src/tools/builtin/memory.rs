//! Memory tools for persistent workspace memory.
//!
//! These tools allow the agent to:
//! - Search past memories, decisions, and context
//! - Read and write files in the workspace
//!
//! # Usage
//!
//! The agent should use `memory_search` before answering questions about
//! prior work, decisions, dates, people, preferences, or todos.
//!
//! Use `memory_write` to persist important facts that should be remembered
//! across sessions.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};
use crate::workspace::{EmbeddingProvider, Workspace, paths};

// ---------------------------------------------------------------------------
// WorkspaceResolver — resolves the correct workspace per request
// ---------------------------------------------------------------------------

/// Resolves a workspace for a given user ID.
///
/// In single-user mode, always returns the same workspace created at startup.
/// In multi-tenant mode, creates and caches per-user workspaces on demand.
#[async_trait]
pub trait WorkspaceResolver: Send + Sync {
    /// Resolve a workspace for the given user ID.
    async fn resolve(&self, user_id: &str) -> Result<Arc<Workspace>, ToolError>;
}

/// Returns the same workspace regardless of user ID (single-user mode).
pub struct FixedWorkspaceResolver {
    workspace: Arc<Workspace>,
}

impl FixedWorkspaceResolver {
    /// Create a resolver that always returns the given workspace.
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self { workspace }
    }
}

#[async_trait]
impl WorkspaceResolver for FixedWorkspaceResolver {
    async fn resolve(&self, _user_id: &str) -> Result<Arc<Workspace>, ToolError> {
        Ok(Arc::clone(&self.workspace))
    }
}

/// Creates and caches per-user workspaces on demand (multi-tenant mode).
pub struct PerUserWorkspaceResolver {
    db: Arc<dyn Database>,
    embeddings: Option<Arc<dyn EmbeddingProvider>>,
    /// Per-user gateway config for workspace_read_scopes and memory_layers.
    user_configs: HashMap<String, PerUserConfig>,
    cache: RwLock<HashMap<String, Arc<Workspace>>>,
}

/// Per-user configuration extracted from UserTokenConfig at startup.
#[derive(Clone)]
struct PerUserConfig {
    workspace_read_scopes: Vec<String>,
    memory_layers: Vec<crate::workspace::layer::MemoryLayer>,
}

impl PerUserWorkspaceResolver {
    /// Create a new per-user workspace resolver.
    pub fn new(
        db: Arc<dyn Database>,
        embeddings: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Self {
        Self {
            db,
            embeddings,
            user_configs: HashMap::new(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Register per-user configuration (workspace_read_scopes, memory_layers).
    pub fn add_user_config(
        &mut self,
        user_id: String,
        workspace_read_scopes: Vec<String>,
        memory_layers: Vec<crate::workspace::layer::MemoryLayer>,
    ) {
        self.user_configs.insert(
            user_id,
            PerUserConfig {
                workspace_read_scopes,
                memory_layers,
            },
        );
    }
}

#[async_trait]
impl WorkspaceResolver for PerUserWorkspaceResolver {
    async fn resolve(&self, user_id: &str) -> Result<Arc<Workspace>, ToolError> {
        // Fast path: check the cache
        {
            let cache = self.cache.read().await;
            if let Some(ws) = cache.get(user_id) {
                return Ok(Arc::clone(ws));
            }
        }

        // Slow path: create a new workspace and cache it
        let mut ws = Workspace::new_with_db(user_id, Arc::clone(&self.db));

        if let Some(config) = self.user_configs.get(user_id) {
            if !config.workspace_read_scopes.is_empty() {
                ws = ws.with_additional_read_scopes(config.workspace_read_scopes.clone());
            }
            ws = ws.with_memory_layers(config.memory_layers.clone());
        }

        if let Some(ref emb) = self.embeddings {
            ws = ws.with_embeddings(Arc::clone(emb));
        }

        let ws = Arc::new(ws);
        self.cache.write().await.insert(user_id.to_string(), Arc::clone(&ws));
        Ok(ws)
    }
}

/// Identity files that the LLM must not overwrite via tool calls.
/// These are loaded into the system prompt and could be used for prompt
/// injection if an attacker tricks the agent into overwriting them.
const PROTECTED_IDENTITY_FILES: &[&str] =
    &[paths::IDENTITY, paths::SOUL, paths::AGENTS, paths::USER];

/// Tool for searching workspace memory.
///
/// Performs hybrid search (FTS + semantic) across all memory documents.
/// The agent should call this tool before answering questions about
/// prior work, decisions, preferences, or any historical context.
pub struct MemorySearchTool {
    resolver: Arc<dyn WorkspaceResolver>,
}

impl MemorySearchTool {
    /// Create a new memory search tool with a resolver.
    pub fn new(resolver: Arc<dyn WorkspaceResolver>) -> Self {
        Self { resolver }
    }

    /// Create from a fixed workspace (backward compatibility).
    pub fn from_workspace(workspace: Arc<Workspace>) -> Self {
        Self {
            resolver: Arc::new(FixedWorkspaceResolver::new(workspace)),
        }
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn name(&self) -> &str {
        "memory_search"
    }

    fn description(&self) -> &str {
        "Search past memories, decisions, and context. MUST be called before answering \
         questions about prior work, decisions, dates, people, preferences, or todos. \
         Returns relevant snippets with relevance scores."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Use natural language to describe what you're looking for."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 20)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
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

        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .min(20) as usize;

        let workspace = self.resolver.resolve(&ctx.user_id).await?;
        let results = workspace
            .search(query, limit)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Search failed: {}", e)))?;

        let result_count = results.len();
        let output = serde_json::json!({
            "query": query,
            "results": results.into_iter().map(|r| serde_json::json!({
                "content": r.content,
                "score": r.score,
                "path": r.document_path,
                "document_id": r.document_id.to_string(),
                "is_hybrid_match": r.is_hybrid(),
            })).collect::<Vec<_>>(),
            "result_count": result_count,
        });

        Ok(ToolOutput::success(output, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal memory, trusted content
    }
}

/// Tool for writing to workspace memory.
///
/// Use this to persist important information that should be remembered
/// across sessions: decisions, preferences, facts, lessons learned.
pub struct MemoryWriteTool {
    resolver: Arc<dyn WorkspaceResolver>,
}

impl MemoryWriteTool {
    /// Create a new memory write tool with a resolver.
    pub fn new(resolver: Arc<dyn WorkspaceResolver>) -> Self {
        Self { resolver }
    }

    /// Create from a fixed workspace (backward compatibility).
    pub fn from_workspace(workspace: Arc<Workspace>) -> Self {
        Self {
            resolver: Arc::new(FixedWorkspaceResolver::new(workspace)),
        }
    }
}

#[async_trait]
impl Tool for MemoryWriteTool {
    fn name(&self) -> &str {
        "memory_write"
    }

    fn description(&self) -> &str {
        "Write to persistent memory (database-backed, NOT the local filesystem). \
         Use for important facts, decisions, preferences, or lessons learned that should \
         be remembered across sessions. Targets: 'memory' for curated long-term facts, \
         'daily_log' for timestamped session notes, 'heartbeat' for the periodic \
         checklist (HEARTBEAT.md), 'bootstrap' to clear the first-run ritual file, \
         or provide a custom path for arbitrary file creation."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to write to memory. Be concise but include relevant context."
                },
                "target": {
                    "type": "string",
                    "description": "Where to write: 'memory' for MEMORY.md, 'daily_log' for today's log, 'heartbeat' for HEARTBEAT.md checklist, 'bootstrap' to clear BOOTSTRAP.md (content is ignored; the file is always cleared), or a path like 'projects/alpha/notes.md'",
                    "default": "daily_log"
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, append to existing content. If false, replace entirely.",
                    "default": true
                },
                "layer": {
                    "type": "string",
                    "description": "Memory layer to write to (e.g. 'private', 'shared', 'finance'). When omitted, writes to the workspace's default scope."
                }
            },
            "required": ["content"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let content = require_str(&params, "content")?;
        let workspace = self.resolver.resolve(&ctx.user_id).await?;

        let target = params
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("daily_log");

        // Bootstrap target: clear BOOTSTRAP.md to mark first-run ritual complete.
        // Handled early because it accepts empty content (unlike other targets).
        if target == "bootstrap" {
            // Write empty content to effectively disable the bootstrap injection.
            // system_prompt_for_context() skips empty files.
            workspace
                .write(paths::BOOTSTRAP, "")
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;

            let output = serde_json::json!({
                "status": "cleared",
                "path": paths::BOOTSTRAP,
                "message": "BOOTSTRAP.md cleared. First-run ritual will not repeat.",
            });

            return Ok(ToolOutput::success(output, start.elapsed()));
        }

        if content.trim().is_empty() {
            return Err(ToolError::InvalidParameters(
                "content cannot be empty".to_string(),
            ));
        }

        let target = params
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("daily_log");

        // Normalize the target path early so protection checks can't be bypassed
        // with trailing slashes, double slashes, or leading slashes.
        let target = target.trim_matches('/');


        // Reject writes to identity files that are loaded into the system prompt.
        // An attacker could use prompt injection to trick the agent into overwriting
        // these, poisoning future conversations.
        if PROTECTED_IDENTITY_FILES.contains(&target) {
            return Err(ToolError::NotAuthorized(format!(
                "writing to '{}' is not allowed (identity file protected from tool writes)",
                target,
            )));
        }

        let append = params
            .get("append")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let layer = params.get("layer").and_then(|v| v.as_str());

        // Resolve the target to a workspace path
        let resolved_path = match target {
            "memory" => paths::MEMORY.to_string(),
            "daily_log" => format!("daily/{}.md", chrono::Utc::now().format("%Y-%m-%d")),
            "heartbeat" => paths::HEARTBEAT.to_string(),
            path => {
                // Second protection check: case-insensitive match after normalization.
                if PROTECTED_IDENTITY_FILES
                    .iter()
                    .any(|p| path.eq_ignore_ascii_case(p))
                {
                    return Err(ToolError::NotAuthorized(format!(
                        "writing to '{}' is not allowed (identity file protected from tool access)",
                        path
                    )));
                }
                path.to_string()
            }
        };

        // When a layer is specified, route through layer-aware methods for ALL targets.
        let layer_result = if let Some(layer_name) = layer {
            let result = if append {
                workspace
                    .append_to_layer(layer_name, &resolved_path, content)
                    .await
                    .map_err(|e| {
                        ToolError::ExecutionFailed(format!("Write failed: {}", e))
                    })?
            } else {
                workspace
                    .write_to_layer(layer_name, &resolved_path, content)
                    .await
                    .map_err(|e| {
                        ToolError::ExecutionFailed(format!("Write failed: {}", e))
                    })?
            };
            Some((result.actual_layer, result.redirected))
        } else {
            // No layer specified -- use default workspace methods
            match target {
                "memory" => {
                    if append {
                        workspace
                            .append_memory(content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    } else {
                        workspace
                            .write(paths::MEMORY, content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    }
                }
                "daily_log" => {
                    workspace
                        .append_daily_log(content)
                        .await
                        .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                }
                "heartbeat" => {
                    if append {
                        workspace
                            .append(paths::HEARTBEAT, content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    } else {
                        workspace
                            .write(paths::HEARTBEAT, content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    }
                }
                _ => {
                    if append {
                        workspace
                            .append(&resolved_path, content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    } else {
                        workspace
                            .write(&resolved_path, content)
                            .await
                            .map_err(|e| ToolError::ExecutionFailed(format!("Write failed: {}", e)))?;
                    }
                }
            }
            None
        };

        let mut output = serde_json::json!({
            "status": "written",
            "path": resolved_path,
            "append": append,
            "content_length": content.len(),
        });
        if let Some((actual_layer, redirected)) = layer_result {
            output["layer"] = serde_json::Value::String(actual_layer);
            output["redirected"] = serde_json::Value::Bool(redirected);
        }

        Ok(ToolOutput::success(output, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal tool
    }

    fn rate_limit_config(&self) -> Option<crate::tools::tool::ToolRateLimitConfig> {
        Some(crate::tools::tool::ToolRateLimitConfig::new(20, 200))
    }
}

/// Tool for reading workspace files.
///
/// Use this to read the full content of any file in the workspace.
pub struct MemoryReadTool {
    resolver: Arc<dyn WorkspaceResolver>,
}

impl MemoryReadTool {
    /// Create a new memory read tool with a resolver.
    pub fn new(resolver: Arc<dyn WorkspaceResolver>) -> Self {
        Self { resolver }
    }

    /// Create from a fixed workspace (backward compatibility).
    pub fn from_workspace(workspace: Arc<Workspace>) -> Self {
        Self {
            resolver: Arc::new(FixedWorkspaceResolver::new(workspace)),
        }
    }
}

#[async_trait]
impl Tool for MemoryReadTool {
    fn name(&self) -> &str {
        "memory_read"
    }

    fn description(&self) -> &str {
        "Read a file from the workspace memory (database-backed storage). \
         Use this to read files shown by memory_tree. NOT for local filesystem files \
         (use read_file for those). Works with identity files, heartbeat checklist, \
         memory, daily logs, or any custom workspace path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (e.g., 'MEMORY.md', 'daily/2024-01-15.md', 'projects/alpha/notes.md')"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let path = require_str(&params, "path")?;

        let workspace = self.resolver.resolve(&ctx.user_id).await?;
        let doc = workspace
            .read(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Read failed: {}", e)))?;

        let output = serde_json::json!({
            "path": doc.path,
            "content": doc.content,
            "word_count": doc.word_count(),
            "updated_at": doc.updated_at.to_rfc3339(),
        });

        Ok(ToolOutput::success(output, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal memory
    }
}

/// Tool for viewing workspace structure as a tree.
///
/// Returns a hierarchical view of files and directories with configurable depth.
pub struct MemoryTreeTool {
    resolver: Arc<dyn WorkspaceResolver>,
}

impl MemoryTreeTool {
    /// Create a new memory tree tool with a resolver.
    pub fn new(resolver: Arc<dyn WorkspaceResolver>) -> Self {
        Self { resolver }
    }

    /// Create from a fixed workspace (backward compatibility).
    pub fn from_workspace(workspace: Arc<Workspace>) -> Self {
        Self {
            resolver: Arc::new(FixedWorkspaceResolver::new(workspace)),
        }
    }

    /// Recursively build tree structure.
    ///
    /// Returns a compact format where directories end with `/` and may have children.
    async fn build_tree(
        workspace: &Workspace,
        path: &str,
        current_depth: usize,
        max_depth: usize,
    ) -> Result<Vec<serde_json::Value>, ToolError> {
        if current_depth > max_depth {
            return Ok(Vec::new());
        }

        let entries = workspace
            .list(path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Tree failed: {}", e)))?;

        let mut result = Vec::new();
        for entry in entries {
            // Directories end with `/`, files don't
            let display_path = if entry.is_directory {
                format!("{}/", entry.name())
            } else {
                entry.name().to_string()
            };

            if entry.is_directory && current_depth < max_depth {
                let children =
                    Box::pin(Self::build_tree(workspace, &entry.path, current_depth + 1, max_depth)).await?;
                if children.is_empty() {
                    result.push(serde_json::Value::String(display_path));
                } else {
                    result.push(serde_json::json!({ display_path: children }));
                }
            } else {
                result.push(serde_json::Value::String(display_path));
            }
        }

        Ok(result)
    }
}

#[async_trait]
impl Tool for MemoryTreeTool {
    fn name(&self) -> &str {
        "memory_tree"
    }

    fn description(&self) -> &str {
        "View the workspace memory structure as a tree (database-backed storage). \
         Use memory_read to read files shown here, NOT read_file. \
         The workspace is separate from the local filesystem."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root path to start from (empty string for workspace root)",
                    "default": ""
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum depth to traverse (1 = immediate children only)",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 10
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");

        let depth = params
            .get("depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .clamp(1, 10) as usize;

        let workspace = self.resolver.resolve(&ctx.user_id).await?;
        let tree = Self::build_tree(&workspace, path, 1, depth).await?;

        // Compact output: just the tree array
        Ok(ToolOutput::success(
            serde_json::Value::Array(tree),
            start.elapsed(),
        ))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal tool
    }
}

#[cfg(all(test, feature = "postgres"))]
mod tests {
    use super::*;

    fn make_test_workspace() -> Arc<Workspace> {
        Arc::new(Workspace::new(
            "test_user",
            deadpool_postgres::Pool::builder(deadpool_postgres::Manager::new(
                tokio_postgres::Config::new(),
                tokio_postgres::NoTls,
            ))
            .build()
            .unwrap(),
        ))
    }

    #[test]
    fn test_memory_search_schema() {
        let workspace = make_test_workspace();
        let tool = MemorySearchTool::from_workspace(workspace);

        assert_eq!(tool.name(), "memory_search");
        assert!(!tool.requires_sanitization());

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["query"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"query".into())
        );
    }

    #[test]
    fn test_memory_write_schema() {
        let workspace = make_test_workspace();
        let tool = MemoryWriteTool::from_workspace(workspace);

        assert_eq!(tool.name(), "memory_write");

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["content"].is_object());
        assert!(schema["properties"]["target"].is_object());
        assert!(schema["properties"]["append"].is_object());
    }

    #[test]
    fn test_memory_read_schema() {
        let workspace = make_test_workspace();
        let tool = MemoryReadTool::from_workspace(workspace);

        assert_eq!(tool.name(), "memory_read");

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"path".into())
        );
    }

    #[test]
    fn test_memory_tree_schema() {
        let workspace = make_test_workspace();
        let tool = MemoryTreeTool::from_workspace(workspace);

        assert_eq!(tool.name(), "memory_tree");

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["path"].is_object());
        assert!(schema["properties"]["depth"].is_object());
        assert_eq!(schema["properties"]["depth"]["default"], 1);
    }
}
