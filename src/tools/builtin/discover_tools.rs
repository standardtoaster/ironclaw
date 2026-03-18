//! discover_tools — LLM-callable tool for discovering available tools.
//!
//! Phase 1: searches already-registered tools in the ToolRegistry.
//! Phase 2: searches the MCP service registry for external services,
//!          connects on demand, and registers tools for the session.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::tools::mcp::service_cache::{CachedTool, ServiceCache};
use crate::tools::mcp::service_registry::{ServiceAuth, ServiceConfig, ServiceRegistry};
use crate::tools::mcp::{McpClient, McpServerConfig};
use crate::tools::mcp::config::McpTransportConfig;
use crate::tools::registry::ToolRegistry;
use crate::tools::tool::{Tool, ToolError, ToolOutput};

/// Built-in tool that lets the LLM discover and load tools on demand.
pub struct DiscoverToolsTool {
    registry: Arc<ToolRegistry>,
    service_registry: Option<Arc<ServiceRegistry>>,
    service_cache: Option<Arc<ServiceCache>>,
}

impl DiscoverToolsTool {
    /// Create a new discover_tools tool with only local registry search (Phase 1).
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            service_registry: None,
            service_cache: None,
        }
    }

    /// Create a new discover_tools tool with both local and MCP service search.
    pub fn with_services(
        registry: Arc<ToolRegistry>,
        service_registry: Option<Arc<ServiceRegistry>>,
        service_cache: Option<Arc<ServiceCache>>,
    ) -> Self {
        Self {
            registry,
            service_registry,
            service_cache,
        }
    }

    /// Build an McpClient from a ServiceConfig, injecting credentials as headers.
    fn build_mcp_client_for_service(service: &ServiceConfig) -> McpClient {
        let mut headers = HashMap::new();
        match &service.auth {
            ServiceAuth::Bearer { credential } => {
                headers.insert(
                    "Authorization".to_string(),
                    format!("Bearer {}", credential),
                );
            }
            ServiceAuth::Header { key, credential } => {
                headers.insert(key.clone(), credential.clone());
            }
        }

        let mcp_config = McpServerConfig {
            name: service.name.clone(),
            url: service.url.clone(),
            transport: Some(McpTransportConfig::Http),
            headers,
            oauth: None,
            enabled: true,
            description: Some(service.description.clone()),
        };

        McpClient::new_with_config(mcp_config)
    }

    /// Fetch tools from an MCP service, using cache when available.
    async fn fetch_and_cache_service_tools(
        &self,
        service: &ServiceConfig,
        user_id: &str,
        refresh: bool,
    ) -> Result<Vec<CachedTool>, ToolError> {
        let cache = match &self.service_cache {
            Some(c) => c,
            None => return Ok(vec![]),
        };

        // Check cache first (unless refresh requested)
        if !refresh
            && let Some(cached) = cache.get(&service.name)
        {
            return Ok(cached
                .into_iter()
                .filter(|t| service.is_tool_allowed(&t.name, user_id))
                .collect());
        }

        // Connect to MCP server and fetch all tools
        let client = Self::build_mcp_client_for_service(service);
        let mcp_tools = client.list_tools().await?;

        // Cache ALL tools (unfiltered) — filtering is per-user at query time
        let all_cached: Vec<CachedTool> = mcp_tools
            .into_iter()
            .map(|t| CachedTool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
            })
            .collect();

        cache.put(&service.name, &all_cached, 86400); // 24h TTL

        // Return filtered for this user
        Ok(all_cached
            .into_iter()
            .filter(|t| service.is_tool_allowed(&t.name, user_id))
            .collect())
    }
}

#[async_trait]
impl Tool for DiscoverToolsTool {
    fn name(&self) -> &str {
        "discover_tools"
    }

    fn description(&self) -> &str {
        "Search for available tools by keyword. Use `load: true` to register tools for use. \
         Searches both locally-registered tools and external MCP services."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords to match against tool names, descriptions, and service metadata"
                },
                "load": {
                    "type": "boolean",
                    "description": "If true, register matching tools so they can be called. If false (default), just list what's available.",
                    "default": false
                },
                "refresh": {
                    "type": "boolean",
                    "description": "If true, refresh cached tool definitions from MCP servers",
                    "default": false
                }
            },
            "required": ["query"]
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

        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameters("Missing 'query' parameter".to_string()))?;
        let load = params
            .get("load")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let refresh = params
            .get("refresh")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let query_lower = query.to_lowercase();
        let mut results: Vec<(String, String)> = vec![];

        // Phase 1: Search already-registered tools
        let all_tools = self.registry.all().await;
        for tool in &all_tools {
            let name = tool.name().to_string();
            let desc = tool.description().to_string();
            if name.to_lowercase().contains(&query_lower)
                || desc.to_lowercase().contains(&query_lower)
            {
                results.push((name, desc));
            }
        }

        // Phase 2: Search service registry for MCP services
        let mut service_results: Vec<(String, String)> = vec![];
        if let Some(svc_registry) = &self.service_registry {
            let user_id = &ctx.user_id;
            let matching_services = svc_registry.search(query, user_id);

            for service in matching_services {
                // Fetch tool definitions (from cache or MCP server)
                let cached_tools = match self
                    .fetch_and_cache_service_tools(service, user_id, refresh)
                    .await
                {
                    Ok(tools) => tools,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to fetch tools from service '{}': {}. Trying cache fallback.",
                            service.name,
                            e
                        );
                        // Fallback to stale cache on error
                        self.service_cache
                            .as_ref()
                            .and_then(|c| c.get_stale(&service.name))
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|t| service.is_tool_allowed(&t.name, user_id))
                            .collect()
                    }
                };

                if cached_tools.is_empty() {
                    continue;
                }

                if load {
                    // Create a persistent McpClient for this service
                    let client = Self::build_mcp_client_for_service(service);

                    match client.create_tools().await {
                        Ok(tools) => {
                            for tool in tools {
                                let tool_name = tool.name().to_string();
                                // Check allow/deny filter (strip service prefix for matching)
                                let unprefixed = tool_name
                                    .strip_prefix(&format!("{}_", service.name))
                                    .unwrap_or(&tool_name);
                                if !service.is_tool_allowed(unprefixed, user_id) {
                                    continue;
                                }
                                let desc = tool.description().to_string();
                                self.registry.register(Arc::clone(&tool)).await;
                                service_results.push((tool_name, desc));
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to create tools for service '{}': {}",
                                service.name,
                                e
                            );
                        }
                    }
                } else {
                    // Search-only: report what's available without registering
                    for t in &cached_tools {
                        let prefixed = format!("{}_{}", service.name, t.name);
                        service_results.push((prefixed, t.description.clone()));
                    }
                }
            }
        }

        results.extend(service_results);

        // Format output
        if results.is_empty() {
            return Ok(ToolOutput::text(
                format!("No tools found matching '{}'.", query),
                start.elapsed(),
            ));
        }

        let mut output = format!("Found {} tool(s) matching '{}':\n\n", results.len(), query);
        for (name, desc) in &results {
            output.push_str(&format!("- **{}**: {}\n", name, desc));
        }

        if !load && self.service_registry.is_some() {
            output.push_str(
                "\nUse `load: true` to register external service tools for this session.",
            );
        }

        Ok(ToolOutput::text(output, start.elapsed()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_includes_refresh_param() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        let schema = tool.parameters_schema();
        let props = schema.get("properties").expect("should have properties");
        assert!(props.get("refresh").is_some());
        assert!(props.get("query").is_some());
        assert!(props.get("load").is_some());
    }

    #[test]
    fn test_schema_requires_query() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        let schema = tool.parameters_schema();
        let required = schema
            .get("required")
            .and_then(|v| v.as_array())
            .expect("should have required");
        assert!(required
            .iter()
            .any(|v| v.as_str() == Some("query")));
    }

    #[test]
    fn test_tool_name_and_description() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        assert_eq!(tool.name(), "discover_tools");
        assert!(!tool.description().is_empty());
    }

    #[tokio::test]
    async fn test_phase1_search_finds_registered_tools() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));

        let ctx = crate::context::JobContext::with_user("test_user", "test", "test");
        let params = serde_json::json!({"query": "echo"});
        let result = tool.execute(params, &ctx).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.result.as_str().unwrap_or("").contains("echo"), "should find echo tool");
    }

    #[tokio::test]
    async fn test_phase1_no_results() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));

        let ctx = crate::context::JobContext::with_user("test_user", "test", "test");
        let params = serde_json::json!({"query": "nonexistent_xyz"});
        let result = tool.execute(params, &ctx).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.result.as_str().unwrap_or("").contains("No tools found"));
    }
}
