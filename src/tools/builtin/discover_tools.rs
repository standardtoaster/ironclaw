//! discover_tools meta-tool.
//!
//! Lets the LLM search for and load registered tools that aren't in
//! the core set. Discovered tools are added to the session and sent
//! to the LLM on subsequent turns.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::tools::registry::ToolRegistry;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

pub struct DiscoverToolsTool {
    registry: Arc<ToolRegistry>,
}

impl DiscoverToolsTool {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for DiscoverToolsTool {
    fn name(&self) -> &str {
        "discover_tools"
    }

    fn description(&self) -> &str {
        "Search for and load available tools by name or capability. \
         Use this when you need a structured tool that isn't in your current set \
         — for example, to manage collections, schedule routines, or use extensions. \
         Check the capability manifest in the system prompt first to see what's available."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you're looking for (e.g., 'grocery', 'schedule', 'install tool')"
                },
                "load": {
                    "type": "boolean",
                    "description": "If true, load matching tools so they're available on the next turn. If false, just list what's available.",
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
        _ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let query = require_str(&params, "query")?;
        let load = params
            .get("load")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let matches = self.registry.search_tools(query).await;

        if matches.is_empty() {
            return Ok(ToolOutput::text(
                format!("No tools found matching \"{query}\"."),
                start.elapsed(),
            ));
        }

        if load {
            for (name, _) in &matches {
                self.registry.mark_discovered(name).await;
            }
        }

        let tool_list: Vec<String> = matches
            .iter()
            .map(|(name, desc)| {
                let short_desc = if desc.len() > 100 {
                    // Find char boundary to avoid slicing mid-character
                    let end = desc
                        .char_indices()
                        .take_while(|(i, _)| *i < 100)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(100);
                    format!("{}...", &desc[..end])
                } else {
                    desc.clone()
                };
                format!("  - {name}: {short_desc}")
            })
            .collect();

        let action = if load { "Loaded" } else { "Found" };
        let suffix = if load {
            " These tools are now available for use."
        } else {
            " Call discover_tools again with load=true to make them available."
        };

        Ok(ToolOutput::text(
            format!(
                "{action} {} tool(s) matching \"{query}\":\n{}\n{suffix}",
                matches.len(),
                tool_list.join("\n"),
            ),
            start.elapsed(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::Tool;

    #[test]
    fn tool_metadata() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        assert_eq!(tool.name(), "discover_tools");
        assert!(!tool.description().is_empty());
        assert!(!tool.requires_sanitization());

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["query"].is_object());
        assert!(schema["properties"]["load"].is_object());
        assert_eq!(schema["required"][0], "query");
    }

    #[tokio::test]
    async fn search_finds_matching_tools() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "echo" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(text.contains("echo"), "should find the echo tool");
        assert!(text.contains("Found"), "should say 'Found' when load=false");
        assert!(
            text.contains("Call discover_tools again with load=true"),
            "should hint about loading"
        );
    }

    #[tokio::test]
    async fn search_with_load_true() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "echo", "load": true });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(text.contains("Loaded"), "should say 'Loaded' when load=true");
        assert!(
            text.contains("now available for use"),
            "should confirm availability"
        );
    }

    #[tokio::test]
    async fn search_no_matches() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "zzz_nonexistent_tool_xyz" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(
            text.contains("No tools found"),
            "should report no matches: {text}"
        );
    }

    #[tokio::test]
    async fn search_is_case_insensitive() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "ECHO" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(
            text.contains("echo"),
            "case-insensitive search should find 'echo'"
        );
    }

    #[tokio::test]
    async fn search_matches_description() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        // "memory" should match tools with "memory" in name or description
        let params = serde_json::json!({ "query": "memory" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(
            text.contains("memory_search") || text.contains("memory_write"),
            "should find memory tools by name/description: {text}"
        );
    }

    #[tokio::test]
    async fn missing_query_returns_error() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({});
        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err(), "missing query should error");
        match result.unwrap_err() {
            ToolError::InvalidParameters(msg) => {
                assert!(msg.contains("query"), "error should mention 'query': {msg}");
            }
            other => panic!("expected InvalidParameters, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_query_matches_all_tools() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        // Empty string is a substring of everything
        let params = serde_json::json!({ "query": "" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        // Should find multiple tools since "" matches all names/descriptions
        assert!(
            text.contains("tool(s) matching"),
            "empty query should match tools: {text}"
        );
    }

    #[tokio::test]
    async fn load_false_is_default() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        // No "load" key at all — should default to false
        let params = serde_json::json!({ "query": "echo" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(
            text.contains("Found"),
            "default load=false should say 'Found': {text}"
        );
    }

    #[tokio::test]
    async fn results_are_sorted_alphabetically() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register_builtin_tools();
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        // "memory" matches multiple tools — verify sorted output
        let params = serde_json::json!({ "query": "memory" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();

        let tool_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.trim_start().starts_with("- "))
            .collect();
        if tool_lines.len() > 1 {
            for pair in tool_lines.windows(2) {
                assert!(
                    pair[0] <= pair[1],
                    "tools should be sorted: {:?} should come before {:?}",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    #[tokio::test]
    async fn long_descriptions_are_truncated() {
        use async_trait::async_trait;

        // Create a tool with a very long description
        struct LongDescTool;
        #[async_trait]
        impl Tool for LongDescTool {
            fn name(&self) -> &str {
                "long_desc_test_tool"
            }
            fn description(&self) -> &str {
                "This is a very long description that exceeds one hundred characters and should be truncated with an ellipsis when displayed in discover_tools output results"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({"type": "object"})
            }
            async fn execute(
                &self,
                _: serde_json::Value,
                _: &JobContext,
            ) -> Result<ToolOutput, ToolError> {
                unreachable!()
            }
        }

        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        registry.register(Arc::new(LongDescTool)).await;
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "long_desc_test" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();

        // Find the tool line and verify truncation
        let tool_line = text
            .lines()
            .find(|l| l.contains("long_desc_test_tool"))
            .expect("should find the tool in output");
        assert!(
            tool_line.contains("..."),
            "long description should be truncated with '...': {tool_line}"
        );
        // The description part (after "tool_name: ") should be ~103 chars max (100 + "...")
        let desc_part = tool_line
            .split(": ")
            .nth(1)
            .expect("should have description after ': '");
        assert!(
            desc_part.len() <= 110,
            "truncated desc should be ~103 chars, got {}: {desc_part}",
            desc_part.len()
        );
    }

    #[tokio::test]
    async fn empty_registry_returns_no_matches() {
        let registry = Arc::new(crate::tools::registry::ToolRegistry::new());
        // Don't register any tools
        let tool = DiscoverToolsTool::new(Arc::clone(&registry));
        let ctx = crate::context::JobContext::new("test-user", "discover test");

        let params = serde_json::json!({ "query": "anything" });
        let output = tool.execute(params, &ctx).await.unwrap();
        let text = output.result.as_str().unwrap();
        assert!(text.contains("No tools found"), "empty registry: {text}");
    }
}
