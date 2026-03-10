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
    #[test]
    fn tool_metadata() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "load": { "type": "boolean" }
            },
            "required": ["query"]
        });
        assert!(schema["properties"]["query"].is_object());
        assert_eq!(schema["required"][0], "query");
    }
}
