//! MCP method dispatch — handles initialize, ping, tools/list, tools/call.

use crate::context::JobContext;
use crate::tools::mcp::protocol::{
    CallToolResult, ContentBlock, InitializeResult, ListToolsResult, ServerCapabilities, ServerInfo,
    ToolsCapability, PROTOCOL_VERSION,
};
use crate::tools::registry::ToolRegistry;

use crate::channels::mcp::translate::{
    tool_error_to_content_blocks, tool_output_to_content_blocks, tool_schema_to_mcp,
};

/// Handle the `initialize` method.
pub fn handle_initialize() -> serde_json::Value {
    let result = InitializeResult {
        protocol_version: Some(PROTOCOL_VERSION.to_string()),
        capabilities: ServerCapabilities {
            tools: Some(ToolsCapability { list_changed: false }),
            resources: None,
            prompts: None,
            logging: None,
        },
        server_info: Some(ServerInfo {
            name: "ironclaw".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        }),
        instructions: Some(
            "IronClaw MCP server. Provides access to memory, collections, \
             calendar, workspace, and other registered tools."
                .to_string(),
        ),
    };
    serde_json::to_value(result).unwrap_or_default()
}

/// Handle the `ping` method.
pub fn handle_ping() -> serde_json::Value {
    serde_json::json!({})
}

/// Handle `tools/list` -- returns all registered tools as MCP tool definitions.
pub async fn handle_tools_list(registry: &ToolRegistry) -> serde_json::Value {
    let tools = registry.all().await;
    let mcp_tools: Vec<_> = tools.iter().map(|t| tool_schema_to_mcp(&t.schema())).collect();
    let result = ListToolsResult { tools: mcp_tools };
    serde_json::to_value(result).unwrap_or_default()
}

/// Handle `tools/call` -- executes a tool by name with the given arguments.
pub async fn handle_tools_call(
    registry: &ToolRegistry,
    ctx: &JobContext,
    params: &serde_json::Value,
) -> serde_json::Value {
    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::to_value(CallToolResult {
                content: vec![ContentBlock::Text {
                    text: "Missing required 'name' parameter".to_string(),
                }],
                is_error: true,
            })
            .unwrap_or_default();
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    let tool = match registry.get(name).await {
        Some(t) => t,
        None => {
            return serde_json::to_value(CallToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("Unknown tool: {name}"),
                }],
                is_error: true,
            })
            .unwrap_or_default();
        }
    };

    match tool.execute(arguments, ctx).await {
        Ok(output) => {
            let content = tool_output_to_content_blocks(&output);
            serde_json::to_value(CallToolResult {
                content,
                is_error: false,
            })
            .unwrap_or_default()
        }
        Err(err) => {
            let (content, is_error) = tool_error_to_content_blocks(&err);
            serde_json::to_value(CallToolResult { content, is_error }).unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::tools::mcp::protocol::PROTOCOL_VERSION;

    #[test]
    fn test_handle_initialize() {
        let result = handle_initialize();
        let init: InitializeResult = serde_json::from_value(result).unwrap();
        assert_eq!(init.protocol_version.as_deref(), Some(PROTOCOL_VERSION));
        assert!(init.capabilities.tools.is_some());
        let info = init.server_info.unwrap();
        assert_eq!(info.name, "ironclaw");
        assert!(init.instructions.is_some());
    }

    #[test]
    fn test_handle_ping() {
        let result = handle_ping();
        assert_eq!(result, serde_json::json!({}));
    }

    #[tokio::test]
    async fn test_handle_tools_list_empty() {
        let registry = ToolRegistry::new();
        let result = handle_tools_list(&registry).await;
        let list: ListToolsResult = serde_json::from_value(result).unwrap();
        assert!(list.tools.is_empty());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::builtin::EchoTool))
            .await;

        let result = handle_tools_list(&registry).await;
        let list: ListToolsResult = serde_json::from_value(result).unwrap();
        assert_eq!(list.tools.len(), 1);
        assert_eq!(list.tools[0].name, "echo");
        assert!(list.tools[0].input_schema["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn test_handle_tools_call_success() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::builtin::EchoTool))
            .await;
        let ctx = JobContext::default();

        let params = serde_json::json!({
            "name": "echo",
            "arguments": { "message": "hello" }
        });
        let result = handle_tools_call(&registry, &ctx, &params).await;
        let call_result: CallToolResult = serde_json::from_value(result).unwrap();
        assert!(!call_result.is_error);
        assert_eq!(call_result.content.len(), 1);
        assert!(call_result.content[0].as_text().unwrap().contains("hello"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_unknown_tool() {
        let registry = ToolRegistry::new();
        let ctx = JobContext::default();

        let params = serde_json::json!({
            "name": "nonexistent",
            "arguments": {}
        });
        let result = handle_tools_call(&registry, &ctx, &params).await;
        let call_result: CallToolResult = serde_json::from_value(result).unwrap();
        assert!(call_result.is_error);
        assert!(call_result.content[0]
            .as_text()
            .unwrap()
            .contains("nonexistent"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_missing_name() {
        let registry = ToolRegistry::new();
        let ctx = JobContext::default();

        let params = serde_json::json!({ "arguments": {} });
        let result = handle_tools_call(&registry, &ctx, &params).await;
        let call_result: CallToolResult = serde_json::from_value(result).unwrap();
        assert!(call_result.is_error);
        assert!(call_result.content[0]
            .as_text()
            .unwrap()
            .contains("Missing"));
    }

    #[tokio::test]
    async fn test_handle_tools_call_no_arguments_defaults_to_empty() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::builtin::EchoTool))
            .await;
        let ctx = JobContext::default();

        // "echo" requires "message" param, so calling without arguments should fail
        // with a tool error (InvalidParameters), not a dispatch error.
        let params = serde_json::json!({ "name": "echo" });
        let result = handle_tools_call(&registry, &ctx, &params).await;
        let call_result: CallToolResult = serde_json::from_value(result).unwrap();
        assert!(call_result.is_error);
        assert!(call_result.content[0]
            .as_text()
            .unwrap()
            .contains("message"));
    }
}
