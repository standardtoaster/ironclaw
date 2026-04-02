//! Translation between IronClaw Tool types and MCP protocol types.

use crate::tools::mcp::protocol::{ContentBlock, McpTool};
use crate::tools::tool::{ToolError, ToolOutput, ToolSchema};

/// Convert an IronClaw ToolSchema to an MCP tool definition.
pub fn tool_schema_to_mcp(schema: &ToolSchema) -> McpTool {
    McpTool {
        name: schema.name.clone(),
        description: schema.description.clone(),
        input_schema: schema.parameters.clone(),
        annotations: None,
    }
}

/// Convert a successful ToolOutput to MCP content blocks.
pub fn tool_output_to_content_blocks(output: &ToolOutput) -> Vec<ContentBlock> {
    let text = serde_json::to_string_pretty(&output.result)
        .unwrap_or_else(|_| output.result.to_string());
    vec![ContentBlock::Text { text }]
}

/// Convert a ToolError to MCP content blocks with is_error flag.
pub fn tool_error_to_content_blocks(err: &ToolError) -> (Vec<ContentBlock>, bool) {
    let text = match err {
        ToolError::NotAuthorized(msg) => format!("Not authorized: {msg}"),
        other => other.to_string(),
    };
    (vec![ContentBlock::Text { text }], true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_tool_schema_to_mcp_tool() {
        let schema = ToolSchema {
            name: "memory_search".to_string(),
            description: "Search workspace memory".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" }
                },
                "required": ["query"]
            }),
        };
        let mcp_tool = tool_schema_to_mcp(&schema);
        assert_eq!(mcp_tool.name, "memory_search");
        assert_eq!(mcp_tool.description, "Search workspace memory");
        assert_eq!(mcp_tool.input_schema["type"], "object");
        assert!(mcp_tool.input_schema["properties"]["query"].is_object());
        assert!(mcp_tool.annotations.is_none());
    }

    #[test]
    fn test_tool_output_to_content_blocks_text() {
        let output = ToolOutput::text("hello world", Duration::from_millis(5));
        let blocks = tool_output_to_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text, "\"hello world\""),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_output_to_content_blocks_json() {
        let output = ToolOutput::success(
            serde_json::json!({"count": 3, "items": ["a", "b", "c"]}),
            Duration::from_millis(10),
        );
        let blocks = tool_output_to_content_blocks(&output);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Text { text } => {
                let parsed: serde_json::Value = serde_json::from_str(text)
                    .expect("content should be valid JSON");
                assert_eq!(parsed["count"], 3);
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_error_to_content_blocks() {
        let err = ToolError::ExecutionFailed("connection refused".into());
        let (blocks, is_error) = tool_error_to_content_blocks(&err);
        assert!(is_error);
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Text { text } => assert!(text.contains("connection refused")),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_error_not_authorized() {
        let err = ToolError::NotAuthorized("no access".into());
        let (blocks, is_error) = tool_error_to_content_blocks(&err);
        assert!(is_error);
        match &blocks[0] {
            ContentBlock::Text { text } => assert!(text.contains("Not authorized")),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_error_invalid_parameters() {
        let err = ToolError::InvalidParameters("missing 'query'".into());
        let (blocks, is_error) = tool_error_to_content_blocks(&err);
        assert!(is_error);
        match &blocks[0] {
            ContentBlock::Text { text } => assert!(text.contains("Invalid parameters")),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_tool_schema_empty_parameters() {
        let schema = ToolSchema::new("ping", "Ping the server");
        let mcp_tool = tool_schema_to_mcp(&schema);
        assert_eq!(mcp_tool.name, "ping");
        assert_eq!(mcp_tool.input_schema["type"], "object");
    }
}
