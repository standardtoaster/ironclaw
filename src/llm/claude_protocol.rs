//! Shared Claude NDJSON protocol types and parsing logic.
//!
//! This module contains the protocol-level types and functions used to
//! communicate with Claude CLI processes via stream-json NDJSON format.
//! Both `ClaudeSidecarProvider` and `ClaudeContainerProvider` share this code.

use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::llm::error::LlmError;
use crate::llm::provider::{ChatMessage, Role, ToolCall};

/// A tool approval request from the Claude CLI process.
///
/// When `skip_permissions` is false, Claude may pause and request approval
/// for tool executions. This struct represents that request, which callers
/// can respond to via the appropriate provider's `approve()` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarApproval {
    /// Unique request ID for this approval.
    pub request_id: String,
    /// Name of the tool requesting approval.
    pub tool_name: String,
    /// Tool parameters.
    pub parameters: serde_json::Value,
}

/// Result of an exchange with a Claude CLI process.
///
/// Either a completed response or a tool approval request that needs
/// user confirmation before the exchange can continue.
#[derive(Debug)]
pub enum ExchangeResult {
    /// The exchange completed with a response.
    Complete {
        content: String,
        tool_calls: Vec<ToolCall>,
        input_tokens: u32,
        output_tokens: u32,
    },
    /// The CLI is waiting for tool approval before continuing.
    NeedApproval(SidecarApproval),
}

/// Messages received from the Claude CLI stdout (stream-json format).
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub enum ClaudeStreamMessage {
    #[serde(rename = "system")]
    System {
        #[serde(default)]
        session_id: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        message: serde_json::Value,
    },
    #[serde(rename = "result")]
    Result {
        result: String,
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default)]
        input_tokens: Option<u32>,
        #[serde(default)]
        output_tokens: Option<u32>,
    },
    /// Tool permission request from Claude (when not using --dangerously-skip-permissions).
    /// The CLI pauses execution and waits for approval on stdin.
    #[serde(rename = "tool_use_permission")]
    ToolUsePermission {
        /// Unique ID for this permission request.
        #[serde(default)]
        id: Option<String>,
        /// Name of the tool requesting permission.
        #[serde(default)]
        tool_name: Option<String>,
        /// Tool input parameters.
        #[serde(default)]
        input: Option<serde_json::Value>,
    },
}

/// Build the text prompt from IronClaw messages.
///
/// The Claude CLI in print mode takes a single prompt string. We concatenate
/// messages with role prefixes to preserve conversation context.
pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                parts.push(format!("[System context]: {}", msg.content));
            }
            Role::User => parts.push(msg.content.clone()),
            Role::Assistant => parts.push(format!("[Previous assistant]: {}", msg.content)),
            Role::Tool => {
                let name = msg.name.as_deref().unwrap_or("tool");
                parts.push(format!("[Tool result from {}]: {}", name, msg.content));
            }
        }
    }
    parts.join("\n\n")
}

/// Extract text content from a Claude assistant message content value.
pub fn extract_text(content: &serde_json::Value) -> String {
    let Some(array) = content.as_array() else {
        return content.as_str().unwrap_or("").to_string();
    };
    array
        .iter()
        .filter_map(|block| {
            if block.get("type")?.as_str()? == "text" {
                block.get("text")?.as_str().map(String::from)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Extract tool calls from a Claude assistant message content array.
pub fn extract_tool_calls(content: &serde_json::Value) -> Vec<ToolCall> {
    let Some(array) = content.as_array() else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|block| {
            if block.get("type")?.as_str()? == "tool_use" {
                Some(ToolCall {
                    id: block.get("id")?.as_str()?.to_string(),
                    name: block.get("name")?.as_str()?.to_string(),
                    arguments: block.get("input")?.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Compute the prompt delta for session-based conversations.
///
/// Given the full message history and the number already sent, returns
/// `(prompt, is_continuation)`:
/// - If `sent >= messages.len()`, falls back to full history (not a continuation).
/// - Otherwise, returns only the new (unsent) messages as a prompt.
pub fn resolve_delta(messages: &[ChatMessage], sent: usize) -> (String, bool) {
    if sent >= messages.len() {
        // No new messages — fall back to full history.
        (messages_to_prompt(messages), false)
    } else {
        let delta = &messages[sent..];
        (messages_to_prompt(delta), true)
    }
}

/// Read NDJSON lines from an async reader until a result or approval request.
///
/// This is the generic protocol reader used by both sidecar and container providers.
/// The `on_system` callback is invoked when a `system` message with a `session_id`
/// is received, allowing the caller to capture it.
///
/// Returns an `ExchangeResult` on success.
pub async fn read_exchange<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
    mut on_system: impl FnMut(String),
) -> Result<ExchangeResult, LlmError> {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();
    let mut buf = String::new();

    loop {
        buf.clear();
        let bytes_read = reader
            .read_line(&mut buf)
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: "claude_protocol".to_string(),
                reason: format!("Failed to read stdout: {}", e),
            })?;

        if bytes_read == 0 {
            return Err(LlmError::RequestFailed {
                provider: "claude_protocol".to_string(),
                reason: "Process exited unexpectedly".to_string(),
            });
        }

        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: ClaudeStreamMessage = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!("Ignoring unparseable output: {} ({})", trimmed, e);
                continue;
            }
        };

        match msg {
            ClaudeStreamMessage::System { session_id } => {
                if let Some(sid) = session_id {
                    on_system(sid);
                }
            }
            ClaudeStreamMessage::Assistant { message } => {
                if let Some(content) = message.get("content") {
                    let text = extract_text(content);
                    if !text.is_empty() {
                        text_content.push_str(&text);
                    }
                    let calls = extract_tool_calls(content);
                    tool_calls.extend(calls);
                }
            }
            ClaudeStreamMessage::Result {
                result,
                input_tokens: it,
                output_tokens: ot,
                ..
            } => {
                if text_content.is_empty() {
                    text_content = result;
                }
                return Ok(ExchangeResult::Complete {
                    content: text_content,
                    tool_calls,
                    input_tokens: it.unwrap_or(0),
                    output_tokens: ot.unwrap_or(0),
                });
            }
            ClaudeStreamMessage::ToolUsePermission {
                id,
                tool_name,
                input,
            } => {
                let request_id =
                    id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                let tool = tool_name.unwrap_or_else(|| "unknown".to_string());
                let params = input.unwrap_or(serde_json::Value::Null);
                tracing::info!(
                    tool = %tool,
                    request_id = %request_id,
                    "Tool permission requested"
                );
                return Ok(ExchangeResult::NeedApproval(SidecarApproval {
                    request_id,
                    tool_name: tool,
                    parameters: params,
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- messages_to_prompt tests ---

    #[test]
    fn test_messages_to_prompt_user_only() {
        let messages = vec![ChatMessage::user("Hello")];
        let prompt = messages_to_prompt(&messages);
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn test_messages_to_prompt_with_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[System context]: You are helpful"));
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_messages_to_prompt_with_assistant() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("Thanks"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[Previous assistant]: Hi there"));
        assert!(prompt.contains("Thanks"));
    }

    #[test]
    fn test_messages_to_prompt_with_tool_result() {
        let messages = vec![
            ChatMessage::user("Search for X"),
            ChatMessage::tool_result("call_1", "search", "found 3 results"),
        ];
        let prompt = messages_to_prompt(&messages);
        assert!(prompt.contains("[Tool result from search]: found 3 results"));
    }

    #[test]
    fn test_messages_to_prompt_empty() {
        let messages: Vec<ChatMessage> = vec![];
        let prompt = messages_to_prompt(&messages);
        assert_eq!(prompt, "");
    }

    // --- resolve_delta tests ---

    #[test]
    fn test_resolve_delta_no_sent_messages() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
            ChatMessage::user("Follow-up"),
        ];
        // With sent=0, all messages are "new" — this is technically a delta.
        // The caller (resolve_prompt) handles the new-session case separately.
        let (prompt, is_continuation) = resolve_delta(&messages, 0);
        assert!(is_continuation);
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("Follow-up"));
    }

    #[test]
    fn test_resolve_delta_some_sent() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
            ChatMessage::user("New question"),
        ];
        let (prompt, is_continuation) = resolve_delta(&messages, 2);
        assert!(is_continuation);
        assert!(prompt.contains("New question"));
        assert!(!prompt.contains("Hello"));
    }

    #[test]
    fn test_resolve_delta_all_sent() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi"),
        ];
        let (prompt, is_continuation) = resolve_delta(&messages, 2);
        assert!(!is_continuation);
        // Falls back to full history.
        assert!(prompt.contains("Hello"));
    }

    #[test]
    fn test_resolve_delta_more_sent_than_available() {
        let messages = vec![ChatMessage::user("Hello")];
        let (prompt, is_continuation) = resolve_delta(&messages, 5);
        assert!(!is_continuation);
        assert!(prompt.contains("Hello"));
    }

    // --- extract_text tests ---

    #[test]
    fn test_extract_text_plain_string() {
        let val = serde_json::json!("plain text");
        assert_eq!(extract_text(&val), "plain text");
    }

    #[test]
    fn test_extract_text_empty_array() {
        let val = serde_json::json!([]);
        assert_eq!(extract_text(&val), "");
    }

    #[test]
    fn test_extract_text_content_blocks() {
        let val = serde_json::json!([
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"}
        ]);
        assert_eq!(extract_text(&val), "Hello world");
    }

    #[test]
    fn test_extract_text_skips_non_text_blocks() {
        let val = serde_json::json!([
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "1", "name": "test", "input": {}}
        ]);
        assert_eq!(extract_text(&val), "Hello");
    }

    // --- extract_tool_calls tests ---

    #[test]
    fn test_extract_tool_calls_no_tools() {
        let val = serde_json::json!([{"type": "text", "text": "hello"}]);
        let calls = extract_tool_calls(&val);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_extract_tool_calls_with_tool() {
        let val = serde_json::json!([
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}}
        ]);
        let calls = extract_tool_calls(&val);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "tu_1");
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_extract_tool_calls_not_array() {
        let val = serde_json::json!("plain string");
        let calls = extract_tool_calls(&val);
        assert!(calls.is_empty());
    }

    // --- ClaudeStreamMessage parsing tests ---

    #[test]
    fn test_parse_result_message() {
        let line = r#"{"type":"result","result":"The answer is 4","session_id":"abc","cost_usd":0.003,"duration_ms":1200,"input_tokens":50,"output_tokens":10}"#;
        let msg: ClaudeStreamMessage = serde_json::from_str(line).expect("parse result");
        match msg {
            ClaudeStreamMessage::Result {
                result,
                input_tokens,
                output_tokens,
                ..
            } => {
                assert_eq!(result, "The answer is 4");
                assert_eq!(input_tokens.unwrap_or(0), 50);
                assert_eq!(output_tokens.unwrap_or(0), 10);
            }
            other => panic!("Expected Result, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_system_init_message() {
        let line = r#"{"type":"system","subtype":"init","session_id":"sess-123"}"#;
        let msg: ClaudeStreamMessage = serde_json::from_str(line).expect("parse system");
        match msg {
            ClaudeStreamMessage::System { session_id } => {
                assert_eq!(session_id, Some("sess-123".to_string()));
            }
            other => panic!("Expected System, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_assistant_text_message() {
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]}}"#;
        let msg: ClaudeStreamMessage = serde_json::from_str(line).expect("parse assistant");
        match msg {
            ClaudeStreamMessage::Assistant { message } => {
                let content = message.get("content").expect("content field");
                let text = extract_text(content);
                assert_eq!(text, "Hello world");
            }
            other => panic!("Expected Assistant, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_tool_use_permission_message() {
        let line = r#"{"type":"tool_use_permission","id":"perm-1","tool_name":"shell","input":{"command":"ls"}}"#;
        let msg: ClaudeStreamMessage = serde_json::from_str(line).expect("parse permission");
        match msg {
            ClaudeStreamMessage::ToolUsePermission {
                id,
                tool_name,
                input,
            } => {
                assert_eq!(id, Some("perm-1".to_string()));
                assert_eq!(tool_name, Some("shell".to_string()));
                assert_eq!(input, Some(serde_json::json!({"command": "ls"})));
            }
            other => panic!("Expected ToolUsePermission, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_tool_use_permission_minimal() {
        let line = r#"{"type":"tool_use_permission"}"#;
        let msg: ClaudeStreamMessage = serde_json::from_str(line).expect("parse minimal");
        match msg {
            ClaudeStreamMessage::ToolUsePermission {
                id,
                tool_name,
                input,
            } => {
                assert!(id.is_none());
                assert!(tool_name.is_none());
                assert!(input.is_none());
            }
            other => panic!("Expected ToolUsePermission, got: {:?}", other),
        }
    }

    // --- read_exchange tests ---

    #[tokio::test]
    async fn test_read_exchange_result() {
        let data = r#"{"type":"system","session_id":"s1"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello"}]}}
{"type":"result","result":"Hello","input_tokens":10,"output_tokens":5}
"#;
        let mut reader = tokio::io::BufReader::new(data.as_bytes());
        let mut captured_session = None;
        let result = read_exchange(&mut reader, |sid| {
            captured_session = Some(sid);
        })
        .await
        .expect("should parse");

        assert_eq!(captured_session, Some("s1".to_string()));
        match result {
            ExchangeResult::Complete {
                content,
                tool_calls,
                input_tokens,
                output_tokens,
            } => {
                assert_eq!(content, "Hello");
                assert!(tool_calls.is_empty());
                assert_eq!(input_tokens, 10);
                assert_eq!(output_tokens, 5);
            }
            _ => panic!("Expected Complete"),
        }
    }

    #[tokio::test]
    async fn test_read_exchange_approval() {
        let data = r#"{"type":"tool_use_permission","id":"perm-1","tool_name":"shell","input":{"cmd":"ls"}}
"#;
        let mut reader = tokio::io::BufReader::new(data.as_bytes());
        let result = read_exchange(&mut reader, |_| {}).await.expect("should parse");

        match result {
            ExchangeResult::NeedApproval(approval) => {
                assert_eq!(approval.request_id, "perm-1");
                assert_eq!(approval.tool_name, "shell");
            }
            _ => panic!("Expected NeedApproval"),
        }
    }

    #[tokio::test]
    async fn test_read_exchange_eof() {
        let data = b"";
        let mut reader = tokio::io::BufReader::new(&data[..]);
        let result = read_exchange(&mut reader, |_| {}).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_exchange_skips_unparseable() {
        let data = r#"not json at all
{"type":"result","result":"ok","input_tokens":1,"output_tokens":1}
"#;
        let mut reader = tokio::io::BufReader::new(data.as_bytes());
        let result = read_exchange(&mut reader, |_| {}).await.expect("should parse");
        match result {
            ExchangeResult::Complete { content, .. } => {
                assert_eq!(content, "ok");
            }
            _ => panic!("Expected Complete"),
        }
    }
}
