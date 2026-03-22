//! Conversation load tool for reading past conversation threads.
//!
//! Allows the agent to load the full message history of a conversation
//! after `memory_search` returns a conversation hit. This provides
//! complete context for answering questions about past discussions.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

/// Tool for loading messages from a past conversation thread.
///
/// Use this after `memory_search` returns a result with a `conversation_id`
/// to read the full discussion in chronological order.
pub struct ConversationLoadTool {
    db: Arc<dyn Database>,
}

impl ConversationLoadTool {
    /// Create a new conversation load tool.
    pub fn new(db: Arc<dyn Database>) -> Self {
        Self { db }
    }
}

#[async_trait]
impl Tool for ConversationLoadTool {
    fn name(&self) -> &str {
        "conversation_load"
    }

    fn description(&self) -> &str {
        "Load messages from a past conversation thread. Use this after memory_search \
         returns a conversation hit to read the full discussion. Returns messages \
         in chronological order."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "UUID of the conversation to load (from memory_search results)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 50, max: 200)",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200
                }
            },
            "required": ["conversation_id"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let conversation_id_str = require_str(&params, "conversation_id")?;
        let conversation_id: uuid::Uuid = conversation_id_str.parse().map_err(|_| {
            ToolError::InvalidParameters(format!(
                "invalid conversation_id '{}': expected a UUID",
                conversation_id_str
            ))
        })?;

        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(50)
            .min(200) as usize;

        // Security: verify this conversation belongs to the requesting user
        let belongs = self
            .db
            .conversation_belongs_to_user(conversation_id, &ctx.user_id)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to verify conversation access: {}", e))
            })?;

        if !belongs {
            return Err(ToolError::NotAuthorized(
                "conversation not found or access denied".to_string(),
            ));
        }

        // Load messages
        let messages = self
            .db
            .list_conversation_messages(conversation_id)
            .await
            .map_err(|e| {
                ToolError::ExecutionFailed(format!("Failed to load conversation: {}", e))
            })?;

        // Apply limit (list_conversation_messages returns all; truncate here)
        let messages: Vec<_> = messages.into_iter().take(limit).collect();
        let message_count = messages.len();

        let output = serde_json::json!({
            "conversation_id": conversation_id.to_string(),
            "message_count": message_count,
            "messages": messages.into_iter().map(|m| serde_json::json!({
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.to_rfc3339(),
            })).collect::<Vec<_>>(),
        });

        Ok(ToolOutput::success(output, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false // Internal conversation data, trusted content
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "libsql")]
    async fn setup_db() -> Arc<dyn Database> {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test_conv.db");
        let backend = crate::db::libsql::LibSqlBackend::new_local(&db_path)
            .await
            .expect("new_local");
        backend.run_migrations().await.expect("migrations");
        // Leak the tempdir so it outlives the test (cleaned up by OS).
        // In-memory DBs don't share state across connections in libsql.
        std::mem::forget(dir);
        Arc::new(backend)
    }

    #[cfg(feature = "libsql")]
    #[tokio::test]
    async fn test_conversation_load_schema() {
        let db = setup_db().await;
        let tool = ConversationLoadTool::new(db);

        assert_eq!(tool.name(), "conversation_load");
        assert!(!tool.requires_sanitization());

        let schema = tool.parameters_schema();
        assert!(schema["properties"]["conversation_id"].is_object());
        assert!(schema["properties"]["limit"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&"conversation_id".into())
        );
    }

    #[cfg(feature = "libsql")]
    #[tokio::test]
    async fn test_conversation_load_rejects_unknown_conversation() {
        let db = setup_db().await;
        let tool = ConversationLoadTool::new(db);

        let ctx = crate::context::JobContext::new("test", "test");
        let params = serde_json::json!({
            "conversation_id": "00000000-0000-0000-0000-000000000001"
        });

        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::NotAuthorized(_) => {} // Expected
            other => panic!("Expected NotAuthorized, got: {:?}", other),
        }
    }

    #[cfg(feature = "libsql")]
    #[tokio::test]
    async fn test_conversation_load_returns_messages() {
        let db = setup_db().await;

        // Create a conversation and add messages
        let conv_id = db
            .create_conversation("test", "test_user", None)
            .await
            .expect("create_conversation");
        db.add_conversation_message(conv_id, "user", "hello world")
            .await
            .expect("add_message");
        db.add_conversation_message(conv_id, "assistant", "hi there")
            .await
            .expect("add_message");

        let tool = ConversationLoadTool::new(Arc::clone(&db));
        let mut ctx = crate::context::JobContext::new("test", "test");
        ctx.user_id = "test_user".to_string();

        let params = serde_json::json!({
            "conversation_id": conv_id.to_string()
        });

        let result = tool.execute(params, &ctx).await.expect("execute");
        let output = &result.result;
        assert_eq!(output["message_count"], 2);
        assert_eq!(output["messages"][0]["role"], "user");
        assert_eq!(output["messages"][0]["content"], "hello world");
        assert_eq!(output["messages"][1]["role"], "assistant");
    }

    #[cfg(feature = "libsql")]
    #[tokio::test]
    async fn test_conversation_load_blocks_cross_user_access() {
        let db = setup_db().await;

        // Create conversation owned by "alice"
        let conv_id = db
            .create_conversation("test", "alice", None)
            .await
            .expect("create_conversation");
        db.add_conversation_message(conv_id, "user", "secret stuff")
            .await
            .expect("add_message");

        let tool = ConversationLoadTool::new(Arc::clone(&db));

        // Try to load as "bob"
        let mut ctx = crate::context::JobContext::new("test", "test");
        ctx.user_id = "bob".to_string();

        let params = serde_json::json!({
            "conversation_id": conv_id.to_string()
        });

        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolError::NotAuthorized(_) => {} // Expected: bob cannot see alice's conversation
            other => panic!("Expected NotAuthorized, got: {:?}", other),
        }
    }
}
