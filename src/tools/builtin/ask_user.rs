//! Ask-user tool: pauses the agent loop to get structured user input.

use std::time::Instant;

use async_trait::async_trait;
use serde_json::json;

use crate::context::JobContext;
use crate::tools::tool::{ApprovalRequirement, Tool, ToolError, ToolOutput, ToolSignal};

/// Tool that asks the user a question and waits for their response.
///
/// Emits a `UserInputNeeded` signal that the dispatcher uses to pause the
/// agent loop. The user's response is injected as the tool result when they
/// reply via the web UI or chat channel.
pub struct AskUserTool;

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the user a question and wait for their response. Use when you need \
         user input to proceed — confirmation, preference, choice between options, \
         or freeform input."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Named choices for the user. Omit for freeform text input."
                },
                "metadata": {
                    "type": "object",
                    "description": "Arbitrary data for client renderers (e.g. cost, model name, turn count)",
                    "properties": {}
                }
            },
            "required": ["question"]
        })
    }

    fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
        // ask_user has its own pause mechanism — doesn't use approval flow
        ApprovalRequirement::Never
    }

    fn requires_sanitization(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();

        let question = params
            .get("question")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameters("question is required".into()))?;

        let options: Option<Vec<String>> = params
            .get("options")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let metadata = params.get("metadata").cloned();

        // Return a placeholder result with a signal for the dispatcher.
        // The dispatcher intercepts the signal before this result reaches the LLM.
        // When the user responds, their answer replaces this placeholder.
        Ok(
            ToolOutput::text("Waiting for user response...", start.elapsed()).with_signal(
                ToolSignal::UserInputNeeded {
                    question: question.to_string(),
                    options,
                    metadata,
                },
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ask_user_parameters_schema() {
        let tool = AskUserTool;
        let schema = tool.parameters_schema();
        let props = schema.get("properties").unwrap();
        assert!(props.get("question").is_some());
        assert!(props.get("options").is_some());
        assert!(props.get("metadata").is_some());

        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.contains(&serde_json::json!("question")));
        assert!(!required.contains(&serde_json::json!("options")));
    }

    #[test]
    fn test_ask_user_requires_no_approval() {
        let tool = AskUserTool;
        let params = serde_json::json!({"question": "test?"});
        assert!(matches!(
            tool.requires_approval(&params),
            ApprovalRequirement::Never
        ));
    }

    #[test]
    fn test_ask_user_tool_name() {
        assert_eq!(AskUserTool.name(), "ask_user");
    }

    #[tokio::test]
    async fn test_ask_user_execute_returns_signal() {
        let tool = AskUserTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({
            "question": "Pick a color",
            "options": ["Red", "Blue", "Green"]
        });

        let output = tool.execute(params, &ctx).await.unwrap();
        assert!(output.signal.is_some());
        match output.signal.unwrap() {
            ToolSignal::UserInputNeeded {
                question, options, ..
            } => {
                assert_eq!(question, "Pick a color");
                assert_eq!(
                    options.unwrap(),
                    vec!["Red".to_string(), "Blue".to_string(), "Green".to_string()]
                );
            }
            _ => panic!("Expected UserInputNeeded signal"),
        }
    }

    #[tokio::test]
    async fn test_ask_user_execute_without_options() {
        let tool = AskUserTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({"question": "What is your name?"});

        let output = tool.execute(params, &ctx).await.unwrap();
        assert!(output.signal.is_some());
        match output.signal.unwrap() {
            ToolSignal::UserInputNeeded {
                question, options, ..
            } => {
                assert_eq!(question, "What is your name?");
                assert!(options.is_none());
            }
            _ => panic!("Expected UserInputNeeded signal"),
        }
    }

    #[tokio::test]
    async fn test_ask_user_missing_question() {
        let tool = AskUserTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({"options": ["A", "B"]});

        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err());
    }
}
