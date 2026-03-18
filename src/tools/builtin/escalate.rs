//! Escalation tools: swap the thread's LLM provider to a higher or lower tier.

use std::time::Instant;

use async_trait::async_trait;
use serde_json::json;

use crate::context::JobContext;
use crate::tools::tool::{ApprovalRequirement, Tool, ToolError, ToolOutput, ToolSignal};

/// Tool that escalates the conversation to a more capable model.
///
/// Emits an `Escalate` signal that the dispatcher uses to swap the thread's
/// active provider and replay the current turn on the new model.
pub struct EscalateTool;

#[async_trait]
impl Tool for EscalateTool {
    fn name(&self) -> &str {
        "escalate"
    }

    fn description(&self) -> &str {
        "Hand this conversation to a more capable model. Call when the request requires \
         multi-step reasoning, complex arithmetic, vision, or anything you're uncertain \
         about. A wrong answer is always worse than escalating."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you're escalating (used for routing and logging)"
                },
                "tier": {
                    "type": "string",
                    "description": "Target tier name. If omitted, escalates to next tier up."
                }
            },
            "required": ["reason"]
        })
    }

    fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
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

        let reason = params
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameters("reason is required".into()))?;

        let tier = params.get("tier").and_then(|v| v.as_str());

        Ok(
            ToolOutput::text(format!("Escalating: {}", reason), start.elapsed()).with_signal(
                ToolSignal::Escalate {
                    reason: reason.to_string(),
                    tier: tier.map(String::from),
                },
            ),
        )
    }
}

/// Tool that de-escalates the conversation back to the default model.
///
/// Emits a `DeEscalate` signal that the dispatcher uses to revert the
/// thread's active provider to the configured default.
pub struct DeescalateTool;

#[async_trait]
impl Tool for DeescalateTool {
    fn name(&self) -> &str {
        "de_escalate"
    }

    fn description(&self) -> &str {
        "Revert to the default local model. Call when the complex portion of the task \
         is complete and subsequent messages can be handled by the efficient local model."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why de-escalating (for logging and UI display)"
                }
            }
        })
    }

    fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
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

        let reason = params
            .get("reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        let msg = match &reason {
            Some(r) => format!("De-escalating: {}", r),
            None => "De-escalating to default model".to_string(),
        };

        Ok(ToolOutput::text(msg, start.elapsed()).with_signal(ToolSignal::DeEscalate { reason }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escalate_parameters_schema() {
        let tool = EscalateTool;
        let schema = tool.parameters_schema();
        let props = schema.get("properties").unwrap();
        assert!(props.get("reason").is_some());
        assert!(props.get("tier").is_some());

        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.contains(&serde_json::json!("reason")));
    }

    #[test]
    fn test_deescalate_parameters_schema() {
        let tool = DeescalateTool;
        let schema = tool.parameters_schema();
        let props = schema.get("properties").unwrap();
        assert!(props.get("reason").is_some());

        // reason is optional for de_escalate
        let required = schema
            .get("required")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(!required.contains(&serde_json::json!("reason")));
    }

    #[test]
    fn test_tool_names() {
        assert_eq!(EscalateTool.name(), "escalate");
        assert_eq!(DeescalateTool.name(), "de_escalate");
    }

    #[test]
    fn test_escalate_requires_no_approval() {
        assert!(matches!(
            EscalateTool.requires_approval(&serde_json::json!({})),
            ApprovalRequirement::Never
        ));
        assert!(matches!(
            DeescalateTool.requires_approval(&serde_json::json!({})),
            ApprovalRequirement::Never
        ));
    }

    #[tokio::test]
    async fn test_escalate_execute_returns_signal() {
        let tool = EscalateTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({
            "reason": "complex math",
            "tier": "claude"
        });

        let output = tool.execute(params, &ctx).await.unwrap();
        assert!(output.signal.is_some());
        match output.signal.unwrap() {
            ToolSignal::Escalate { reason, tier } => {
                assert_eq!(reason, "complex math");
                assert_eq!(tier.as_deref(), Some("claude"));
            }
            _ => panic!("Expected Escalate signal"),
        }
    }

    #[tokio::test]
    async fn test_escalate_execute_no_tier() {
        let tool = EscalateTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({"reason": "uncertain"});

        let output = tool.execute(params, &ctx).await.unwrap();
        match output.signal.unwrap() {
            ToolSignal::Escalate { tier, .. } => {
                assert!(tier.is_none());
            }
            _ => panic!("Expected Escalate signal"),
        }
    }

    #[tokio::test]
    async fn test_escalate_missing_reason() {
        let tool = EscalateTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({"tier": "claude"});

        let result = tool.execute(params, &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deescalate_execute_returns_signal() {
        let tool = DeescalateTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({"reason": "task complete"});

        let output = tool.execute(params, &ctx).await.unwrap();
        assert!(output.signal.is_some());
        match output.signal.unwrap() {
            ToolSignal::DeEscalate { reason } => {
                assert_eq!(reason.as_deref(), Some("task complete"));
            }
            _ => panic!("Expected DeEscalate signal"),
        }
    }

    #[tokio::test]
    async fn test_deescalate_execute_no_reason() {
        let tool = DeescalateTool;
        let ctx = JobContext::default();
        let params = serde_json::json!({});

        let output = tool.execute(params, &ctx).await.unwrap();
        match output.signal.unwrap() {
            ToolSignal::DeEscalate { reason } => {
                assert!(reason.is_none());
            }
            _ => panic!("Expected DeEscalate signal"),
        }
    }
}
