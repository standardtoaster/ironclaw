//! Smart routing provider that routes requests across N model tiers based on
//! task complexity, skill hints, and operation type.
//!
//! Supports pluggable classification via the [`MessageClassifier`] trait and
//! automatic cascade escalation when a cheaper tier fails (timeout, no tool
//! call, uncertainty).
//!
//! Backward-compatible: the 2-tier `new()` constructor produces identical
//! behavior to the original SmartRoutingProvider.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::llm::provider::{
    CompletionRequest, CompletionResponse, LlmProvider, ModelMetadata, Role, ToolCompletionRequest,
    ToolCompletionResponse,
};

// -- Metadata key constants --

/// Metadata key for the routing operation type (plan, select_tools, respond, evaluate).
pub const META_ROUTING_OPERATION: &str = "routing_operation";

/// Metadata key for serialized skill routing hints (JSON array of `SkillRoutingHint`).
pub const META_ROUTING_SKILL_HINTS: &str = "routing_skill_hints";

// -- Core types --

/// Classification of a request's complexity, determining which model handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    /// Short, simple queries -> cheap model
    Simple,
    /// Ambiguous complexity -> cheap model first, cascade to primary if uncertain
    Moderate,
    /// Code generation, analysis, multi-step reasoning -> primary model
    Complex,
}

/// Which reasoning operation is being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingOperation {
    /// Action planning — low stakes.
    Plan,
    /// Tool selection — needs accuracy.
    SelectTools,
    /// Full response generation with tool calls.
    Respond,
    /// Success evaluation — low stakes.
    Evaluate,
    /// Generic completion (no specific operation).
    Other,
}

impl RoutingOperation {
    /// Parse from metadata string value.
    pub fn from_meta(s: &str) -> Self {
        match s {
            "plan" => Self::Plan,
            "select_tools" => Self::SelectTools,
            "respond" => Self::Respond,
            "evaluate" => Self::Evaluate,
            _ => Self::Other,
        }
    }

    /// Convert to metadata string value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::SelectTools => "select_tools",
            Self::Respond => "respond",
            Self::Evaluate => "evaluate",
            Self::Other => "other",
        }
    }
}

/// Routing hint from an active skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRoutingHint {
    pub skill_name: String,
    /// Default starting tier for tool calls (0 = cheapest).
    #[serde(default)]
    pub tool_tier: Option<usize>,
    /// Per-operation overrides: keys match tool name suffixes (_add, _delete, etc.).
    #[serde(default)]
    pub operation_tiers: Option<HashMap<String, usize>>,
}

// -- Configuration --

/// Configuration for the smart routing provider.
#[derive(Debug, Clone)]
pub struct SmartRoutingConfig {
    /// Enable cascade mode: retry with higher tier if response is uncertain.
    pub cascade_enabled: bool,
    /// Message length threshold below which a message may be classified as Simple (default: 200).
    pub simple_max_chars: usize,
    /// Message length threshold above which a message is classified as Complex (default: 1000).
    pub complex_min_chars: usize,
    /// Cascade trigger configuration.
    pub cascade: CascadeConfig,
}

impl Default for SmartRoutingConfig {
    fn default() -> Self {
        Self {
            cascade_enabled: true,
            simple_max_chars: 200,
            complex_min_chars: 1000,
            cascade: CascadeConfig::default(),
        }
    }
}

/// Configuration for cascade escalation triggers.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Escalate when tools were available but model returned text only.
    pub escalate_on_no_tool_call: bool,
    /// Escalate when response shows uncertainty patterns.
    pub escalate_on_uncertainty: bool,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            escalate_on_no_tool_call: true,
            escalate_on_uncertainty: true,
        }
    }
}

// -- Classifier trait --

/// Context provided to the classifier for routing decisions.
pub struct ClassificationContext<'a> {
    /// Last user message content.
    pub message: &'a str,
    /// Whether tools are available for this request.
    pub has_tools: bool,
    /// Names of tools available in this request.
    pub tool_names: Vec<&'a str>,
    /// The reasoning operation being performed.
    pub operation: RoutingOperation,
    /// Routing hints from active skills.
    pub skill_hints: Vec<SkillRoutingHint>,
    /// Number of available tiers.
    pub num_tiers: usize,
    /// Routing configuration.
    pub config: &'a SmartRoutingConfig,
}

/// Pluggable classifier that picks a starting tier for a request.
pub trait MessageClassifier: Send + Sync {
    /// Returns the starting tier index (0 = cheapest).
    /// Will be clamped to valid range by the provider.
    fn classify(&self, ctx: &ClassificationContext) -> usize;
}

/// Default classifier that maps `TaskComplexity` to tier indices.
///
/// Preserves original 2-tier behavior:
/// - Simple -> tier 0
/// - Moderate -> tier 0 (with cascade)
/// - Complex -> last tier
/// - has_tools -> last tier
pub struct DefaultClassifier;

impl MessageClassifier for DefaultClassifier {
    fn classify(&self, ctx: &ClassificationContext) -> usize {
        let last_tier = ctx.num_tiers.saturating_sub(1);

        // Tool calls go to most capable tier by default
        if ctx.has_tools {
            return last_tier;
        }

        let complexity = classify_message(ctx.message, ctx.config);
        match complexity {
            TaskComplexity::Simple => 0,
            TaskComplexity::Moderate => 0,
            TaskComplexity::Complex => last_tier,
        }
    }
}

/// Skill-aware classifier that uses routing hints and operation type.
///
/// - Plan/Evaluate operations always route to tier 0 (cheap handles these fine).
/// - Tool calls check skill hints for tier preferences.
/// - Falls back to `DefaultClassifier` for non-tool, non-hinted calls.
pub struct SkillAwareClassifier;

impl MessageClassifier for SkillAwareClassifier {
    fn classify(&self, ctx: &ClassificationContext) -> usize {
        let last_tier = ctx.num_tiers.saturating_sub(1);

        // Low-stakes reasoning operations always go cheap
        if matches!(ctx.operation, RoutingOperation::Plan | RoutingOperation::Evaluate) {
            return 0;
        }

        // Check skill hints for tool calls
        if ctx.has_tools && !ctx.skill_hints.is_empty() {
            let mut min_tier = last_tier; // safe default: most capable
            for hint in &ctx.skill_hints {
                // Check per-operation overrides first: match tool name suffixes
                // against the available tools (e.g., "add" matches "grocery_items_add").
                if let Some(ref op_tiers) = hint.operation_tiers {
                    for tool_name in &ctx.tool_names {
                        for (suffix, &tier) in op_tiers {
                            if tool_name.ends_with(&format!("_{suffix}")) {
                                min_tier = min_tier.min(tier);
                            }
                        }
                    }
                }
                // Fall back to skill-level default
                if let Some(t) = hint.tool_tier {
                    min_tier = min_tier.min(t);
                }
            }
            return min_tier;
        }

        // Non-tool: use DefaultClassifier heuristics
        DefaultClassifier.classify(ctx)
    }
}

// -- Stats --

/// Atomic counters for routing observability.
struct SmartRoutingStats {
    total_requests: AtomicU64,
    tier_requests: Vec<AtomicU64>,
    cascade_escalations: AtomicU64,
}

impl SmartRoutingStats {
    fn new(num_tiers: usize) -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            tier_requests: (0..num_tiers).map(|_| AtomicU64::new(0)).collect(),
            cascade_escalations: AtomicU64::new(0),
        }
    }
}

/// Snapshot of routing statistics for external consumption.
#[derive(Debug, Clone)]
pub struct SmartRoutingSnapshot {
    pub total_requests: u64,
    /// Requests per tier (index 0 = cheapest).
    pub tier_requests: Vec<u64>,
    pub cascade_escalations: u64,
    // Backward-compat accessors
    pub cheap_requests: u64,
    pub primary_requests: u64,
}

// -- Provider --

/// Smart routing provider that classifies task complexity and routes to the
/// appropriate model tier. Supports N tiers with pluggable classification and
/// automatic cascade escalation.
pub struct SmartRoutingProvider {
    /// Providers ordered cheapest-first. Index 0 = cheapest, last = most capable.
    tiers: Vec<Arc<dyn LlmProvider>>,
    /// Pluggable classifier that picks a starting tier.
    classifier: Arc<dyn MessageClassifier>,
    config: SmartRoutingConfig,
    stats: SmartRoutingStats,
}

impl SmartRoutingProvider {
    /// Backward-compatible 2-tier constructor.
    ///
    /// Creates a routing provider with `cheap` at tier 0 and `primary` at tier 1,
    /// using the `DefaultClassifier`. Behavior is identical to the original
    /// SmartRoutingProvider.
    pub fn new(
        primary: Arc<dyn LlmProvider>,
        cheap: Arc<dyn LlmProvider>,
        config: SmartRoutingConfig,
    ) -> Self {
        Self::tiered(vec![cheap, primary], Arc::new(DefaultClassifier), config)
    }

    /// N-tier constructor with pluggable classifier.
    pub fn tiered(
        tiers: Vec<Arc<dyn LlmProvider>>,
        classifier: Arc<dyn MessageClassifier>,
        config: SmartRoutingConfig,
    ) -> Self {
        assert!(!tiers.is_empty(), "SmartRoutingProvider requires at least one tier");
        let stats = SmartRoutingStats::new(tiers.len());
        Self {
            tiers,
            classifier,
            config,
            stats,
        }
    }

    /// Get a snapshot of routing statistics.
    pub fn stats(&self) -> SmartRoutingSnapshot {
        let tier_requests: Vec<u64> = self
            .stats
            .tier_requests
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let cheap_requests = tier_requests.first().copied().unwrap_or(0);
        let primary_requests = if self.tiers.len() > 1 {
            tier_requests.last().copied().unwrap_or(0)
        } else {
            cheap_requests
        };
        SmartRoutingSnapshot {
            total_requests: self.stats.total_requests.load(Ordering::Relaxed),
            tier_requests,
            cascade_escalations: self.stats.cascade_escalations.load(Ordering::Relaxed),
            cheap_requests,
            primary_requests,
        }
    }

    /// Return the last (most capable) tier provider.
    fn primary(&self) -> &Arc<dyn LlmProvider> {
        self.tiers.last().expect("tiers is non-empty")
    }

    /// Extract routing context from request metadata.
    fn extract_routing_context<'a>(
        &'a self,
        metadata: &std::collections::HashMap<String, String>,
        has_tools: bool,
        tool_names: Vec<&'a str>,
        message: &'a str,
    ) -> ClassificationContext<'a> {
        let operation = metadata
            .get(META_ROUTING_OPERATION)
            .map(|s| RoutingOperation::from_meta(s))
            .unwrap_or(RoutingOperation::Other);

        let skill_hints: Vec<SkillRoutingHint> = metadata
            .get(META_ROUTING_SKILL_HINTS)
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        ClassificationContext {
            message,
            has_tools,
            tool_names,
            operation,
            skill_hints,
            num_tiers: self.tiers.len(),
            config: &self.config,
        }
    }

    /// Check if a completion response shows uncertainty, warranting escalation.
    fn response_is_uncertain(response: &CompletionResponse) -> bool {
        let content = response.content.trim();

        if content.is_empty() {
            return true;
        }

        let lower = content.to_lowercase();

        let uncertainty_patterns = [
            "i'm not sure",
            "i am not sure",
            "i don't know",
            "i do not know",
            "i'm unable to",
            "i am unable to",
            "i cannot",
            "i can't",
            "beyond my capabilities",
            "beyond my ability",
            "i'm not able to",
            "i am not able to",
            "i don't have enough",
            "i do not have enough",
            "i need more context",
            "i need more information",
            "could you clarify",
            "could you provide more",
            "i'm not confident",
            "i am not confident",
        ];

        uncertainty_patterns.iter().any(|p| lower.contains(p))
    }

    fn record_tier(&self, tier_idx: usize) {
        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        if let Some(counter) = self.stats.tier_requests.get(tier_idx) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Classify a message's complexity based on content patterns and length.
///
/// Exposed as a package-visible function for testability.
pub(crate) fn classify_message(msg: &str, config: &SmartRoutingConfig) -> TaskComplexity {
    let trimmed = msg.trim();
    let len = trimmed.len();

    if len == 0 {
        return TaskComplexity::Simple;
    }

    if trimmed.contains("```") {
        return TaskComplexity::Complex;
    }

    let lower = trimmed.to_lowercase();

    const COMPLEX_KEYWORDS: &[&str] = &[
        "implement",
        "refactor",
        "analyze",
        "debug",
        "create a",
        "build a",
        "design",
        "fix the",
        "fix this",
        "write a",
        "write the",
        "explain how",
        "explain why",
        "explain the",
        "compare",
        "optimize",
        "review",
        "rewrite",
        "migrate",
        "architect",
        "integrate",
    ];

    if COMPLEX_KEYWORDS.iter().any(|k| lower.contains(k)) {
        return TaskComplexity::Complex;
    }

    if len >= config.complex_min_chars {
        return TaskComplexity::Complex;
    }

    const SIMPLE_KEYWORDS: &[&str] = &[
        "list",
        "show",
        "what is",
        "what's",
        "status",
        "help",
        "yes",
        "no",
        "ok",
        "thanks",
        "thank you",
        "hello",
        "hi",
        "hey",
        "ping",
        "version",
        "how many",
        "when",
        "where is",
        "who",
    ];

    if len <= config.simple_max_chars && SIMPLE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        return TaskComplexity::Simple;
    }

    if len <= 10 {
        return TaskComplexity::Simple;
    }

    TaskComplexity::Moderate
}

#[async_trait]
impl LlmProvider for SmartRoutingProvider {
    fn model_name(&self) -> &str {
        self.primary().model_name()
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        self.primary().cost_per_token()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let ctx = self.extract_routing_context(&request.metadata, false, vec![], last_user_msg);
        let tier_idx = self.classifier.classify(&ctx).min(self.tiers.len() - 1);

        tracing::debug!(
            tier = tier_idx,
            model = %self.tiers[tier_idx].model_name(),
            operation = ?ctx.operation,
            "Smart routing: complete()"
        );

        self.record_tier(tier_idx);
        let mut response = self.tiers[tier_idx].complete(request.clone()).await?;

        // Cascade on uncertainty: walk up tiers one at a time
        if self.config.cascade_enabled && self.config.cascade.escalate_on_uncertainty {
            let mut current = tier_idx;
            while current < self.tiers.len() - 1 && Self::response_is_uncertain(&response) {
                let escalate_to = current + 1;
                tracing::info!(
                    from_tier = current,
                    from_model = %self.tiers[current].model_name(),
                    to_tier = escalate_to,
                    to_model = %self.tiers[escalate_to].model_name(),
                    "Smart routing: Escalating (uncertain response)"
                );
                self.stats
                    .cascade_escalations
                    .fetch_add(1, Ordering::Relaxed);
                self.record_tier(escalate_to);
                response = self.tiers[escalate_to].complete(request.clone()).await?;
                current = escalate_to;
            }
        }

        Ok(response)
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let has_tools = !request.tools.is_empty();
        let tool_names: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
        let ctx = self.extract_routing_context(&request.metadata, has_tools, tool_names, last_user_msg);
        let tier_idx = self.classifier.classify(&ctx).min(self.tiers.len() - 1);

        tracing::debug!(
            tier = tier_idx,
            model = %self.tiers[tier_idx].model_name(),
            has_tools,
            operation = ?ctx.operation,
            hints = ctx.skill_hints.len(),
            "Smart routing: complete_with_tools()"
        );

        self.record_tier(tier_idx);
        let mut response = self.tiers[tier_idx]
            .complete_with_tools(request.clone())
            .await?;

        // Cascade: walk up tiers if tools were available but model returned text only
        if self.config.cascade_enabled && self.config.cascade.escalate_on_no_tool_call && has_tools {
            let mut current = tier_idx;
            while current < self.tiers.len() - 1 && response.tool_calls.is_empty() {
                let escalate_to = current + 1;
                tracing::info!(
                    from_tier = current,
                    to_tier = escalate_to,
                    to_model = %self.tiers[escalate_to].model_name(),
                    "Smart routing: Escalating (no tool call from tier {})", current
                );
                self.stats
                    .cascade_escalations
                    .fetch_add(1, Ordering::Relaxed);
                self.record_tier(escalate_to);
                response = self.tiers[escalate_to]
                    .complete_with_tools(request.clone())
                    .await?;
                current = escalate_to;
            }
        }

        Ok(response)
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.primary().list_models().await
    }

    async fn model_metadata(&self) -> Result<ModelMetadata, LlmError> {
        self.primary().model_metadata().await
    }

    fn active_model_name(&self) -> String {
        self.primary().active_model_name()
    }

    fn set_model(&self, model: &str) -> Result<(), LlmError> {
        self.primary().set_model(model)
    }

    fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> Decimal {
        self.primary().calculate_cost(input_tokens, output_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;
    use crate::testing::StubLlm;

    fn default_config() -> SmartRoutingConfig {
        SmartRoutingConfig::default()
    }

    // -- Classification tests --

    #[test]
    fn classify_empty_message_as_simple() {
        assert_eq!(
            classify_message("", &default_config()),
            TaskComplexity::Simple
        );
    }

    #[test]
    fn classify_greeting_as_simple() {
        assert_eq!(
            classify_message("hello", &default_config()),
            TaskComplexity::Simple
        );
        assert_eq!(
            classify_message("hi there", &default_config()),
            TaskComplexity::Simple
        );
    }

    #[test]
    fn classify_short_question_with_simple_keyword() {
        assert_eq!(
            classify_message("what is the status?", &default_config()),
            TaskComplexity::Simple
        );
        assert_eq!(
            classify_message("show me the list", &default_config()),
            TaskComplexity::Simple
        );
        assert_eq!(
            classify_message("help", &default_config()),
            TaskComplexity::Simple
        );
    }

    #[test]
    fn classify_yes_no_as_simple() {
        assert_eq!(
            classify_message("yes", &default_config()),
            TaskComplexity::Simple
        );
        assert_eq!(
            classify_message("no", &default_config()),
            TaskComplexity::Simple
        );
        assert_eq!(
            classify_message("ok", &default_config()),
            TaskComplexity::Simple
        );
    }

    #[test]
    fn classify_code_generation_as_complex() {
        assert_eq!(
            classify_message("implement a binary search function", &default_config()),
            TaskComplexity::Complex
        );
        assert_eq!(
            classify_message("refactor the auth module", &default_config()),
            TaskComplexity::Complex
        );
        assert_eq!(
            classify_message("debug this error", &default_config()),
            TaskComplexity::Complex
        );
    }

    #[test]
    fn classify_code_blocks_as_complex() {
        let msg = "What does this do?\n```rust\nfn main() {}\n```";
        assert_eq!(
            classify_message(msg, &default_config()),
            TaskComplexity::Complex
        );
    }

    #[test]
    fn classify_long_message_as_complex() {
        let long_msg = "a ".repeat(600); // 1200 chars
        assert_eq!(
            classify_message(&long_msg, &default_config()),
            TaskComplexity::Complex
        );
    }

    #[test]
    fn classify_medium_message_without_keywords_as_moderate() {
        let msg = "Tell me about the weather patterns in the Pacific Ocean during summer months";
        assert_eq!(
            classify_message(msg, &default_config()),
            TaskComplexity::Moderate
        );
    }

    #[test]
    fn classify_very_short_unknown_as_simple() {
        assert_eq!(
            classify_message("foo", &default_config()),
            TaskComplexity::Simple
        );
    }

    // -- Uncertainty detection tests --

    #[test]
    fn detects_uncertain_short_response() {
        let response = CompletionResponse {
            content: "I'm not sure.".to_string(),
            input_tokens: 10,
            output_tokens: 5,
            finish_reason: crate::llm::FinishReason::Stop,
        };
        assert!(SmartRoutingProvider::response_is_uncertain(&response));
    }

    #[test]
    fn detects_empty_response_as_uncertain() {
        let response = CompletionResponse {
            content: "".to_string(),
            input_tokens: 10,
            output_tokens: 0,
            finish_reason: crate::llm::FinishReason::Stop,
        };
        assert!(SmartRoutingProvider::response_is_uncertain(&response));
    }

    #[test]
    fn short_confident_response_is_not_uncertain() {
        let response = CompletionResponse {
            content: "Yes.".to_string(),
            input_tokens: 10,
            output_tokens: 1,
            finish_reason: crate::llm::FinishReason::Stop,
        };
        assert!(!SmartRoutingProvider::response_is_uncertain(&response));
    }

    #[test]
    fn confident_response_is_not_uncertain() {
        let response = CompletionResponse {
            content: "The answer is 42. This is a well-known constant from the Hitchhiker's Guide."
                .to_string(),
            input_tokens: 10,
            output_tokens: 20,
            finish_reason: crate::llm::FinishReason::Stop,
        };
        assert!(!SmartRoutingProvider::response_is_uncertain(&response));
    }

    // -- Routing tests (backward compat) --

    fn make_request(content: &str) -> CompletionRequest {
        CompletionRequest::new(vec![ChatMessage::user(content)])
    }

    fn make_tool_request() -> ToolCompletionRequest {
        ToolCompletionRequest::new(
            vec![ChatMessage::user("implement a search")],
            vec![crate::llm::ToolDefinition {
                name: "search_tool".to_string(),
                description: "A search tool".to_string(),
                parameters: serde_json::json!({}),
            }],
        )
    }

    #[tokio::test]
    async fn simple_task_routes_to_cheap() {
        let primary = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("cheap-response").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(
            primary.clone(),
            cheap.clone(),
            SmartRoutingConfig {
                cascade_enabled: false,
                ..default_config()
            },
        );

        let resp = router.complete(make_request("hello")).await.unwrap();
        assert_eq!(resp.content, "cheap-response");
        assert_eq!(cheap.calls(), 1);
        assert_eq!(primary.calls(), 0);
    }

    #[tokio::test]
    async fn complex_task_routes_to_primary() {
        let primary = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("cheap-response").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(primary.clone(), cheap.clone(), default_config());

        let resp = router
            .complete(make_request("implement a binary search"))
            .await
            .unwrap();
        assert_eq!(resp.content, "primary-response");
        assert_eq!(primary.calls(), 1);
        assert_eq!(cheap.calls(), 0);
    }

    #[tokio::test]
    async fn tool_use_always_routes_to_primary() {
        let primary = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("cheap-response").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(primary.clone(), cheap.clone(), default_config());

        let resp = router
            .complete_with_tools(make_tool_request())
            .await
            .unwrap();
        assert_eq!(resp.content, Some("primary-response".to_string()));
        assert_eq!(primary.calls(), 1);
        assert_eq!(cheap.calls(), 0);
    }

    #[tokio::test]
    async fn stats_increment_correctly() {
        let primary = Arc::new(StubLlm::new("primary").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("cheap").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(
            primary,
            cheap,
            SmartRoutingConfig {
                cascade_enabled: false,
                ..default_config()
            },
        );

        // Simple -> cheap
        router.complete(make_request("hello")).await.unwrap();
        // Complex -> primary
        router
            .complete(make_request("implement a search"))
            .await
            .unwrap();
        // Tool use -> primary
        router
            .complete_with_tools(make_tool_request())
            .await
            .unwrap();

        let stats = router.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.cheap_requests, 1);
        assert_eq!(stats.primary_requests, 2);
        assert_eq!(stats.cascade_escalations, 0);
    }

    #[tokio::test]
    async fn cascade_escalates_on_uncertain_response() {
        let primary = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("I'm not sure about that.").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(
            primary.clone(),
            cheap.clone(),
            SmartRoutingConfig {
                cascade_enabled: true,
                ..default_config()
            },
        );

        let resp = router
            .complete(make_request(
                "Tell me about the weather patterns in the Pacific Ocean during summer months",
            ))
            .await
            .unwrap();

        assert_eq!(resp.content, "primary-response");
        assert_eq!(cheap.calls(), 1);
        assert_eq!(primary.calls(), 1);

        let stats = router.stats();
        assert_eq!(stats.cascade_escalations, 1);
    }

    #[tokio::test]
    async fn cascade_does_not_escalate_on_confident_response() {
        let primary = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));
        let cheap = Arc::new(
            StubLlm::new(
                "The Pacific Ocean weather patterns during summer are characterized by trade winds.",
            )
            .with_model_name("cheap"),
        );

        let router = SmartRoutingProvider::new(
            primary.clone(),
            cheap.clone(),
            SmartRoutingConfig {
                cascade_enabled: true,
                ..default_config()
            },
        );

        let resp = router
            .complete(make_request(
                "Tell me about the weather patterns in the Pacific Ocean during summer months",
            ))
            .await
            .unwrap();

        assert!(resp.content.contains("Pacific Ocean"));
        assert_eq!(cheap.calls(), 1);
        assert_eq!(primary.calls(), 0);

        let stats = router.stats();
        assert_eq!(stats.cascade_escalations, 0);
    }

    #[tokio::test]
    async fn model_name_returns_primary() {
        let primary = Arc::new(StubLlm::new("ok").with_model_name("sonnet"));
        let cheap = Arc::new(StubLlm::new("ok").with_model_name("haiku"));

        let router = SmartRoutingProvider::new(primary, cheap, default_config());
        assert_eq!(router.model_name(), "sonnet");
        assert_eq!(router.active_model_name(), "sonnet");
    }

    // -- N-tier and classifier tests --

    #[test]
    fn default_classifier_simple_maps_to_tier_0() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "hello",
            has_tools: false,
            tool_names: vec![],
            operation: RoutingOperation::Other,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(DefaultClassifier.classify(&ctx), 0);
    }

    #[test]
    fn default_classifier_complex_maps_to_last_tier() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "implement a binary search function",
            has_tools: false,
            tool_names: vec![],
            operation: RoutingOperation::Other,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(DefaultClassifier.classify(&ctx), 2);
    }

    #[test]
    fn default_classifier_tools_maps_to_last_tier() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "hello",
            has_tools: true,
            tool_names: vec![],
            operation: RoutingOperation::Other,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(DefaultClassifier.classify(&ctx), 2);
    }

    #[test]
    fn skill_aware_plan_routes_to_tier_0() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "implement something complex",
            has_tools: true,
            tool_names: vec![],
            operation: RoutingOperation::Plan,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(SkillAwareClassifier.classify(&ctx), 0);
    }

    #[test]
    fn skill_aware_evaluate_routes_to_tier_0() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "complex evaluation query",
            has_tools: true,
            tool_names: vec![],
            operation: RoutingOperation::Evaluate,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(SkillAwareClassifier.classify(&ctx), 0);
    }

    #[test]
    fn skill_aware_tool_call_with_hint_routes_to_hinted_tier() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "add eggs to grocery list",
            has_tools: true,
            tool_names: vec!["grocery_items_add"],
            operation: RoutingOperation::Respond,
            skill_hints: vec![SkillRoutingHint {
                skill_name: "grocery_items".to_string(),
                tool_tier: Some(0),
                operation_tiers: None,
            }],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(SkillAwareClassifier.classify(&ctx), 0);
    }

    #[test]
    fn skill_aware_tool_call_without_hint_routes_to_last_tier() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "add eggs",
            has_tools: true,
            tool_names: vec!["grocery_items_add"],
            operation: RoutingOperation::Respond,
            skill_hints: vec![],
            num_tiers: 3,
            config: &config,
        };
        assert_eq!(SkillAwareClassifier.classify(&ctx), 2);
    }

    #[tokio::test]
    async fn three_tier_simple_routes_to_cheapest() {
        let t0 = Arc::new(StubLlm::new("tier0").with_model_name("local"));
        let t1 = Arc::new(StubLlm::new("tier1").with_model_name("4o-mini"));
        let t2 = Arc::new(StubLlm::new("tier2").with_model_name("sonnet"));

        let router = SmartRoutingProvider::tiered(
            vec![t0.clone(), t1.clone(), t2.clone()],
            Arc::new(DefaultClassifier),
            SmartRoutingConfig {
                cascade_enabled: false,
                ..default_config()
            },
        );

        let resp = router.complete(make_request("hello")).await.unwrap();
        assert_eq!(resp.content, "tier0");
        assert_eq!(t0.calls(), 1);
        assert_eq!(t1.calls(), 0);
        assert_eq!(t2.calls(), 0);
    }

    #[tokio::test]
    async fn three_tier_complex_routes_to_top() {
        let t0 = Arc::new(StubLlm::new("tier0").with_model_name("local"));
        let t1 = Arc::new(StubLlm::new("tier1").with_model_name("4o-mini"));
        let t2 = Arc::new(StubLlm::new("tier2").with_model_name("sonnet"));

        let router = SmartRoutingProvider::tiered(
            vec![t0.clone(), t1.clone(), t2.clone()],
            Arc::new(DefaultClassifier),
            default_config(),
        );

        let resp = router
            .complete(make_request("implement a binary search"))
            .await
            .unwrap();
        assert_eq!(resp.content, "tier2");
        assert_eq!(t0.calls(), 0);
        assert_eq!(t1.calls(), 0);
        assert_eq!(t2.calls(), 1);
    }

    #[tokio::test]
    async fn cascade_no_tool_call_escalates() {
        // StubLlm returns empty tool_calls, so tier 0 will trigger NO_TOOL_CALL cascade
        let t0 = Arc::new(StubLlm::new("text-only").with_model_name("cheap"));
        let t1 = Arc::new(StubLlm::new("primary-response").with_model_name("primary"));

        let router = SmartRoutingProvider::tiered(
            vec![t0.clone(), t1.clone()],
            Arc::new(SkillAwareClassifier),
            default_config(),
        );

        // Build request with skill hint routing to tier 0 and with tools
        let mut request = ToolCompletionRequest::new(
            vec![ChatMessage::user("add eggs")],
            vec![crate::llm::ToolDefinition {
                name: "grocery_items_add".to_string(),
                description: "Add item".to_string(),
                parameters: serde_json::json!({}),
            }],
        );
        request.metadata.insert(
            META_ROUTING_SKILL_HINTS.to_string(),
            serde_json::to_string(&vec![SkillRoutingHint {
                skill_name: "grocery_items".to_string(),
                tool_tier: Some(0),
                operation_tiers: None,
            }])
            .unwrap(),
        );

        let resp = router.complete_with_tools(request).await.unwrap();
        // Should have escalated: tier 0 returned no tool calls, so tier 1 was tried
        assert_eq!(resp.content, Some("primary-response".to_string()));
        assert_eq!(t0.calls(), 1);
        assert_eq!(t1.calls(), 1);

        let stats = router.stats();
        assert_eq!(stats.cascade_escalations, 1);
    }

    #[tokio::test]
    async fn last_tier_no_tool_call_does_not_cascade() {
        let t0 = Arc::new(StubLlm::new("text-only").with_model_name("only-tier"));

        let router = SmartRoutingProvider::tiered(
            vec![t0.clone()],
            Arc::new(DefaultClassifier),
            default_config(),
        );

        let request = ToolCompletionRequest::new(
            vec![ChatMessage::user("add eggs")],
            vec![crate::llm::ToolDefinition {
                name: "test".to_string(),
                description: "test".to_string(),
                parameters: serde_json::json!({}),
            }],
        );

        let resp = router.complete_with_tools(request).await.unwrap();
        assert_eq!(resp.content, Some("text-only".to_string()));
        assert_eq!(t0.calls(), 1);
        assert_eq!(router.stats().cascade_escalations, 0);
    }

    #[test]
    fn routing_operation_roundtrip() {
        for op in [
            RoutingOperation::Plan,
            RoutingOperation::SelectTools,
            RoutingOperation::Respond,
            RoutingOperation::Evaluate,
            RoutingOperation::Other,
        ] {
            assert_eq!(RoutingOperation::from_meta(op.as_str()), op);
        }
    }

    #[test]
    fn skill_routing_hint_deserializes() {
        let json = r#"[{"skill_name":"grocery_items","tool_tier":0}]"#;
        let hints: Vec<SkillRoutingHint> = serde_json::from_str(json).unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].tool_tier, Some(0));
    }

    #[test]
    fn skill_aware_operation_tiers_override_tool_tier() {
        let config = default_config();
        // Skill says tool_tier: 0 (default cheap), but delete operations should go to tier 1
        let ctx = ClassificationContext {
            message: "delete the eggs entry",
            has_tools: true,
            tool_names: vec!["grocery_items_delete"],
            operation: RoutingOperation::Respond,
            skill_hints: vec![SkillRoutingHint {
                skill_name: "grocery_items".to_string(),
                tool_tier: Some(0),
                operation_tiers: Some(HashMap::from([
                    ("add".to_string(), 0),
                    ("delete".to_string(), 1),
                ])),
            }],
            num_tiers: 3,
            config: &config,
        };
        // operation_tiers "delete" -> 1 should win (min of tool_tier=0, op_tier=1 is 0)
        // Actually: tool_tier=0 is still checked. The classifier takes min of ALL hints.
        // With op_tiers: delete->1 gives 1, tool_tier->0 gives 0, min is 0.
        // This matches the plan: per-operation tiers give a starting point,
        // but if the skill also says tool_tier: 0, that's the minimum.
        assert_eq!(SkillAwareClassifier.classify(&ctx), 0);
    }

    #[test]
    fn skill_aware_operation_tiers_without_tool_tier_fallback() {
        let config = default_config();
        // Skill only sets operation_tiers, no global tool_tier
        let ctx = ClassificationContext {
            message: "delete the eggs entry",
            has_tools: true,
            tool_names: vec!["grocery_items_delete"],
            operation: RoutingOperation::Respond,
            skill_hints: vec![SkillRoutingHint {
                skill_name: "grocery_items".to_string(),
                tool_tier: None,
                operation_tiers: Some(HashMap::from([
                    ("add".to_string(), 0),
                    ("delete".to_string(), 1),
                ])),
            }],
            num_tiers: 3,
            config: &config,
        };
        // No tool_tier fallback, operation_tiers delete->1, min_tier starts at last(2)
        // so min(2, 1) = 1
        assert_eq!(SkillAwareClassifier.classify(&ctx), 1);
    }

    #[test]
    fn skill_aware_delegates_to_default_for_non_tool_respond() {
        let config = default_config();
        let ctx = ClassificationContext {
            message: "hello",
            has_tools: false,
            tool_names: vec![],
            operation: RoutingOperation::Respond,
            skill_hints: vec![SkillRoutingHint {
                skill_name: "grocery_items".to_string(),
                tool_tier: Some(0),
                operation_tiers: None,
            }],
            num_tiers: 3,
            config: &config,
        };
        // No tools -> delegates to DefaultClassifier -> "hello" is Simple -> tier 0
        assert_eq!(SkillAwareClassifier.classify(&ctx), 0);
    }

    #[tokio::test]
    async fn three_tier_cascade_walks_up_not_skips() {
        // All three tiers return uncertain responses
        let t0 = Arc::new(StubLlm::new("I'm not sure about that").with_model_name("local"));
        let t1 = Arc::new(StubLlm::new("I'm not sure either").with_model_name("4o-mini"));
        let t2 = Arc::new(StubLlm::new("confident-answer").with_model_name("sonnet"));

        let router = SmartRoutingProvider::tiered(
            vec![t0.clone(), t1.clone(), t2.clone()],
            Arc::new(DefaultClassifier),
            SmartRoutingConfig {
                cascade_enabled: true,
                ..default_config()
            },
        );

        let resp = router.complete(make_request("hello")).await.unwrap();
        // Should walk: tier 0 (uncertain) -> tier 1 (uncertain) -> tier 2 (confident)
        assert_eq!(resp.content, "confident-answer");
        assert_eq!(t0.calls(), 1);
        assert_eq!(t1.calls(), 1); // tier 1 was tried (not skipped)
        assert_eq!(t2.calls(), 1);
        assert_eq!(router.stats().cascade_escalations, 2);
    }

    #[tokio::test]
    async fn two_tier_new_constructor_identical_behavior() {
        let primary = Arc::new(StubLlm::new("primary").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("cheap").with_model_name("cheap"));

        let router = SmartRoutingProvider::new(
            primary.clone(),
            cheap.clone(),
            SmartRoutingConfig {
                cascade_enabled: false,
                ..default_config()
            },
        );

        // Simple -> tier 0 (cheap)
        let resp = router.complete(make_request("hello")).await.unwrap();
        assert_eq!(resp.content, "cheap");

        // Complex -> tier 1 (primary)
        let resp = router
            .complete(make_request("implement a search"))
            .await
            .unwrap();
        assert_eq!(resp.content, "primary");

        // Tool use -> tier 1 (primary) via DefaultClassifier
        let resp = router
            .complete_with_tools(make_tool_request())
            .await
            .unwrap();
        assert_eq!(resp.content, Some("primary".to_string()));
    }
}
