//! Smart routing provider that routes requests across N model tiers based on
//! task complexity, skill hints, and operation type.
//!
//! Uses a 13-dimension complexity scorer (from PR #208 by @onlyamicrowave) to analyze prompts
//! across reasoning, code, multi-step, domain-specific, creativity, precision, safety, and other
//! dimensions. Pattern overrides provide fast-path routing for obvious cases (greetings → cheap,
//! security audits → primary).
//!
//! Supports pluggable classification via the [`MessageClassifier`] trait and
//! automatic cascade escalation when a cheaper tier fails (timeout, no tool
//! call, uncertainty).
//!
//! Backward-compatible: the 2-tier `new()` constructor produces identical
//! behavior to the original SmartRoutingProvider.
//!
//! # Complexity Tiers
//!
//! The scorer produces a 0-100 score mapped to four tiers:
//! - **Flash** (0-15): Greetings, quick lookups → cheap model
//! - **Standard** (16-40): Writing, comparisons → cheap model
//! - **Pro** (41-65): Multi-step analysis, code review → cheap with cascade, or primary
//! - **Frontier** (66+): Security audits, critical decisions → primary model

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use regex::Regex;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::llm::error::LlmError;
use crate::llm::provider::{
    CompletionRequest, CompletionResponse, LlmProvider, ModelMetadata, Role, ToolCompletionRequest,
    ToolCompletionResponse,
};

// ---------------------------------------------------------------------------
// Complexity tiers & scoring
// ---------------------------------------------------------------------------

/// Complexity tier produced by the 13-dimension scorer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Simple requests: greetings, quick lookups (score 0-15).
    Flash,
    /// Standard tasks: writing, comparisons (score 16-40).
    Standard,
    /// Complex work: multi-step analysis, code review (score 41-65).
    Pro,
    /// Critical tasks: security audits, high-stakes decisions (score 66+).
    Frontier,
}

impl Tier {
    /// Convert a complexity score to a tier.
    pub fn from_score(score: u32) -> Self {
        match score {
            0..=15 => Tier::Flash,
            16..=40 => Tier::Standard,
            41..=65 => Tier::Pro,
            _ => Tier::Frontier,
        }
    }

    /// Get a representative score for this tier (used when score is not computed).
    pub fn to_score(self) -> u32 {
        match self {
            Tier::Flash => 8,
            Tier::Standard => 28,
            Tier::Pro => 52,
            Tier::Frontier => 80,
        }
    }

    /// Tier name as string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Flash => "flash",
            Tier::Standard => "standard",
            Tier::Pro => "pro",
            Tier::Frontier => "frontier",
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Weights for each of the 13 scoring dimensions.
#[derive(Debug, Clone)]
pub struct ScorerWeights {
    pub reasoning_words: f32,
    pub token_estimate: f32,
    pub code_indicators: f32,
    pub multi_step: f32,
    pub domain_specific: f32,
    pub ambiguity: f32,
    pub creativity: f32,
    pub precision: f32,
    pub context_dependency: f32,
    pub tool_likelihood: f32,
    pub safety_sensitivity: f32,
    pub question_complexity: f32,
    pub sentence_complexity: f32,
}

impl Default for ScorerWeights {
    fn default() -> Self {
        Self {
            reasoning_words: 0.14,
            token_estimate: 0.12,
            code_indicators: 0.10,
            multi_step: 0.10,
            domain_specific: 0.10,
            ambiguity: 0.05,
            creativity: 0.07,
            precision: 0.06,
            context_dependency: 0.05,
            tool_likelihood: 0.05,
            safety_sensitivity: 0.04,
            question_complexity: 0.07,
            sentence_complexity: 0.05,
        }
    }
}

/// Default domain-specific keywords for complexity scoring.
pub const DEFAULT_DOMAIN_KEYWORDS: &[&str] = &[
    // Infrastructure
    "kubernetes",
    "k8s",
    "docker",
    "terraform",
    "nginx",
    "apache",
    "linux",
    "unix",
    "bash",
    "shell",
    // Languages & frameworks
    "solidity",
    "rust",
    "typescript",
    "react",
    "nextjs",
    "vue",
    "angular",
    "svelte",
    // Databases
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    // APIs & protocols
    "graphql",
    "grpc",
    "protobuf",
    "websocket",
    "oauth",
    "jwt",
    "cors",
    "csrf",
    "xss",
    "sql.?injection",
    "api",
    "rest",
    "http",
    "https",
    "tcp",
    "udp",
    "dns",
    "cdn",
    // Cloud & deployment
    "aws",
    "gcp",
    "azure",
    "vercel",
    "netlify",
    "cloudflare",
    "ci/cd",
    "devops",
    // Version control
    "git",
    "github",
    "gitlab",
    // Web3 general
    "blockchain",
    "web3",
    "defi",
    "nft",
    "smart.?contract",
    // Ethereum
    "ethereum",
    "evm",
    "anchor",
    // NEAR ecosystem
    "near",
    "near.?sdk",
    "near.?api",
    "testnet",
    "mainnet",
    "meteor",
    "ledger",
    "cold.?wallet",
    "rpc",
    "indexer",
    "relayer",
    "cross.?chain",
    "intents",
    // Fogo/SVM
    "fogo",
    "svm",
    "firedancer",
    "paymaster",
    "gasless",
    "sessions.?sdk",
    // Rust/NEAR tooling
    "cargo.?near",
    "workspaces",
    "sandbox",
    // Project-specific
    "lobo",
    "trezu",
    "multisig",
    "treasury",
    "openclaw",
    "ironclaw",
];

/// Configuration for the complexity scorer.
#[derive(Debug, Clone, Default)]
pub struct ScorerConfig {
    /// Weights for each scoring dimension.
    pub weights: ScorerWeights,
    /// Custom domain-specific keywords (overrides defaults if provided).
    /// Each entry is a word or regex pattern fragment.
    pub domain_keywords: Option<Vec<String>>,
}

/// Build a domain regex from a keyword list, with fallback on invalid patterns.
///
/// An empty keyword list falls back to the default keywords so scoring
/// doesn't break when `domain_keywords: Some(vec![])` is configured.
fn build_domain_regex(keywords: &[&str]) -> Regex {
    if keywords.is_empty() {
        return RE_DOMAIN_DEFAULT.clone();
    }
    let pattern = format!(r"(?i)\b({})\b", keywords.join("|"));
    Regex::new(&pattern).unwrap_or_else(|e| {
        tracing::warn!(error = %e, "Invalid domain keywords pattern, using minimal fallback");
        Regex::new(r"(?i)\b(api|code|deploy)\b").expect("fallback regex is valid")
    })
}

/// Breakdown of complexity score by dimension.
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    /// Total complexity score (0-100).
    pub total: u32,
    /// Computed tier.
    pub tier: Tier,
    /// Per-dimension scores (0-100 each).
    pub components: HashMap<String, u32>,
    /// Human-readable hints about why this score.
    pub hints: Vec<String>,
}

// ---------------------------------------------------------------------------
// Static regex patterns (compiled once via LazyLock)
// ---------------------------------------------------------------------------

use std::sync::LazyLock;

static RE_REASONING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(why|how|explain|analyze|analyse|compare|contrast|evaluate|assess|reason|think|consider|implications?|consequences?|trade-?offs?|pros?\s*(and|&)\s*cons?|advantages?|disadvantages?|benefits?|drawbacks?|differs?|difference|versus|vs\.?|better|worse|optimal|best|worst)\b"
    ).expect("RE_REASONING is a valid regex")
});

static RE_MULTI_STEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(first|then|next|after|before|finally|step|steps|phase|stages?|process|workflow|sequence|procedure|pipeline|chain|series|order|followed by)\b"
    ).expect("RE_MULTI_STEP is a valid regex")
});

static RE_CREATIVITY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(write|create|generate|compose|design|imagine|brainstorm|ideate|draft|invent|story|poem|essay|article|blog|content|narrative|script|summarize|summarise|rewrite|paraphrase|translate|adapt|tweet|post|thread|outline|structure|format|style|tone|voice)\b"
    ).expect("RE_CREATIVITY is a valid regex")
});

static RE_PRECISION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(\d{4}|\d+\.\d+|exactly|precisely|specific|accurate|correct|verify|confirm|date|time|number|calculate|compute|measure|count)\b"
    ).expect("RE_PRECISION is a valid regex")
});

static RE_CODE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(`{1,3}|```|function|const|let|var|import|export|class|def |async|await|=>|\.ts|\.js|\.py|\.rs|\.go|\.sol|\(\)|\[\]|\{\}|<[A-Z][a-z]+>|useState|useEffect|npm|yarn|pnpm|cargo|pip|implement|rebase|merge|commit|branch|PR|pull.?request|columns?|migrations?|module|refactor|debug|fix|bug|error|schema|database|query)"
    ).expect("RE_CODE is a valid regex")
});

static RE_TOOL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(file|read|write|search|fetch|run|execute|check|look up|find|open|save|send|post|get|download|upload|install|deploy|build|compile|test|add|update|remove|delete|modify|change|edit|create|resolve|push|pull|clone)\b"
    ).expect("RE_TOOL is a valid regex")
});

static RE_SAFETY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(password|secret|private|confidential|medical|legal|financial|personal|sensitive|ssn|credit.?card|auth|token|key|encrypt|decrypt|hash|vulnerability|exploit|attack|breach)\b"
    ).expect("RE_SAFETY is a valid regex")
});

static RE_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(previous|earlier|above|before|last|that|those|it|they|we discussed|you said|mentioned|remember|recall|as I said|like I mentioned)\b"
    ).expect("RE_CONTEXT is a valid regex")
});

static RE_VAGUE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(it|this|that|something|stuff|thing|things)\b")
        .expect("RE_VAGUE is a valid regex")
});

static RE_OPEN_ENDED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(why|how|what if|explain|describe|elaborate|discuss)\b")
        .expect("RE_OPEN_ENDED is a valid regex")
});

static RE_CONJUNCTIONS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(and|but|or|however|therefore|because|although|while|whereas|moreover|furthermore)\b",
    )
    .expect("RE_CONJUNCTIONS is a valid regex")
});

static RE_TIER_HINT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[tier:(flash|standard|pro|frontier)\]")
        .expect("RE_TIER_HINT is a valid regex")
});

/// Default domain regex, compiled once from `DEFAULT_DOMAIN_KEYWORDS`.
static RE_DOMAIN_DEFAULT: LazyLock<Regex> =
    LazyLock::new(|| build_domain_regex(DEFAULT_DOMAIN_KEYWORDS));

// ---------------------------------------------------------------------------
// Pattern overrides (fast-path before scoring)
// ---------------------------------------------------------------------------

/// A compiled pattern override entry.
struct PatternOverride {
    regex: Regex,
    tier: Tier,
}

/// Default pattern overrides, compiled once.
static DEFAULT_OVERRIDES: LazyLock<Vec<PatternOverride>> = LazyLock::new(|| {
    vec![
        // Flash tier: greetings and acknowledgments
        PatternOverride {
            regex: Regex::new(
                r"(?i)^(hi|hello|hey|thanks|ok|sure|yes|no|yep|nope|cool|nice|great|got it)$",
            )
            .expect("greeting pattern is valid"),
            tier: Tier::Flash,
        },
        // Flash tier: quick lookups (end-anchored to avoid matching complex questions
        // like "What time complexity is merge sort?")
        PatternOverride {
            regex: Regex::new(
                r"(?i)^what(?:'s|\s+is)?\s+(?:the\s+)?(time|date|day|weather)\b(?:\s+(?:is\s+it|today|now|in\s+\S+))?[?.!]*$",
            )
            .expect("lookup pattern is valid"),
            tier: Tier::Flash,
        },
        // Frontier tier: security audits
        PatternOverride {
            regex: Regex::new(r"(?i)security.*(audit|review|scan)")
                .expect("security audit pattern is valid"),
            tier: Tier::Frontier,
        },
        PatternOverride {
            regex: Regex::new(r"(?i)vulnerabilit(y|ies).*(review|scan|check|audit)")
                .expect("vulnerability pattern is valid"),
            tier: Tier::Frontier,
        },
        // Pro tier: production deployments
        PatternOverride {
            regex: Regex::new(r"(?i)deploy.*(mainnet|production)")
                .expect("deploy pattern is valid"),
            tier: Tier::Pro,
        },
        PatternOverride {
            regex: Regex::new(r"(?i)production.*(deploy|release|push)")
                .expect("production pattern is valid"),
            tier: Tier::Pro,
        },
    ]
});

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/// Count regex matches in text.
fn count_matches(re: &Regex, text: &str) -> usize {
    re.find_iter(text).count()
}

/// Score a prompt's complexity across 13 dimensions.
///
/// Returns a `ScoreBreakdown` with a total score (0-100) and per-dimension breakdown.
pub fn score_complexity(prompt: &str) -> ScoreBreakdown {
    score_complexity_with_config(prompt, &ScorerConfig::default())
}

/// Score with custom configuration (weights + domain keywords).
///
/// If you will call this repeatedly with the same config, prefer
/// [`score_complexity_with_regex`] and pre-build the domain regex once.
pub fn score_complexity_with_config(prompt: &str, config: &ScorerConfig) -> ScoreBreakdown {
    let domain_regex = match &config.domain_keywords {
        Some(custom) => {
            let refs: Vec<&str> = custom.iter().map(|s| s.as_str()).collect();
            build_domain_regex(&refs)
        }
        None => RE_DOMAIN_DEFAULT.clone(),
    };
    score_complexity_internal(prompt, &config.weights, &domain_regex)
}

/// Score with a pre-compiled domain regex (avoids rebuilding per call).
pub fn score_complexity_with_regex(
    prompt: &str,
    weights: &ScorerWeights,
    domain_regex: &Regex,
) -> ScoreBreakdown {
    score_complexity_internal(prompt, weights, domain_regex)
}

/// Internal scoring implementation.
fn score_complexity_internal(
    prompt: &str,
    weights: &ScorerWeights,
    domain_regex: &Regex,
) -> ScoreBreakdown {
    let mut hints = Vec::new();
    let mut components = HashMap::new();

    // Check for explicit tier hint (e.g. "[tier:flash]")
    if let Some(caps) = RE_TIER_HINT.captures(prompt) {
        let tier_str = caps.get(1).expect("capture group 1 exists").as_str();
        let tier = match tier_str.to_lowercase().as_str() {
            "flash" => Tier::Flash,
            "standard" => Tier::Standard,
            "pro" => Tier::Pro,
            "frontier" => Tier::Frontier,
            // The regex only captures valid tiers, so this is defensive.
            other => {
                tracing::error!(tier = %other, "Unexpected tier in hint despite regex constraint");
                Tier::Standard
            }
        };
        hints.push(format!("Explicit tier hint: {tier}"));
        return ScoreBreakdown {
            total: tier.to_score(),
            tier,
            components,
            hints,
        };
    }

    // Token estimate (based on char count): <20 chars = 0, >=520 chars = 100
    let char_count = prompt.len();
    let token_score = ((char_count as i32 - 20).max(0) as f32 / 5.0).min(100.0) as u32;
    components.insert("token_estimate".to_string(), token_score);
    if char_count > 200 {
        hints.push(format!("Long prompt ({char_count} chars)"));
    }

    // Reasoning words
    let reasoning_count = count_matches(&RE_REASONING, prompt);
    let reasoning_score = (reasoning_count * 50).min(100) as u32;
    components.insert("reasoning_words".to_string(), reasoning_score);
    if reasoning_count >= 2 {
        hints.push(format!("reasoning_words: {reasoning_count} matches"));
    }

    // Multi-step
    let multi_step_count = count_matches(&RE_MULTI_STEP, prompt);
    let multi_step_score = (multi_step_count * 50).min(100) as u32;
    components.insert("multi_step".to_string(), multi_step_score);
    if multi_step_count >= 2 {
        hints.push(format!("multi_step: {multi_step_count} matches"));
    }

    // Creativity
    let creativity_count = count_matches(&RE_CREATIVITY, prompt);
    let creativity_score = (creativity_count * 50).min(100) as u32;
    components.insert("creativity".to_string(), creativity_score);
    if creativity_count >= 2 {
        hints.push(format!("creativity: {creativity_count} matches"));
    }

    // Precision
    let precision_count = count_matches(&RE_PRECISION, prompt);
    let precision_score = (precision_count * 50).min(100) as u32;
    components.insert("precision".to_string(), precision_score);

    // Code indicators
    let code_count = count_matches(&RE_CODE, prompt);
    let code_score = (code_count * 50).min(100) as u32;
    components.insert("code_indicators".to_string(), code_score);
    if code_count >= 2 {
        hints.push(format!("code_indicators: {code_count} matches"));
    }

    // Tool likelihood
    let tool_count = count_matches(&RE_TOOL, prompt);
    let tool_score = (tool_count * 50).min(100) as u32;
    components.insert("tool_likelihood".to_string(), tool_score);

    // Safety sensitivity
    let safety_count = count_matches(&RE_SAFETY, prompt);
    let safety_score = (safety_count * 50).min(100) as u32;
    components.insert("safety_sensitivity".to_string(), safety_score);
    if safety_count >= 1 {
        hints.push(format!("safety_sensitivity: {safety_count} matches"));
    }

    // Context dependency
    let context_count = count_matches(&RE_CONTEXT, prompt);
    let context_score = (context_count * 50).min(100) as u32;
    components.insert("context_dependency".to_string(), context_score);

    // Domain specific
    let domain_count = count_matches(domain_regex, prompt);
    let domain_score = (domain_count * 50).min(100) as u32;
    components.insert("domain_specific".to_string(), domain_score);
    if domain_count >= 2 {
        hints.push(format!("domain_specific: {domain_count} matches"));
    }

    // Ambiguity (vague pronouns)
    let vague_count = count_matches(&RE_VAGUE, prompt);
    let ambiguity_score = (vague_count * 25).min(100) as u32;
    components.insert("ambiguity".to_string(), ambiguity_score);

    // Question complexity
    let question_marks = prompt.matches('?').count();
    let open_ended_count = count_matches(&RE_OPEN_ENDED, prompt);
    let question_score = ((question_marks * 20) + (open_ended_count * 25)).min(100) as u32;
    components.insert("question_complexity".to_string(), question_score);
    if question_marks >= 2 {
        hints.push(format!("Multiple questions: {question_marks}"));
    }

    // Sentence complexity (commas, semicolons, conjunctions)
    let commas = prompt.matches(',').count();
    let semicolons = prompt.matches(';').count();
    let conjunctions = count_matches(&RE_CONJUNCTIONS, prompt);
    let clauses = commas + (semicolons * 2) + conjunctions;
    let sentence_score = (clauses * 12).min(100) as u32;
    components.insert("sentence_complexity".to_string(), sentence_score);
    if clauses >= 5 {
        hints.push(format!("Complex structure: {clauses} clauses"));
    }

    // Calculate weighted total using data-driven iteration
    let total: f32 = [
        ("reasoning_words", weights.reasoning_words),
        ("token_estimate", weights.token_estimate),
        ("code_indicators", weights.code_indicators),
        ("multi_step", weights.multi_step),
        ("domain_specific", weights.domain_specific),
        ("ambiguity", weights.ambiguity),
        ("creativity", weights.creativity),
        ("precision", weights.precision),
        ("context_dependency", weights.context_dependency),
        ("tool_likelihood", weights.tool_likelihood),
        ("safety_sensitivity", weights.safety_sensitivity),
        ("question_complexity", weights.question_complexity),
        ("sentence_complexity", weights.sentence_complexity),
    ]
    .iter()
    .map(|(name, weight)| components.get(*name).copied().unwrap_or(0) as f32 * weight)
    .sum();

    // Multi-dimensional boost: +30% when 3+ dimensions fire above threshold
    let triggered_dimensions = components.values().filter(|&&v| v > 20).count();
    let total = if triggered_dimensions >= 3 {
        hints.push(format!(
            "Multi-dimensional ({triggered_dimensions} triggers)"
        ));
        total * 1.3
    } else if triggered_dimensions >= 2 {
        total * 1.15
    } else {
        total
    };

    // Clamp to 0-100
    let total = (total as u32).clamp(0, 100);
    let tier = Tier::from_score(total);

    ScoreBreakdown {
        total,
        tier,
        components,
        hints,
    }
}

// -- Metadata key constants --

/// Metadata key for the routing operation type (plan, select_tools, respond, evaluate).
pub const META_ROUTING_OPERATION: &str = "routing_operation";

/// Metadata key for serialized skill routing hints (JSON array of `SkillRoutingHint`).
pub const META_ROUTING_SKILL_HINTS: &str = "routing_skill_hints";

// ---------------------------------------------------------------------------
// TaskComplexity (provider-level classification)
// ---------------------------------------------------------------------------

/// Classification of a request's complexity, determining which model handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    /// Short, simple queries -> cheap model (Flash + Standard tiers)
    Simple,
    /// Ambiguous complexity -> cheap model first, cascade to primary if uncertain (Pro tier)
    Moderate,
    /// Code generation, analysis, multi-step reasoning -> primary model (Frontier tier)
    Complex,
}

impl From<Tier> for TaskComplexity {
    fn from(tier: Tier) -> Self {
        match tier {
            Tier::Flash | Tier::Standard => TaskComplexity::Simple,
            Tier::Pro => TaskComplexity::Moderate,
            Tier::Frontier => TaskComplexity::Complex,
        }
    }
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

// ---------------------------------------------------------------------------
// SmartRoutingConfig & Provider
// ---------------------------------------------------------------------------

/// Configuration for the smart routing provider.
#[derive(Debug, Clone)]
pub struct SmartRoutingConfig {
    /// Enable cascade mode: retry with higher tier if response is uncertain.
    pub cascade_enabled: bool,
    /// Custom domain keywords for the scorer (None uses defaults).
    pub domain_keywords: Option<Vec<String>>,
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
            domain_keywords: None,
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

/// Smart routing provider that classifies task complexity and routes to the
/// appropriate model tier. Supports N tiers with pluggable classification and
/// automatic cascade escalation.
///
/// - `complete()` — scores complexity across 13 dimensions, checks pattern overrides, then
///   routes to cheap or primary model. Moderate tasks use cascade (try cheap, escalate if uncertain).
/// - `complete_with_tools()` — routes via classifier; cascades on no-tool-call or uncertainty.
pub struct SmartRoutingProvider {
    /// Providers ordered cheapest-first. Index 0 = cheapest, last = most capable.
    tiers: Vec<Arc<dyn LlmProvider>>,
    /// Pluggable classifier that picks a starting tier.
    classifier: Arc<dyn MessageClassifier>,
    config: SmartRoutingConfig,
    scorer_config: ScorerConfig,
    /// Pre-compiled domain regex (built once at construction time).
    domain_regex: Regex,
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
        let scorer_config = ScorerConfig {
            weights: ScorerWeights::default(),
            domain_keywords: config.domain_keywords.clone(),
        };
        let domain_regex = match &scorer_config.domain_keywords {
            Some(custom) => {
                let refs: Vec<&str> = custom.iter().map(|s| s.as_str()).collect();
                build_domain_regex(&refs)
            }
            None => RE_DOMAIN_DEFAULT.clone(),
        };
        let stats = SmartRoutingStats::new(tiers.len());
        Self {
            tiers,
            classifier,
            config,
            scorer_config,
            domain_regex,
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

    /// Classify the complexity of a request based on its last user message.
    ///
    /// Priority: explicit tier hints > pattern overrides > 13-dimension scorer.
    fn classify(&self, request: &CompletionRequest) -> TaskComplexity {
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .unwrap_or("");

        // Normalize: trim whitespace so anchored regexes and token scoring are consistent.
        let last_user_msg = last_user_msg.trim();

        // Highest priority: explicit tier hints (e.g. "[tier:flash]")
        if let Some(caps) = RE_TIER_HINT.captures(last_user_msg) {
            let tier_str = caps.get(1).expect("capture group 1 exists").as_str();
            let tier = match tier_str.to_lowercase().as_str() {
                "flash" => Tier::Flash,
                "standard" => Tier::Standard,
                "pro" => Tier::Pro,
                "frontier" => Tier::Frontier,
                other => {
                    tracing::error!(tier = %other, "Unexpected tier in hint despite regex constraint");
                    Tier::Standard
                }
            };
            let complexity = TaskComplexity::from(tier);
            tracing::debug!(
                %tier,
                ?complexity,
                "Smart routing: explicit tier hint"
            );
            return complexity;
        }

        // Fast-path: check pattern overrides
        for po in DEFAULT_OVERRIDES.iter() {
            if po.regex.is_match(last_user_msg) {
                let complexity = TaskComplexity::from(po.tier);
                tracing::debug!(
                    tier = %po.tier,
                    ?complexity,
                    "Smart routing: pattern override matched"
                );
                return complexity;
            }
        }

        // Full 13-dimension scoring (uses pre-compiled domain regex)
        let breakdown = score_complexity_with_regex(
            last_user_msg,
            &self.scorer_config.weights,
            &self.domain_regex,
        );
        let complexity = TaskComplexity::from(breakdown.tier);
        tracing::debug!(
            score = breakdown.total,
            tier = %breakdown.tier,
            ?complexity,
            hints = ?breakdown.hints,
            "Smart routing: scored complexity"
        );
        complexity
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

    fn cache_write_multiplier(&self) -> Decimal {
        self.primary().cache_write_multiplier()
    }

    fn cache_read_discount(&self) -> Decimal {
        self.primary().cache_read_discount()
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

    fn effective_model_name(&self, requested_model: Option<&str>) -> String {
        self.primary.effective_model_name(requested_model)
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

    // -----------------------------------------------------------------------
    // Score complexity: tier boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn score_empty_prompt_is_flash() {
        let result = score_complexity("");
        assert_eq!(result.tier, Tier::Flash);
        assert!(result.total <= 15);
    }

    #[test]
    fn score_simple_greeting_is_flash() {
        let result = score_complexity("Hi");
        assert_eq!(result.tier, Tier::Flash);
        assert!(result.total <= 15);
    }

    #[test]
    fn score_quick_question_is_flash_or_standard() {
        let result = score_complexity("What time is it?");
        assert!(
            result.tier == Tier::Flash || result.tier == Tier::Standard,
            "Expected Flash or Standard, got {:?} (score {})",
            result.tier,
            result.total
        );
    }

    #[test]
    fn score_code_task_is_standard_or_higher() {
        let result = score_complexity("Implement a function to sort an array in TypeScript");
        assert!(
            result.tier == Tier::Standard || result.tier == Tier::Pro,
            "Expected Standard or Pro, got {:?} (score {})",
            result.tier,
            result.total
        );
    }

    #[test]
    fn score_complex_analysis_is_at_least_standard() {
        let result = score_complexity(
            "Explain why React uses a virtual DOM and compare it to Svelte's approach. \
             Consider the trade-offs for performance and developer experience.",
        );
        assert!(
            result.total >= 20,
            "Expected score >= 20, got {}",
            result.total
        );
        assert!(
            result.tier == Tier::Standard || result.tier == Tier::Pro,
            "Expected Standard or Pro, got {:?}",
            result.tier
        );
    }

    #[test]
    fn score_security_audit_prompt_is_at_least_standard() {
        let result = score_complexity(
            "Analyze this Solidity contract for reentrancy vulnerabilities, \
             check for authentication bypass, and provide a security audit report.",
        );
        assert!(
            result.total >= 16,
            "Expected score >= 16, got {}",
            result.total
        );
    }

    // -----------------------------------------------------------------------
    // Score complexity: individual dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn score_reasoning_dimension() {
        let result = score_complexity("Why is this better? Explain the trade-offs and compare");
        let reasoning = result
            .components
            .get("reasoning_words")
            .copied()
            .unwrap_or(0);
        assert!(
            reasoning >= 100,
            "Expected reasoning >= 100, got {reasoning}"
        );
    }

    #[test]
    fn score_multi_step_dimension() {
        let result = score_complexity(
            "First, read the file at src/auth.ts. Then analyze it for security issues. \
             After that, write a detailed report.",
        );
        let multi_step = result.components.get("multi_step").copied().unwrap_or(0);
        assert!(
            multi_step >= 100,
            "Expected multi_step >= 100, got {multi_step}"
        );
        assert!(result.hints.iter().any(|h| h.contains("multi_step")));
    }

    #[test]
    fn score_code_dimension() {
        let result = score_complexity("Fix the bug in the async function, refactor the module");
        let code = result
            .components
            .get("code_indicators")
            .copied()
            .unwrap_or(0);
        assert!(code >= 50, "Expected code_indicators >= 50, got {code}");
    }

    #[test]
    fn score_safety_dimension() {
        let result = score_complexity("Store the password and encrypt the auth token");
        let safety = result
            .components
            .get("safety_sensitivity")
            .copied()
            .unwrap_or(0);
        assert!(safety >= 100, "Expected safety >= 100, got {safety}");
    }

    #[test]
    fn score_domain_dimension() {
        let result = score_complexity("Deploy the kubernetes cluster on aws with terraform");
        let domain = result
            .components
            .get("domain_specific")
            .copied()
            .unwrap_or(0);
        assert!(
            domain >= 100,
            "Expected domain_specific >= 100, got {domain}"
        );
    }

    #[test]
    fn score_creativity_dimension() {
        let result = score_complexity("Write a blog post about design patterns, then summarize");
        let creativity = result.components.get("creativity").copied().unwrap_or(0);
        assert!(
            creativity >= 100,
            "Expected creativity >= 100, got {creativity}"
        );
    }

    #[test]
    fn score_question_complexity_dimension() {
        let result = score_complexity("Why does this fail? How can I fix it? What if I try X?");
        let qc = result
            .components
            .get("question_complexity")
            .copied()
            .unwrap_or(0);
        assert!(qc >= 60, "Expected question_complexity >= 60, got {qc}");
        assert!(
            result
                .hints
                .iter()
                .any(|h| h.contains("Multiple questions"))
        );
    }

    #[test]
    fn score_sentence_complexity_dimension() {
        let result = score_complexity(
            "This is complex, because it has commas, and conjunctions, \
             however it also has semicolons; moreover, it keeps going, and going",
        );
        let sc = result
            .components
            .get("sentence_complexity")
            .copied()
            .unwrap_or(0);
        assert!(sc >= 60, "Expected sentence_complexity >= 60, got {sc}");
    }

    #[test]
    fn score_token_estimate_for_long_prompt() {
        let long_prompt = "a ".repeat(300); // 600 chars
        let result = score_complexity(&long_prompt);
        let token = result
            .components
            .get("token_estimate")
            .copied()
            .unwrap_or(0);
        assert!(token >= 80, "Expected token_estimate >= 80, got {token}");
    }

    #[test]
    fn score_token_estimate_for_short_prompt() {
        let result = score_complexity("hi");
        let token = result
            .components
            .get("token_estimate")
            .copied()
            .unwrap_or(0);
        assert_eq!(token, 0, "Expected token_estimate == 0, got {token}");
    }

    // -----------------------------------------------------------------------
    // Score complexity: multi-dimensional boost
    // -----------------------------------------------------------------------

    #[test]
    fn score_multi_dimensional_boost() {
        // This triggers reasoning, multi-step, code, domain, creativity, safety
        let result = score_complexity(
            "First, explain why the kubernetes deployment fails. \
             Then refactor the auth module to fix the vulnerability. \
             After that, write a security report comparing the approaches.",
        );
        assert!(
            result.hints.iter().any(|h| h.contains("Multi-dimensional")),
            "Expected multi-dimensional boost, hints: {:?}",
            result.hints
        );
    }

    // -----------------------------------------------------------------------
    // Score complexity: explicit tier hint
    // -----------------------------------------------------------------------

    #[test]
    fn score_explicit_tier_hint_flash() {
        let result = score_complexity("[tier:flash] This looks complex but override to flash");
        assert_eq!(result.tier, Tier::Flash);
        assert!(
            result
                .hints
                .iter()
                .any(|h| h.contains("Explicit tier hint"))
        );
    }

    #[test]
    fn score_explicit_tier_hint_frontier() {
        let result = score_complexity("[tier:frontier] Simple question but I want the best");
        assert_eq!(result.tier, Tier::Frontier);
    }

    #[test]
    fn score_explicit_tier_hint_case_insensitive() {
        let result = score_complexity("[tier:PRO] some message");
        assert_eq!(result.tier, Tier::Pro);
    }

    // -----------------------------------------------------------------------
    // Score complexity: custom domain keywords
    // -----------------------------------------------------------------------

    #[test]
    fn score_custom_domain_keywords_override_defaults() {
        // Default keywords should match "kubernetes"
        let default_result = score_complexity("How do I deploy kubernetes?");
        let default_domain = default_result
            .components
            .get("domain_specific")
            .copied()
            .unwrap_or(0);
        assert!(
            default_domain > 0,
            "Default keywords should match 'kubernetes'"
        );

        // Custom keywords that DON'T include kubernetes
        let config = ScorerConfig {
            weights: ScorerWeights::default(),
            domain_keywords: Some(vec!["mycompany".to_string(), "myproduct".to_string()]),
        };
        let custom_result = score_complexity_with_config("How do I deploy kubernetes?", &config);
        let custom_domain = custom_result
            .components
            .get("domain_specific")
            .copied()
            .unwrap_or(0);
        assert_eq!(
            custom_domain, 0,
            "Custom keywords shouldn't match 'kubernetes'"
        );

        // Custom keywords should match their own terms
        let custom_result2 =
            score_complexity_with_config("Tell me about myproduct features", &config);
        let custom_domain2 = custom_result2
            .components
            .get("domain_specific")
            .copied()
            .unwrap_or(0);
        assert!(
            custom_domain2 > 0,
            "Custom keywords should match 'myproduct'"
        );
    }

    // -----------------------------------------------------------------------
    // Score complexity: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn score_whitespace_only_is_flash() {
        let result = score_complexity("   \n\t  ");
        assert_eq!(result.tier, Tier::Flash);
    }

    #[test]
    fn score_single_word_no_keywords() {
        let result = score_complexity("banana");
        assert!(
            result.tier == Tier::Flash || result.tier == Tier::Standard,
            "Single non-keyword word should be Flash or Standard, got {:?}",
            result.tier
        );
    }

    #[test]
    fn score_very_long_prompt_is_at_least_standard() {
        let long = "Tell me about ".to_string() + &"things ".repeat(200);
        let result = score_complexity(&long);
        assert!(
            result.total >= 16,
            "Very long prompt should score at least Standard, got {}",
            result.total
        );
    }

    #[test]
    fn score_all_dimensions_have_entries() {
        let result = score_complexity(
            "First, explain why the function fails. Then write a fix and deploy it.",
        );
        let expected_keys = [
            "reasoning_words",
            "token_estimate",
            "code_indicators",
            "multi_step",
            "domain_specific",
            "ambiguity",
            "creativity",
            "precision",
            "context_dependency",
            "tool_likelihood",
            "safety_sensitivity",
            "question_complexity",
            "sentence_complexity",
        ];
        for key in &expected_keys {
            assert!(
                result.components.contains_key(*key),
                "Missing component: {key}"
            );
        }
    }

    #[test]
    fn score_is_clamped_to_100() {
        // Trigger every dimension hard
        let prompt = "First, explain why the kubernetes docker terraform deployment on aws fails. \
             Then analyze the security vulnerability and compare the trade-offs. \
             After that, write a detailed blog post report with code examples: \
             ```rust\nfn main() {}\n``` \
             Calculate exactly how many steps are needed? Why? How? \
             Deploy to production mainnet. Review the authentication token password.";
        let result = score_complexity(prompt);
        assert!(
            result.total <= 100,
            "Score should be clamped to 100, got {}",
            result.total
        );
    }

    // -----------------------------------------------------------------------
    // Pattern overrides
    // -----------------------------------------------------------------------

    #[test]
    fn pattern_override_greeting_is_simple() {
        let primary = Arc::new(StubLlm::new("p").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("c").with_model_name("cheap"));
        let provider = SmartRoutingProvider::new(primary, cheap, default_config());

        let req = CompletionRequest::new(vec![ChatMessage::user("Hi")]);
        let complexity = provider.classify(&req);
        assert_eq!(complexity, TaskComplexity::Simple);
    }

    #[test]
    fn pattern_override_security_audit_is_complex() {
        let primary = Arc::new(StubLlm::new("p").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("c").with_model_name("cheap"));
        let provider = SmartRoutingProvider::new(primary, cheap, default_config());

        let req = CompletionRequest::new(vec![ChatMessage::user(
            "Please do a security audit of this contract",
        )]);
        let complexity = provider.classify(&req);
        assert_eq!(complexity, TaskComplexity::Complex);
    }

    #[test]
    fn pattern_override_production_deploy_is_moderate() {
        let primary = Arc::new(StubLlm::new("p").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("c").with_model_name("cheap"));
        let provider = SmartRoutingProvider::new(primary, cheap, default_config());

        let req = CompletionRequest::new(vec![ChatMessage::user("Deploy this to production")]);
        let complexity = provider.classify(&req);
        assert_eq!(complexity, TaskComplexity::Moderate);
    }

    #[test]
    fn pattern_override_time_question_is_simple() {
        let primary = Arc::new(StubLlm::new("p").with_model_name("primary"));
        let cheap = Arc::new(StubLlm::new("c").with_model_name("cheap"));
        let provider = SmartRoutingProvider::new(primary, cheap, default_config());

        let req = CompletionRequest::new(vec![ChatMessage::user("What time is it?")]);
        let complexity = provider.classify(&req);
        assert_eq!(complexity, TaskComplexity::Simple);
    }

    #[test]
    fn pattern_override_time_does_not_match_complex_questions() {
        // The quick-lookup override regex should NOT match "What time complexity..."
        // because it's end-anchored. Verify the regex itself doesn't fire.
        let overrides = &*DEFAULT_OVERRIDES;
        let lookup_override = overrides
            .iter()
            .find(|po| po.tier == Tier::Flash && po.regex.as_str().contains("time"))
            .expect("time lookup override exists");

        assert!(
            !lookup_override
                .regex
                .is_match("What time complexity is merge sort?"),
            "Time override should not match 'What time complexity is merge sort?'"
        );
        // But it should still match actual time lookups
        assert!(lookup_override.regex.is_match("What time is it?"));
        assert!(lookup_override.regex.is_match("what's the date today?"));
    }

    #[test]
    fn empty_domain_keywords_uses_defaults() {
        // An empty custom keywords list should fall back to defaults, not produce
        // a broken regex that matches empty strings everywhere.
        let config = ScorerConfig {
            domain_keywords: Some(vec![]),
            ..ScorerConfig::default()
        };
        let result = score_complexity_with_config("deploy kubernetes to mainnet", &config);
        // Should still detect domain keywords via the default fallback
        assert!(
            result
                .components
                .get("domain_specific")
                .copied()
                .unwrap_or(0)
                > 0,
            "Empty custom keywords should fall back to defaults"
        );
    }

    // -----------------------------------------------------------------------
    // Tier → TaskComplexity mapping
    // -----------------------------------------------------------------------

    #[test]
    fn tier_to_task_complexity_mapping() {
        assert_eq!(TaskComplexity::from(Tier::Flash), TaskComplexity::Simple);
        assert_eq!(TaskComplexity::from(Tier::Standard), TaskComplexity::Simple);
        assert_eq!(TaskComplexity::from(Tier::Pro), TaskComplexity::Moderate);
        assert_eq!(
            TaskComplexity::from(Tier::Frontier),
            TaskComplexity::Complex
        );
    }

    #[test]
    fn tier_from_score_boundaries() {
        assert_eq!(Tier::from_score(0), Tier::Flash);
        assert_eq!(Tier::from_score(15), Tier::Flash);
        assert_eq!(Tier::from_score(16), Tier::Standard);
        assert_eq!(Tier::from_score(40), Tier::Standard);
        assert_eq!(Tier::from_score(41), Tier::Pro);
        assert_eq!(Tier::from_score(65), Tier::Pro);
        assert_eq!(Tier::from_score(66), Tier::Frontier);
        assert_eq!(Tier::from_score(100), Tier::Frontier);
    }

    #[test]
    fn tier_display() {
        assert_eq!(Tier::Flash.as_str(), "flash");
        assert_eq!(Tier::Frontier.to_string(), "frontier");
    }

    // -----------------------------------------------------------------------
    // Uncertainty detection
    // -----------------------------------------------------------------------
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
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
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
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
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
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
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
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        };
        assert!(!SmartRoutingProvider::response_is_uncertain(&response));
    }

    // -----------------------------------------------------------------------
    // Provider routing tests
    // -----------------------------------------------------------------------
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

        // Security audit triggers Frontier via pattern override → Complex → primary
        let resp = router
            .complete(make_request(
                "Please do a security audit of this smart contract",
            ))
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

        // Simple → cheap (greeting pattern override)
        router.complete(make_request("hello")).await.unwrap();
        // Complex → primary (security audit pattern override → Frontier)
        router
            .complete(make_request("security audit review"))
            .await
            .unwrap();
        // Tool use → primary
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

        // A Pro-tier task (triggers Moderate → cascade)
        let resp = router
            .complete(make_request("Deploy this to production"))
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
            StubLlm::new("Deployed successfully to production mainnet.").with_model_name("cheap"),
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
            .complete(make_request("Deploy this to production"))
            .await
            .unwrap();

        // Should NOT have escalated
        assert!(resp.content.contains("Deployed successfully"));
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

    #[tokio::test]
    async fn tier_hint_overrides_pattern_override() {
        // "[tier:flash] security audit review" has both a Flash tier hint and
        // a Frontier pattern override. Tier hints should win.
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

        let resp = router
            .complete(make_request("[tier:flash] security audit review"))
            .await
            .unwrap();

        // Tier hint -> Flash -> Simple -> cheap model
        assert_eq!(resp.content, "cheap");
        assert_eq!(cheap.calls(), 1);
        assert_eq!(primary.calls(), 0);
    }

    #[tokio::test]
    async fn trimmed_greeting_matches_override() {
        // Trailing whitespace should not prevent the greeting override from matching.
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

        let resp = router
            .complete(make_request("  hello  \n"))
            .await
            .unwrap();

        // Should match greeting override -> Flash -> Simple -> cheap model
        assert_eq!(resp.content, "cheap");
        assert_eq!(cheap.calls(), 1);
        assert_eq!(primary.calls(), 0);
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

        let resp = router
            .complete(make_request("hello"))
            .await
            .unwrap();

        // Simple greeting -> tier 0 (cheapest)
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
