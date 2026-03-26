use std::time::Duration;

use crate::config::helpers::{parse_bool_env, parse_option_env, parse_optional_env};
use crate::error::ConfigError;
use crate::settings::Settings;

/// Agent behavior configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub name: String,
    pub max_parallel_jobs: usize,
    pub job_timeout: Duration,
    pub stuck_threshold: Duration,
    pub repair_check_interval: Duration,
    pub max_repair_attempts: u32,
    /// Whether to use planning before tool execution.
    pub use_planning: bool,
    /// Session idle timeout. Sessions inactive longer than this are pruned.
    pub session_idle_timeout: Duration,
    /// Allow chat to use filesystem/shell tools directly (bypass sandbox).
    pub allow_local_tools: bool,
    /// Maximum daily LLM spend in cents (e.g. 10000 = $100). None = unlimited.
    pub max_cost_per_day_cents: Option<u64>,
    /// Maximum LLM/tool actions per hour. None = unlimited.
    pub max_actions_per_hour: Option<u64>,
    /// Maximum tool-call iterations per agentic loop invocation. Default 50.
    pub max_tool_iterations: usize,
    /// When true, skip tool approval checks entirely. For benchmarks/CI.
    pub auto_approve_tools: bool,
    /// Default timezone for new sessions (IANA name, e.g. "America/New_York").
    pub default_timezone: String,
    /// Maximum tokens per job (0 = unlimited).
    pub max_tokens_per_job: u64,
    /// How tool definitions are sent to the LLM.
    ///
    /// `Compressed` (default): all tools visible with name+description only; full
    /// schemas fetched on demand when a tool is called.
    /// `Full`: all tools visible with complete parameter schemas (legacy behavior).
    pub tool_description_mode: ToolDescriptionMode,
}

/// Controls how tool parameter schemas are presented to the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDescriptionMode {
    /// Name + description only; schemas fetched on demand.
    Compressed,
    /// Full parameter schemas included in every LLM request (legacy).
    Full,
}

impl AgentConfig {
    /// Create a test-friendly config without reading env vars.
    #[cfg(feature = "libsql")]
    pub fn for_testing() -> Self {
        Self {
            name: "test-rig".to_string(),
            max_parallel_jobs: 1,
            job_timeout: Duration::from_secs(30),
            stuck_threshold: Duration::from_secs(300),
            repair_check_interval: Duration::from_secs(3600),
            max_repair_attempts: 0,
            use_planning: false,
            session_idle_timeout: Duration::from_secs(3600),
            allow_local_tools: true,
            max_cost_per_day_cents: None,
            max_actions_per_hour: None,
            max_tool_iterations: 10,
            auto_approve_tools: true,
            default_timezone: "UTC".to_string(),
            max_tokens_per_job: 0,
            tool_description_mode: ToolDescriptionMode::Compressed,
        }
    }

    pub(crate) fn resolve(settings: &Settings) -> Result<Self, ConfigError> {
        Ok(Self {
            name: parse_optional_env("AGENT_NAME", settings.agent.name.clone())?,
            max_parallel_jobs: parse_optional_env(
                "AGENT_MAX_PARALLEL_JOBS",
                settings.agent.max_parallel_jobs as usize,
            )?,
            job_timeout: Duration::from_secs(parse_optional_env(
                "AGENT_JOB_TIMEOUT_SECS",
                settings.agent.job_timeout_secs,
            )?),
            stuck_threshold: Duration::from_secs(parse_optional_env(
                "AGENT_STUCK_THRESHOLD_SECS",
                settings.agent.stuck_threshold_secs,
            )?),
            repair_check_interval: Duration::from_secs(parse_optional_env(
                "SELF_REPAIR_CHECK_INTERVAL_SECS",
                settings.agent.repair_check_interval_secs,
            )?),
            max_repair_attempts: parse_optional_env(
                "SELF_REPAIR_MAX_ATTEMPTS",
                settings.agent.max_repair_attempts,
            )?,
            use_planning: parse_bool_env("AGENT_USE_PLANNING", settings.agent.use_planning)?,
            session_idle_timeout: Duration::from_secs(parse_optional_env(
                "SESSION_IDLE_TIMEOUT_SECS",
                settings.agent.session_idle_timeout_secs,
            )?),
            allow_local_tools: parse_bool_env("ALLOW_LOCAL_TOOLS", false)?,
            max_cost_per_day_cents: parse_option_env("MAX_COST_PER_DAY_CENTS")?,
            max_actions_per_hour: parse_option_env("MAX_ACTIONS_PER_HOUR")?,
            max_tool_iterations: parse_optional_env(
                "AGENT_MAX_TOOL_ITERATIONS",
                settings.agent.max_tool_iterations,
            )?,
            auto_approve_tools: parse_bool_env(
                "AGENT_AUTO_APPROVE_TOOLS",
                settings.agent.auto_approve_tools,
            )?,
            default_timezone: {
                let tz: String = parse_optional_env(
                    "DEFAULT_TIMEZONE",
                    settings.agent.default_timezone.clone(),
                )?;
                if crate::timezone::parse_timezone(&tz).is_none() {
                    return Err(ConfigError::InvalidValue {
                        key: "DEFAULT_TIMEZONE".into(),
                        message: format!("invalid IANA timezone: '{tz}'"),
                    });
                }
                tz
            },
            max_tokens_per_job: parse_optional_env(
                "AGENT_MAX_TOKENS_PER_JOB",
                settings.agent.max_tokens_per_job,
            )?,
            tool_description_mode: match std::env::var("TOOL_DESCRIPTION_MODE")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str()
            {
                "full" => ToolDescriptionMode::Full,
                _ => ToolDescriptionMode::Compressed,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_timezone_rejects_invalid() {
        let mut settings = Settings::default();
        settings.agent.default_timezone = "Fake/Zone".to_string();

        let result = AgentConfig::resolve(&settings);
        assert!(result.is_err(), "invalid IANA timezone should be rejected");
    }

    #[test]
    fn test_default_timezone_accepts_valid() {
        let settings = Settings::default(); // default is "UTC"
        let config = AgentConfig::resolve(&settings).expect("resolve");
        assert_eq!(config.default_timezone, "UTC");
    }

    #[test]
    fn test_tool_description_mode_from_env() {
        // SAFETY: test-only; modifies process env for config resolution.
        // Combined into one test to avoid parallel env-var races.
        let settings = Settings::default();

        // Default (unset) → Compressed
        unsafe {
            std::env::remove_var("TOOL_DESCRIPTION_MODE");
        }
        let config = AgentConfig::resolve(&settings).expect("resolve default");
        assert_eq!(
            config.tool_description_mode,
            ToolDescriptionMode::Compressed,
            "should default to Compressed when TOOL_DESCRIPTION_MODE is unset"
        );

        // Explicit "full" → Full
        unsafe {
            std::env::set_var("TOOL_DESCRIPTION_MODE", "full");
        }
        let config = AgentConfig::resolve(&settings).expect("resolve full");
        assert_eq!(config.tool_description_mode, ToolDescriptionMode::Full);

        // Clean up
        unsafe {
            std::env::remove_var("TOOL_DESCRIPTION_MODE");
        }
    }
}
