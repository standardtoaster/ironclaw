use std::path::PathBuf;

use secrecy::SecretString;

use crate::bootstrap::ironclaw_base_dir;
use crate::config::helpers::{optional_env, parse_optional_env};
use crate::error::ConfigError;
use crate::llm::config::*;
use crate::llm::registry::{ProviderProtocol, ProviderRegistry};
use crate::llm::session::SessionConfig;
use crate::settings::Settings;

/// Per-tier configuration for N-tier model routing.
///
/// Each tier represents a backend+model combination ordered by cost (cheapest first).
/// Parsed from `LLM_ROUTING_TIERS` + per-tier env vars.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Tier name (e.g., "local", "standard", "premium").
    pub name: String,
    /// Which backend to use for this tier.
    pub backend: LlmBackend,
    /// Model identifier.
    pub model: String,
    /// Optional base URL override.
    pub base_url: Option<String>,
    /// Optional API key.
    pub api_key: Option<SecretString>,
}

/// Which LLM backend to use. Configurable per-user via `GATEWAY_USER_TOKENS`.
///
/// Users can override with `LLM_BACKEND` env var to use their own API keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmBackend {
    /// NEAR AI proxy (default) -- session or API key auth
    #[default]
    NearAi,
    /// Direct OpenAI API
    OpenAi,
    /// Direct Anthropic API
    Anthropic,
    /// Local Ollama instance
    Ollama,
    /// Any OpenAI-compatible endpoint (e.g. vLLM, LiteLLM, Together)
    OpenAiCompatible,
    /// Tinfoil private inference
    Tinfoil,
}

impl std::str::FromStr for LlmBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nearai" | "near_ai" | "near" => Ok(Self::NearAi),
            "openai" | "open_ai" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "ollama" => Ok(Self::Ollama),
            "openai_compatible" | "openai-compatible" | "compatible" => Ok(Self::OpenAiCompatible),
            "tinfoil" => Ok(Self::Tinfoil),
            _ => Err(format!(
                "invalid LLM backend '{}', expected one of: nearai, openai, anthropic, ollama, openai_compatible, tinfoil",
                s
            )),
        }
    }
}

impl std::fmt::Display for LlmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NearAi => write!(f, "nearai"),
            Self::OpenAi => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Ollama => write!(f, "ollama"),
            Self::OpenAiCompatible => write!(f, "openai_compatible"),
            Self::Tinfoil => write!(f, "tinfoil"),
        }
    }
}

impl LlmConfig {
    /// Create a test-friendly config without reading env vars.
    #[cfg(feature = "libsql")]
    pub fn for_testing() -> Self {
        Self {
            backend: "nearai".to_string(),
            session: SessionConfig {
                auth_base_url: "http://localhost:0".to_string(),
                session_path: std::env::temp_dir().join("ironclaw-test-session.json"),
            },
            nearai: NearAiConfig {
                model: "test-model".to_string(),
                cheap_model: None,
                base_url: "http://localhost:0".to_string(),
                api_key: None,
                fallback_model: None,
                max_retries: 0,
                circuit_breaker_threshold: None,
                circuit_breaker_recovery_secs: 30,
                response_cache_enabled: false,
                response_cache_ttl_secs: 3600,
                response_cache_max_entries: 100,
                failover_cooldown_secs: 300,
                failover_cooldown_threshold: 3,
                smart_routing_cascade: false,
            },
            provider: None,
            bedrock: None,
            request_timeout_secs: 120,
            routing_tiers: Vec::new(),
        }
    }

    /// Resolve a model name from env var -> settings.selected_model -> hardcoded default.
    fn resolve_model(
        env_var: &str,
        settings: &Settings,
        default: &str,
    ) -> Result<String, ConfigError> {
        Ok(optional_env(env_var)?
            .or_else(|| settings.selected_model.clone())
            .unwrap_or_else(|| default.to_string()))
    }

    pub(crate) fn resolve(settings: &Settings) -> Result<Self, ConfigError> {
        let registry = ProviderRegistry::load();

        // Determine backend: env var > settings > default ("nearai")
        let backend = if let Some(b) = optional_env("LLM_BACKEND")? {
            b
        } else if let Some(ref b) = settings.llm_backend {
            b.clone()
        } else {
            "nearai".to_string()
        };

        // Validate the backend is known
        let backend_lower = backend.to_lowercase();
        let is_nearai =
            backend_lower == "nearai" || backend_lower == "near_ai" || backend_lower == "near";
        let is_bedrock =
            backend_lower == "bedrock" || backend_lower == "aws_bedrock" || backend_lower == "aws";

        if !is_nearai && !is_bedrock && registry.find(&backend_lower).is_none() {
            tracing::warn!(
                "Unknown LLM backend '{}'. Will attempt as openai_compatible fallback.",
                backend
            );
        }

        // Session config (used by NearAI provider for OAuth/session-token auth)
        let session = SessionConfig {
            auth_base_url: optional_env("NEARAI_AUTH_URL")?
                .unwrap_or_else(|| "https://private.near.ai".to_string()),
            session_path: optional_env("NEARAI_SESSION_PATH")?
                .map(PathBuf::from)
                .unwrap_or_else(default_session_path),
        };

        // Always resolve NEAR AI config (used for embeddings even when not the primary backend)
        let nearai_api_key = optional_env("NEARAI_API_KEY")?.map(SecretString::from);
        let nearai = NearAiConfig {
            model: Self::resolve_model("NEARAI_MODEL", settings, "zai-org/GLM-latest")?,
            cheap_model: optional_env("NEARAI_CHEAP_MODEL")?,
            base_url: optional_env("NEARAI_BASE_URL")?.unwrap_or_else(|| {
                if nearai_api_key.is_some() {
                    "https://cloud-api.near.ai".to_string()
                } else {
                    "https://private.near.ai".to_string()
                }
            }),
            api_key: nearai_api_key,
            fallback_model: optional_env("NEARAI_FALLBACK_MODEL")?,
            max_retries: parse_optional_env("NEARAI_MAX_RETRIES", 3)?,
            circuit_breaker_threshold: optional_env("CIRCUIT_BREAKER_THRESHOLD")?
                .map(|s| s.parse())
                .transpose()
                .map_err(|e| ConfigError::InvalidValue {
                    key: "CIRCUIT_BREAKER_THRESHOLD".to_string(),
                    message: format!("must be a positive integer: {e}"),
                })?,
            circuit_breaker_recovery_secs: parse_optional_env("CIRCUIT_BREAKER_RECOVERY_SECS", 30)?,
            response_cache_enabled: parse_optional_env("RESPONSE_CACHE_ENABLED", false)?,
            response_cache_ttl_secs: parse_optional_env("RESPONSE_CACHE_TTL_SECS", 3600)?,
            response_cache_max_entries: parse_optional_env("RESPONSE_CACHE_MAX_ENTRIES", 1000)?,
            failover_cooldown_secs: parse_optional_env("LLM_FAILOVER_COOLDOWN_SECS", 300)?,
            failover_cooldown_threshold: parse_optional_env("LLM_FAILOVER_THRESHOLD", 3)?,
            smart_routing_cascade: parse_optional_env("SMART_ROUTING_CASCADE", true)?,
        };

        // Resolve registry provider config (for non-NearAI, non-Bedrock backends)
        let provider = if is_nearai || is_bedrock {
            None
        } else {
            Some(Self::resolve_registry_provider(
                &backend_lower,
                &registry,
                settings,
            )?)
        };

        let bedrock = if is_bedrock {
            let explicit_region =
                optional_env("BEDROCK_REGION")?.or_else(|| settings.bedrock_region.clone());
            if explicit_region.is_none() {
                tracing::info!("BEDROCK_REGION not set, defaulting to us-east-1");
            }
            let region = explicit_region.unwrap_or_else(|| "us-east-1".to_string());
            let model = optional_env("BEDROCK_MODEL")?
                .or_else(|| settings.selected_model.clone())
                .ok_or_else(|| ConfigError::MissingRequired {
                    key: "BEDROCK_MODEL".to_string(),
                    hint: "Set BEDROCK_MODEL when LLM_BACKEND=bedrock".to_string(),
                })?;
            let cross_region = optional_env("BEDROCK_CROSS_REGION")?
                .or_else(|| settings.bedrock_cross_region.clone());
            if let Some(ref cr) = cross_region
                && !matches!(cr.as_str(), "us" | "eu" | "apac" | "global")
            {
                return Err(ConfigError::InvalidValue {
                    key: "BEDROCK_CROSS_REGION".to_string(),
                    message: format!(
                        "'{}' is not valid, expected one of: us, eu, apac, global",
                        cr
                    ),
                });
            }
            let profile = optional_env("AWS_PROFILE")?.or_else(|| settings.bedrock_profile.clone());
            Some(BedrockConfig {
                region,
                model,
                cross_region,
                profile,
            })
        } else {
            None
        };

        let request_timeout_secs = parse_optional_env("LLM_REQUEST_TIMEOUT_SECS", 120)?;

        // N-tier routing: LLM_ROUTING_TIERS=local,standard,premium
        let routing_tiers = parse_routing_tiers()?;

        Ok(Self {
            backend: if is_nearai {
                "nearai".to_string()
            } else if is_bedrock {
                "bedrock".to_string()
            } else if let Some(ref p) = provider {
                p.provider_id.clone()
            } else {
                backend_lower
            },
            session,
            nearai,
            provider,
            bedrock,
            request_timeout_secs,
            routing_tiers,
        })
    }

    /// Resolve a `RegistryProviderConfig` from the registry and env vars.
    fn resolve_registry_provider(
        backend: &str,
        registry: &ProviderRegistry,
        settings: &Settings,
    ) -> Result<RegistryProviderConfig, ConfigError> {
        // Look up provider definition. Fall back to openai_compatible if unknown.
        let def = registry
            .find(backend)
            .or_else(|| registry.find("openai_compatible"));

        let (
            canonical_id,
            protocol,
            api_key_env,
            base_url_env,
            model_env,
            default_model,
            default_base_url,
            extra_headers_env,
            api_key_required,
            base_url_required,
            unsupported_params,
        ) = if let Some(def) = def {
            (
                def.id.as_str(),
                def.protocol,
                def.api_key_env.as_deref(),
                def.base_url_env.as_deref(),
                def.model_env.as_str(),
                def.default_model.as_str(),
                def.default_base_url.as_deref(),
                def.extra_headers_env.as_deref(),
                def.api_key_required,
                def.base_url_required,
                def.unsupported_params.clone(),
            )
        } else {
            // Absolute fallback: treat as generic openai_completions
            (
                backend,
                ProviderProtocol::OpenAiCompletions,
                Some("LLM_API_KEY"),
                Some("LLM_BASE_URL"),
                "LLM_MODEL",
                "default",
                None,
                Some("LLM_EXTRA_HEADERS"),
                false,
                true,
                Vec::new(),
            )
        };

        // Resolve API key from env
        let api_key = if let Some(env_var) = api_key_env {
            optional_env(env_var)?.map(SecretString::from)
        } else {
            None
        };

        if api_key_required && api_key.is_none() {
            // Don't hard-fail here. The key might be injected later from the secrets store
            // via inject_llm_keys_from_secrets(). Log a warning instead.
            if let Some(env_var) = api_key_env {
                tracing::debug!(
                    "API key not found in {env_var} for backend '{backend}'. \
                     Will be injected from secrets store if available."
                );
            }
        }

        // Resolve base URL: env var > settings (backward compat) > registry default
        let base_url = if let Some(env_var) = base_url_env {
            optional_env(env_var)?
        } else {
            None
        }
        .or_else(|| {
            // Backward compat: check legacy settings fields
            match backend {
                "ollama" => settings.ollama_base_url.clone(),
                "openai_compatible" | "openrouter" => settings.openai_compatible_base_url.clone(),
                _ => None,
            }
        })
        .or_else(|| default_base_url.map(String::from))
        .unwrap_or_default();

        if base_url_required
            && base_url.is_empty()
            && let Some(env_var) = base_url_env
        {
            return Err(ConfigError::MissingRequired {
                key: env_var.to_string(),
                hint: format!("Set {env_var} when LLM_BACKEND={backend}"),
            });
        }

        // Resolve model
        let model = Self::resolve_model(model_env, settings, default_model)?;

        // Resolve extra headers
        let extra_headers = if let Some(env_var) = extra_headers_env {
            optional_env(env_var)?
                .map(|val| parse_extra_headers(&val))
                .transpose()?
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Resolve OAuth token (Anthropic-specific: `claude login` flow).
        // Only check for OAuth token when the provider is actually Anthropic.
        let oauth_token = if canonical_id == "anthropic" {
            optional_env("ANTHROPIC_OAUTH_TOKEN")?.map(SecretString::from)
        } else {
            None
        };
        let api_key = if api_key.is_none() && oauth_token.is_some() {
            // OAuth token present but no API key: use a placeholder so the
            // config block is populated. The provider factory will route to
            // the OAuth provider instead of rig-core's x-api-key client.
            Some(SecretString::from(OAUTH_PLACEHOLDER.to_string()))
        } else {
            api_key
        };

        // Resolve Anthropic prompt cache retention from env (default: Short).
        let cache_retention: CacheRetention = if canonical_id == "anthropic" {
            optional_env("ANTHROPIC_CACHE_RETENTION")?
                .and_then(|val| match val.parse::<CacheRetention>() {
                    Ok(r) => Some(r),
                    Err(e) => {
                        tracing::warn!(
                            "Invalid ANTHROPIC_CACHE_RETENTION: {e}; defaulting to short"
                        );
                        None
                    }
                })
                .unwrap_or_default()
        } else {
            CacheRetention::default()
        };

        Ok(RegistryProviderConfig {
            protocol,
            provider_id: canonical_id.to_string(),
            api_key,
            base_url,
            model,
            extra_headers,
            oauth_token,
            cache_retention,
            unsupported_params,
        })
    }
}

/// Parse N-tier routing config from env vars.
///
/// Format:
/// ```text
/// LLM_ROUTING_TIERS=local,standard,premium
/// LLM_TIER_LOCAL_BACKEND=ollama
/// LLM_TIER_LOCAL_MODEL=llama3.1:8b
/// LLM_TIER_STANDARD_BACKEND=openai_compatible
/// LLM_TIER_STANDARD_MODEL=openai/gpt-4o-mini
/// LLM_TIER_STANDARD_BASE_URL=https://openrouter.ai/api/v1
/// LLM_TIER_STANDARD_API_KEY=sk-or-...
/// ```
fn parse_routing_tiers() -> Result<Vec<TierConfig>, ConfigError> {
    let Some(tiers_str) = optional_env("LLM_ROUTING_TIERS")? else {
        return Ok(Vec::new());
    };

    let tier_names: Vec<&str> = tiers_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if tier_names.len() < 2 {
        return Err(ConfigError::InvalidValue {
            key: "LLM_ROUTING_TIERS".to_string(),
            message: "at least 2 tiers are required for N-tier routing".to_string(),
        });
    }

    let mut tiers = Vec::with_capacity(tier_names.len());
    for name in tier_names {
        let prefix = format!("LLM_TIER_{}", name.to_uppercase());

        let backend_key = format!("{prefix}_BACKEND");
        let backend_str = optional_env(&backend_key)?.ok_or_else(|| ConfigError::MissingRequired {
            key: backend_key.clone(),
            hint: format!("Each tier in LLM_ROUTING_TIERS needs {prefix}_BACKEND"),
        })?;
        let backend: LlmBackend = backend_str.parse().map_err(|e| ConfigError::InvalidValue {
            key: backend_key,
            message: e,
        })?;

        let model_key = format!("{prefix}_MODEL");
        let model = optional_env(&model_key)?.ok_or_else(|| ConfigError::MissingRequired {
            key: model_key,
            hint: format!("Each tier in LLM_ROUTING_TIERS needs {prefix}_MODEL"),
        })?;

        let base_url = optional_env(&format!("{prefix}_BASE_URL"))?;
        let api_key = optional_env(&format!("{prefix}_API_KEY"))?.map(SecretString::from);

        tiers.push(TierConfig {
            name: name.to_string(),
            backend,
            model,
            base_url,
            api_key,
        });
    }

    Ok(tiers)
}

/// Parse `LLM_EXTRA_HEADERS` value into a list of (key, value) pairs.
///
/// Format: `Key1:Value1,Key2:Value2` (colon-separated, not `=`, because
/// header values often contain `=`).
fn parse_extra_headers(val: &str) -> Result<Vec<(String, String)>, ConfigError> {
    if val.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut headers = Vec::new();
    for pair in val.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let Some((key, value)) = pair.split_once(':') else {
            return Err(ConfigError::InvalidValue {
                key: "LLM_EXTRA_HEADERS".to_string(),
                message: format!("malformed header entry '{}', expected Key:Value", pair),
            });
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(ConfigError::InvalidValue {
                key: "LLM_EXTRA_HEADERS".to_string(),
                message: format!("empty header name in entry '{}'", pair),
            });
        }
        headers.push((key.to_string(), value.trim().to_string()));
    }
    Ok(headers)
}

/// Get the default session file path (~/.ironclaw/session.json).
pub fn default_session_path() -> PathBuf {
    ironclaw_base_dir().join("session.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::helpers::ENV_MUTEX;
    use crate::settings::Settings;

    /// Clear all openai-compatible-related env vars.
    fn clear_openai_compatible_env() {
        // SAFETY: Only called under ENV_MUTEX in tests.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("LLM_BASE_URL");
            std::env::remove_var("LLM_MODEL");
        }
    }

    #[test]
    fn openai_compatible_uses_selected_model_when_llm_model_unset() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();

        let settings = Settings {
            llm_backend: Some("openai_compatible".to_string()),
            openai_compatible_base_url: Some("https://openrouter.ai/api/v1".to_string()),
            selected_model: Some("openai/gpt-5.1-codex".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(provider.model, "openai/gpt-5.1-codex");
    }

    #[test]
    fn openai_compatible_llm_model_env_overrides_selected_model() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("LLM_MODEL", "openai/gpt-5-codex");
        }

        let settings = Settings {
            llm_backend: Some("openai_compatible".to_string()),
            openai_compatible_base_url: Some("https://openrouter.ai/api/v1".to_string()),
            selected_model: Some("openai/gpt-5.1-codex".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(provider.model, "openai/gpt-5-codex");

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_MODEL");
        }
    }

    #[test]
    fn test_extra_headers_parsed() {
        let result = parse_extra_headers("HTTP-Referer:https://myapp.com,X-Title:MyApp").unwrap();
        assert_eq!(
            result,
            vec![
                ("HTTP-Referer".to_string(), "https://myapp.com".to_string()),
                ("X-Title".to_string(), "MyApp".to_string()),
            ]
        );
    }

    #[test]
    fn test_extra_headers_empty_string() {
        let result = parse_extra_headers("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_extra_headers_whitespace_only() {
        let result = parse_extra_headers("  ").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_extra_headers_malformed() {
        let result = parse_extra_headers("NoColonHere");
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_headers_empty_key() {
        let result = parse_extra_headers(":value");
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_headers_value_with_colons() {
        let result = parse_extra_headers("Authorization:Bearer abc:def").unwrap();
        assert_eq!(
            result,
            vec![("Authorization".to_string(), "Bearer abc:def".to_string())]
        );
    }

    #[test]
    fn test_extra_headers_trailing_comma() {
        let result = parse_extra_headers("X-Title:MyApp,").unwrap();
        assert_eq!(result, vec![("X-Title".to_string(), "MyApp".to_string())]);
    }

    #[test]
    fn test_extra_headers_with_spaces() {
        let result =
            parse_extra_headers(" HTTP-Referer : https://myapp.com , X-Title : MyApp ").unwrap();
        assert_eq!(
            result,
            vec![
                ("HTTP-Referer".to_string(), "https://myapp.com".to_string()),
                ("X-Title".to_string(), "MyApp".to_string()),
            ]
        );
    }

    /// Clear all routing tier env vars.
    fn clear_routing_tier_env() {
        // SAFETY: Only called under ENV_MUTEX in tests.
        unsafe {
            std::env::remove_var("LLM_ROUTING_TIERS");
            for name in ["LOCAL", "STANDARD", "PREMIUM"] {
                std::env::remove_var(format!("LLM_TIER_{name}_BACKEND"));
                std::env::remove_var(format!("LLM_TIER_{name}_MODEL"));
                std::env::remove_var(format!("LLM_TIER_{name}_BASE_URL"));
                std::env::remove_var(format!("LLM_TIER_{name}_API_KEY"));
            }
        }
    }

    #[test]
    fn test_routing_tiers_not_set_returns_empty() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_routing_tier_env();

        let tiers = parse_routing_tiers().expect("should succeed");
        assert!(tiers.is_empty());
    }

    #[test]
    fn test_routing_tiers_single_tier_rejected() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_routing_tier_env();
        unsafe {
            std::env::set_var("LLM_ROUTING_TIERS", "local");
            std::env::set_var("LLM_TIER_LOCAL_BACKEND", "ollama");
            std::env::set_var("LLM_TIER_LOCAL_MODEL", "llama3.1:8b");
        }

        let result = parse_routing_tiers();
        assert!(result.is_err());

        clear_routing_tier_env();
    }

    #[test]
    fn test_routing_tiers_two_tiers_parsed() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_routing_tier_env();
        unsafe {
            std::env::set_var("LLM_ROUTING_TIERS", "local,standard");
            std::env::set_var("LLM_TIER_LOCAL_BACKEND", "ollama");
            std::env::set_var("LLM_TIER_LOCAL_MODEL", "llama3.1:8b");
            std::env::set_var("LLM_TIER_STANDARD_BACKEND", "openai_compatible");
            std::env::set_var("LLM_TIER_STANDARD_MODEL", "openai/gpt-4o-mini");
            std::env::set_var("LLM_TIER_STANDARD_BASE_URL", "https://openrouter.ai/api/v1");
            std::env::set_var("LLM_TIER_STANDARD_API_KEY", "sk-or-test");
        }

        let tiers = parse_routing_tiers().expect("should succeed");
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].name, "local");
        assert_eq!(tiers[0].backend, LlmBackend::Ollama);
        assert_eq!(tiers[0].model, "llama3.1:8b");
        assert!(tiers[0].api_key.is_none());
        assert_eq!(tiers[1].name, "standard");
        assert_eq!(tiers[1].backend, LlmBackend::OpenAiCompatible);
        assert_eq!(tiers[1].model, "openai/gpt-4o-mini");
        assert!(tiers[1].api_key.is_some());

        clear_routing_tier_env();
    }

    #[test]
    fn test_routing_tiers_missing_backend_errors() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_routing_tier_env();
        unsafe {
            std::env::set_var("LLM_ROUTING_TIERS", "local,standard");
            std::env::set_var("LLM_TIER_LOCAL_BACKEND", "ollama");
            std::env::set_var("LLM_TIER_LOCAL_MODEL", "llama3.1:8b");
            // Missing STANDARD backend/model
        }

        let result = parse_routing_tiers();
        assert!(result.is_err());

        clear_routing_tier_env();
    }

    #[test]
    fn test_routing_tiers_three_tiers_parsed() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_routing_tier_env();
        unsafe {
            std::env::set_var("LLM_ROUTING_TIERS", "local, standard, premium");
            std::env::set_var("LLM_TIER_LOCAL_BACKEND", "ollama");
            std::env::set_var("LLM_TIER_LOCAL_MODEL", "llama3.1:8b");
            std::env::set_var("LLM_TIER_STANDARD_BACKEND", "openai_compatible");
            std::env::set_var("LLM_TIER_STANDARD_MODEL", "openai/gpt-4o-mini");
            std::env::set_var("LLM_TIER_STANDARD_BASE_URL", "https://openrouter.ai/api/v1");
            std::env::set_var("LLM_TIER_PREMIUM_BACKEND", "openai_compatible");
            std::env::set_var("LLM_TIER_PREMIUM_MODEL", "anthropic/claude-sonnet-4");
            std::env::set_var("LLM_TIER_PREMIUM_BASE_URL", "https://openrouter.ai/api/v1");
        }

        let tiers = parse_routing_tiers().expect("should succeed");
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0].name, "local");
        assert_eq!(tiers[1].name, "standard");
        assert_eq!(tiers[2].name, "premium");

        clear_routing_tier_env();
    }

    /// Clear all ollama-related env vars.
    fn clear_ollama_env() {
        // SAFETY: Only called under ENV_MUTEX in tests.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("OLLAMA_BASE_URL");
            std::env::remove_var("OLLAMA_MODEL");
        }
    }

    #[test]
    fn ollama_uses_selected_model_when_ollama_model_unset() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_ollama_env();

        let settings = Settings {
            llm_backend: Some("ollama".to_string()),
            selected_model: Some("llama3.2".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(provider.model, "llama3.2");
    }

    #[test]
    fn ollama_model_env_overrides_selected_model() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_ollama_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("OLLAMA_MODEL", "mistral:latest");
        }

        let settings = Settings {
            llm_backend: Some("ollama".to_string()),
            selected_model: Some("llama3.2".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(provider.model, "mistral:latest");

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("OLLAMA_MODEL");
        }
    }

    #[test]
    fn openai_compatible_preserves_dotted_model_name() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();

        let settings = Settings {
            llm_backend: Some("openai_compatible".to_string()),
            openai_compatible_base_url: Some("http://localhost:11434/v1".to_string()),
            selected_model: Some("llama3.2".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(
            provider.model, "llama3.2",
            "model name with dot must not be truncated"
        );
    }

    #[test]
    fn registry_provider_resolves_groq() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("GROQ_API_KEY");
            std::env::remove_var("GROQ_MODEL");
        }

        let settings = Settings {
            llm_backend: Some("groq".to_string()),
            selected_model: Some("llama-3.3-70b-versatile".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        assert_eq!(cfg.backend, "groq");
        let provider = cfg.provider.expect("provider config should be present");
        assert_eq!(provider.provider_id, "groq");
        assert_eq!(provider.model, "llama-3.3-70b-versatile");
        assert_eq!(provider.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(provider.protocol, ProviderProtocol::OpenAiCompletions);
    }

    #[test]
    fn registry_provider_resolves_tinfoil() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("TINFOIL_API_KEY");
            std::env::remove_var("TINFOIL_MODEL");
        }

        let settings = Settings {
            llm_backend: Some("tinfoil".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        assert_eq!(cfg.backend, "tinfoil");
        let provider = cfg.provider.expect("provider config should be present");
        assert_eq!(provider.base_url, "https://inference.tinfoil.sh/v1");
        assert_eq!(provider.model, "kimi-k2-5");
        assert!(
            provider
                .unsupported_params
                .contains(&"temperature".to_string()),
            "tinfoil should propagate unsupported_params from registry"
        );
    }

    #[test]
    fn nearai_backend_has_no_registry_provider() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
        }

        let settings = Settings::default();
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        assert_eq!(cfg.backend, "nearai");
        assert!(cfg.provider.is_none());
    }

    #[test]
    fn backend_alias_normalized_to_canonical_id() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("LLM_BACKEND", "open_ai");
            std::env::set_var("OPENAI_API_KEY", "test-key");
        }

        let settings = Settings::default();
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        assert_eq!(
            cfg.backend, "openai",
            "alias 'open_ai' should be normalized to canonical 'openai'"
        );
        let provider = cfg.provider.expect("should have provider config");
        assert_eq!(provider.provider_id, "openai");

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("OPENAI_API_KEY");
        }
    }

    #[test]
    fn unknown_backend_falls_back_to_openai_compatible() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("LLM_BACKEND", "some_custom_provider");
            std::env::set_var("LLM_BASE_URL", "http://localhost:8080/v1");
        }

        let settings = Settings::default();
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        assert_eq!(cfg.backend, "openai_compatible");
        let provider = cfg.provider.expect("should have provider config");
        assert_eq!(provider.provider_id, "openai_compatible");
        assert_eq!(provider.base_url, "http://localhost:8080/v1");

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("LLM_BASE_URL");
        }
    }

    #[test]
    fn nearai_aliases_all_resolve_to_nearai() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");

        for alias in &["nearai", "near_ai", "near"] {
            // SAFETY: Under ENV_MUTEX.
            unsafe {
                std::env::set_var("LLM_BACKEND", alias);
            }
            let settings = Settings::default();
            let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
            assert_eq!(
                cfg.backend, "nearai",
                "alias '{alias}' should resolve to 'nearai'"
            );
            assert!(
                cfg.provider.is_none(),
                "nearai should not have a registry provider"
            );
        }

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
        }
    }

    #[test]
    fn base_url_resolution_priority() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_openai_compatible_env();

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("LLM_BACKEND", "openai_compatible");
            std::env::set_var("LLM_BASE_URL", "http://env-url/v1");
        }

        let settings = Settings {
            llm_backend: Some("openai_compatible".to_string()),
            openai_compatible_base_url: Some("http://settings-url/v1".to_string()),
            ..Default::default()
        };

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("should have provider config");
        assert_eq!(
            provider.base_url, "http://env-url/v1",
            "env var should take priority over settings"
        );

        // Now without env var, settings should win over registry default
        unsafe {
            std::env::remove_var("LLM_BASE_URL");
        }

        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("should have provider config");
        assert_eq!(
            provider.base_url, "http://settings-url/v1",
            "settings should take priority over registry default"
        );

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
        }
    }

    // ── OAuth resolution tests ──────────────────────────────────────

    /// Clear all Anthropic-related env vars.
    fn clear_anthropic_env() {
        // SAFETY: Only called under ENV_MUTEX in tests.
        unsafe {
            std::env::remove_var("LLM_BACKEND");
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("ANTHROPIC_OAUTH_TOKEN");
            std::env::remove_var("ANTHROPIC_MODEL");
            std::env::remove_var("ANTHROPIC_BASE_URL");
        }
    }

    #[test]
    fn anthropic_oauth_token_sets_placeholder_api_key() {
        use secrecy::ExposeSecret;

        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_anthropic_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat01-test-token");
        }

        let settings = Settings {
            llm_backend: Some("anthropic".to_string()),
            ..Default::default()
        };
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(
            provider
                .api_key
                .as_ref()
                .map(|k| k.expose_secret().to_string()),
            Some(OAUTH_PLACEHOLDER.to_string()),
            "api_key should be the OAuth placeholder when only OAuth token is set"
        );
        assert!(
            provider.oauth_token.is_some(),
            "oauth_token should be populated"
        );
        assert_eq!(
            provider.oauth_token.as_ref().unwrap().expose_secret(),
            "sk-ant-oat01-test-token"
        );

        clear_anthropic_env();
    }

    #[test]
    fn anthropic_api_key_takes_priority_over_oauth() {
        use secrecy::ExposeSecret;

        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_anthropic_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-real-key");
            std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat01-test-token");
        }

        let settings = Settings {
            llm_backend: Some("anthropic".to_string()),
            ..Default::default()
        };
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert_eq!(
            provider
                .api_key
                .as_ref()
                .map(|k| k.expose_secret().to_string()),
            Some("sk-ant-real-key".to_string()),
            "real API key should take priority over OAuth placeholder"
        );
        assert!(
            provider.oauth_token.is_some(),
            "oauth_token should still be populated"
        );

        clear_anthropic_env();
    }

    #[test]
    fn non_anthropic_provider_has_no_oauth_token() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_anthropic_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat01-test-token");
        }

        let settings = Settings {
            llm_backend: Some("openai".to_string()),
            ..Default::default()
        };
        let cfg = LlmConfig::resolve(&settings).expect("resolve should succeed");
        let provider = cfg.provider.expect("provider config should be present");

        assert!(
            provider.oauth_token.is_none(),
            "non-Anthropic providers should not pick up ANTHROPIC_OAUTH_TOKEN"
        );

        clear_anthropic_env();
    }

    // ── Cache retention tests ───────────────────────────────────────

    #[test]
    fn cache_retention_from_str_primary_values() {
        assert_eq!(
            "none".parse::<CacheRetention>().unwrap(),
            CacheRetention::None
        );
        assert_eq!(
            "short".parse::<CacheRetention>().unwrap(),
            CacheRetention::Short
        );
        assert_eq!(
            "long".parse::<CacheRetention>().unwrap(),
            CacheRetention::Long
        );
    }

    #[test]
    fn cache_retention_from_str_aliases() {
        assert_eq!(
            "off".parse::<CacheRetention>().unwrap(),
            CacheRetention::None
        );
        assert_eq!(
            "disabled".parse::<CacheRetention>().unwrap(),
            CacheRetention::None
        );
        assert_eq!(
            "5m".parse::<CacheRetention>().unwrap(),
            CacheRetention::Short
        );
        assert_eq!(
            "ephemeral".parse::<CacheRetention>().unwrap(),
            CacheRetention::Short
        );
        assert_eq!(
            "1h".parse::<CacheRetention>().unwrap(),
            CacheRetention::Long
        );
    }

    #[test]
    fn cache_retention_from_str_case_insensitive() {
        assert_eq!(
            "NONE".parse::<CacheRetention>().unwrap(),
            CacheRetention::None
        );
        assert_eq!(
            "Short".parse::<CacheRetention>().unwrap(),
            CacheRetention::Short
        );
        assert_eq!(
            "LONG".parse::<CacheRetention>().unwrap(),
            CacheRetention::Long
        );
        assert_eq!(
            "Ephemeral".parse::<CacheRetention>().unwrap(),
            CacheRetention::Short
        );
    }

    #[test]
    fn cache_retention_from_str_invalid() {
        let err = "bogus".parse::<CacheRetention>().unwrap_err();
        assert!(
            err.contains("bogus"),
            "error should mention the invalid value"
        );
    }

    #[test]
    fn cache_retention_display_round_trip() {
        for variant in [
            CacheRetention::None,
            CacheRetention::Short,
            CacheRetention::Long,
        ] {
            let s = variant.to_string();
            let parsed: CacheRetention = s.parse().unwrap();
            assert_eq!(parsed, variant, "round-trip failed for {s}");
        }
    }

    #[test]
    fn test_request_timeout_defaults_to_120() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::remove_var("LLM_REQUEST_TIMEOUT_SECS");
        }
        let config = LlmConfig::resolve(&Settings::default()).expect("resolve");
        assert_eq!(config.request_timeout_secs, 120);
    }

    #[test]
    fn test_request_timeout_configurable() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("LLM_REQUEST_TIMEOUT_SECS", "300");
        }
        let config = LlmConfig::resolve(&Settings::default()).expect("resolve");
        assert_eq!(config.request_timeout_secs, 300);
        // SAFETY: Cleanup
        unsafe {
            std::env::remove_var("LLM_REQUEST_TIMEOUT_SECS");
        }
    }
}
