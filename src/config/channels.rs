use std::collections::HashMap;
use std::path::PathBuf;

use secrecy::SecretString;
use serde::Deserialize;

use crate::bootstrap::ironclaw_base_dir;
use crate::config::helpers::{optional_env, parse_bool_env, parse_optional_env};
use crate::config::LlmBackend;
use crate::error::ConfigError;
use crate::settings::Settings;

/// Channel configurations.
#[derive(Debug, Clone)]
pub struct ChannelsConfig {
    pub cli: CliConfig,
    pub http: Option<HttpConfig>,
    pub gateway: Option<GatewayConfig>,
    pub signal: Option<SignalConfig>,
    /// Directory containing WASM channel modules (default: ~/.ironclaw/channels/).
    pub wasm_channels_dir: std::path::PathBuf,
    /// Whether WASM channels are enabled.
    pub wasm_channels_enabled: bool,
    /// Telegram owner user ID. When set, the bot only responds to this user.
    pub telegram_owner_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub host: String,
    pub port: u16,
    pub webhook_secret: Option<SecretString>,
    pub user_id: String,
}

/// Web gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    /// Bearer token for authentication. Random hex generated at startup if unset.
    pub auth_token: Option<String>,
    pub user_id: String,
    /// Additional user scopes for workspace reads.
    ///
    /// When set, the workspace will be able to read (search, read, list) from
    /// these additional user scopes while writes remain isolated to `user_id`.
    /// Parsed from `WORKSPACE_READ_SCOPES` (comma-separated).
    pub workspace_read_scopes: Vec<String>,
    /// Memory layer definitions (JSON in env var, or from external config).
    pub memory_layers: Vec<crate::workspace::layer::MemoryLayer>,
    /// Multi-user token map. When set, each token maps to a user identity.
    /// Parsed from `GATEWAY_USER_TOKENS` (JSON string). When absent, falls back
    /// to single-user mode via `auth_token` + `user_id`.
    pub user_tokens: Option<HashMap<String, UserTokenConfig>>,
}

/// Per-user token configuration for multi-user mode.
#[derive(Debug, Clone, Deserialize)]
pub struct UserTokenConfig {
    pub user_id: String,
    #[serde(default)]
    pub workspace_read_scopes: Vec<String>,
    /// LLM backend override for this user (e.g. "anthropic", "ollama", "openai").
    #[serde(default)]
    pub llm_backend: Option<String>,
    /// LLM model override for this user (e.g. "claude-haiku-4-5-20251001").
    #[serde(default)]
    pub llm_model: Option<String>,
    /// LLM API key for this user's provider.
    #[serde(default, deserialize_with = "deserialize_optional_secret")]
    pub llm_api_key: Option<SecretString>,
    /// LLM base URL override for this user's provider.
    #[serde(default)]
    pub llm_base_url: Option<String>,
}

/// Resolved per-user LLM configuration.
///
/// Only constructed when all required fields (`llm_backend` + `llm_model`) are
/// present on a `UserTokenConfig`. The API key and base URL are optional
/// depending on the backend (e.g. Ollama needs no key).
#[derive(Debug, Clone)]
pub struct UserLlmConfig {
    pub backend: LlmBackend,
    pub model: String,
    pub api_key: Option<SecretString>,
    pub base_url: Option<String>,
}

impl UserTokenConfig {
    /// Try to extract a resolved `UserLlmConfig` from this token config.
    ///
    /// Returns `Some` when at least `llm_backend` and `llm_model` are set.
    /// Returns an error if the backend string is invalid.
    pub fn llm_config(&self) -> Result<Option<UserLlmConfig>, String> {
        match (&self.llm_backend, &self.llm_model) {
            (Some(backend_str), Some(model)) => {
                let backend: LlmBackend = backend_str.parse().map_err(|e: String| {
                    format!("user '{}': {}", self.user_id, e)
                })?;
                Ok(Some(UserLlmConfig {
                    backend,
                    model: model.clone(),
                    api_key: self.llm_api_key.clone(),
                    base_url: self.llm_base_url.clone(),
                }))
            }
            (Some(_), None) | (None, Some(_)) => Err(format!(
                "user '{}': llm_backend and llm_model must both be set (or both omitted)",
                self.user_id
            )),
            (None, None) => Ok(None),
        }
    }
}

/// Deserialize an optional `SecretString` from a JSON string.
fn deserialize_optional_secret<'de, D>(deserializer: D) -> Result<Option<SecretString>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    Ok(opt.map(SecretString::from))
}

/// Signal channel configuration (signal-cli daemon HTTP/JSON-RPC).
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Base URL of the signal-cli daemon HTTP endpoint (e.g. `http://127.0.0.1:8080`).
    pub http_url: String,
    /// Signal account identifier (E.164 phone number, e.g. `+1234567890`).
    pub account: String,
    /// Users allowed to interact with the bot in DMs.
    ///
    /// Each entry is one of:
    /// - `*` — allow everyone
    /// - E.164 phone number (e.g. `+1234567890`)
    /// - bare UUID (e.g. `a1b2c3d4-e5f6-7890-abcd-ef1234567890`)
    /// - `uuid:<id>` prefix form (e.g. `uuid:a1b2c3d4-e5f6-7890-abcd-ef1234567890`)
    ///
    /// An empty list denies all senders (secure by default).
    pub allow_from: Vec<String>,
    /// Groups allowed to interact with the bot.
    ///
    /// - Empty list — deny all group messages (DMs only, secure by default).
    /// - `*` — allow all groups.
    /// - Specific group IDs — allow only those groups.
    pub allow_from_groups: Vec<String>,
    /// DM policy: "open", "allowlist", or "pairing". Default: "pairing".
    ///
    /// - "open" — allow all DM senders (ignores allow_from for DMs)
    /// - "allowlist" — only allow senders in allow_from list
    /// - "pairing" — allowlist + send pairing reply to unknown users
    pub dm_policy: String,
    /// Group policy: "allowlist", "open", or "disabled". Default: "allowlist".
    ///
    /// - "disabled" — deny all group messages
    /// - "allowlist" — check allow_from_groups and group_allow_from
    /// - "open" — accept all group messages (respects allow_from_groups for group ID)
    pub group_policy: String,
    /// Allow list for group message senders. If empty, inherits from allow_from.
    pub group_allow_from: Vec<String>,
    /// Skip messages that contain only attachments (no text).
    pub ignore_attachments: bool,
    /// Skip story messages.
    pub ignore_stories: bool,
}

impl ChannelsConfig {
    pub(crate) fn resolve(settings: &Settings) -> Result<Self, ConfigError> {
        let http = if optional_env("HTTP_PORT")?.is_some() || optional_env("HTTP_HOST")?.is_some() {
            Some(HttpConfig {
                host: optional_env("HTTP_HOST")?.unwrap_or_else(|| "0.0.0.0".to_string()),
                port: parse_optional_env("HTTP_PORT", 8080)?,
                webhook_secret: optional_env("HTTP_WEBHOOK_SECRET")?.map(SecretString::from),
                user_id: optional_env("HTTP_USER_ID")?.unwrap_or_else(|| "http".to_string()),
            })
        } else {
            None
        };

        let gateway_enabled = parse_bool_env("GATEWAY_ENABLED", true)?;
        let gateway = if gateway_enabled {
            let user_id = optional_env("GATEWAY_USER_ID")?.unwrap_or_else(|| "default".to_string());

            let memory_layers: Vec<crate::workspace::layer::MemoryLayer> =
                match optional_env("MEMORY_LAYERS")? {
                    Some(json_str) => {
                        serde_json::from_str(&json_str).map_err(|e| ConfigError::InvalidValue {
                            key: "MEMORY_LAYERS".to_string(),
                            message: format!("must be valid JSON array of layer objects: {e}"),
                        })?
                    }
                    None => crate::workspace::layer::MemoryLayer::default_for_user(&user_id),
                };

            // Validate layer names and scopes
            for layer in &memory_layers {
                if layer.name.trim().is_empty() {
                    return Err(ConfigError::InvalidValue {
                        key: "MEMORY_LAYERS".to_string(),
                        message: "layer name must not be empty".to_string(),
                    });
                }
                if layer.name.len() > 64 {
                    return Err(ConfigError::InvalidValue {
                        key: "MEMORY_LAYERS".to_string(),
                        message: format!(
                            "layer name '{}' exceeds 64 characters",
                            layer.name
                        ),
                    });
                }
                if !layer
                    .name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
                {
                    return Err(ConfigError::InvalidValue {
                        key: "MEMORY_LAYERS".to_string(),
                        message: format!(
                            "layer name '{}' contains invalid characters \
                             (allowed: a-z, A-Z, 0-9, _, -)",
                            layer.name
                        ),
                    });
                }
                if layer.scope.trim().is_empty() {
                    return Err(ConfigError::InvalidValue {
                        key: "MEMORY_LAYERS".to_string(),
                        message: format!(
                            "layer '{}' has an empty scope",
                            layer.name
                        ),
                    });
                }
            }

            // Check for duplicate layer names
            {
                let mut seen = std::collections::HashSet::new();
                for layer in &memory_layers {
                    if !seen.insert(&layer.name) {
                        return Err(ConfigError::InvalidValue {
                            key: "MEMORY_LAYERS".to_string(),
                            message: format!("duplicate layer name '{}'", layer.name),
                        });
                    }
                }
            }

            let user_tokens: Option<HashMap<String, UserTokenConfig>> =
                match optional_env("GATEWAY_USER_TOKENS")? {
                    Some(json_str) => {
                        let tokens: HashMap<String, UserTokenConfig> =
                            serde_json::from_str(&json_str).map_err(|e| {
                                ConfigError::InvalidValue {
                                    key: "GATEWAY_USER_TOKENS".to_string(),
                                    message: format!(
                                        "must be valid JSON object mapping tokens to user configs: {e}"
                                    ),
                                }
                            })?;
                        if tokens.is_empty() {
                            return Err(ConfigError::InvalidValue {
                                key: "GATEWAY_USER_TOKENS".to_string(),
                                message: "token map is empty — remove the variable to use single-user mode".to_string(),
                            });
                        }
                        for (tok, cfg) in &tokens {
                            if cfg.user_id.trim().is_empty() {
                                return Err(ConfigError::InvalidValue {
                                    key: "GATEWAY_USER_TOKENS".to_string(),
                                    message: format!(
                                        "token '{}...' has an empty user_id",
                                        &tok[..tok.len().min(8)]
                                    ),
                                });
                            }
                        }
                        Some(tokens)
                    }
                    None => None,
                };


            let workspace_read_scopes: Vec<String> = optional_env("WORKSPACE_READ_SCOPES")?
                .map(|s| {
                    s.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            for scope in &workspace_read_scopes {
                if scope.len() > 128 {
                    return Err(ConfigError::InvalidValue {
                        key: "WORKSPACE_READ_SCOPES".to_string(),
                        message: format!(
                            "scope '{}...' exceeds 128 characters",
                            &scope[..32]
                        ),
                    });
                }
            }


            Some(GatewayConfig {
                host: optional_env("GATEWAY_HOST")?.unwrap_or_else(|| "127.0.0.1".to_string()),
                port: parse_optional_env("GATEWAY_PORT", 3000)?,
                auth_token: optional_env("GATEWAY_AUTH_TOKEN")?,
                user_id,
                workspace_read_scopes,
                memory_layers,
                user_tokens,
            })
        } else {
            None
        };

        let signal = if let Some(http_url) = optional_env("SIGNAL_HTTP_URL")? {
            let account = optional_env("SIGNAL_ACCOUNT")?.ok_or(ConfigError::InvalidValue {
                key: "SIGNAL_ACCOUNT".to_string(),
                message: "SIGNAL_ACCOUNT is required when SIGNAL_HTTP_URL is set".to_string(),
            })?;
            let allow_from = match std::env::var_os("SIGNAL_ALLOW_FROM") {
                None => vec![account.clone()],
                Some(val) => {
                    let s = val.to_string_lossy();
                    s.split(',')
                        .map(|e| e.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                }
            };
            let dm_policy =
                optional_env("SIGNAL_DM_POLICY")?.unwrap_or_else(|| "pairing".to_string());
            let group_policy =
                optional_env("SIGNAL_GROUP_POLICY")?.unwrap_or_else(|| "allowlist".to_string());
            Some(SignalConfig {
                http_url,
                account,
                allow_from,
                allow_from_groups: optional_env("SIGNAL_ALLOW_FROM_GROUPS")?
                    .map(|s| {
                        s.split(',')
                            .map(|e| e.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
                dm_policy,
                group_policy,
                group_allow_from: optional_env("SIGNAL_GROUP_ALLOW_FROM")?
                    .map(|s| {
                        s.split(',')
                            .map(|e| e.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
                ignore_attachments: optional_env("SIGNAL_IGNORE_ATTACHMENTS")?
                    .map(|s| s.to_lowercase() == "true" || s == "1")
                    .unwrap_or(false),
                ignore_stories: optional_env("SIGNAL_IGNORE_STORIES")?
                    .map(|s| s.to_lowercase() == "true" || s == "1")
                    .unwrap_or(true),
            })
        } else {
            None
        };

        let cli_enabled = optional_env("CLI_ENABLED")?
            .map(|s| s.to_lowercase() != "false" && s != "0")
            .unwrap_or(true);

        Ok(Self {
            cli: CliConfig {
                enabled: cli_enabled,
            },
            http,
            gateway,
            signal,
            wasm_channels_dir: optional_env("WASM_CHANNELS_DIR")?
                .map(PathBuf::from)
                .unwrap_or_else(default_channels_dir),
            wasm_channels_enabled: parse_bool_env("WASM_CHANNELS_ENABLED", true)?,
            telegram_owner_id: optional_env("TELEGRAM_OWNER_ID")?
                .map(|s| s.parse())
                .transpose()
                .map_err(|e: std::num::ParseIntError| ConfigError::InvalidValue {
                    key: "TELEGRAM_OWNER_ID".to_string(),
                    message: format!("must be an integer: {e}"),
                })?
                .or(settings.channels.telegram_owner_id),
        })
    }
}

/// Get the default channels directory (~/.ironclaw/channels/).
fn default_channels_dir() -> PathBuf {
    ironclaw_base_dir().join("channels")
}
