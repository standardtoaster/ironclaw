use crate::config::helpers::optional_env;
use crate::error::ConfigError;
use crate::workspace::layer::MemoryLayer;

/// Workspace memory configuration.
///
/// Controls memory layer definitions for privacy-aware writes and
/// cross-scope read access for multi-user deployments.
#[derive(Debug, Clone)]
pub struct WorkspaceConfig {
    pub memory_layers: Vec<MemoryLayer>,
    /// Additional user scopes the workspace can read from.
    ///
    /// When set, search/read/list operations span these scopes in addition
    /// to the primary user scope. Writes remain isolated to the primary scope.
    /// Parsed from `WORKSPACE_READ_SCOPES` (comma-separated).
    pub read_scopes: Vec<String>,
}

impl WorkspaceConfig {
    pub(crate) fn resolve(user_id: &str) -> Result<Self, ConfigError> {
        let memory_layers: Vec<MemoryLayer> = match optional_env("MEMORY_LAYERS")? {
            Some(json_str) => {
                serde_json::from_str(&json_str).map_err(|e| ConfigError::InvalidValue {
                    key: "MEMORY_LAYERS".to_string(),
                    message: format!("must be valid JSON array of layer objects: {e}"),
                })?
            }
            None => MemoryLayer::default_for_user(user_id),
        };

        // Validate layer names and scopes
        for layer in &memory_layers {
            if layer.name.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "MEMORY_LAYERS".to_string(),
                    message: "layer name must not be empty".to_string(),
                });
            }
            if layer.scope.trim().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: "MEMORY_LAYERS".to_string(),
                    message: format!("layer '{}' has an empty scope", layer.name),
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

        let read_scopes = optional_env("WORKSPACE_READ_SCOPES")?
            .map(|s| {
                s.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            memory_layers,
            read_scopes,
        })
    }
}
