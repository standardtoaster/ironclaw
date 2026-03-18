use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub name: String,
    pub description: String,
    pub url: String,
    pub auth: ServiceAuth,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default = "default_allow")]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default = "default_tier")]
    pub tier: String,
    #[serde(default)]
    pub lens_overrides: HashMap<String, LensOverride>,
}

fn default_allow() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_tier() -> String {
    "read".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServiceAuth {
    #[serde(rename = "bearer")]
    Bearer { credential: String },
    #[serde(rename = "header")]
    Header { key: String, credential: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensOverride {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub tier: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_service_config() {
        let json = serde_json::json!([{
            "name": "home_assistant",
            "description": "Smart home control",
            "url": "http://192.168.1.3:8123/api/mcp",
            "auth": {"type": "bearer", "credential": "HA_TOKEN"},
            "scopes": ["andrew", "grace"],
            "keywords": ["light", "home", "sensor"],
            "allow": ["*"],
            "deny": ["delete_*"],
            "tier": "read",
            "lens_overrides": {}
        }]);
        let configs: Vec<ServiceConfig> = serde_json::from_value(json).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "home_assistant");
        assert_eq!(configs[0].keywords, vec!["light", "home", "sensor"]);
    }
}
