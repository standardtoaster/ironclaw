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

pub struct ServiceRegistry {
    services: Vec<ServiceConfig>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self { services: vec![] }
    }

    pub fn from_configs(services: Vec<ServiceConfig>) -> Self {
        Self { services }
    }

    pub fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let services: Vec<ServiceConfig> = serde_json::from_str(&contents)?;
        Ok(Self { services })
    }

    /// Search services by query string, filtered by user scope.
    /// Matches against keywords (any query word matches any keyword), description (substring),
    /// and service name (substring).
    pub fn search(&self, query: &str, user_id: &str) -> Vec<&ServiceConfig> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        self.services
            .iter()
            .filter(|s| self.user_in_scope(s, user_id))
            .filter(|s| {
                // Check keyword match (any query word matches any keyword)
                let keyword_match = s.keywords.iter().any(|kw| {
                    let kw_lower = kw.to_lowercase();
                    query_words.iter().any(|qw| kw_lower.contains(qw))
                });
                // Check description match (substring)
                let desc_match = s.description.to_lowercase().contains(&query_lower);
                // Check name match
                let name_match = s.name.to_lowercase().contains(&query_lower);
                keyword_match || desc_match || name_match
            })
            .collect()
    }

    /// Get a specific service by name, checking user scope.
    pub fn get(&self, name: &str, user_id: &str) -> Option<&ServiceConfig> {
        self.services
            .iter()
            .find(|s| s.name == name && self.user_in_scope(s, user_id))
    }

    fn user_in_scope(&self, service: &ServiceConfig, user_id: &str) -> bool {
        service.scopes.is_empty() || service.scopes.iter().any(|s| s == user_id)
    }
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

    #[test]
    fn test_keyword_search_matches() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("home_assistant", vec!["light", "home", "sensor"]),
            make_test_config("media", vec!["movie", "show", "plex"]),
        ]);
        let results = registry.search("light", "andrew");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "home_assistant");
    }

    #[test]
    fn test_keyword_search_description_fallback() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("home_assistant", vec!["light"]),
        ]);
        // "Smart home control" is in description -- should match
        let results = registry.search("smart home", "andrew");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_keyword_search_respects_scope() {
        let mut config = make_test_config("media", vec!["movie"]);
        config.scopes = vec!["andrew".to_string()];
        let registry = ServiceRegistry::from_configs(vec![config]);

        let andrew_results = registry.search("movie", "andrew");
        assert_eq!(andrew_results.len(), 1);

        let grace_results = registry.search("movie", "grace");
        assert_eq!(grace_results.len(), 0);
    }

    #[test]
    fn test_empty_scopes_means_all_users() {
        let mut config = make_test_config("media", vec!["movie"]);
        config.scopes = vec![];
        let registry = ServiceRegistry::from_configs(vec![config]);

        let results = registry.search("movie", "anyone");
        assert_eq!(results.len(), 1);
    }

    fn make_test_config(name: &str, keywords: Vec<&str>) -> ServiceConfig {
        ServiceConfig {
            name: name.to_string(),
            description: format!("Smart home control for {}", name),
            url: "http://localhost:8123/api/mcp".to_string(),
            auth: ServiceAuth::Bearer {
                credential: "TEST_TOKEN".to_string(),
            },
            scopes: vec!["andrew".to_string(), "grace".to_string()],
            keywords: keywords.into_iter().map(String::from).collect(),
            allow: vec!["*".to_string()],
            deny: vec![],
            tier: "read".to_string(),
            lens_overrides: HashMap::new(),
        }
    }
}
