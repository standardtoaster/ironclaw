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

impl ServiceConfig {
    /// Check if a tool name passes the allow/deny filters for a given user.
    /// Deny always takes precedence. Per-lens overrides are checked after base filters.
    pub fn is_tool_allowed(&self, tool_name: &str, user_id: &str) -> bool {
        // Base allow/deny
        if !self.matches_allow(&self.allow, tool_name) {
            return false;
        }
        if self.matches_deny(&self.deny, tool_name) {
            return false;
        }
        // Per-lens overrides
        if let Some(lens_override) = self.lens_overrides.get(user_id) {
            if !lens_override.allow.is_empty()
                && !self.matches_allow(&lens_override.allow, tool_name)
            {
                return false;
            }
            if self.matches_deny(&lens_override.deny, tool_name) {
                return false;
            }
        }
        true
    }

    fn matches_allow(&self, patterns: &[String], name: &str) -> bool {
        if patterns.is_empty() {
            return true;
        }
        patterns.iter().any(|p| glob_match(p, name))
    }

    fn matches_deny(&self, patterns: &[String], name: &str) -> bool {
        patterns.iter().any(|p| glob_match(p, name))
    }
}

/// Simple glob matching supporting only `*` as wildcard.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else if let Some(suffix) = pattern.strip_prefix('*') {
        name.ends_with(suffix)
    } else {
        pattern == name
    }
}

pub struct ServiceRegistry {
    services: Vec<ServiceConfig>,
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
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

    /// Return the number of registered services.
    pub fn len(&self) -> usize {
        self.services.len()
    }

    /// Return true if no services are registered.
    pub fn is_empty(&self) -> bool {
        self.services.is_empty()
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

    #[test]
    fn test_tool_allowed_by_default() {
        let config = make_test_config("ha", vec!["light"]);
        assert!(config.is_tool_allowed("turn_on", "andrew"));
    }

    #[test]
    fn test_tool_denied_by_pattern() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.deny = vec!["delete_*".to_string()];
        assert!(!config.is_tool_allowed("delete_entity", "andrew"));
        assert!(config.is_tool_allowed("turn_on", "andrew"));
    }

    #[test]
    fn test_deny_takes_precedence() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.allow = vec!["*".to_string()];
        config.deny = vec!["turn_*".to_string()];
        assert!(!config.is_tool_allowed("turn_on", "andrew"));
    }

    #[test]
    fn test_lens_override_deny() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.lens_overrides.insert(
            "grace".to_string(),
            LensOverride {
                allow: vec![],
                deny: vec!["automation_*".to_string()],
                tier: None,
            },
        );
        assert!(config.is_tool_allowed("automation_create", "andrew"));
        assert!(!config.is_tool_allowed("automation_create", "grace"));
    }

    // ---- glob_match edge cases ----

    #[test]
    fn test_glob_match_exact() {
        assert!(glob_match("turn_on", "turn_on"));
        assert!(!glob_match("turn_on", "turn_off"));
    }

    #[test]
    fn test_glob_match_prefix_wildcard() {
        assert!(glob_match("delete_*", "delete_entity"));
        assert!(glob_match("delete_*", "delete_"));
        assert!(!glob_match("delete_*", "remove_entity"));
    }

    #[test]
    fn test_glob_match_suffix_wildcard() {
        assert!(glob_match("*_entity", "delete_entity"));
        assert!(glob_match("*_entity", "create_entity"));
        assert!(!glob_match("*_entity", "delete_item"));
    }

    #[test]
    fn test_glob_match_star_alone_matches_all() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", ""));
    }

    #[test]
    fn test_glob_match_empty_pattern() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "something"));
    }

    #[test]
    fn test_glob_match_case_sensitive() {
        assert!(!glob_match("Delete_*", "delete_entity"));
        assert!(glob_match("Delete_*", "Delete_entity"));
    }

    // ---- is_tool_allowed edge cases ----

    #[test]
    fn test_tool_allowed_with_empty_allow_list() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.allow = vec![];
        assert!(config.is_tool_allowed("turn_on", "andrew"));
    }

    #[test]
    fn test_tool_allowed_with_specific_allow() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.allow = vec!["turn_*".to_string(), "get_*".to_string()];
        assert!(config.is_tool_allowed("turn_on", "andrew"));
        assert!(config.is_tool_allowed("get_status", "andrew"));
        assert!(!config.is_tool_allowed("delete_entity", "andrew"));
    }

    #[test]
    fn test_tool_multiple_deny_patterns() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.deny = vec!["delete_*".to_string(), "remove_*".to_string(), "destroy_*".to_string()];
        assert!(!config.is_tool_allowed("delete_entity", "andrew"));
        assert!(!config.is_tool_allowed("remove_entity", "andrew"));
        assert!(!config.is_tool_allowed("destroy_all", "andrew"));
        assert!(config.is_tool_allowed("turn_on", "andrew"));
    }

    #[test]
    fn test_lens_override_allow_restricts() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.lens_overrides.insert(
            "grace".to_string(),
            LensOverride {
                allow: vec!["get_*".to_string()],
                deny: vec![],
                tier: None,
            },
        );
        assert!(config.is_tool_allowed("turn_on", "andrew"));
        assert!(config.is_tool_allowed("get_status", "andrew"));
        assert!(config.is_tool_allowed("get_status", "grace"));
        assert!(!config.is_tool_allowed("turn_on", "grace"));
    }

    #[test]
    fn test_lens_override_deny_and_allow_combined() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.lens_overrides.insert(
            "grace".to_string(),
            LensOverride {
                allow: vec!["*".to_string()],
                deny: vec!["automation_*".to_string(), "delete_*".to_string()],
                tier: None,
            },
        );
        assert!(config.is_tool_allowed("turn_on", "grace"));
        assert!(!config.is_tool_allowed("automation_create", "grace"));
        assert!(!config.is_tool_allowed("delete_entity", "grace"));
    }

    // ---- ServiceRegistry edge cases ----

    #[test]
    fn test_search_by_service_name() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("home_assistant", vec!["light"]),
        ]);
        let results = registry.search("home_assistant", "andrew");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_case_insensitive() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("home_assistant", vec!["Light", "HOME"]),
        ]);
        let results = registry.search("LIGHT", "andrew");
        assert_eq!(results.len(), 1);
        let results = registry.search("home", "andrew");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_multi_word_query() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("home_assistant", vec!["light", "sensor"]),
            make_test_config("media", vec!["movie", "music"]),
        ]);
        let results = registry.search("light sensor", "andrew");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "home_assistant");
    }

    #[test]
    fn test_search_no_matching_services() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("ha", vec!["light"]),
        ]);
        let results = registry.search("zzz_no_match", "andrew");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_get_service_by_name() {
        let registry = ServiceRegistry::from_configs(vec![
            make_test_config("ha", vec!["light"]),
            make_test_config("media", vec!["movie"]),
        ]);
        let result = registry.get("ha", "andrew");
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "ha");

        let result = registry.get("nonexistent", "andrew");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_respects_scope() {
        let mut config = make_test_config("ha", vec!["light"]);
        config.scopes = vec!["andrew".to_string()];
        let registry = ServiceRegistry::from_configs(vec![config]);

        assert!(registry.get("ha", "andrew").is_some());
        assert!(registry.get("ha", "grace").is_none());
    }

    #[test]
    fn test_empty_registry() {
        let registry = ServiceRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.search("anything", "anyone").len(), 0);
        assert!(registry.get("anything", "anyone").is_none());
    }

    #[test]
    fn test_load_from_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("services.json");
        let json = serde_json::json!([{
            "name": "ha",
            "description": "Home Assistant",
            "url": "http://localhost:8123/api/mcp",
            "auth": {"type": "bearer", "credential": "token"},
            "keywords": ["light"]
        }]);
        std::fs::write(&config_path, serde_json::to_string(&json).unwrap()).unwrap();

        let registry = ServiceRegistry::load_from_file(&config_path).unwrap();
        assert_eq!(registry.len(), 1);
        let service = registry.get("ha", "anyone").unwrap();
        assert_eq!(service.allow, vec!["*"]);
        assert_eq!(service.tier, "read");
        assert!(service.deny.is_empty());
    }

    #[test]
    fn test_load_from_file_invalid_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let config_path = dir.path().join("bad.json");
        std::fs::write(&config_path, "not json").unwrap();
        assert!(ServiceRegistry::load_from_file(&config_path).is_err());
    }

    #[test]
    fn test_load_from_file_missing() {
        let path = std::path::Path::new("/tmp/nonexistent_mcp_config_12345.json");
        assert!(ServiceRegistry::load_from_file(path).is_err());
    }

    #[test]
    fn test_multiple_services_different_scopes() {
        let mut ha = make_test_config("ha", vec!["light"]);
        ha.scopes = vec!["andrew".to_string(), "grace".to_string()];
        let mut admin = make_test_config("admin", vec!["config"]);
        admin.scopes = vec!["andrew".to_string()];
        let mut public = make_test_config("weather", vec!["weather"]);
        public.scopes = vec![];

        let registry = ServiceRegistry::from_configs(vec![ha, admin, public]);

        assert_eq!(registry.search("light", "andrew").len(), 1);
        assert_eq!(registry.search("config", "andrew").len(), 1);
        assert_eq!(registry.search("weather", "andrew").len(), 1);

        assert_eq!(registry.search("light", "grace").len(), 1);
        assert_eq!(registry.search("config", "grace").len(), 0);
        assert_eq!(registry.search("weather", "grace").len(), 1);

        assert_eq!(registry.search("light", "unknown").len(), 0);
        assert_eq!(registry.search("weather", "unknown").len(), 1);
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
