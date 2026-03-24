use crate::tools::mcp::service_registry::glob_match;

/// Per-tool scope metadata attached at registration time.
/// Checked on every invocation to enforce per-lens access control.
#[derive(Debug, Clone)]
pub struct ToolScope {
    pub service_name: String,
    pub allowed_users: Vec<String>,
    pub denied_patterns: Vec<String>,
}

impl ToolScope {
    pub fn is_allowed(&self, tool_name: &str, user_id: &str) -> bool {
        // Check user scope
        if !self.allowed_users.is_empty()
            && !self.allowed_users.iter().any(|u| u == user_id)
        {
            return false;
        }
        // Check deny patterns (strip service prefix for matching)
        let unprefixed = tool_name
            .strip_prefix(&format!("{}_", self.service_name))
            .unwrap_or(tool_name);
        if self.denied_patterns.iter().any(|p| glob_match(p, unprefixed)) {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_check_allows_valid_user() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec!["andrew".to_string()],
            denied_patterns: vec![],
        };
        assert!(scope.is_allowed("ha_turn_on", "andrew"));
    }

    #[test]
    fn test_scope_check_rejects_invalid_user() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec!["andrew".to_string()],
            denied_patterns: vec![],
        };
        assert!(!scope.is_allowed("ha_turn_on", "grace"));
    }

    #[test]
    fn test_scope_check_rejects_denied_tool() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec!["andrew".to_string()],
            denied_patterns: vec!["delete_*".to_string()],
        };
        assert!(!scope.is_allowed("ha_delete_entity", "andrew"));
    }

    #[test]
    fn test_empty_allowed_users_means_all() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec![],
            denied_patterns: vec![],
        };
        assert!(scope.is_allowed("ha_turn_on", "anyone"));
    }

    #[test]
    fn test_multiple_deny_patterns() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec![],
            denied_patterns: vec!["delete_*".to_string(), "remove_*".to_string()],
        };
        assert!(!scope.is_allowed("ha_delete_entity", "andrew"));
        assert!(!scope.is_allowed("ha_remove_entity", "andrew"));
        assert!(scope.is_allowed("ha_turn_on", "andrew"));
    }

    #[test]
    fn test_tool_without_service_prefix() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec![],
            denied_patterns: vec!["delete_*".to_string()],
        };
        assert!(!scope.is_allowed("delete_entity", "andrew"));
    }

    #[test]
    fn test_user_and_deny_both_checked() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec!["andrew".to_string()],
            denied_patterns: vec!["delete_*".to_string()],
        };
        assert!(!scope.is_allowed("ha_delete_entity", "andrew"));
        assert!(!scope.is_allowed("ha_turn_on", "grace"));
        assert!(scope.is_allowed("ha_turn_on", "andrew"));
    }

    #[test]
    fn test_multiple_allowed_users() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec!["andrew".to_string(), "grace".to_string()],
            denied_patterns: vec![],
        };
        assert!(scope.is_allowed("ha_turn_on", "andrew"));
        assert!(scope.is_allowed("ha_turn_on", "grace"));
        assert!(!scope.is_allowed("ha_turn_on", "stranger"));
    }

    #[test]
    fn test_deny_pattern_exact_match() {
        let scope = ToolScope {
            service_name: "ha".to_string(),
            allowed_users: vec![],
            denied_patterns: vec!["turn_on".to_string()],
        };
        assert!(!scope.is_allowed("ha_turn_on", "andrew"));
        assert!(scope.is_allowed("ha_turn_off", "andrew"));
    }
}
