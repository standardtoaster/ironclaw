//! Skills system for IronClaw.
//!
//! This module contains main-crate skill logic that depends on types from
//! other `src/` modules (e.g. `crate::llm::ToolDefinition`, `crate::secrets`).
//! For core skill types, parsing, and registry, import from `ironclaw_skills` directly.
//!
//! The `attenuation` submodule remains here because it depends on
//! `crate::llm::ToolDefinition` which is a main-crate type.
//!
//! # V1 migration notes
//!
//! The following items in this module exist **only for the v1 agent** (`src/agent/`).
//! Once the v1 agent is removed and all users are on ENGINE_V2, they can be deleted:
//!
//! - **`attenuation` module** — Trust-based tool filtering. In v2, the Python
//!   orchestrator handles skill trust via the `format_skills()` function and
//!   the policy engine handles tool access via capability leases.
//! - **`register_skill_credentials()`** — Registers credential mappings from v1
//!   `LoadedSkill` into `SharedCredentialRegistry`. In v2, credentials are declared
//!   in the SKILL.md frontmatter and registered at migration time in `skill_migration.rs`.
//! - **`credential_spec_to_mapping()` / `convert_credential_location()`** — Conversion
//!   helpers used by `register_skill_credentials()`. Same lifecycle.
//! - **This entire module** — Once v1 is gone, the remaining local items
//!   can be deleted and this file removed.
//!
//! The `ironclaw_skills` crate itself remains (types, parser, validation, v2 types).

pub mod attenuation;
pub mod bundled;
pub mod script_runner;

// Items from `ironclaw_skills` are no longer glob-re-exported.
// Callers should import from `ironclaw_skills` directly.

// Re-export attenuation at the same path as before.
pub use attenuation::{AttenuationResult, attenuate_tools};
pub use script_runner::run_activation_script;

use crate::secrets::{CredentialLocation, CredentialMapping};
use ironclaw_skills::{LoadedSkill, SkillCredentialLocation, SkillCredentialSpec};

/// Convert a skill credential location to the main crate's [`CredentialLocation`].
fn convert_credential_location(loc: &SkillCredentialLocation) -> CredentialLocation {
    match loc {
        SkillCredentialLocation::Bearer => CredentialLocation::AuthorizationBearer,
        SkillCredentialLocation::BasicAuth { username } => CredentialLocation::AuthorizationBasic {
            username: username.clone(),
        },
        SkillCredentialLocation::Header { name, prefix } => CredentialLocation::Header {
            name: name.clone(),
            prefix: prefix.clone(),
        },
        SkillCredentialLocation::QueryParam { name } => {
            CredentialLocation::QueryParam { name: name.clone() }
        }
    }
}

/// Convert a [`SkillCredentialSpec`] to a [`CredentialMapping`] for the
/// [`SharedCredentialRegistry`](crate::tools::wasm::SharedCredentialRegistry).
pub fn credential_spec_to_mapping(spec: &SkillCredentialSpec) -> CredentialMapping {
    CredentialMapping {
        secret_name: spec.name.clone(),
        location: convert_credential_location(&spec.location),
        host_patterns: spec.hosts.clone(),
    }
}

/// Register credential mappings from loaded skills into the shared registry.
///
/// Validates each spec before registration; invalid specs are logged and skipped.
pub fn register_skill_credentials(
    skills: &[LoadedSkill],
    registry: &crate::tools::wasm::SharedCredentialRegistry,
) {
    let mut count = 0usize;
    for skill in skills {
        for spec in &skill.manifest.credentials {
            let errors = ironclaw_skills::validation::validate_credential_spec(spec);
            if !errors.is_empty() {
                tracing::warn!(
                    skill = %skill.name(),
                    credential = %spec.name,
                    errors = ?errors,
                    "Skipping invalid credential spec"
                );
                continue;
            }
            let mapping = credential_spec_to_mapping(spec);
            tracing::debug!(
                skill = %skill.name(),
                credential = %spec.name,
                hosts = ?spec.hosts,
                "Registering skill credential mapping"
            );
            registry.add_mappings(std::iter::once(mapping));
            count += 1;
        }
    }
    if count > 0 {
        tracing::debug!(count, "Registered skill credential mappings");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironclaw_skills::{
        ActivationCriteria, ActivationScript, SkillManifest, SkillScope,
    };

    #[test]
    fn test_convert_bearer_location() {
        let loc = ironclaw_skills::SkillCredentialLocation::Bearer;
        let converted = convert_credential_location(&loc);
        assert!(matches!(
            converted,
            crate::secrets::CredentialLocation::AuthorizationBearer
        ));
    }

    #[test]
    fn test_convert_basic_auth_location() {
        let loc = ironclaw_skills::SkillCredentialLocation::BasicAuth {
            username: "admin".to_string(),
        };
        let converted = convert_credential_location(&loc);
        match converted {
            crate::secrets::CredentialLocation::AuthorizationBasic { username } => {
                assert_eq!(username, "admin");
            }
            _ => panic!("expected AuthorizationBasic"),
        }
    }

    #[test]
    fn test_convert_header_location() {
        let loc = ironclaw_skills::SkillCredentialLocation::Header {
            name: "X-API-Key".to_string(),
            prefix: Some("Token".to_string()),
        };
        let converted = convert_credential_location(&loc);
        match converted {
            crate::secrets::CredentialLocation::Header { name, prefix } => {
                assert_eq!(name, "X-API-Key");
                assert_eq!(prefix, Some("Token".to_string()));
            }
            _ => panic!("expected Header"),
        }
    }

    #[test]
    fn test_convert_query_param_location() {
        let loc = ironclaw_skills::SkillCredentialLocation::QueryParam {
            name: "key".to_string(),
        };
        let converted = convert_credential_location(&loc);
        match converted {
            crate::secrets::CredentialLocation::QueryParam { name } => {
                assert_eq!(name, "key");
            }
            _ => panic!("expected QueryParam"),
        }
    }

    #[test]
    fn test_credential_spec_to_mapping() {
        let spec = ironclaw_skills::SkillCredentialSpec {
            name: "github_token".to_string(),
            provider: "github".to_string(),
            location: ironclaw_skills::SkillCredentialLocation::Bearer,
            hosts: vec!["api.github.com".to_string(), "*.github.com".to_string()],
            oauth: None,
            setup_instructions: None,
        };
        let mapping = super::credential_spec_to_mapping(&spec);
        assert_eq!(mapping.secret_name, "github_token");
        assert!(matches!(
            mapping.location,
            crate::secrets::CredentialLocation::AuthorizationBearer
        ));
        assert_eq!(mapping.host_patterns.len(), 2);
        assert_eq!(mapping.host_patterns[0], "api.github.com");
    }

    #[test]
    fn test_register_skill_credentials_valid() {
        use ironclaw_skills::types::*;
        use std::path::PathBuf;

        let skill = ironclaw_skills::LoadedSkill {
            manifest: SkillManifest {
                name: "test-api".to_string(),
                version: "1.0.0".to_string(),
                description: "Test".to_string(),
                activation: ActivationCriteria::default(),
                credentials: vec![SkillCredentialSpec {
                    name: "test_token".to_string(),
                    provider: "test".to_string(),
                    location: SkillCredentialLocation::Bearer,
                    hosts: vec!["api.test.com".to_string()],
                    oauth: None,
                    setup_instructions: None,
                }],
                metadata: None,
                scope: None,
            },
            prompt_content: "test".to_string(),
            trust: SkillTrust::Trusted,
            source: SkillSource::User(PathBuf::from("/tmp/test")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_exclude_keywords: vec![],
            lowercased_tags: vec![],
        };

        let registry = crate::tools::wasm::SharedCredentialRegistry::new();
        register_skill_credentials(&[skill], &registry);

        assert!(registry.has_credentials_for_host("api.test.com"));
        assert!(!registry.has_credentials_for_host("other.host.com"));
    }

    #[test]
    fn test_register_skill_credentials_invalid_skipped() {
        use ironclaw_skills::types::*;
        use std::path::PathBuf;

        let skill = ironclaw_skills::LoadedSkill {
            manifest: SkillManifest {
                name: "bad-skill".to_string(),
                version: "1.0.0".to_string(),
                description: "Test".to_string(),
                activation: ActivationCriteria::default(),
                credentials: vec![SkillCredentialSpec {
                    name: "INVALID_NAME".to_string(), // uppercase = invalid
                    provider: "test".to_string(),
                    location: SkillCredentialLocation::Bearer,
                    hosts: vec!["api.test.com".to_string()],
                    oauth: None,
                    setup_instructions: None,
                }],
                metadata: None,
                scope: None,
            },
            prompt_content: "test".to_string(),
            trust: SkillTrust::Trusted,
            source: SkillSource::User(PathBuf::from("/tmp/test")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_exclude_keywords: vec![],
            lowercased_tags: vec![],
        };

        let registry = crate::tools::wasm::SharedCredentialRegistry::new();
        register_skill_credentials(&[skill], &registry);

        // Invalid spec should be skipped — host should NOT be registered
        assert!(!registry.has_credentials_for_host("api.test.com"));
    }

    // ── SkillScope unit tests ──────────────────────────────────────────

    #[test]
    fn test_skill_scope_single_matches() {
        let scope = SkillScope::Single("alice".to_string());
        assert!(scope.matches("alice"));
    }

    #[test]
    fn test_skill_scope_single_no_match() {
        let scope = SkillScope::Single("alice".to_string());
        assert!(!scope.matches("bob"));
    }

    #[test]
    fn test_skill_scope_single_empty_user_id() {
        let scope = SkillScope::Single("alice".to_string());
        assert!(!scope.matches(""));
    }

    #[test]
    fn test_skill_scope_single_case_sensitive() {
        let scope = SkillScope::Single("Alice".to_string());
        assert!(!scope.matches("alice"), "scope matching should be case-sensitive");
        assert!(scope.matches("Alice"));
    }

    #[test]
    fn test_skill_scope_multiple_matches_first() {
        let scope = SkillScope::Multiple(vec!["alice".to_string(), "bob".to_string()]);
        assert!(scope.matches("alice"));
    }

    #[test]
    fn test_skill_scope_multiple_matches_last() {
        let scope = SkillScope::Multiple(vec!["alice".to_string(), "bob".to_string()]);
        assert!(scope.matches("bob"));
    }

    #[test]
    fn test_skill_scope_multiple_no_match() {
        let scope = SkillScope::Multiple(vec!["alice".to_string(), "bob".to_string()]);
        assert!(!scope.matches("charlie"));
    }

    #[test]
    fn test_skill_scope_multiple_empty_list() {
        let scope = SkillScope::Multiple(vec![]);
        assert!(!scope.matches("alice"), "empty scope list should match nobody");
    }

    #[test]
    fn test_skill_scope_serde_single_from_yaml() {
        let yaml = "\"user1\"";
        let scope: SkillScope = serde_yml::from_str(yaml).expect("parse failed");
        assert_eq!(scope, SkillScope::Single("user1".to_string()));
        assert!(scope.matches("user1"));
    }

    #[test]
    fn test_skill_scope_serde_multiple_from_yaml() {
        let yaml = "[\"alice\", \"bob\"]";
        let scope: SkillScope = serde_yml::from_str(yaml).expect("parse failed");
        match &scope {
            SkillScope::Multiple(v) => {
                assert_eq!(v.len(), 2);
                assert!(scope.matches("alice"));
                assert!(scope.matches("bob"));
            }
            _ => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn test_skill_scope_serde_roundtrip_single() {
        let scope = SkillScope::Single("user1".to_string());
        let serialized = serde_json::to_string(&scope).unwrap();
        let deserialized: SkillScope = serde_json::from_str(&serialized).unwrap();
        assert_eq!(scope, deserialized);
    }

    #[test]
    fn test_skill_scope_serde_roundtrip_multiple() {
        let scope = SkillScope::Multiple(vec!["a".to_string(), "b".to_string()]);
        let serialized = serde_json::to_string(&scope).unwrap();
        let deserialized: SkillScope = serde_json::from_str(&serialized).unwrap();
        assert_eq!(scope, deserialized);
    }

    #[test]
    fn test_skill_manifest_with_scope_in_yaml() {
        let yaml = r#"
name: scoped-skill
version: "1.0"
description: A skill scoped to one user
scope: "andrew"
activation:
  keywords: [test]
"#;
        let manifest: SkillManifest = serde_yml::from_str(yaml).expect("parse failed");
        assert_eq!(manifest.scope, Some(SkillScope::Single("andrew".to_string())));
    }

    #[test]
    fn test_skill_manifest_with_multi_scope_in_yaml() {
        let yaml = r#"
name: shared-skill
version: "1.0"
description: A skill scoped to multiple users
scope:
  - alice
  - bob
activation:
  keywords: [test]
"#;
        let manifest: SkillManifest = serde_yml::from_str(yaml).expect("parse failed");
        match &manifest.scope {
            Some(SkillScope::Multiple(v)) => {
                assert_eq!(v.len(), 2);
                assert!(v.contains(&"alice".to_string()));
                assert!(v.contains(&"bob".to_string()));
            }
            other => panic!("expected Some(Multiple), got {:?}", other),
        }
    }

    #[test]
    fn test_skill_manifest_without_scope_defaults_to_none() {
        let yaml = r#"
name: universal-skill
version: "1.0"
description: Available to all users
activation:
  keywords: [test]
"#;
        let manifest: SkillManifest = serde_yml::from_str(yaml).expect("parse failed");
        assert_eq!(manifest.scope, None);
    }

    // ── ActivationScript unit tests ────────────────────────────────────

    #[test]
    fn test_activation_script_default_timeout() {
        let script = ActivationScript {
            language: "bash".to_string(),
            source: Some("echo hi".to_string()),
            source_file: None,
            timeout_ms: 5000,
            max_output_bytes: 4096,
        };
        assert_eq!(script.timeout_ms, 5000);
        assert_eq!(script.max_output_bytes, 4096);
    }

    #[test]
    fn test_activation_script_serde_roundtrip() {
        let script = ActivationScript {
            language: "python".to_string(),
            source: Some("print('hello')".to_string()),
            source_file: None,
            timeout_ms: 3000,
            max_output_bytes: 2048,
        };
        let json = serde_json::to_string(&script).unwrap();
        let deserialized: ActivationScript = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.language, "python");
        assert_eq!(deserialized.source, Some("print('hello')".to_string()));
        assert_eq!(deserialized.timeout_ms, 3000);
    }

    #[test]
    fn test_activation_criteria_with_script_in_yaml() {
        let yaml = r#"
keywords: [deploy]
script:
  language: bash
  source: "echo $IRONCLAW_USER_ID"
  timeout_ms: 2000
  max_output_bytes: 1024
"#;
        let criteria: ActivationCriteria = serde_yml::from_str(yaml).expect("parse failed");
        assert!(criteria.script.is_some());
        let script = criteria.script.unwrap();
        assert_eq!(script.language, "bash");
        assert_eq!(script.timeout_ms, 2000);
    }

    #[test]
    fn test_activation_criteria_with_tools_prefix_in_yaml() {
        let yaml = r#"
keywords: [collection]
tools_prefix: "collection_"
"#;
        let criteria: ActivationCriteria = serde_yml::from_str(yaml).expect("parse failed");
        assert_eq!(criteria.tools_prefix, Some("collection_".to_string()));
    }

    #[test]
    fn test_activation_criteria_tools_prefix_defaults_to_none() {
        let yaml = r#"
keywords: [test]
"#;
        let criteria: ActivationCriteria = serde_yml::from_str(yaml).expect("parse failed");
        assert_eq!(criteria.tools_prefix, None);
    }
}
