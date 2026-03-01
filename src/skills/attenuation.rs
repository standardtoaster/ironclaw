//! Trust-based tool filtering (authority attenuation).
//!
//! The core defense mechanism: the minimum trust level of any active skill
//! determines a *tool ceiling* -- tools above the ceiling are removed from
//! the LLM's tool list entirely. The LLM cannot be manipulated into calling
//! a tool it doesn't know exists.
//!
//! | Trust State        | Tool Ceiling                                      |
//! |--------------------|---------------------------------------------------|
//! | No skills active   | All tools (normal behavior)                       |
//! | Trusted only       | All tools (user placed these, full trust)         |
//! | Installed present  | Read-only tools ONLY                              |

use std::collections::HashSet;

use crate::llm::ToolDefinition;
use crate::skills::{LoadedSkill, SkillTrust};
use crate::tools::ToolGroup;

/// Tools that are always safe -- read-only, no side effects.
///
/// **Maintenance note**: This list is intentionally hardcoded and conservative.
/// When adding new tools to IronClaw, they default to *excluded* from the
/// read-only list (i.e., blocked under Installed ceilings). A tool
/// should only be added here if it is provably free of side effects -- it must
/// not write files, make network requests, execute commands, or modify any state.
/// Review by the security team is required before expanding this list.
///
const READ_ONLY_TOOLS: &[&str] = &[
    "memory_search",
    "memory_read",
    "memory_tree",
    "time",
    "echo",
    "json",
    "skill_list",
    "skill_search",
];

/// Result of tool attenuation, including transparency information.
#[derive(Debug, Clone)]
pub struct AttenuationResult {
    /// The filtered tool definitions to send to the LLM.
    pub tools: Vec<ToolDefinition>,
    /// The minimum trust level across all active skills.
    pub min_trust: SkillTrust,
    /// Human-readable explanation of what was removed and why.
    pub explanation: String,
    /// Names of tools that were removed.
    pub removed_tools: Vec<String>,
}

/// Filter tool definitions based on the trust level of active skills.
///
/// This is the hard security gate: tools above the trust ceiling are removed
/// from the tool list before it reaches the LLM. The LLM cannot call tools
/// it doesn't know exist, regardless of what a skill prompt instructs.
pub fn attenuate_tools(
    tools: &[ToolDefinition],
    active_skills: &[LoadedSkill],
) -> AttenuationResult {
    // No active skills = no attenuation
    if active_skills.is_empty() {
        return AttenuationResult {
            tools: tools.to_vec(),
            min_trust: SkillTrust::Trusted,
            explanation: "No skills active, all tools available".to_string(),
            removed_tools: vec![],
        };
    }

    // Compute minimum trust across all active skills
    let min_trust = active_skills
        .iter()
        .map(|s| s.trust)
        .min()
        .unwrap_or(SkillTrust::Trusted);

    match min_trust {
        SkillTrust::Trusted => {
            // Trusted skills have full trust -- no filtering
            AttenuationResult {
                tools: tools.to_vec(),
                min_trust,
                explanation: "All active skills are trusted (full trust), all tools available"
                    .to_string(),
                removed_tools: vec![],
            }
        }
        SkillTrust::Installed => {
            // Installed: read-only tools ONLY
            let mut kept = Vec::new();
            let mut removed = Vec::new();

            for tool in tools {
                if READ_ONLY_TOOLS.contains(&tool.name.as_str()) {
                    kept.push(tool.clone());
                } else {
                    removed.push(tool.name.clone());
                }
            }

            let explanation = format!(
                "Installed skill present: restricted to read-only tools, removed {} tool(s): {}",
                removed.len(),
                removed.join(", ")
            );

            AttenuationResult {
                tools: kept,
                min_trust,
                explanation,
                removed_tools: removed,
            }
        }
    }
}

// ==================== Tool Visibility Tiers ====================

/// Map a tool name to its visibility group.
///
/// Built-in tools are assigned centrally here. Unknown tools (WASM, MCP,
/// dynamic) default to `Core` (always visible) since they were explicitly
/// installed or generated.
pub fn tool_group_for_name(name: &str) -> ToolGroup {
    match name {
        // Core (always visible)
        "time" | "message" => ToolGroup::Core,

        // Memory (always visible)
        "memory_search" | "memory_write" | "memory_read" | "memory_tree" => ToolGroup::Memory,

        // Collections management (always visible)
        "collections_list" | "collections_register" | "collections_drop" | "collections_alter" => {
            ToolGroup::Collections
        }

        // Dev (on-demand)
        "shell" | "read_file" | "write_file" | "list_dir" | "apply_patch" => ToolGroup::Dev,

        // Jobs (on-demand)
        "create_job" | "list_jobs" | "job_status" | "cancel_job" | "job_events" | "job_prompt" => {
            ToolGroup::Jobs
        }

        // Extensions (on-demand)
        "tool_search" | "tool_install" | "tool_auth" | "tool_activate" | "tool_list"
        | "tool_remove" => ToolGroup::Extensions,

        // Skills (on-demand)
        "skill_list" | "skill_search" | "skill_install" | "skill_remove" => ToolGroup::Skills,

        // Routines (on-demand)
        "routine_create" | "routine_list" | "routine_update" | "routine_delete"
        | "routine_history" => ToolGroup::Routines,

        // Utility (on-demand)
        "echo" | "json" | "http" | "web_fetch" | "build_software" => ToolGroup::Utility,

        // Unknown/dynamic tools: always visible
        _ => ToolGroup::Core,
    }
}

/// Keywords in skill prompt content that activate each on-demand tool group.
///
/// Uses tool names rather than generic words to avoid false-positive activation.
/// Case-insensitive substring match against the skill's prompt content.
const GROUP_KEYWORDS: &[(ToolGroup, &[&str])] = &[
    (
        ToolGroup::Dev,
        &[
            "shell",
            "read_file",
            "write_file",
            "list_dir",
            "apply_patch",
        ],
    ),
    (
        ToolGroup::Jobs,
        &["create_job", "list_jobs", "job_status", "cancel_job"],
    ),
    (
        ToolGroup::Extensions,
        &[
            "tool_install",
            "tool_search",
            "tool_auth",
            "tool_activate",
            "mcp server",
        ],
    ),
    (
        ToolGroup::Skills,
        &[
            "skill_list",
            "skill_search",
            "skill_install",
            "skill_remove",
        ],
    ),
    (
        ToolGroup::Routines,
        &[
            "routine_create",
            "routine_list",
            "routine_update",
            "cron",
        ],
    ),
    (
        ToolGroup::Utility,
        &["echo", "build_software"],
    ),
];

/// Scan active skill prompts and return the set of on-demand tool groups
/// that should be made visible.
fn activated_groups(active_skills: &[LoadedSkill]) -> HashSet<ToolGroup> {
    let mut groups = HashSet::new();

    for skill in active_skills {
        let content_lower = skill.prompt_content.to_lowercase();
        for &(group, keywords) in GROUP_KEYWORDS {
            if keywords
                .iter()
                .any(|kw: &&str| content_lower.contains(*kw))
            {
                groups.insert(group);
            }
        }
    }

    groups
}

/// Filter tool definitions based on visibility tiers.
///
/// When skills are active, only tools in always-visible groups or in groups
/// activated by skill content are shown. When no skills are active, all tools
/// are returned (backward compatibility).
pub fn filter_tools_by_visibility(
    tools: &[ToolDefinition],
    active_skills: &[LoadedSkill],
) -> Vec<ToolDefinition> {
    if active_skills.is_empty() {
        return tools.to_vec();
    }

    let active_groups = activated_groups(active_skills);

    let mut kept = Vec::new();
    let mut removed = Vec::new();

    for tool in tools {
        let group = tool_group_for_name(&tool.name);
        if group.is_always_visible() || active_groups.contains(&group) {
            kept.push(tool.clone());
        } else {
            removed.push(tool.name.clone());
        }
    }

    if !removed.is_empty() {
        tracing::info!(
            removed_count = removed.len(),
            active_groups = ?active_groups,
            removed = ?removed,
            "Tool visibility filtering: hidden {} on-demand tool(s)",
            removed.len()
        );
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::{ActivationCriteria, SkillManifest, SkillSource};
    use std::path::PathBuf;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{} tool", name),
            parameters: serde_json::json!({}),
        }
    }

    fn make_skill_with_trust(name: &str, trust: SkillTrust) -> LoadedSkill {
        LoadedSkill {
            manifest: SkillManifest {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: String::new(),
                activation: ActivationCriteria::default(),
                metadata: None,
            },
            prompt_content: "test".to_string(),
            trust,
            source: SkillSource::User(PathBuf::from("/tmp")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_tags: vec![],
        }
    }

    fn all_tools() -> Vec<ToolDefinition> {
        vec![
            make_tool("shell"),
            make_tool("http"),
            make_tool("memory_write"),
            make_tool("memory_search"),
            make_tool("memory_read"),
            make_tool("memory_tree"),
            make_tool("time"),
            make_tool("echo"),
            make_tool("json"),
        ]
    }

    #[test]
    fn test_no_skills_returns_all_tools() {
        let tools = all_tools();
        let result = attenuate_tools(&tools, &[]);
        assert_eq!(result.tools.len(), tools.len());
        assert!(result.removed_tools.is_empty());
    }

    #[test]
    fn test_trusted_skills_no_filtering() {
        let tools = all_tools();
        let skills = vec![make_skill_with_trust("trusted_skill", SkillTrust::Trusted)];
        let result = attenuate_tools(&tools, &skills);
        assert_eq!(result.tools.len(), tools.len());
        assert!(result.removed_tools.is_empty());
        assert_eq!(result.min_trust, SkillTrust::Trusted);
    }

    #[test]
    fn test_installed_only_read_only() {
        let tools = all_tools();
        let skills = vec![make_skill_with_trust(
            "installed_skill",
            SkillTrust::Installed,
        )];
        let result = attenuate_tools(&tools, &skills);

        let kept_names: Vec<&str> = result.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(!kept_names.contains(&"shell"));
        assert!(!kept_names.contains(&"http"));
        assert!(!kept_names.contains(&"memory_write"));
        assert!(kept_names.contains(&"memory_search"));
        assert!(kept_names.contains(&"memory_read"));
        assert!(kept_names.contains(&"time"));
        assert_eq!(result.min_trust, SkillTrust::Installed);
    }

    #[test]
    fn test_mixed_trust_drops_to_lowest() {
        let tools = all_tools();
        let skills = vec![
            make_skill_with_trust("trusted_skill", SkillTrust::Trusted),
            make_skill_with_trust("installed_skill", SkillTrust::Installed),
        ];
        let result = attenuate_tools(&tools, &skills);

        // Mixed: installed + trusted = installed ceiling
        assert_eq!(result.min_trust, SkillTrust::Installed);
        let kept_names: Vec<&str> = result.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(!kept_names.contains(&"shell"));
    }

    #[test]
    fn test_attenuation_result_has_explanation() {
        let tools = vec![make_tool("shell"), make_tool("time")];
        let skills = vec![make_skill_with_trust("installed", SkillTrust::Installed)];
        let result = attenuate_tools(&tools, &skills);

        assert!(!result.explanation.is_empty());
        assert!(result.removed_tools.contains(&"shell".to_string()));
        assert!(!result.removed_tools.contains(&"time".to_string()));
    }

    // ==================== Visibility Tier Tests ====================

    fn make_skill_with_content(name: &str, content: &str) -> LoadedSkill {
        LoadedSkill {
            manifest: SkillManifest {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                description: String::new(),
                activation: ActivationCriteria::default(),
                metadata: None,
            },
            prompt_content: content.to_string(),
            trust: SkillTrust::Trusted,
            source: SkillSource::User(PathBuf::from("/tmp")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_tags: vec![],
        }
    }

    fn mixed_tools() -> Vec<ToolDefinition> {
        vec![
            // Core (always visible)
            make_tool("time"),
            make_tool("message"),
            // Memory (always visible)
            make_tool("memory_search"),
            make_tool("memory_write"),
            // Collections (always visible)
            make_tool("collections_register"),
            // Dev (on-demand)
            make_tool("shell"),
            make_tool("read_file"),
            // Jobs (on-demand)
            make_tool("create_job"),
            make_tool("list_jobs"),
            // Extensions (on-demand)
            make_tool("tool_install"),
            // Skills (on-demand)
            make_tool("skill_list"),
            // Routines (on-demand)
            make_tool("routine_create"),
            // Utility (on-demand)
            make_tool("echo"),
            make_tool("json"),
        ]
    }

    #[test]
    fn test_visibility_no_skills_returns_all() {
        let tools = mixed_tools();
        let result = filter_tools_by_visibility(&tools, &[]);
        assert_eq!(result.len(), tools.len());
    }

    #[test]
    fn test_visibility_skill_hides_on_demand() {
        let tools = mixed_tools();
        // Skill that mentions nothing about on-demand tool groups
        let skills = vec![make_skill_with_content(
            "grocery",
            "Track your grocery items. Use collections_register to create a list.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Always visible
        assert!(names.contains(&"time"));
        assert!(names.contains(&"message"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_write"));
        assert!(names.contains(&"collections_register"));

        // On-demand: hidden since skill doesn't reference their tool names
        assert!(!names.contains(&"shell"));
        assert!(!names.contains(&"read_file"));
        assert!(!names.contains(&"create_job"));
        assert!(!names.contains(&"list_jobs"));
        assert!(!names.contains(&"tool_install"));
        assert!(!names.contains(&"skill_list"));
        assert!(!names.contains(&"routine_create"));
        assert!(!names.contains(&"echo"));
        assert!(!names.contains(&"json"));
    }

    #[test]
    fn test_visibility_skill_activates_group_by_keyword() {
        let tools = mixed_tools();
        // Skill that references shell and create_job tool names
        let skills = vec![make_skill_with_content(
            "dev-helper",
            "Use the shell to run commands. Use create_job for background work.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Dev group activated by "shell"
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"read_file"));
        // Jobs group activated by "create_job"
        assert!(names.contains(&"create_job"));
        assert!(names.contains(&"list_jobs"));
        // Still hidden
        assert!(!names.contains(&"tool_install"));
        assert!(!names.contains(&"routine_create"));
    }

    #[test]
    fn test_visibility_case_insensitive() {
        let tools = mixed_tools();
        let skills = vec![make_skill_with_content(
            "scheduler",
            "Use ROUTINE_CREATE for recurring tasks. Set up CRON schedules.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"routine_create"));
    }

    #[test]
    fn test_visibility_generic_words_dont_trigger() {
        let tools = mixed_tools();
        // Skill that uses generic words like "json", "skill", "schedule"
        // but doesn't reference actual tool names
        let skills = vec![make_skill_with_content(
            "grocery",
            "Track your grocery items. Uses JSON schema for validation. \
             This skill helps you manage shopping lists.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Generic words shouldn't trigger on-demand groups
        assert!(!names.contains(&"routine_create"));
        assert!(!names.contains(&"skill_list"));
        // "echo" is a keyword for Utility, but "echo" doesn't appear in prompt
        assert!(!names.contains(&"echo"));
    }

    #[test]
    fn test_visibility_multiple_skills_combine() {
        let tools = mixed_tools();
        let skills = vec![
            make_skill_with_content("dev", "Use the shell to run builds."),
            make_skill_with_content("ext", "Use tool_install to add an MCP server."),
        ];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Dev from first skill
        assert!(names.contains(&"shell"));
        // Extensions from second skill
        assert!(names.contains(&"tool_install"));
        // Routines still hidden
        assert!(!names.contains(&"routine_create"));
    }

    #[test]
    fn test_tool_group_mapping() {
        assert!(tool_group_for_name("time").is_always_visible());
        assert!(tool_group_for_name("memory_search").is_always_visible());
        assert!(tool_group_for_name("collections_list").is_always_visible());
        assert!(!tool_group_for_name("shell").is_always_visible());
        assert!(!tool_group_for_name("create_job").is_always_visible());
        assert!(!tool_group_for_name("routine_create").is_always_visible());
        // Unknown tools default to Core (always visible)
        assert!(tool_group_for_name("my_custom_wasm_tool").is_always_visible());
    }

    #[test]
    fn test_visibility_unknown_tools_always_visible() {
        let tools = vec![
            make_tool("time"),
            make_tool("my_wasm_tool"),
            make_tool("custom_mcp_tool"),
            make_tool("shell"),
        ];
        let skills = vec![make_skill_with_content(
            "basic",
            "A simple skill with no tool references.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Unknown tools are always visible (Core group)
        assert!(names.contains(&"my_wasm_tool"));
        assert!(names.contains(&"custom_mcp_tool"));
        // Known on-demand tool is hidden
        assert!(!names.contains(&"shell"));
    }

    /// Every built-in tool name should map to its expected group. This catches
    /// misclassification and serves as a reference for which tools are in which tier.
    #[test]
    fn test_all_builtin_tools_have_correct_group() {
        let expected: &[(&str, ToolGroup)] = &[
            // Core (always visible)
            ("time", ToolGroup::Core),
            ("message", ToolGroup::Core),
            // Memory (always visible)
            ("memory_search", ToolGroup::Memory),
            ("memory_write", ToolGroup::Memory),
            ("memory_read", ToolGroup::Memory),
            ("memory_tree", ToolGroup::Memory),
            // Collections (always visible)
            ("collections_list", ToolGroup::Collections),
            ("collections_register", ToolGroup::Collections),
            ("collections_drop", ToolGroup::Collections),
            ("collections_alter", ToolGroup::Collections),
            // Dev (on-demand)
            ("shell", ToolGroup::Dev),
            ("read_file", ToolGroup::Dev),
            ("write_file", ToolGroup::Dev),
            ("list_dir", ToolGroup::Dev),
            ("apply_patch", ToolGroup::Dev),
            // Jobs (on-demand)
            ("create_job", ToolGroup::Jobs),
            ("list_jobs", ToolGroup::Jobs),
            ("job_status", ToolGroup::Jobs),
            ("cancel_job", ToolGroup::Jobs),
            ("job_events", ToolGroup::Jobs),
            ("job_prompt", ToolGroup::Jobs),
            // Extensions (on-demand)
            ("tool_search", ToolGroup::Extensions),
            ("tool_install", ToolGroup::Extensions),
            ("tool_auth", ToolGroup::Extensions),
            ("tool_activate", ToolGroup::Extensions),
            ("tool_list", ToolGroup::Extensions),
            ("tool_remove", ToolGroup::Extensions),
            // Skills (on-demand)
            ("skill_list", ToolGroup::Skills),
            ("skill_search", ToolGroup::Skills),
            ("skill_install", ToolGroup::Skills),
            ("skill_remove", ToolGroup::Skills),
            // Routines (on-demand)
            ("routine_create", ToolGroup::Routines),
            ("routine_list", ToolGroup::Routines),
            ("routine_update", ToolGroup::Routines),
            ("routine_delete", ToolGroup::Routines),
            ("routine_history", ToolGroup::Routines),
            // Utility (on-demand)
            ("echo", ToolGroup::Utility),
            ("json", ToolGroup::Utility),
            ("http", ToolGroup::Utility),
            ("web_fetch", ToolGroup::Utility),
            ("build_software", ToolGroup::Utility),
        ];

        for &(tool_name, expected_group) in expected {
            let actual = tool_group_for_name(tool_name);
            assert_eq!(
                actual, expected_group,
                "Tool '{}' expected {:?} but got {:?}",
                tool_name, expected_group, actual
            );
        }
    }

    /// Dynamic/external tools should fall through to Core (always visible).
    #[test]
    fn test_dynamic_tools_default_to_core() {
        assert_eq!(tool_group_for_name("my_wasm_tool"), ToolGroup::Core);
        assert_eq!(tool_group_for_name("custom_mcp_server"), ToolGroup::Core);
        assert_eq!(tool_group_for_name("grocery_items_add"), ToolGroup::Core);
    }

    /// Empty skill prompt should activate no on-demand groups.
    #[test]
    fn test_visibility_empty_prompt_activates_nothing() {
        let tools = mixed_tools();
        let skills = vec![make_skill_with_content("empty", "")];
        let result = filter_tools_by_visibility(&tools, &skills);
        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();

        // Only always-visible tools should be present
        assert!(names.contains(&"time"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"collections_register"));
        assert!(!names.contains(&"shell"));
        assert!(!names.contains(&"create_job"));
        assert!(!names.contains(&"routine_create"));
    }

    /// Skill that references every group keyword should activate all groups.
    #[test]
    fn test_visibility_all_groups_activated() {
        let tools = mixed_tools();
        let skills = vec![make_skill_with_content(
            "kitchen-sink",
            "Use shell to run commands. Use create_job for background work. \
             Use tool_install for extensions. Use skill_list to browse. \
             Use routine_create for scheduling. Use echo for testing.",
        )];
        let result = filter_tools_by_visibility(&tools, &skills);
        // Every tool should be visible when all groups are active
        assert_eq!(result.len(), tools.len());
    }

    /// Visibility filtering and trust attenuation work correctly together.
    /// Visibility runs first (hiding irrelevant tools), then trust attenuation
    /// further restricts based on skill trust level.
    #[test]
    fn test_visibility_and_attenuation_compose() {
        let tools = mixed_tools();

        // Installed skill that mentions shell — activates Dev group
        let skills = vec![LoadedSkill {
            manifest: SkillManifest {
                name: "dev-helper".to_string(),
                version: "1.0.0".to_string(),
                description: String::new(),
                activation: ActivationCriteria::default(),
                metadata: None,
            },
            prompt_content: "Use the shell to run commands.".to_string(),
            trust: SkillTrust::Installed,
            source: SkillSource::User(PathBuf::from("/tmp")),
            content_hash: "sha256:000".to_string(),
            compiled_patterns: vec![],
            lowercased_keywords: vec![],
            lowercased_tags: vec![],
        }];

        // Step 1: Visibility filtering
        let after_visibility = filter_tools_by_visibility(&tools, &skills);
        let vis_names: Vec<&str> = after_visibility.iter().map(|t| t.name.as_str()).collect();
        // Shell should be visible (Dev group activated)
        assert!(vis_names.contains(&"shell"));
        // Jobs should be hidden (not activated)
        assert!(!vis_names.contains(&"create_job"));

        // Step 2: Trust attenuation on the visibility-filtered set
        let after_attenuation = attenuate_tools(&after_visibility, &skills);
        let att_names: Vec<&str> = after_attenuation
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        // Shell should now be removed by trust attenuation (not in READ_ONLY_TOOLS)
        assert!(!att_names.contains(&"shell"));
        // Read-only tools survive both filters
        assert!(att_names.contains(&"memory_search"));
        assert!(att_names.contains(&"time"));
        // Installed trust level
        assert_eq!(after_attenuation.min_trust, SkillTrust::Installed);
    }

    /// Verify each on-demand group has at least one keyword that activates it.
    #[test]
    fn test_every_on_demand_group_is_activatable() {
        let on_demand_groups = [
            (ToolGroup::Dev, "shell"),
            (ToolGroup::Jobs, "create_job"),
            (ToolGroup::Extensions, "tool_install"),
            (ToolGroup::Skills, "skill_list"),
            (ToolGroup::Routines, "routine_create"),
            (ToolGroup::Utility, "echo"),
        ];

        for (expected_group, keyword) in on_demand_groups {
            let skills = vec![make_skill_with_content(
                "test",
                &format!("Use {} to do things.", keyword),
            )];
            let groups = activated_groups(&skills);
            assert!(
                groups.contains(&expected_group),
                "Keyword '{}' should activate {:?} but got {:?}",
                keyword,
                expected_group,
                groups
            );
        }
    }
}
