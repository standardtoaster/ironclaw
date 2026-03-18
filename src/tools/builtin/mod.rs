//! Built-in tools that come with the agent.

pub mod ask_user;
pub mod collections;
mod create_workspace;
mod delegate_workspace;
mod discover_tools;
mod list_workspaces;
mod search_workspace_history;
mod set_workspace_topic;
mod workspace_summary;
mod echo;
pub mod escalate;
pub mod extension_tools;
mod file;
mod http;
mod job;
mod json;
mod memory;
mod message;
pub mod path_utils;
mod restart;
pub mod routine;
pub mod secrets_tools;
pub(crate) mod shell;
pub mod skill_tools;
mod time;

pub use collections::{
    CollectionDropTool, CollectionListTool, CollectionRegisterTool, CollectionsAlterTool,
    generate_collection_tools,
};
pub use create_workspace::CreateWorkspaceTool;
pub use delegate_workspace::DelegateToWorkspaceTool;
pub use discover_tools::DiscoverToolsTool;
pub use list_workspaces::ListWorkspacesTool;
pub use search_workspace_history::SearchWorkspaceHistoryTool;
pub use set_workspace_topic::SetWorkspaceTopicTool;
pub use workspace_summary::WorkspaceSummaryTool;
pub use echo::EchoTool;
pub use escalate::{DeescalateTool, EscalateTool};
pub use extension_tools::{
    ExtensionInfoTool, ToolActivateTool, ToolAuthTool, ToolInstallTool, ToolListTool,
    ToolRemoveTool, ToolSearchTool, ToolUpgradeTool,
};
pub use file::{ApplyPatchTool, ListDirTool, ReadFileTool, WriteFileTool};
pub use http::HttpTool;
pub use job::{
    CancelJobTool, CreateJobTool, JobEventsTool, JobPromptTool, JobStatusTool, ListJobsTool,
    PromptQueue, SchedulerSlot,
};
pub use json::JsonTool;
pub use memory::{MemoryReadTool, MemorySearchTool, MemoryTreeTool, MemoryWriteTool};
pub use message::MessageTool;
pub use restart::RestartTool;
pub use routine::{
    RoutineCreateTool, RoutineDeleteTool, RoutineFireTool, RoutineHistoryTool, RoutineListTool,
    RoutineUpdateTool,
};
pub use secrets_tools::{SecretDeleteTool, SecretListTool};
pub use shell::ShellTool;
pub use skill_tools::{SkillInstallTool, SkillListTool, SkillRemoveTool, SkillSearchTool};
pub use time::TimeTool;
mod html_converter;
pub mod image_analyze;
pub mod image_edit;
pub mod image_gen;

pub use html_converter::convert_html_to_markdown;
pub use image_analyze::ImageAnalyzeTool;
pub use image_edit::ImageEditTool;
pub use ask_user::AskUserTool;
pub use image_gen::ImageGenerateTool;

/// Detect image media type from file extension via `mime_guess`.
/// Falls back to `image/jpeg` for unrecognized or non-image extensions.
pub(crate) fn media_type_from_path(path: &str) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .filter(|m| m.starts_with("image/"))
        .unwrap_or("image/jpeg")
        .to_string()
}
