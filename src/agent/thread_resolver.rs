//! Generic thread resolution trait.
//!
//! Allows pluggable logic for deciding which conversation thread a message
//! should be processed in, and injecting additional context into the system
//! prompt. The default behavior (no resolver) processes messages in their
//! session's active thread with no extra context.
//!
//! Implementations can use this to build workspace routing, topic-based
//! context switching, or any other message-to-thread mapping strategy.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

/// Result of resolving which thread a message should be processed in.
#[derive(Debug, Default)]
pub struct ThreadResolution {
    /// Which conversation thread to process the message in.
    /// If `None`, use the session's default thread (no redirect).
    pub thread_id: Option<Uuid>,

    /// Extra context to inject into the system prompt.
    /// Appended after the identity section, before skills.
    pub context: Option<String>,

    /// Routing metadata for SSE events and logging.
    pub metadata: HashMap<String, String>,
}

/// Information about a workspace created or updated by the organizer.
#[derive(Debug)]
pub struct WorkspaceInfo {
    /// Workspace ID.
    pub id: Uuid,
    /// Short topic label.
    pub topic: String,
    /// Rich description used for embedding.
    pub description: String,
}

/// Result of running the organizer.
#[derive(Debug, Default)]
pub struct OrganizeResult {
    /// Workspaces created during this run.
    pub created: Vec<WorkspaceInfo>,
    /// Workspaces whose descriptions were updated.
    pub updated: Vec<WorkspaceInfo>,
    /// Number of messages analyzed.
    pub messages_analyzed: usize,
}

/// Resolves which thread a message should be processed in.
///
/// Called by the dispatcher before thread hydration and the agentic loop.
/// Implementations can redirect messages to different threads (e.g. topic
/// workspaces) and inject context into the system prompt.
#[async_trait]
pub trait ThreadResolver: Send + Sync {
    /// Resolve which thread to use for this message.
    ///
    /// # Arguments
    /// * `user_id` — the authenticated user
    /// * `message_content` — the user's message text
    /// * `default_thread_id` — the session's current thread
    ///
    /// # Returns
    /// A `ThreadResolution` indicating where to route and what context to add.
    /// Return `ThreadResolution::default()` to use the default thread with no
    /// extra context (passthrough behavior).
    async fn resolve(
        &self,
        user_id: &str,
        message_content: &str,
        default_thread_id: Uuid,
    ) -> Result<ThreadResolution, ResolverError>;

    /// Run the organizer: analyze recent messages and create/update workspaces.
    ///
    /// Called by the `/organize` command, cron routines, and the test harness.
    /// All entry points use this same function.
    async fn organize(&self, user_id: &str) -> Result<OrganizeResult, ResolverError>;

    /// Notify the resolver that a message was processed in a specific thread.
    ///
    /// Called after the agentic loop completes. Implementations use this to
    /// update stickiness state (active workspace, last message time).
    async fn notify_routed(&self, user_id: &str, thread_id: Uuid) {
        // Default: no-op. Implementations override for stickiness tracking.
        let _ = (user_id, thread_id);
    }
}

/// Errors from thread resolution or organization.
#[derive(Debug, thiserror::Error)]
pub enum ResolverError {
    #[error("embedding failed: {0}")]
    Embedding(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("LLM classification failed: {0}")]
    Classification(String),

    #[error("{0}")]
    Other(String),
}
