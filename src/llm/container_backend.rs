//! Trait abstraction for Claude Code session management backends.
//!
//! `ContainerBackend` defines the interface for managing Claude Code sessions.
//! Two implementations exist:
//! - `ContainerPool` (Docker): manages Docker containers via bollard
//! - `SupervisorBackend` (supervisor): delegates to a compute supervisor HTTP API

use async_trait::async_trait;
use uuid::Uuid;

use crate::llm::claude_protocol::ExchangeResult;
use crate::llm::error::LlmError;

/// Info about a created session (returned by supervisor backend).
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Session identifier assigned by the backend.
    pub session_id: String,
    /// URL of the channel endpoint (supervisor backend provides this).
    pub channel_url: Option<String>,
}

/// Abstraction over how Claude Code sessions are managed.
///
/// Docker backend creates containers; supervisor backend calls an HTTP API.
/// Both track per-thread sessions and support NDJSON-based exchanges.
#[async_trait]
pub trait ContainerBackend: Send + Sync {
    /// Ensure a session exists for this thread. Creates one if needed.
    async fn get_or_create(&self, thread_id: Uuid) -> Result<(), LlmError>;

    /// Send a prompt and get a response.
    ///
    /// For Docker: write to stdin, read NDJSON from stdout.
    /// For supervisor: POST to percy-channel, wait for callback.
    async fn exchange(
        &self,
        thread_id: Uuid,
        prompt: &str,
    ) -> Result<ExchangeResult, LlmError>;

    /// Remove/destroy a session.
    async fn remove_session(&self, thread_id: Uuid) -> Result<(), LlmError>;

    /// Shut down all sessions.
    async fn shutdown_all(&self) -> Result<(), LlmError>;

    /// Number of active sessions.
    async fn session_count(&self) -> usize;

    /// Messages sent in a session (for delta tracking).
    async fn messages_sent(&self, thread_id: &Uuid) -> Option<usize>;

    /// Update messages_sent counter.
    async fn set_messages_sent(&self, thread_id: &Uuid, count: usize);

    /// Whether the session for this thread was resumed from a previous CLI session.
    /// When true, the caller should skip replaying existing messages — Claude
    /// already has the conversation history from the resumed session.
    async fn is_resumed(&self, thread_id: &Uuid) -> bool {
        false
    }
}
