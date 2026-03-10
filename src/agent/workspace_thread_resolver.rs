//! Workspace-aware implementation of [`ThreadResolver`].
//!
//! Routes messages to workspace threads by embedding similarity, with
//! conversation stickiness and collision handling. Delegates to the
//! auto-organizer for workspace creation/updates.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::agent::thread_resolver::{
    OrganizeResult, ResolverError, ThreadResolution, ThreadResolver,
};
use crate::agent::workspace_router::WorkspaceRouter;
use crate::db::{AgentWorkspace, AgentWorkspaceStore};

/// Per-user stickiness state for conversation continuity.
struct StickinessState {
    /// Currently active workspace (most recent routing destination).
    active_workspace_id: Option<Uuid>,
    /// Conversation ID of the active workspace (for routing).
    active_conversation_id: Option<Uuid>,
    /// Topic of the active workspace (for logging).
    active_topic: Option<String>,
    /// When the last message was processed.
    last_message_at: Option<Instant>,
}

/// Configuration for the workspace thread resolver.
pub struct WorkspaceResolverConfig {
    /// Embedding similarity above which a match is considered strong (0.75).
    pub high_threshold: f64,
    /// Embedding similarity below which we fall back to default (0.5).
    pub low_threshold: f64,
    /// Time gap after which stickiness resets (30 min).
    pub stickiness_timeout: Duration,
    /// Maximum word count for a message to be considered "reply-like".
    pub reply_max_words: usize,
}

impl Default for WorkspaceResolverConfig {
    fn default() -> Self {
        Self {
            high_threshold: 0.75,
            low_threshold: 0.5,
            stickiness_timeout: Duration::from_secs(30 * 60),
            reply_max_words: 15,
        }
    }
}

/// Implements `ThreadResolver` with workspace routing, stickiness, and auto-organization.
pub struct WorkspaceThreadResolver {
    router: Arc<WorkspaceRouter>,
    db: Arc<dyn AgentWorkspaceStore>,
    config: WorkspaceResolverConfig,
    /// Per-user stickiness tracking (in-memory, resets on restart).
    stickiness: RwLock<HashMap<String, StickinessState>>,
}

impl WorkspaceThreadResolver {
    pub fn new(
        router: Arc<WorkspaceRouter>,
        db: Arc<dyn AgentWorkspaceStore>,
        config: WorkspaceResolverConfig,
    ) -> Self {
        Self {
            router,
            db,
            config,
            stickiness: RwLock::new(HashMap::new()),
        }
    }
}

/// Reply detection heuristics.
fn is_reply_like(message: &str, max_words: usize) -> bool {
    let words: Vec<&str> = message.split_whitespace().collect();
    if words.len() > max_words {
        return false;
    }

    // Pronouns and anaphora that reference prior context
    const ANAPHORA: &[&str] = &[
        "it", "that", "there", "this", "them", "those", "the one", "she", "he", "they",
    ];
    // Follow-up markers
    const FOLLOW_UP: &[&str] = &[
        "actually", "also", "what about", "and", "oh", "how about", "but", "or", "plus",
        "btw", "by the way",
    ];

    let lower = message.to_lowercase();

    for word in ANAPHORA {
        // Check as a whole word (not substring of a larger word)
        if lower
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .any(|w| w == *word)
        {
            return true;
        }
    }

    let words_lower: Vec<&str> = lower.split_whitespace().collect();
    for marker in FOLLOW_UP {
        // For single-word markers, check word boundaries
        if !marker.contains(' ') {
            if words_lower.contains(marker) {
                return true;
            }
        } else {
            // Multi-word markers: check as substring (they're specific enough)
            if lower.contains(marker) {
                return true;
            }
        }
    }

    // Very short messages (< 5 words) are usually continuations
    if words.len() < 5 {
        return true;
    }

    false
}

#[async_trait]
impl ThreadResolver for WorkspaceThreadResolver {
    async fn resolve(
        &self,
        user_id: &str,
        message_content: &str,
        _default_thread_id: Uuid,
    ) -> Result<ThreadResolution, ResolverError> {
        // Check stickiness: if there's an active workspace and message looks like a reply,
        // continue in the same workspace.
        let sticky_result = {
            let state = self.stickiness.read().await;
            if let Some(s) = state.get(user_id) {
                if let (Some(ws_id), Some(conv_id)) =
                    (s.active_workspace_id, s.active_conversation_id)
                {
                    // Check time gap
                    let timed_out = s
                        .last_message_at
                        .is_some_and(|t| t.elapsed() > self.config.stickiness_timeout);

                    if !timed_out && is_reply_like(message_content, self.config.reply_max_words) {
                        Some((ws_id, conv_id, s.active_topic.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((ws_id, conv_id, topic)) = sticky_result {
            tracing::info!(
                workspace_id = %ws_id,
                "ThreadResolver: sticky continuation (reply-like message)"
            );
            let mut metadata = HashMap::new();
            metadata.insert("workspace_id".to_string(), ws_id.to_string());
            if let Some(ref t) = topic {
                metadata.insert("topic".to_string(), t.clone());
            }
            metadata.insert("routing_reason".to_string(), "sticky".to_string());

            return Ok(ThreadResolution {
                thread_id: Some(conv_id),
                context: self.build_workspace_context(ws_id, topic.as_deref()).await,
                metadata,
            });
        }

        // Embedding-based routing: find the best matching workspace
        let match_result = self
            .router
            .route(user_id, message_content)
            .await
            .map_err(|e| ResolverError::Embedding(e.to_string()))?;

        match match_result {
            Some(ws) if ws.topic != "general" => {
                tracing::info!(
                    workspace_id = %ws.id,
                    topic = %ws.topic,
                    "ThreadResolver: embedding match"
                );

                // Touch workspace
                let _ = self.db.touch_agent_workspace(ws.id).await;

                let mut metadata = HashMap::new();
                metadata.insert("workspace_id".to_string(), ws.id.to_string());
                metadata.insert("topic".to_string(), ws.topic.clone());
                metadata.insert("routing_reason".to_string(), "embedding".to_string());

                Ok(ThreadResolution {
                    thread_id: Some(ws.conversation_id),
                    context: self
                        .build_workspace_context(ws.id, Some(&ws.topic))
                        .await,
                    metadata,
                })
            }
            _ => {
                // No match or matched "general" — try to get/create default workspace
                match self.get_or_create_default(user_id).await {
                    Ok(Some(ws)) => {
                        let mut metadata = HashMap::new();
                        metadata.insert("workspace_id".to_string(), ws.id.to_string());
                        metadata.insert("topic".to_string(), ws.topic.clone());
                        metadata.insert("routing_reason".to_string(), "default".to_string());

                        Ok(ThreadResolution {
                            thread_id: Some(ws.conversation_id),
                            context: None, // No special context for default workspace
                            metadata,
                        })
                    }
                    Ok(None) => {
                        tracing::info!("ThreadResolver: no default workspace, using session thread");
                        Ok(ThreadResolution::default())
                    }
                    Err(e) => {
                        tracing::warn!("ThreadResolver: default workspace error: {}", e);
                        Ok(ThreadResolution::default())
                    }
                }
            }
        }
    }

    async fn organize(&self, _user_id: &str) -> Result<OrganizeResult, ResolverError> {
        // TODO: Implement LLM-based auto-organizer.
        // For now, return empty result (no workspaces created/updated).
        Ok(OrganizeResult::default())
    }

    async fn notify_routed(&self, user_id: &str, thread_id: Uuid) {
        // Update stickiness: the workspace this message was processed in
        // becomes the active workspace for future reply detection.
        let workspace = match self.db.get_agent_workspace_by_conversation(thread_id).await {
            Ok(Some(ws)) => ws,
            _ => return, // Not a workspace thread, nothing to track
        };

        let mut state = self.stickiness.write().await;
        let entry = state.entry(user_id.to_string()).or_insert(StickinessState {
            active_workspace_id: None,
            active_conversation_id: None,
            active_topic: None,
            last_message_at: None,
        });
        entry.active_workspace_id = Some(workspace.id);
        entry.active_conversation_id = Some(workspace.conversation_id);
        entry.active_topic = Some(workspace.topic);
        entry.last_message_at = Some(Instant::now());
    }
}

impl WorkspaceThreadResolver {
    /// Build context string for a workspace to inject into the system prompt.
    async fn build_workspace_context(
        &self,
        workspace_id: Uuid,
        topic: Option<&str>,
    ) -> Option<String> {
        let ws = match self.db.get_agent_workspace(workspace_id).await {
            Ok(Some(ws)) => ws,
            _ => return None,
        };

        let mut ctx = String::new();
        ctx.push_str("\n\n## Workspace Context\n");
        if let Some(t) = topic {
            ctx.push_str(&format!("**Topic:** {t}\n"));
        } else if !ws.topic.is_empty() {
            ctx.push_str(&format!("**Topic:** {}\n", ws.topic));
        }
        ctx.push_str(&format!("**Turn count:** {}\n", ws.turn_count));
        ctx.push_str(&format!("**Workspace ID:** {}\n", ws.id));
        ctx.push_str(
            "\nYou are continuing a workspace conversation. \
             Stay focused on the topic above. If the user's message \
             seems unrelated, acknowledge the topic shift.\n",
        );
        Some(ctx)
    }

    /// Get or create the default "general" workspace for a user.
    async fn get_or_create_default(
        &self,
        user_id: &str,
    ) -> Result<Option<AgentWorkspace>, ResolverError> {
        let workspaces = self
            .db
            .list_agent_workspaces(user_id, Some("active"))
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        if let Some(existing) = workspaces.iter().find(|ws| ws.topic == "general") {
            return Ok(Some(existing.clone()));
        }

        // Create default workspace — need a ConversationStore for this, but we
        // only have AgentWorkspaceStore. For now, return None (no default created).
        // The caller falls back to the session's ephemeral thread.
        //
        // In production, the workspace_router in agent_loop.rs handles default
        // workspace creation via get_or_create_default_workspace() which has
        // access to the full Database trait.
        tracing::debug!(
            "WorkspaceThreadResolver: no default workspace for {}, cannot create (need ConversationStore)",
            user_id
        );
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_like_short_message() {
        assert!(is_reply_like("yes", 15));
        assert!(is_reply_like("ok thanks", 15));
        assert!(is_reply_like("sure", 15));
    }

    #[test]
    fn reply_like_pronouns() {
        assert!(is_reply_like("what about that one near the station?", 15));
        assert!(is_reply_like("can she do it tomorrow?", 15));
        assert!(is_reply_like("is it expensive?", 15));
    }

    #[test]
    fn reply_like_follow_up_markers() {
        assert!(is_reply_like("actually make it two nights", 15));
        assert!(is_reply_like("also check the weather", 15));
        assert!(is_reply_like("what about the restaurants?", 15));
        assert!(is_reply_like("oh and when does it start?", 15));
    }

    #[test]
    fn not_reply_like_long_message() {
        assert!(!is_reply_like(
            "I want to plan a completely new trip to somewhere warm with beaches and good food and culture",
            15
        ));
    }

    #[test]
    fn not_reply_like_new_topic() {
        // Long enough and no reply markers
        assert!(!is_reply_like(
            "when do the school entrance exams happen for private schools in London?",
            15
        ));
    }
}
