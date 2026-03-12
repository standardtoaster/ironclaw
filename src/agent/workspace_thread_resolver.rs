//! Workspace-aware implementation of [`ThreadResolver`].
//!
//! Routes messages to workspace threads by embedding similarity, with
//! conversation stickiness and collision handling. Delegates to the
//! auto-organizer for workspace creation/updates.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::agent::organizer_runner::OrganizerSignal;
use crate::agent::thread_resolver::{
    OrganizeResult, ResolverError, ThreadResolution, ThreadResolver, WorkspaceInfo,
};
use crate::agent::workspace_router::WorkspaceRouter;
use crate::db::{AgentWorkspace, AgentWorkspaceStore, ConversationStore};
use crate::llm::{ChatMessage, CompletionRequest, LlmProvider};

/// Default workspace topic name.
pub const DEFAULT_WORKSPACE_TOPIC: &str = "general";

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
    /// Model override for the organizer LLM call (`ORGANIZER_MODEL`).
    /// When set, passed via `CompletionRequest::with_model()`.
    /// Only effective with providers that support per-request model overrides.
    pub organizer_model: Option<String>,
    /// Temperature for organizer LLM calls (default: 0.3).
    pub organizer_temperature: f32,
    /// Max tokens for organizer LLM responses (default: 2000).
    pub organizer_max_tokens: u32,
}

impl Default for WorkspaceResolverConfig {
    fn default() -> Self {
        Self {
            high_threshold: 0.75,
            low_threshold: 0.5,
            stickiness_timeout: Duration::from_secs(30 * 60),
            reply_max_words: 15,
            organizer_model: None,
            organizer_temperature: 0.3_f32,
            organizer_max_tokens: 2000,
        }
    }
}

/// Implements `ThreadResolver` with workspace routing, stickiness, and auto-organization.
pub struct WorkspaceThreadResolver {
    router: Arc<WorkspaceRouter>,
    db: Arc<dyn AgentWorkspaceStore>,
    conversations: Arc<dyn ConversationStore>,
    llm: Option<Arc<dyn LlmProvider>>,
    config: WorkspaceResolverConfig,
    /// Per-user stickiness tracking (in-memory, resets on restart).
    stickiness: RwLock<HashMap<String, StickinessState>>,
    /// Per-user watermark: turn_count of the general workspace at last organize run.
    /// Resets on restart (first run after restart always processes).
    organize_watermarks: RwLock<HashMap<String, i32>>,
    /// Channel to signal the organizer runner when unrouted messages land in general.
    organize_tx: Option<tokio::sync::mpsc::Sender<OrganizerSignal>>,
}

impl WorkspaceThreadResolver {
    pub fn new(
        router: Arc<WorkspaceRouter>,
        db: Arc<dyn AgentWorkspaceStore>,
        conversations: Arc<dyn ConversationStore>,
        llm: Option<Arc<dyn LlmProvider>>,
        config: WorkspaceResolverConfig,
    ) -> Self {
        Self {
            router,
            db,
            conversations,
            llm,
            config,
            stickiness: RwLock::new(HashMap::new()),
            organize_watermarks: RwLock::new(HashMap::new()),
            organize_tx: None,
        }
    }

    /// Set the channel used to signal the organizer runner when unrouted messages
    /// land in the general workspace.
    pub fn with_organizer_channel(
        mut self,
        tx: tokio::sync::mpsc::Sender<OrganizerSignal>,
    ) -> Self {
        self.organize_tx = Some(tx);
        self
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
        // Embedding-based routing with confidence gap check.
        // This runs FIRST — a confident embedding match always wins over stickiness.
        let route_result = self
            .router
            .route_with_confidence(user_id, message_content)
            .await
            .map_err(|e| ResolverError::Embedding(e.to_string()))?;

        // Helper: get current stickiness state (non-timed-out, reply-like).
        let get_sticky_state = |require_reply_like: bool| async move {
            let state = self.stickiness.read().await;
            let s = state.get(user_id)?;
            let (ws_id, conv_id) = match (s.active_workspace_id, s.active_conversation_id) {
                (Some(w), Some(c)) => (w, c),
                _ => return None,
            };
            let timed_out = s
                .last_message_at
                .is_some_and(|t| t.elapsed() > self.config.stickiness_timeout);
            if timed_out {
                return None;
            }
            if require_reply_like
                && !is_reply_like(message_content, self.config.reply_max_words)
            {
                return None;
            }
            Some((ws_id, conv_id, s.active_topic.clone()))
        };

        // If no confident match, try stickiness or momentum tiebreak.
        if route_result.workspace.is_none()
            && let Some((ws_id, conv_id, topic)) = get_sticky_state(false).await
        {
                // Momentum tiebreak: if ambiguous and the active workspace is a
                // competitive candidate (above threshold and within 0.10 of the
                // top score), prefer it.
                let active_is_competitive = route_result.ambiguous
                    && route_result.candidates.iter().any(|(id, score)| {
                        *id == ws_id
                            && *score >= self.config.low_threshold
                            && (route_result.score - score) < 0.10
                    });

                let reason = if active_is_competitive {
                    "momentum"
                } else if route_result.ambiguous {
                    // Active workspace not competitive or not among candidates —
                    // the user has changed topic. Don't use stickiness.
                    "none"
                } else {
                    // Below threshold — only use stickiness for reply-like messages.
                    if is_reply_like(message_content, self.config.reply_max_words) {
                        "sticky"
                    } else {
                        "none"
                    }
                };

                if reason != "none" {
                    tracing::info!(
                        workspace_id = %ws_id,
                        score = route_result.score,
                        gap = route_result.gap,
                        reason,
                        "ThreadResolver: using stickiness/momentum"
                    );
                    let mut metadata = HashMap::new();
                    metadata.insert("workspace_id".to_string(), ws_id.to_string());
                    if let Some(ref t) = topic {
                        metadata.insert("topic".to_string(), t.clone());
                    }
                    metadata.insert("routing_reason".to_string(), reason.to_string());

                    return Ok(ThreadResolution {
                        thread_id: Some(conv_id),
                        context: self.build_workspace_context(ws_id, topic.as_deref()).await,
                        metadata,
                    });
                }
            }
        // Fall through to embedding match or default workspace below

        match route_result.workspace {
            Some(ws) if ws.topic != DEFAULT_WORKSPACE_TOPIC => {
                tracing::info!(
                    workspace_id = %ws.id,
                    topic = %ws.topic,
                    score = route_result.score,
                    gap = route_result.gap,
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
                // No match or matched "general" — try to get/create default workspace.
                // Signal the organizer that an unrouted message landed in general.
                if let Some(ref tx) = self.organize_tx {
                    let _ = tx.try_send(OrganizerSignal {
                        user_id: user_id.to_string(),
                    });
                }

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

    async fn organize(&self, user_id: &str) -> Result<OrganizeResult, ResolverError> {
        let llm = match &self.llm {
            Some(llm) => Arc::clone(llm),
            None => {
                tracing::warn!("Organizer: no LLM available, skipping");
                return Ok(OrganizeResult::default());
            }
        };

        // 1. Find default workspace
        let workspaces = self
            .db
            .list_agent_workspaces(user_id, Some("active"))
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        let default_ws = match workspaces.iter().find(|ws| ws.topic == DEFAULT_WORKSPACE_TOPIC) {
            Some(ws) => ws,
            None => {
                tracing::info!("Organizer: no default workspace, nothing to organize");
                return Ok(OrganizeResult::default());
            }
        };

        // 1b. Watermark check: skip if no new messages since last organize run
        let current_turn_count = default_ws.turn_count;
        let last_watermark = self
            .organize_watermarks
            .read()
            .await
            .get(user_id)
            .copied()
            .unwrap_or(-1);
        if current_turn_count <= last_watermark {
            tracing::debug!(
                user_id,
                turn_count = current_turn_count,
                watermark = last_watermark,
                "Organizer: no new messages since last run, skipping"
            );
            return Ok(OrganizeResult::default());
        }

        // 2. Read last 20 messages from default workspace conversation (paginated, DESC order)
        let (mut messages, _has_more) = self
            .conversations
            .list_conversation_messages_paginated(default_ws.conversation_id, None, 20)
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        // Paginated returns newest-first; reverse to chronological for the LLM
        messages.reverse();

        let recent: Vec<_> = messages
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .collect();

        if recent.is_empty() {
            return Ok(OrganizeResult::default());
        }

        // 3. Build context about existing workspaces for dedup
        let existing_names: Vec<String> = workspaces
            .iter()
            .filter(|ws| ws.topic != DEFAULT_WORKSPACE_TOPIC)
            .map(|ws| ws.topic.clone())
            .collect();

        // 4. Format messages for LLM
        let mut formatted = String::new();
        for (i, msg) in recent.iter().enumerate() {
            let role_label = if msg.role == "user" { "User" } else { "Assistant" };
            // Truncate long messages to avoid blowing up context
            let content = if msg.content.len() > 300 {
                let boundary = msg.content.char_indices()
                    .take_while(|(idx, _)| *idx < 300)
                    .last()
                    .map(|(idx, c)| idx + c.len_utf8())
                    .unwrap_or(300.min(msg.content.len()));
                format!("{}...", &msg.content[..boundary])
            } else {
                msg.content.clone()
            };
            formatted.push_str(&format!("[{i}] {role_label}: {content}\n"));
        }

        let existing_ctx = if existing_names.is_empty() {
            String::new()
        } else {
            format!(
                "\nExisting workspaces (do not duplicate): {}\n",
                existing_names.join(", ")
            )
        };

        let user_prompt = format!("{existing_ctx}\nMessages:\n{formatted}");

        // 5. Call LLM
        let mut request = CompletionRequest::new(vec![
            ChatMessage::system(ORGANIZER_PROMPT.to_string()),
            ChatMessage::user(user_prompt),
        ])
        .with_max_tokens(self.config.organizer_max_tokens)
        .with_temperature(self.config.organizer_temperature);

        if let Some(ref model) = self.config.organizer_model {
            request = request.with_model(model.clone());
        }

        let response = match llm.complete(request).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Organizer: LLM call failed: {e}");
                return Ok(OrganizeResult::default());
            }
        };

        // 6. Clean thinking tags, then parse JSON response
        let cleaned = crate::llm::clean_response(&response.content);
        let projects = match parse_organizer_response(&cleaned) {
            Some(p) => p,
            None => {
                tracing::warn!(
                    "Organizer: failed to parse LLM response as JSON: {}",
                    &cleaned[..cleaned.len().min(200)]
                );
                return Ok(OrganizeResult::default());
            }
        };

        // 7. Create workspaces for each project (with dedup)
        let mut result = OrganizeResult {
            messages_analyzed: recent.len(),
            ..Default::default()
        };

        // Build lowercase set for dedup (existing + created-this-batch)
        let mut known_topics: std::collections::HashSet<String> = existing_names
            .iter()
            .map(|n| n.to_lowercase())
            .collect();

        for project in projects {
            // Dedup: check if a workspace with this name already exists (case-insensitive)
            if known_topics.contains(&project.name.to_lowercase()) {
                tracing::info!(
                    name = %project.name,
                    "Organizer: workspace already exists, skipping"
                );
                continue;
            }

            let topic_for_embed = format!("{}: {}", project.name, project.description);
            let topic_for_embed =
                &topic_for_embed[..topic_for_embed.len().min(500)];

            // Create conversation for new workspace
            let conversation_id = match self
                .conversations
                .create_conversation("workspace", user_id, None)
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        name = %project.name,
                        "Organizer: failed to create conversation: {e}"
                    );
                    continue;
                }
            };

            // Create workspace
            let ws = match self
                .db
                .create_agent_workspace(user_id, conversation_id)
                .await
            {
                Ok(ws) => ws,
                Err(e) => {
                    tracing::warn!(
                        name = %project.name,
                        "Organizer: failed to create workspace: {e}"
                    );
                    continue;
                }
            };

            // Embed and store topic
            let embedding = match self.router.embed(topic_for_embed).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        name = %project.name,
                        "Organizer: failed to embed topic: {e}"
                    );
                    continue;
                }
            };

            if let Err(e) = self
                .db
                .update_agent_workspace_topic(ws.id, topic_for_embed, &embedding)
                .await
            {
                tracing::warn!(
                    name = %project.name,
                    "Organizer: failed to set topic: {e}"
                );
                continue;
            }

            // Pre-seed: write a summary message into the new workspace conversation
            let summary = format!(
                "This workspace was created by the organizer for: {}. {}",
                project.name, project.description
            );
            let _ = self
                .conversations
                .add_conversation_message(conversation_id, "system", &summary)
                .await;

            tracing::info!(
                workspace_id = %ws.id,
                name = %project.name,
                "Organizer: created workspace"
            );

            known_topics.insert(project.name.to_lowercase());

            result.created.push(WorkspaceInfo {
                id: ws.id,
                topic: project.name.clone(),
                description: project.description.clone(),
            });
        }

        // Update watermark after successful organize
        self.organize_watermarks
            .write()
            .await
            .insert(user_id.to_string(), current_turn_count);

        Ok(result)
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

        if let Some(existing) = workspaces.iter().find(|ws| ws.topic == DEFAULT_WORKSPACE_TOPIC) {
            return Ok(Some(existing.clone()));
        }

        // Create default workspace with a backing conversation.
        tracing::info!(
            "WorkspaceThreadResolver: creating default '{}' workspace for {}",
            DEFAULT_WORKSPACE_TOPIC,
            user_id
        );

        let conversation_id = self
            .conversations
            .create_conversation("workspace", user_id, None)
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        let ws = self
            .db
            .create_agent_workspace(user_id, conversation_id)
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        // Embed and set the topic to "general" so it's recognized as the default workspace.
        let embedding = self
            .router
            .embed("general conversation and miscellaneous topics")
            .await
            .map_err(|e| ResolverError::Embedding(e.to_string()))?;
        self.db
            .update_agent_workspace_topic(ws.id, DEFAULT_WORKSPACE_TOPIC, &embedding)
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        // Re-fetch to get updated fields
        let workspaces = self
            .db
            .list_agent_workspaces(user_id, Some("active"))
            .await
            .map_err(|e| ResolverError::Database(e.to_string()))?;

        Ok(workspaces.into_iter().find(|w| w.id == ws.id))
    }
}

const ORGANIZER_PROMPT: &str = "\
You are a conversation organizer. Below are recent messages from a personal \
assistant conversation. Identify any ongoing topics that the user is likely \
to revisit — even if only mentioned briefly so far.

A topic qualifies if the user would benefit from having a dedicated workspace \
for it. Two messages about the same trip, event, or goal is enough. Each \
distinct topic gets its own workspace — e.g. two different trips should be \
two separate projects, not combined.

NOT projects: single factual questions, unit conversions, quick calculations.

For each project:
- name: short label (2-5 words)
- description: one sentence summary

Respond ONLY with a JSON array. No explanation, no markdown.

[{\"name\": \"Kitchen renovation\", \"description\": \"Planning kitchen remodel including contractor quotes and timeline\"}]";

/// A project identified by the organizer LLM.
#[derive(Debug, Deserialize)]
struct OrganizerProject {
    name: String,
    description: String,
}

/// Parse the organizer LLM response, extracting a JSON array from the text.
fn parse_organizer_response(content: &str) -> Option<Vec<OrganizerProject>> {
    // Try direct parse first
    if let Ok(projects) = serde_json::from_str::<Vec<OrganizerProject>>(content.trim()) {
        return Some(projects);
    }

    // Try extracting JSON from markdown code blocks or surrounding text
    let trimmed = content.trim();

    // Look for ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let json_start = start + 7;
        if let Some(end) = trimmed[json_start..].find("```") {
            let json_str = &trimmed[json_start..json_start + end].trim();
            if let Ok(projects) = serde_json::from_str::<Vec<OrganizerProject>>(json_str) {
                return Some(projects);
            }
        }
    }

    // Look for ``` ... ``` blocks (without language tag)
    if let Some(start) = trimmed.find("```") {
        let json_start = start + 3;
        // Skip the optional language tag line
        let after_tag = if let Some(nl) = trimmed[json_start..].find('\n') {
            json_start + nl + 1
        } else {
            json_start
        };
        if let Some(end) = trimmed[after_tag..].find("```") {
            let json_str = trimmed[after_tag..after_tag + end].trim();
            if let Ok(projects) = serde_json::from_str::<Vec<OrganizerProject>>(json_str) {
                return Some(projects);
            }
        }
    }

    // Look for first [ ... last ]
    let bracket_start = trimmed.find('[');
    let bracket_end = trimmed.rfind(']');
    if let (Some(start), Some(end)) = (bracket_start, bracket_end)
        && end > start
        && let Ok(projects) = serde_json::from_str::<Vec<OrganizerProject>>(&trimmed[start..=end])
    {
        return Some(projects);
    }

    None
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

    #[test]
    fn parse_organizer_valid_json() {
        let input = r#"[{"name": "Japan trip", "description": "Planning a trip to Japan"}]"#;
        let projects = parse_organizer_response(input).expect("should parse");
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "Japan trip");
    }

    #[test]
    fn parse_organizer_markdown_code_block() {
        let input = "Here are the projects:\n```json\n[{\"name\": \"School apps\", \"description\": \"Applying to schools\"}]\n```\n";
        let projects = parse_organizer_response(input).expect("should parse");
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "School apps");
    }

    #[test]
    fn parse_organizer_embedded_json() {
        let input = "I found these projects:\n[{\"name\": \"Trip\", \"description\": \"A trip\"}]\nThat's all.";
        let projects = parse_organizer_response(input).expect("should parse");
        assert_eq!(projects.len(), 1);
    }

    #[test]
    fn parse_organizer_empty_array() {
        let input = "[]";
        let projects = parse_organizer_response(input).expect("should parse");
        assert!(projects.is_empty());
    }

    #[test]
    fn parse_organizer_malformed_graceful() {
        let input = "I don't see any projects here, sorry!";
        assert!(parse_organizer_response(input).is_none());
    }

    #[test]
    fn parse_organizer_multiple_projects() {
        let input = r#"[
            {"name": "Japan trip", "description": "Planning Japan"},
            {"name": "School applications", "description": "Schools for Emma"}
        ]"#;
        let projects = parse_organizer_response(input).expect("should parse");
        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].name, "Japan trip");
        assert_eq!(projects[1].name, "School applications");
    }

    #[test]
    fn parse_organizer_backwards_compatible_with_indices() {
        // Old LLM responses with message_indices should still parse (serde ignores unknown fields)
        let input = r#"[{"name": "Trip", "description": "A trip", "message_indices": [0, 1]}]"#;
        let projects = parse_organizer_response(input).expect("should parse");
        assert_eq!(projects.len(), 1);
    }
}
