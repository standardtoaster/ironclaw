//! Routes incoming prompts to existing workspaces by embedding similarity.

use std::sync::Arc;

use crate::db::{AgentWorkspace, AgentWorkspaceStore};
use crate::workspace::{EmbeddingError, EmbeddingProvider};

/// Errors that can occur during workspace routing.
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("database error: {0}")]
    Database(#[from] crate::error::DatabaseError),
}

/// Routes prompts to the best-matching workspace via embedding similarity.
pub struct WorkspaceRouter {
    db: Arc<dyn AgentWorkspaceStore>,
    embedder: Arc<dyn EmbeddingProvider>,
    threshold: f64,
}

impl WorkspaceRouter {
    pub fn new(
        db: Arc<dyn AgentWorkspaceStore>,
        embedder: Arc<dyn EmbeddingProvider>,
        threshold: f64,
    ) -> Self {
        Self {
            db,
            embedder,
            threshold,
        }
    }

    /// Route a prompt to the best matching workspace, or `None` if no match
    /// exceeds the similarity threshold.
    pub async fn route(
        &self,
        user_id: &str,
        prompt: &str,
    ) -> Result<Option<AgentWorkspace>, RouterError> {
        let embedding = self.embedder.embed(prompt).await?;
        let workspace = self
            .db
            .find_matching_workspace(user_id, &embedding, self.threshold)
            .await?;
        Ok(workspace)
    }

    /// Route with an optional hint prepended to the prompt for embedding.
    ///
    /// The hint biases the embedding toward a topic (e.g. a workspace name)
    /// without changing the user-visible prompt.
    pub async fn route_with_hint(
        &self,
        user_id: &str,
        prompt: &str,
        hint: Option<&str>,
    ) -> Result<Option<AgentWorkspace>, RouterError> {
        let text = match hint {
            Some(h) => format!("{h}: {prompt}"),
            None => prompt.to_string(),
        };
        let embedding = self.embedder.embed(&text).await?;
        let workspace = self
            .db
            .find_matching_workspace(user_id, &embedding, self.threshold)
            .await?;
        Ok(workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use chrono::Utc;
    use std::sync::Mutex;
    use uuid::Uuid;

    use crate::error::DatabaseError;

    // ── Fake embedder ──────────────────────────────────────────────────

    /// Returns a deterministic, normalised embedding derived from the text hash.
    /// Uses the same LCG approach as `MockEmbeddings` in `workspace/embeddings.rs`.
    struct FakeEmbedder {
        dim: usize,
    }

    impl FakeEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }

        /// Produce the raw embedding vector for `text` (shared helper so the
        /// test can pre-compute the "stored" embedding for a workspace).
        fn embed_sync(&self, text: &str) -> Vec<f32> {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            let hash = hasher.finish();

            let mut embedding = Vec::with_capacity(self.dim);
            let mut seed = hash;
            for _ in 0..self.dim {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let value = (seed as f32 / u64::MAX as f32) * 2.0 - 1.0;
                embedding.push(value);
            }

            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for x in &mut embedding {
                    *x /= magnitude;
                }
            }
            embedding
        }
    }

    #[async_trait]
    impl EmbeddingProvider for FakeEmbedder {
        fn dimension(&self) -> usize {
            self.dim
        }
        fn model_name(&self) -> &str {
            "fake"
        }
        fn max_input_length(&self) -> usize {
            10_000
        }
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            Ok(self.embed_sync(text))
        }
    }

    // ── Fake workspace store ───────────────────────────────────────────

    struct FakeWorkspaceStore {
        /// Each entry is (workspace, embedding).
        workspaces: Mutex<Vec<(AgentWorkspace, Vec<f32>)>>,
    }

    impl FakeWorkspaceStore {
        fn new() -> Self {
            Self {
                workspaces: Mutex::new(Vec::new()),
            }
        }

        fn insert(&self, ws: AgentWorkspace, embedding: Vec<f32>) {
            self.workspaces.lock().unwrap().push((ws, embedding));
        }
    }

    /// Cosine similarity between two unit (or non-unit) vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    #[async_trait]
    impl AgentWorkspaceStore for FakeWorkspaceStore {
        async fn create_agent_workspace(
            &self,
            _user_id: &str,
            _conversation_id: Uuid,
        ) -> Result<AgentWorkspace, DatabaseError> {
            unimplemented!("not needed for router tests")
        }

        async fn update_agent_workspace_topic(
            &self,
            _id: Uuid,
            _topic: &str,
            _embedding: &[f32],
        ) -> Result<(), DatabaseError> {
            unimplemented!("not needed for router tests")
        }

        async fn find_matching_workspace(
            &self,
            user_id: &str,
            embedding: &[f32],
            threshold: f64,
        ) -> Result<Option<AgentWorkspace>, DatabaseError> {
            let guard = self.workspaces.lock().unwrap();
            let mut best: Option<(f64, &AgentWorkspace)> = None;
            for (ws, ws_emb) in guard.iter() {
                if ws.user_id != user_id {
                    continue;
                }
                let sim = cosine_similarity(embedding, ws_emb);
                if sim >= threshold && (best.is_none() || sim > best.unwrap().0) {
                    best = Some((sim, ws));
                }
            }
            Ok(best.map(|(_, ws)| ws.clone()))
        }

        async fn get_agent_workspace(
            &self,
            id: Uuid,
        ) -> Result<Option<AgentWorkspace>, DatabaseError> {
            let guard = self.workspaces.lock().unwrap();
            Ok(guard.iter().find(|(ws, _)| ws.id == id).map(|(ws, _)| ws.clone()))
        }

        async fn list_agent_workspaces(
            &self,
            user_id: &str,
            _status: Option<&str>,
        ) -> Result<Vec<AgentWorkspace>, DatabaseError> {
            let guard = self.workspaces.lock().unwrap();
            Ok(guard
                .iter()
                .filter(|(ws, _)| ws.user_id == user_id)
                .map(|(ws, _)| ws.clone())
                .collect())
        }

        async fn touch_agent_workspace(&self, _id: Uuid) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn update_agent_workspace_status(
            &self,
            _id: Uuid,
            _status: &str,
        ) -> Result<(), DatabaseError> {
            Ok(())
        }

        async fn archive_stale_workspaces(
            &self,
            _user_id: &str,
            _stale_days: i64,
        ) -> Result<u64, DatabaseError> {
            Ok(0)
        }
    }

    // ── Helper ─────────────────────────────────────────────────────────

    fn make_workspace(user_id: &str) -> AgentWorkspace {
        AgentWorkspace {
            id: Uuid::new_v4(),
            user_id: user_id.to_string(),
            topic: String::new(),
            conversation_id: Uuid::new_v4(),
            status: "active".to_string(),
            last_accessed: Utc::now(),
            turn_count: 0,
            created_at: Utc::now(),
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn router_error_display() {
        let err = RouterError::Embedding(EmbeddingError::AuthFailed);
        assert!(err.to_string().contains("embedding failed"));

        let err = RouterError::Database(crate::error::DatabaseError::NotFound {
            entity: "workspace".into(),
            id: "test".into(),
        });
        assert!(err.to_string().contains("database error"));
    }

    #[tokio::test]
    async fn test_route_finds_matching_workspace() {
        let embedder = Arc::new(FakeEmbedder::new(64));
        let store = Arc::new(FakeWorkspaceStore::new());

        // Store a workspace with the embedding for "grocery list"
        let ws = make_workspace("user-1");
        let ws_id = ws.id;
        let emb = embedder.embed_sync("grocery list");
        store.insert(ws, emb);

        let router = WorkspaceRouter::new(store, embedder.clone(), 0.99);

        // Route the exact same prompt — should match (cosine sim == 1.0)
        let result = router.route("user-1", "grocery list").await.unwrap();
        assert!(result.is_some(), "expected a matching workspace");
        assert_eq!(result.unwrap().id, ws_id);
    }

    #[tokio::test]
    async fn test_route_returns_none_for_unrelated() {
        let embedder = Arc::new(FakeEmbedder::new(64));
        let store = Arc::new(FakeWorkspaceStore::new());

        let ws = make_workspace("user-1");
        let emb = embedder.embed_sync("grocery list");
        store.insert(ws, emb);

        // High threshold; a completely different prompt should not match
        let router = WorkspaceRouter::new(store, embedder.clone(), 0.95);
        let result = router
            .route("user-1", "quantum physics homework")
            .await
            .unwrap();
        assert!(result.is_none(), "unrelated prompt should not match");
    }

    #[tokio::test]
    async fn test_route_with_hint_prepends() {
        let embedder = Arc::new(FakeEmbedder::new(64));
        let store = Arc::new(FakeWorkspaceStore::new());

        // Store a workspace keyed on the *hinted* text
        let ws = make_workspace("user-1");
        let ws_id = ws.id;
        let emb = embedder.embed_sync("grocery: buy milk");
        store.insert(ws, emb);

        let router = WorkspaceRouter::new(store, embedder.clone(), 0.99);

        // Without hint: embedding of "buy milk" != "grocery: buy milk"
        let without = router
            .route_with_hint("user-1", "buy milk", None)
            .await
            .unwrap();
        assert!(
            without.is_none(),
            "without hint the embedding should differ"
        );

        // With hint: embedding of "grocery: buy milk" should match exactly
        let with_hint = router
            .route_with_hint("user-1", "buy milk", Some("grocery"))
            .await
            .unwrap();
        assert!(with_hint.is_some(), "with hint should match");
        assert_eq!(with_hint.unwrap().id, ws_id);
    }

    #[tokio::test]
    async fn test_route_respects_threshold() {
        let embedder = Arc::new(FakeEmbedder::new(64));
        let store = Arc::new(FakeWorkspaceStore::new());

        let ws = make_workspace("user-1");
        let emb = embedder.embed_sync("grocery list");
        store.insert(ws, emb);

        // With a threshold of 1.0 (exact match only), a slightly different
        // prompt should fail to match even though cosine sim is close.
        let router = WorkspaceRouter::new(store, embedder.clone(), 1.0);
        let result = router.route("user-1", "grocery lists").await.unwrap();
        assert!(
            result.is_none(),
            "threshold 1.0 should reject non-identical embeddings"
        );
    }

    #[tokio::test]
    async fn test_route_empty_store() {
        let embedder = Arc::new(FakeEmbedder::new(64));
        let store = Arc::new(FakeWorkspaceStore::new());

        let router = WorkspaceRouter::new(store, embedder.clone(), 0.5);
        let result = router.route("user-1", "anything").await.unwrap();
        assert!(result.is_none(), "empty store should return None");
    }
}
