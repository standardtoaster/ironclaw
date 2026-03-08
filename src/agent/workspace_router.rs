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

    // Basic compile-time validation that the types work together.
    // Full integration tests require a real DB backend.
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
}
