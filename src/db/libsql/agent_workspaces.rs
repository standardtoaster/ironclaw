//! AgentWorkspaceStore implementation for LibSqlBackend.
//!
//! Since libSQL doesn't have pgvector, `find_matching_workspace` always returns
//! `Ok(None)`. Topic updates store the text but ignore the embedding.

use async_trait::async_trait;
use chrono::Utc;
use libsql::params;
use uuid::Uuid;

use super::{LibSqlBackend, fmt_ts, get_i64, get_text, get_ts};
use crate::db::{AgentWorkspace, AgentWorkspaceStore, WorkspaceMessageResult};
use crate::error::DatabaseError;

/// Column list for agent_workspaces (matches positional access in `row_to_agent_workspace`).
const AGENT_WORKSPACE_COLUMNS: &str = "\
    id, user_id, topic, conversation_id, status, \
    last_accessed, turn_count, created_at";

fn row_to_agent_workspace(row: &libsql::Row) -> AgentWorkspace {
    AgentWorkspace {
        id: get_text(row, 0).parse().unwrap_or_default(),
        user_id: get_text(row, 1),
        topic: get_text(row, 2),
        conversation_id: get_text(row, 3).parse().unwrap_or_default(),
        status: get_text(row, 4),
        last_accessed: get_ts(row, 5),
        turn_count: get_i64(row, 6) as i32,
        created_at: get_ts(row, 7),
    }
}

#[async_trait]
impl AgentWorkspaceStore for LibSqlBackend {
    async fn create_agent_workspace(
        &self,
        user_id: &str,
        conversation_id: Uuid,
    ) -> Result<AgentWorkspace, DatabaseError> {
        let conn = self.connect().await?;
        let id = Uuid::new_v4();
        let now = fmt_ts(&Utc::now());
        conn.execute(
            r#"
            INSERT INTO agent_workspaces (id, user_id, conversation_id, last_accessed, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                id.to_string(),
                user_id,
                conversation_id.to_string(),
                now.as_str(),
                now.as_str(),
                now.as_str(),
            ],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;

        // Read back the inserted row.
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {} FROM agent_workspaces WHERE id = ?1",
                    AGENT_WORKSPACE_COLUMNS
                ),
                params![id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(row_to_agent_workspace(&row)),
            None => Err(DatabaseError::Query(
                "failed to read back inserted agent_workspace".to_string(),
            )),
        }
    }

    async fn update_agent_workspace_topic(
        &self,
        id: Uuid,
        topic: &str,
        _embedding: &[f32],
    ) -> Result<(), DatabaseError> {
        // libSQL has no pgvector — store the topic text only, ignore the embedding.
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE agent_workspaces SET topic = ?2 WHERE id = ?1",
            params![id.to_string(), topic],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn find_matching_workspace(
        &self,
        _user_id: &str,
        _embedding: &[f32],
        _threshold: f64,
    ) -> Result<Option<AgentWorkspace>, DatabaseError> {
        // No pgvector in libSQL — vector similarity search is not supported.
        Ok(None)
    }

    async fn get_agent_workspace(
        &self,
        id: Uuid,
    ) -> Result<Option<AgentWorkspace>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {} FROM agent_workspaces WHERE id = ?1",
                    AGENT_WORKSPACE_COLUMNS
                ),
                params![id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(row_to_agent_workspace(&row))),
            None => Ok(None),
        }
    }

    async fn get_agent_workspace_by_conversation(
        &self,
        conversation_id: Uuid,
    ) -> Result<Option<AgentWorkspace>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                &format!(
                    "SELECT {} FROM agent_workspaces WHERE conversation_id = ?1",
                    AGENT_WORKSPACE_COLUMNS
                ),
                params![conversation_id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(row_to_agent_workspace(&row))),
            None => Ok(None),
        }
    }

    async fn list_agent_workspaces(
        &self,
        user_id: &str,
        status: Option<&str>,
    ) -> Result<Vec<AgentWorkspace>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = match status {
            Some(s) => {
                conn.query(
                    &format!(
                        "SELECT {} FROM agent_workspaces WHERE user_id = ?1 AND status = ?2 ORDER BY last_accessed DESC",
                        AGENT_WORKSPACE_COLUMNS
                    ),
                    params![user_id, s],
                )
                .await
                .map_err(|e| DatabaseError::Query(e.to_string()))?
            }
            None => {
                conn.query(
                    &format!(
                        "SELECT {} FROM agent_workspaces WHERE user_id = ?1 ORDER BY last_accessed DESC",
                        AGENT_WORKSPACE_COLUMNS
                    ),
                    params![user_id],
                )
                .await
                .map_err(|e| DatabaseError::Query(e.to_string()))?
            }
        };

        let mut workspaces = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            workspaces.push(row_to_agent_workspace(&row));
        }
        Ok(workspaces)
    }

    async fn touch_agent_workspace(&self, id: Uuid) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        conn.execute(
            "UPDATE agent_workspaces SET turn_count = turn_count + 1, last_accessed = ?2 WHERE id = ?1",
            params![id.to_string(), now],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn update_agent_workspace_status(
        &self,
        id: Uuid,
        status: &str,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE agent_workspaces SET status = ?2 WHERE id = ?1",
            params![id.to_string(), status],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn archive_stale_workspaces(
        &self,
        user_id: &str,
        stale_days: i64,
    ) -> Result<u64, DatabaseError> {
        let conn = self.connect().await?;
        let cutoff = Utc::now() - chrono::Duration::days(stale_days);
        let cutoff_str = fmt_ts(&cutoff);
        let n = conn
            .execute(
                r#"
                UPDATE agent_workspaces SET status = 'archived'
                WHERE user_id = ?1
                  AND status = 'active'
                  AND last_accessed < ?2
                "#,
                params![user_id, cutoff_str],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(n)
    }

    async fn search_workspace_messages(
        &self,
        user_id: &str,
        query: &str,
        workspace_id: Option<Uuid>,
        limit: i64,
    ) -> Result<Vec<WorkspaceMessageResult>, DatabaseError> {
        let conn = self.connect().await?;
        let search_pattern = format!("%{query}%");

        let mut rows = if let Some(ws_id) = workspace_id {
            conn.query(
                r#"
                SELECT m.content, m.role, m.created_at, w.topic, w.id as workspace_id
                FROM conversation_messages m
                JOIN agent_workspaces w ON w.conversation_id = m.conversation_id
                WHERE w.user_id = ?1
                  AND w.id = ?4
                  AND m.role IN ('user', 'assistant')
                  AND m.content LIKE ?2
                ORDER BY m.created_at DESC
                LIMIT ?3
                "#,
                params![user_id, search_pattern, limit, ws_id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        } else {
            conn.query(
                r#"
                SELECT m.content, m.role, m.created_at, w.topic, w.id as workspace_id
                FROM conversation_messages m
                JOIN agent_workspaces w ON w.conversation_id = m.conversation_id
                WHERE w.user_id = ?1
                  AND m.role IN ('user', 'assistant')
                  AND m.content LIKE ?2
                ORDER BY m.created_at DESC
                LIMIT ?3
                "#,
                params![user_id, search_pattern, limit],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        };

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            results.push(WorkspaceMessageResult {
                content: get_text(&row, 0),
                role: get_text(&row, 1),
                created_at: get_ts(&row, 2),
                topic: get_text(&row, 3),
                workspace_id: get_text(&row, 4).parse().unwrap_or_default(),
            });
        }
        Ok(results)
    }
}
