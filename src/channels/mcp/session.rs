//! MCP session management for Streamable HTTP transport.
//!
//! Tracks active sessions identified by UUID. Each session is created on
//! `initialize` and can be cleaned up via DELETE /mcp.

use std::collections::HashMap;
use std::sync::Mutex;

use uuid::Uuid;

/// Per-session state (currently minimal — just tracks existence).
#[derive(Debug, Clone)]
pub struct McpSession {
    /// User who owns this session.
    pub user_id: String,
    /// When the session was created.
    pub created_at: std::time::Instant,
}

/// Thread-safe store for active MCP sessions.
#[derive(Debug, Default)]
pub struct McpSessionStore {
    sessions: Mutex<HashMap<Uuid, McpSession>>,
}

impl McpSessionStore {
    /// Create a new empty session store.
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new session and return its ID.
    pub fn create(&self, user_id: &str) -> Uuid {
        let id = Uuid::new_v4();
        let session = McpSession {
            user_id: user_id.to_string(),
            created_at: std::time::Instant::now(),
        };
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.insert(id, session);
        id
    }

    /// Check if a session exists.
    pub fn exists(&self, id: &Uuid) -> bool {
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.contains_key(id)
    }

    /// Remove a session. Returns true if the session existed.
    pub fn remove(&self, id: &Uuid) -> bool {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.remove(id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_exists() {
        let store = McpSessionStore::new();
        let id = store.create("andrew");
        assert!(store.exists(&id));
    }

    #[test]
    fn test_remove() {
        let store = McpSessionStore::new();
        let id = store.create("andrew");
        assert!(store.remove(&id));
        assert!(!store.exists(&id));
    }

    #[test]
    fn test_remove_nonexistent() {
        let store = McpSessionStore::new();
        let id = Uuid::new_v4();
        assert!(!store.remove(&id));
    }

    #[test]
    fn test_default() {
        let store = McpSessionStore::default();
        assert!(!store.exists(&Uuid::new_v4()));
    }
}
