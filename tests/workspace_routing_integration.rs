//! Integration tests for the workspace routing system.
//!
//! These tests exercise the workspace store, router, and queue working together
//! against a real PostgreSQL database with pgvector. They validate the full flow:
//! create workspace → set topic with embedding → route by similarity → queue messages.
//!
//! Requires: PostgreSQL with pgvector extension, reachable at DATABASE_URL
//! (default: postgres://localhost/ironclaw_test). Tests skip gracefully if unavailable.
//!
//! Run with: cargo test --test workspace_routing_integration

#![cfg(feature = "postgres")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use secrecy::SecretString;
use tokio::sync::oneshot;
use ironclaw::agent::workspace_queue::{MessagePriority, WorkspaceMessage, WorkspaceQueueManager};
use ironclaw::agent::workspace_router::WorkspaceRouter;
use ironclaw::config::{DatabaseBackend, DatabaseConfig, SslMode};
use ironclaw::db::postgres::PgBackend;
use ironclaw::db::{AgentWorkspaceStore, ConversationStore, Database};
use ironclaw::workspace::{EmbeddingProvider, MockEmbeddings};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost/ironclaw_test".to_string())
}

fn test_db_config() -> DatabaseConfig {
    DatabaseConfig {
        backend: DatabaseBackend::Postgres,
        url: SecretString::from(database_url()),
        pool_size: 4,
        ssl_mode: SslMode::Disable,
        libsql_path: None,
        libsql_url: None,
        libsql_auth_token: None,
    }
}

async fn setup() -> Option<Arc<PgBackend>> {
    let config = test_db_config();
    match PgBackend::new(&config).await {
        Ok(backend) => {
            if backend.run_migrations().await.is_err() {
                eprintln!("skipping: migration failed");
                return None;
            }
            Some(Arc::new(backend))
        }
        Err(e) => {
            eprintln!("skipping: database unavailable ({e})");
            None
        }
    }
}

async fn cleanup(db: &PgBackend, user_id: &str) {
    let pool = db.pool();
    let conn = pool.get().await.expect("cleanup: get connection");
    // Workspaces cascade-delete when conversations are deleted.
    conn.execute(
        "DELETE FROM agent_workspaces WHERE user_id = $1",
        &[&user_id],
    )
    .await
    .ok();
    conn.execute(
        "DELETE FROM conversations WHERE user_id = $1",
        &[&user_id],
    )
    .await
    .ok();
}

fn mock_embedder(dim: usize) -> Arc<MockEmbeddings> {
    Arc::new(MockEmbeddings::new(dim))
}

// ---------------------------------------------------------------------------
// Tests: Workspace store basics (against real Postgres)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_create_and_get_workspace() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_create";
    cleanup(&db, user).await;

    // Must create conversation first (FK constraint).
    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    assert_eq!(ws.user_id, user);
    assert_eq!(ws.conversation_id, conv_id);
    assert_eq!(ws.status, "active");
    assert_eq!(ws.turn_count, 0);
    assert!(ws.topic.is_empty());

    // Get by ID.
    let fetched = db.get_agent_workspace(ws.id).await.unwrap().unwrap();
    assert_eq!(fetched.id, ws.id);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_list_workspaces_by_status() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_list_status";
    cleanup(&db, user).await;

    let c1 = db.create_conversation("test", user, None).await.unwrap();
    let c2 = db.create_conversation("test", user, None).await.unwrap();
    let ws1 = db.create_agent_workspace(user, c1).await.unwrap();
    let ws2 = db.create_agent_workspace(user, c2).await.unwrap();

    // Archive one.
    db.update_agent_workspace_status(ws2.id, "archived")
        .await
        .unwrap();

    let active = db.list_agent_workspaces(user, Some("active")).await.unwrap();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].id, ws1.id);

    let archived = db
        .list_agent_workspaces(user, Some("archived"))
        .await
        .unwrap();
    assert_eq!(archived.len(), 1);
    assert_eq!(archived[0].id, ws2.id);

    let all = db.list_agent_workspaces(user, None).await.unwrap();
    assert_eq!(all.len(), 2);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_touch_increments_turn_count() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_touch";
    cleanup(&db, user).await;

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();
    assert_eq!(ws.turn_count, 0);

    db.touch_agent_workspace(ws.id).await.unwrap();
    db.touch_agent_workspace(ws.id).await.unwrap();
    db.touch_agent_workspace(ws.id).await.unwrap();

    let updated = db.get_agent_workspace(ws.id).await.unwrap().unwrap();
    assert_eq!(updated.turn_count, 3);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_unique_constraint_user_conversation() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_unique";
    cleanup(&db, user).await;

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    db.create_agent_workspace(user, conv_id).await.unwrap();

    // Second workspace on same conversation should fail.
    let result = db.create_agent_workspace(user, conv_id).await;
    assert!(result.is_err(), "duplicate (user, conversation) should fail");

    cleanup(&db, user).await;
}

// ---------------------------------------------------------------------------
// Tests: Topic embedding + similarity routing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_set_topic_and_find_exact_match() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_exact_match";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    // Set topic with embedding.
    let embedding = embedder.embed("home automation porch lights").await.unwrap();
    db.update_agent_workspace_topic(ws.id, "home automation porch lights", &embedding)
        .await
        .unwrap();

    // MockEmbeddings is deterministic — same text → same vector → similarity 1.0.
    let found = db
        .find_matching_workspace(user, &embedding, 0.99)
        .await
        .unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, ws.id);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_no_match_for_unrelated_prompt() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_no_match";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    let topic_emb = embedder.embed("grocery shopping list").await.unwrap();
    db.update_agent_workspace_topic(ws.id, "grocery shopping list", &topic_emb)
        .await
        .unwrap();

    // Different text → different hash-based embedding → low similarity.
    let query_emb = embedder.embed("quantum physics homework").await.unwrap();
    let found = db
        .find_matching_workspace(user, &query_emb, 0.5)
        .await
        .unwrap();
    assert!(found.is_none());

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_router_finds_best_workspace() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_router_best";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    // Create two workspaces with different topics.
    let c1 = db.create_conversation("test", user, None).await.unwrap();
    let c2 = db.create_conversation("test", user, None).await.unwrap();
    let ws1 = db.create_agent_workspace(user, c1).await.unwrap();
    let ws2 = db.create_agent_workspace(user, c2).await.unwrap();

    let e1 = embedder.embed("home automation lights").await.unwrap();
    db.update_agent_workspace_topic(ws1.id, "home automation lights", &e1)
        .await
        .unwrap();

    let e2 = embedder.embed("grocery shopping").await.unwrap();
    db.update_agent_workspace_topic(ws2.id, "grocery shopping", &e2)
        .await
        .unwrap();

    // Router with very high threshold — only exact text match works with MockEmbeddings.
    let router = WorkspaceRouter::new(
        Arc::clone(&db) as Arc<dyn AgentWorkspaceStore>,
        embedder.clone() as Arc<dyn ironclaw::workspace::EmbeddingProvider>,
        0.99,
    );

    // Exact match on workspace 1.
    let result = router.route(user, "home automation lights").await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().id, ws1.id);

    // Exact match on workspace 2.
    let result = router.route(user, "grocery shopping").await.unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().id, ws2.id);

    // No match for unrelated text.
    let result = router
        .route(user, "something completely different")
        .await
        .unwrap();
    assert!(result.is_none());

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_router_with_hint_prepends_to_prompt() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_router_hint";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    // Topic is the hint-prepended text that the router will construct.
    let topic_text = "home automation: adjust the porch light brightness";
    let e = embedder.embed(topic_text).await.unwrap();
    db.update_agent_workspace_topic(ws.id, topic_text, &e)
        .await
        .unwrap();

    let router = WorkspaceRouter::new(
        Arc::clone(&db) as Arc<dyn AgentWorkspaceStore>,
        embedder.clone() as Arc<dyn ironclaw::workspace::EmbeddingProvider>,
        0.99,
    );

    // With hint, the router prepends "home automation: " to the prompt.
    let result = router
        .route_with_hint(
            user,
            "adjust the porch light brightness",
            Some("home automation"),
        )
        .await
        .unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap().id, ws.id);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_archived_workspaces_excluded_from_routing() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_archived_excluded";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    let e = embedder.embed("nanny schedule tracking").await.unwrap();
    db.update_agent_workspace_topic(ws.id, "nanny schedule tracking", &e)
        .await
        .unwrap();

    // Archive it.
    db.update_agent_workspace_status(ws.id, "archived")
        .await
        .unwrap();

    // Routing should not find archived workspace.
    let found = db.find_matching_workspace(user, &e, 0.5).await.unwrap();
    assert!(found.is_none());

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_user_isolation_in_routing() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let alice = "ws_test_isolation_alice";
    let bob = "ws_test_isolation_bob";
    cleanup(&db, alice).await;
    cleanup(&db, bob).await;

    let embedder = mock_embedder(128);

    // Alice creates a workspace.
    let c1 = db.create_conversation("test", alice, None).await.unwrap();
    let ws = db.create_agent_workspace(alice, c1).await.unwrap();
    let e = embedder.embed("private journal").await.unwrap();
    db.update_agent_workspace_topic(ws.id, "private journal", &e)
        .await
        .unwrap();

    // Bob should not find Alice's workspace.
    let found = db.find_matching_workspace(bob, &e, 0.5).await.unwrap();
    assert!(found.is_none());

    // Alice finds it fine.
    let found = db.find_matching_workspace(alice, &e, 0.5).await.unwrap();
    assert!(found.is_some());

    cleanup(&db, alice).await;
    cleanup(&db, bob).await;
}

// ---------------------------------------------------------------------------
// Tests: Archival
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_archive_stale_workspaces() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_archive_stale";
    cleanup(&db, user).await;

    let c1 = db.create_conversation("test", user, None).await.unwrap();
    let c2 = db.create_conversation("test", user, None).await.unwrap();
    let ws1 = db.create_agent_workspace(user, c1).await.unwrap();
    let _ws2 = db.create_agent_workspace(user, c2).await.unwrap();

    // Manually backdate ws1's last_accessed to 100 days ago.
    let pool = db.pool();
    let conn = pool.get().await.unwrap();
    conn.execute(
        "UPDATE agent_workspaces SET last_accessed = now() - interval '100 days' WHERE id = $1",
        &[&ws1.id],
    )
    .await
    .unwrap();

    // Archive workspaces stale for > 30 days.
    let count = db.archive_stale_workspaces(user, 30).await.unwrap();
    assert_eq!(count, 1);

    // ws1 is archived, ws2 is still active.
    let active = db.list_agent_workspaces(user, Some("active")).await.unwrap();
    assert_eq!(active.len(), 1);

    let archived = db
        .list_agent_workspaces(user, Some("archived"))
        .await
        .unwrap();
    assert_eq!(archived.len(), 1);
    assert_eq!(archived[0].id, ws1.id);

    cleanup(&db, user).await;
}

// ---------------------------------------------------------------------------
// Tests: Router + Queue working together
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_full_flow_route_then_queue() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_full_flow";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    // Create workspace with topic.
    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();
    let e = embedder.embed("budget tracking finances").await.unwrap();
    db.update_agent_workspace_topic(ws.id, "budget tracking finances", &e)
        .await
        .unwrap();

    // Route to it.
    let router = WorkspaceRouter::new(
        Arc::clone(&db) as Arc<dyn AgentWorkspaceStore>,
        embedder.clone() as Arc<dyn ironclaw::workspace::EmbeddingProvider>,
        0.99,
    );
    let matched = router
        .route(user, "budget tracking finances")
        .await
        .unwrap()
        .expect("should match");

    // Queue a message to the matched workspace.
    let queue = WorkspaceQueueManager::new(10);
    let (tx, rx) = oneshot::channel();
    queue
        .enqueue(
            matched.id,
            WorkspaceMessage {
                prompt: "what's my spending this month?".to_string(),
                priority: MessagePriority::User,
                response_tx: tx,
                ttl: None,
                enqueued_at: Instant::now(),
                metadata: None,
            },
        )
        .await
        .unwrap();

    // Dequeue should return the message.
    let msg = queue.dequeue(matched.id).await.unwrap();
    assert_eq!(msg.prompt, "what's my spending this month?");

    // Simulate sending a response back.
    msg.response_tx.send(Ok("You spent $342.".to_string())).ok();
    let response = rx.await.unwrap();
    assert_eq!(response.unwrap(), "You spent $342.");

    // Touch the workspace (simulating turn completion).
    db.touch_agent_workspace(matched.id).await.unwrap();
    let updated = db.get_agent_workspace(matched.id).await.unwrap().unwrap();
    assert_eq!(updated.turn_count, 1);

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_queue_priority_with_routed_workspace() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_queue_priority";
    cleanup(&db, user).await;

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    let queue = WorkspaceQueueManager::new(10);

    // Enqueue routine, then delegated, then user — all to the same workspace.
    let (tx1, _) = oneshot::channel();
    let (tx2, _) = oneshot::channel();
    let (tx3, _) = oneshot::channel();

    queue
        .enqueue(
            ws.id,
            WorkspaceMessage {
                prompt: "routine: check sensors".to_string(),
                priority: MessagePriority::Routine,
                response_tx: tx1,
                ttl: None,
                enqueued_at: Instant::now(),
                metadata: None,
            },
        )
        .await
        .unwrap();
    queue
        .enqueue(
            ws.id,
            WorkspaceMessage {
                prompt: "delegated: summarize logs".to_string(),
                priority: MessagePriority::Delegated,
                response_tx: tx2,
                ttl: None,
                enqueued_at: Instant::now(),
                metadata: None,
            },
        )
        .await
        .unwrap();
    queue
        .enqueue(
            ws.id,
            WorkspaceMessage {
                prompt: "user: turn off the lights".to_string(),
                priority: MessagePriority::User,
                response_tx: tx3,
                ttl: None,
                enqueued_at: Instant::now(),
                metadata: None,
            },
        )
        .await
        .unwrap();

    // Should come out in priority order: User > Delegated > Routine.
    let m1 = queue.dequeue(ws.id).await.unwrap();
    assert!(m1.prompt.starts_with("user:"));

    let m2 = queue.dequeue(ws.id).await.unwrap();
    assert!(m2.prompt.starts_with("delegated:"));

    let m3 = queue.dequeue(ws.id).await.unwrap();
    assert!(m3.prompt.starts_with("routine:"));

    assert!(queue.dequeue(ws.id).await.is_none());

    cleanup(&db, user).await;
}

#[tokio::test]
async fn test_ttl_expiry_in_queue() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_ttl_expiry";
    cleanup(&db, user).await;

    let conv_id = db.create_conversation("test", user, None).await.unwrap();
    let ws = db.create_agent_workspace(user, conv_id).await.unwrap();

    let queue = WorkspaceQueueManager::new(10);

    // Enqueue with already-expired TTL.
    let (tx1, _) = oneshot::channel();
    queue
        .enqueue(
            ws.id,
            WorkspaceMessage {
                prompt: "motion detected".to_string(),
                priority: MessagePriority::Routine,
                response_tx: tx1,
                ttl: Some(Duration::from_millis(1)),
                enqueued_at: Instant::now() - Duration::from_secs(10),
                metadata: None,
            },
        )
        .await
        .unwrap();

    // Enqueue a fresh message.
    let (tx2, _) = oneshot::channel();
    queue
        .enqueue(
            ws.id,
            WorkspaceMessage {
                prompt: "price alert still valid".to_string(),
                priority: MessagePriority::Routine,
                response_tx: tx2,
                ttl: None,
                enqueued_at: Instant::now(),
                metadata: None,
            },
        )
        .await
        .unwrap();

    // Expired message should be skipped; only the fresh one dequeues.
    let msg = queue.dequeue(ws.id).await.unwrap();
    assert_eq!(msg.prompt, "price alert still valid");

    assert!(queue.dequeue(ws.id).await.is_none());

    cleanup(&db, user).await;
}

// ---------------------------------------------------------------------------
// Tests: Multiple workspaces scenario (simulates real usage)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_multi_workspace_conversation_scenario() {
    let db = match setup().await {
        Some(d) => d,
        None => return,
    };
    let user = "ws_test_multi_scenario";
    cleanup(&db, user).await;

    let embedder = mock_embedder(128);

    // Simulate: user asks about 3 different topics over time.
    let topics = [
        ("home automation porch lights", "HA workspace"),
        ("grocery shopping weekly list", "Grocery workspace"),
        ("nanny schedule and payments", "Nanny workspace"),
    ];

    let mut workspace_ids = Vec::new();
    for (topic, _label) in &topics {
        let conv_id = db.create_conversation("test", user, None).await.unwrap();
        let ws = db.create_agent_workspace(user, conv_id).await.unwrap();
        let e = embedder.embed(topic).await.unwrap();
        db.update_agent_workspace_topic(ws.id, topic, &e)
            .await
            .unwrap();
        workspace_ids.push(ws.id);
    }

    let router = WorkspaceRouter::new(
        Arc::clone(&db) as Arc<dyn AgentWorkspaceStore>,
        embedder.clone() as Arc<dyn ironclaw::workspace::EmbeddingProvider>,
        0.99,
    );

    // Each topic routes to its own workspace (exact match via MockEmbeddings).
    for (i, (topic, label)) in topics.iter().enumerate() {
        let result = router.route(user, topic).await.unwrap();
        assert!(
            result.is_some(),
            "{label} should match for prompt '{topic}'"
        );
        assert_eq!(result.unwrap().id, workspace_ids[i], "{label} wrong ID");
    }

    // Touch workspaces to simulate turns.
    for id in &workspace_ids {
        db.touch_agent_workspace(*id).await.unwrap();
    }

    // All 3 are active.
    let active = db.list_agent_workspaces(user, Some("active")).await.unwrap();
    assert_eq!(active.len(), 3);

    // Archive the nanny workspace. It should disappear from routing.
    db.update_agent_workspace_status(workspace_ids[2], "archived")
        .await
        .unwrap();

    let nanny_emb = embedder.embed(topics[2].0).await.unwrap();
    let result = db
        .find_matching_workspace(user, &nanny_emb, 0.5)
        .await
        .unwrap();
    assert!(result.is_none(), "archived workspace should not route");

    // But the other two still work.
    let ha_result = router.route(user, topics[0].0).await.unwrap();
    assert!(ha_result.is_some());

    cleanup(&db, user).await;
}
