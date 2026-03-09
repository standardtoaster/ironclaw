//! Per-workspace async message queue with priority ordering.
//!
//! Ensures single-writer access to workspace conversations. Each workspace
//! gets its own queue. Messages are priority-ordered: User > Delegated > Routine.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify, RwLock, oneshot};
use uuid::Uuid;

use crate::agent::scheduler::Scheduler;
use crate::db::Database;

/// Message priority levels. Higher value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Routine = 0,
    Delegated = 1,
    User = 2,
}

/// A message destined for a workspace queue.
pub struct WorkspaceMessage {
    pub prompt: String,
    pub priority: MessagePriority,
    pub response_tx: oneshot::Sender<Result<String, String>>,
    pub ttl: Option<Duration>,
    pub enqueued_at: Instant,
    pub metadata: Option<serde_json::Value>,
}

impl std::fmt::Debug for WorkspaceMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceMessage")
            .field("priority", &self.priority)
            .field("ttl", &self.ttl)
            .field("enqueued_at", &self.enqueued_at)
            .finish()
    }
}

/// Whether a workspace is currently processing a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceStatus {
    Idle,
    Busy { queue_depth: usize },
}

/// Errors from queue operations.
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("queue full for workspace {0}")]
    QueueFull(Uuid),
}

/// Internal per-workspace queue state.
struct WorkspaceQueue {
    messages: Mutex<VecDeque<WorkspaceMessage>>,
    busy: Mutex<bool>,
    notify: Notify,
}

impl WorkspaceQueue {
    fn new() -> Self {
        Self {
            messages: Mutex::new(VecDeque::new()),
            busy: Mutex::new(false),
            notify: Notify::new(),
        }
    }
}

/// Manages per-workspace message queues with priority ordering and TTL expiry.
pub struct WorkspaceQueueManager {
    queues: RwLock<HashMap<Uuid, Arc<WorkspaceQueue>>>,
    max_queue_depth: usize,
}

impl WorkspaceQueueManager {
    /// Create a new queue manager with the given max depth per workspace.
    pub fn new(max_queue_depth: usize) -> Self {
        Self {
            queues: RwLock::new(HashMap::new()),
            max_queue_depth,
        }
    }

    /// Get or create the queue for a workspace.
    async fn get_or_create_queue(&self, workspace_id: Uuid) -> Arc<WorkspaceQueue> {
        // Fast path: read lock
        {
            let queues = self.queues.read().await;
            if let Some(q) = queues.get(&workspace_id) {
                return Arc::clone(q);
            }
        }
        // Slow path: write lock
        let mut queues = self.queues.write().await;
        Arc::clone(
            queues
                .entry(workspace_id)
                .or_insert_with(|| Arc::new(WorkspaceQueue::new())),
        )
    }

    /// Enqueue a message to a workspace's queue.
    ///
    /// User-priority messages always get in. Routine messages are rejected
    /// with `QueueError::QueueFull` when the queue is at capacity.
    pub async fn enqueue(
        &self,
        workspace_id: Uuid,
        message: WorkspaceMessage,
    ) -> Result<(), QueueError> {
        let queue = self.get_or_create_queue(workspace_id).await;
        let mut messages = queue.messages.lock().await;

        if messages.len() >= self.max_queue_depth
            && message.priority != MessagePriority::User
        {
            return Err(QueueError::QueueFull(workspace_id));
        }

        // Insert in priority-sorted position (highest priority at front).
        // Within same priority, append (FIFO).
        let pos = messages
            .iter()
            .position(|m| m.priority < message.priority)
            .unwrap_or(messages.len());
        messages.insert(pos, message);

        // Wake any waiting processor.
        queue.notify.notify_one();
        Ok(())
    }

    /// Dequeue the highest-priority non-expired message from a workspace.
    ///
    /// Expired messages (past TTL) are silently dropped from any position.
    pub async fn dequeue(&self, workspace_id: Uuid) -> Option<WorkspaceMessage> {
        let queue = self.get_or_create_queue(workspace_id).await;
        let mut messages = queue.messages.lock().await;
        let now = Instant::now();

        // Drop all expired messages from any position in the queue.
        messages.retain(|msg| {
            if let Some(ttl) = msg.ttl {
                msg.enqueued_at + ttl >= now
            } else {
                true
            }
        });

        messages.pop_front()
    }

    /// Check whether a workspace is currently busy.
    pub async fn workspace_status(&self, workspace_id: Uuid) -> WorkspaceStatus {
        let queue = self.get_or_create_queue(workspace_id).await;
        let busy = *queue.busy.lock().await;
        if busy {
            let depth = queue.messages.lock().await.len();
            WorkspaceStatus::Busy {
                queue_depth: depth,
            }
        } else {
            WorkspaceStatus::Idle
        }
    }

    /// Mark a workspace as actively processing a turn.
    pub async fn mark_busy(&self, workspace_id: Uuid) {
        let queue = self.get_or_create_queue(workspace_id).await;
        *queue.busy.lock().await = true;
    }

    /// Mark a workspace as idle (done processing).
    pub async fn mark_idle(&self, workspace_id: Uuid) {
        let queue = self.get_or_create_queue(workspace_id).await;
        *queue.busy.lock().await = false;
    }

    /// Process a single message by dispatching a job to the workspace's conversation.
    ///
    /// Returns `Ok(result_text)` on success, `Err(error_string)` on failure.
    pub async fn process_message(
        scheduler: &Arc<Scheduler>,
        db: &Arc<dyn Database>,
        workspace_id: Uuid,
        conversation_id: Uuid,
        msg: WorkspaceMessage,
    ) {
        let user_id = msg
            .metadata
            .as_ref()
            .and_then(|m| m.get("user_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();

        let title = msg
            .metadata
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("Workspace task")
            .to_string();

        let job_metadata = msg
            .metadata
            .as_ref()
            .and_then(|m| m.get("job_metadata"))
            .cloned();

        let result = async {
            let job_id = scheduler
                .dispatch_job_to_conversation(
                    &user_id,
                    conversation_id,
                    &title,
                    &msg.prompt,
                    job_metadata,
                )
                .await
                .map_err(|e| format!("failed to dispatch job: {e}"))?;

            scheduler
                .await_job(job_id, Duration::from_secs(300))
                .await
                .map_err(|e| format!("workspace job failed or timed out: {e}"))
        }
        .await;

        // Touch the workspace to update last_accessed (best-effort).
        if let Err(e) = db.touch_agent_workspace(workspace_id).await {
            tracing::warn!(
                workspace_id = %workspace_id,
                "failed to touch workspace: {e}"
            );
        }

        // Send result back through the response channel.
        let _ = msg.response_tx.send(result);
    }

    /// Start processing the queue for a workspace.
    ///
    /// If the workspace is already busy, this is a no-op (messages will be
    /// picked up by the existing processing loop). If idle, spawns a tokio
    /// task that drains the queue until empty, then marks the workspace idle.
    pub async fn start_processing(
        self: &Arc<Self>,
        workspace_id: Uuid,
        conversation_id: Uuid,
        scheduler: Arc<Scheduler>,
        db: Arc<dyn Database>,
    ) {
        let queue = self.get_or_create_queue(workspace_id).await;

        // Only start if not already busy.
        {
            let mut busy = queue.busy.lock().await;
            if *busy {
                return;
            }
            *busy = true;
        }

        let mgr = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                match mgr.dequeue(workspace_id).await {
                    Some(msg) => {
                        Self::process_message(
                            &scheduler,
                            &db,
                            workspace_id,
                            conversation_id,
                            msg,
                        )
                        .await;
                    }
                    None => {
                        // Queue is empty — mark idle and exit.
                        mgr.mark_idle(workspace_id).await;
                        break;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(
        prompt: &str,
        priority: MessagePriority,
        ttl: Option<Duration>,
    ) -> (WorkspaceMessage, oneshot::Receiver<Result<String, String>>) {
        let (tx, rx) = oneshot::channel();
        let msg = WorkspaceMessage {
            prompt: prompt.to_string(),
            priority,
            response_tx: tx,
            ttl,
            enqueued_at: Instant::now(),
            metadata: None,
        };
        (msg, rx)
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        let (routine, _r1) = make_message("routine", MessagePriority::Routine, None);
        let (delegated, _r2) = make_message("delegated", MessagePriority::Delegated, None);
        let (user, _r3) = make_message("user", MessagePriority::User, None);

        // Enqueue in reverse priority order.
        mgr.enqueue(ws, routine).await.unwrap();
        mgr.enqueue(ws, delegated).await.unwrap();
        mgr.enqueue(ws, user).await.unwrap();

        // Dequeue should return highest priority first.
        let m1 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m1.prompt, "user");
        assert_eq!(m1.priority, MessagePriority::User);

        let m2 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m2.prompt, "delegated");
        assert_eq!(m2.priority, MessagePriority::Delegated);

        let m3 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m3.prompt, "routine");
        assert_eq!(m3.priority, MessagePriority::Routine);

        assert!(mgr.dequeue(ws).await.is_none());
    }

    #[tokio::test]
    async fn test_fifo_within_same_priority() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        let (m1, _r1) = make_message("first", MessagePriority::User, None);
        let (m2, _r2) = make_message("second", MessagePriority::User, None);
        let (m3, _r3) = make_message("third", MessagePriority::User, None);

        mgr.enqueue(ws, m1).await.unwrap();
        mgr.enqueue(ws, m2).await.unwrap();
        mgr.enqueue(ws, m3).await.unwrap();

        assert_eq!(mgr.dequeue(ws).await.unwrap().prompt, "first");
        assert_eq!(mgr.dequeue(ws).await.unwrap().prompt, "second");
        assert_eq!(mgr.dequeue(ws).await.unwrap().prompt, "third");
    }

    #[tokio::test]
    async fn test_ttl_expiry() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        // Create a message with an already-expired TTL.
        let (tx, _rx) = oneshot::channel();
        let expired = WorkspaceMessage {
            prompt: "expired".to_string(),
            priority: MessagePriority::User,
            response_tx: tx,
            ttl: Some(Duration::from_millis(0)),
            enqueued_at: Instant::now() - Duration::from_secs(1),
            metadata: None,
        };
        mgr.enqueue(ws, expired).await.unwrap();

        // Fresh message behind it.
        let (fresh, _rx2) = make_message("fresh", MessagePriority::Routine, None);
        mgr.enqueue(ws, fresh).await.unwrap();

        // The expired message should be skipped.
        let m = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m.prompt, "fresh");
    }

    #[tokio::test]
    async fn test_queue_depth_limit_rejects_routine() {
        let mgr = WorkspaceQueueManager::new(2);
        let ws = Uuid::new_v4();

        let (m1, _r1) = make_message("a", MessagePriority::Routine, None);
        let (m2, _r2) = make_message("b", MessagePriority::Routine, None);
        let (m3, _r3) = make_message("c", MessagePriority::Routine, None);

        mgr.enqueue(ws, m1).await.unwrap();
        mgr.enqueue(ws, m2).await.unwrap();

        // Third routine message should be rejected.
        let err = mgr.enqueue(ws, m3).await.unwrap_err();
        assert!(matches!(err, QueueError::QueueFull(_)));
    }

    #[tokio::test]
    async fn test_queue_depth_limit_accepts_user() {
        let mgr = WorkspaceQueueManager::new(2);
        let ws = Uuid::new_v4();

        let (m1, _r1) = make_message("a", MessagePriority::Routine, None);
        let (m2, _r2) = make_message("b", MessagePriority::Routine, None);
        let (m3, _r3) = make_message("urgent", MessagePriority::User, None);

        mgr.enqueue(ws, m1).await.unwrap();
        mgr.enqueue(ws, m2).await.unwrap();

        // User message should still be accepted even when full.
        mgr.enqueue(ws, m3).await.unwrap();

        // User message comes out first due to priority.
        let m = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m.prompt, "urgent");
    }

    #[tokio::test]
    async fn test_workspace_status() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        assert_eq!(mgr.workspace_status(ws).await, WorkspaceStatus::Idle);

        mgr.mark_busy(ws).await;
        let status = mgr.workspace_status(ws).await;
        assert!(matches!(status, WorkspaceStatus::Busy { queue_depth: 0 }));

        mgr.mark_idle(ws).await;
        assert_eq!(mgr.workspace_status(ws).await, WorkspaceStatus::Idle);
    }

    #[tokio::test]
    async fn test_separate_workspaces_are_independent() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws1 = Uuid::new_v4();
        let ws2 = Uuid::new_v4();

        let (m1, _r1) = make_message("ws1-msg", MessagePriority::User, None);
        let (m2, _r2) = make_message("ws2-msg", MessagePriority::User, None);

        mgr.enqueue(ws1, m1).await.unwrap();
        mgr.enqueue(ws2, m2).await.unwrap();

        assert_eq!(mgr.dequeue(ws1).await.unwrap().prompt, "ws1-msg");
        assert_eq!(mgr.dequeue(ws2).await.unwrap().prompt, "ws2-msg");
        assert!(mgr.dequeue(ws1).await.is_none());
        assert!(mgr.dequeue(ws2).await.is_none());
    }

    /// Helper to create a message with a backdated enqueued_at for TTL testing.
    fn make_expired_message(
        prompt: &str,
        priority: MessagePriority,
        ttl: Duration,
        age: Duration,
    ) -> (WorkspaceMessage, oneshot::Receiver<Result<String, String>>) {
        let (tx, rx) = oneshot::channel();
        let msg = WorkspaceMessage {
            prompt: prompt.to_string(),
            priority,
            response_tx: tx,
            ttl: Some(ttl),
            enqueued_at: Instant::now() - age,
            metadata: None,
        };
        (msg, rx)
    }

    #[tokio::test]
    async fn test_ttl_expiry_non_front_position() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        // User message (no TTL, valid) — will be at front due to higher priority.
        let (user_msg, _r1) = make_message("user", MessagePriority::User, None);
        // Routine message with expired TTL — will be behind the User message.
        let (expired_routine, _r2) = make_expired_message(
            "expired-routine",
            MessagePriority::Routine,
            Duration::from_millis(0),
            Duration::from_secs(1),
        );

        mgr.enqueue(ws, user_msg).await.unwrap();
        mgr.enqueue(ws, expired_routine).await.unwrap();

        // First dequeue gets the User message.
        let m1 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m1.prompt, "user");

        // Second dequeue should return None — the expired Routine must be skipped.
        assert!(mgr.dequeue(ws).await.is_none());
    }

    #[tokio::test]
    async fn test_response_channel_on_ttl_expiry() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        // Create an expired message and hold onto its response receiver.
        let (expired, rx) = make_expired_message(
            "expired",
            MessagePriority::User,
            Duration::from_millis(0),
            Duration::from_secs(1),
        );
        mgr.enqueue(ws, expired).await.unwrap();

        // Dequeue skips it (returns None), dropping the response_tx.
        assert!(mgr.dequeue(ws).await.is_none());

        // The receiver should get a RecvError because the sender was dropped.
        assert!(rx.await.is_err());
    }

    #[tokio::test]
    async fn test_delegated_rejected_when_full() {
        let mgr = WorkspaceQueueManager::new(2);
        let ws = Uuid::new_v4();

        let (m1, _r1) = make_message("a", MessagePriority::Routine, None);
        let (m2, _r2) = make_message("b", MessagePriority::Routine, None);
        mgr.enqueue(ws, m1).await.unwrap();
        mgr.enqueue(ws, m2).await.unwrap();

        // Delegated message should be rejected (only User is exempt).
        let (delegated, _r3) = make_message("delegated", MessagePriority::Delegated, None);
        let err = mgr.enqueue(ws, delegated).await.unwrap_err();
        assert!(matches!(err, QueueError::QueueFull(_)));
    }

    #[tokio::test]
    async fn test_concurrent_enqueue_dequeue() {
        let mgr = Arc::new(WorkspaceQueueManager::new(100));
        let ws = Uuid::new_v4();
        let num_messages = 50;

        // Spawn producers.
        let mut handles = Vec::new();
        for i in 0..num_messages {
            let mgr = Arc::clone(&mgr);
            handles.push(tokio::spawn(async move {
                let (msg, _rx) = make_message(
                    &format!("msg-{i}"),
                    MessagePriority::User,
                    None,
                );
                mgr.enqueue(ws, msg).await.unwrap();
            }));
        }

        // Wait for all enqueues to finish.
        for h in handles {
            h.await.unwrap();
        }

        // Spawn consumers that drain the queue.
        let mut consumer_handles = Vec::new();
        for _ in 0..num_messages {
            let mgr = Arc::clone(&mgr);
            consumer_handles.push(tokio::spawn(async move {
                mgr.dequeue(ws).await
            }));
        }

        let mut dequeued = Vec::new();
        for h in consumer_handles {
            if let Some(msg) = h.await.unwrap() {
                dequeued.push(msg.prompt);
            }
        }

        // All messages should have been dequeued exactly once.
        assert_eq!(dequeued.len(), num_messages);

        // Queue should now be empty.
        assert!(mgr.dequeue(ws).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_empty_after_all_expired() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        for i in 0..3 {
            let (msg, _rx) = make_expired_message(
                &format!("expired-{i}"),
                MessagePriority::User,
                Duration::from_millis(0),
                Duration::from_secs(1),
            );
            mgr.enqueue(ws, msg).await.unwrap();
        }

        // All messages are expired — dequeue should return None.
        assert!(mgr.dequeue(ws).await.is_none());
    }

    #[tokio::test]
    async fn test_mixed_ttl_and_no_ttl() {
        let mgr = WorkspaceQueueManager::new(10);
        let ws = Uuid::new_v4();

        // Expired User message (TTL=0, enqueued 1s ago).
        let (expired, _r1) = make_expired_message(
            "expired-user",
            MessagePriority::User,
            Duration::from_millis(0),
            Duration::from_secs(1),
        );
        // No-TTL Delegated message (never expires).
        let (no_ttl, _r2) = make_message("no-ttl-delegated", MessagePriority::Delegated, None);
        // Valid TTL Routine message (TTL=1h, enqueued just now).
        let (tx, _r3) = oneshot::channel();
        let valid_ttl = WorkspaceMessage {
            prompt: "valid-ttl-routine".to_string(),
            priority: MessagePriority::Routine,
            response_tx: tx,
            ttl: Some(Duration::from_secs(3600)),
            enqueued_at: Instant::now(),
            metadata: None,
        };

        mgr.enqueue(ws, expired).await.unwrap();
        mgr.enqueue(ws, no_ttl).await.unwrap();
        mgr.enqueue(ws, valid_ttl).await.unwrap();

        // Expired User is dropped. Remaining in priority order: Delegated, then Routine.
        let m1 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m1.prompt, "no-ttl-delegated");

        let m2 = mgr.dequeue(ws).await.unwrap();
        assert_eq!(m2.prompt, "valid-ttl-routine");

        assert!(mgr.dequeue(ws).await.is_none());
    }
}
