//! Passive ingestion buffer.
//!
//! Accumulates `suppress_response=true` messages and flushes them as
//! batches for efficient LLM processing. Instead of one LLM call per
//! passive message, the buffer collects messages and presents them as
//! a single batch when a silence gap or size limit is reached.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

/// Key for buffering — one buffer per unique source.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct BufferKey {
    pub channel: String,
    pub source_id: String,
    pub user_id: String,
}

/// A buffered passive message.
#[derive(Debug, Clone)]
pub struct BufferedMessage {
    pub content: String,
    pub sender: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Per-source buffer state.
#[derive(Debug)]
struct SourceBuffer {
    messages: Vec<BufferedMessage>,
    last_arrival: Instant,
    last_flush: Instant,
}

/// Passive message buffer with per-source batching.
pub struct PassiveBuffer {
    buffers: HashMap<BufferKey, SourceBuffer>,
    /// Silence duration before auto-flush.
    silence_timeout: Duration,
    /// Max messages per source before forced flush.
    max_batch_size: usize,
}

impl PassiveBuffer {
    pub fn new(silence_timeout: Duration, max_batch_size: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            silence_timeout,
            max_batch_size,
        }
    }

    /// Push a passive message into the buffer.
    pub fn push(&mut self, key: BufferKey, msg: BufferedMessage) {
        let now = Instant::now();
        let buf = self.buffers.entry(key).or_insert_with(|| SourceBuffer {
            messages: Vec::new(),
            last_arrival: now,
            last_flush: now,
        });
        buf.messages.push(msg);
        buf.last_arrival = Instant::now();
    }

    /// Check if any source buffer is ready to flush.
    pub fn has_ready(&self) -> bool {
        let now = Instant::now();
        self.buffers.values().any(|buf| self.is_ready(buf, now))
    }

    /// Get keys of all buffers ready to flush, ordered by last-flush time (oldest first).
    pub fn ready_keys(&self) -> Vec<BufferKey> {
        let now = Instant::now();
        let mut ready: Vec<_> = self
            .buffers
            .iter()
            .filter(|(_, buf)| self.is_ready(buf, now))
            .map(|(key, buf)| (key.clone(), buf.last_flush))
            .collect();
        ready.sort_by_key(|(_, last_flush)| *last_flush);
        ready.into_iter().map(|(key, _)| key).collect()
    }

    /// Drain a source buffer and return its messages.
    pub fn drain(&mut self, key: &BufferKey) -> Vec<BufferedMessage> {
        if let Some(buf) = self.buffers.get_mut(key) {
            buf.last_flush = Instant::now();
            std::mem::take(&mut buf.messages)
        } else {
            Vec::new()
        }
    }

    /// Get the deadline for the next flush (for use with `tokio::select!` timeout).
    /// Returns `None` if no messages are buffered.
    pub fn next_flush_deadline(&self) -> Option<Instant> {
        self.buffers
            .values()
            .filter(|buf| !buf.messages.is_empty())
            .map(|buf| buf.last_arrival + self.silence_timeout)
            .min()
    }

    /// Format a batch of messages as a text block for LLM processing.
    pub fn format_batch(key: &BufferKey, messages: &[BufferedMessage]) -> String {
        let mut out = format!(
            "[Passive ingestion batch — {} message(s) from {}/{}]\n\n",
            messages.len(),
            key.channel,
            key.source_id,
        );
        for msg in messages {
            let sender = msg.sender.as_deref().unwrap_or("unknown");
            let ts = msg.timestamp.format("%H:%M:%S");
            out.push_str(&format!("{ts} {sender}: {}\n", msg.content));
        }
        out.push_str(
            "\nProcess this batch. Extract actionable items, update relevant \
             collections, and note any context worth remembering. Do not respond to the chat.",
        );
        out
    }

    /// Number of buffered messages across all sources.
    pub fn total_pending(&self) -> usize {
        self.buffers.values().map(|b| b.messages.len()).sum()
    }

    /// Remove empty source buffers to prevent unbounded growth.
    pub fn cleanup_empty(&mut self) {
        self.buffers.retain(|_, buf| !buf.messages.is_empty());
    }

    /// Check if a single source buffer is ready to flush.
    fn is_ready(&self, buf: &SourceBuffer, now: Instant) -> bool {
        !buf.messages.is_empty()
            && (buf.messages.len() >= self.max_batch_size
                || now.duration_since(buf.last_arrival) >= self.silence_timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(source: &str) -> BufferKey {
        BufferKey {
            channel: "whatsapp".to_string(),
            source_id: source.to_string(),
            user_id: "user_a".to_string(),
        }
    }

    fn test_key_with_user(source: &str, user: &str) -> BufferKey {
        BufferKey {
            channel: "whatsapp".to_string(),
            source_id: source.to_string(),
            user_id: user.to_string(),
        }
    }

    fn test_msg(content: &str) -> BufferedMessage {
        BufferedMessage {
            content: content.to_string(),
            sender: Some("sarah".to_string()),
            timestamp: Utc::now(),
        }
    }

    fn test_msg_with_sender(content: &str, sender: Option<&str>) -> BufferedMessage {
        BufferedMessage {
            content: content.to_string(),
            sender: sender.map(|s| s.to_string()),
            timestamp: Utc::now(),
        }
    }

    // --- Existing tests ---

    #[test]
    fn test_push_and_pending() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(10), 20);
        assert_eq!(buf.total_pending(), 0);
        buf.push(test_key("group1"), test_msg("hello"));
        assert_eq!(buf.total_pending(), 1);
        buf.push(test_key("group1"), test_msg("world"));
        assert_eq!(buf.total_pending(), 2);
    }

    #[test]
    fn test_flush_on_max_batch_size() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 3);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        buf.push(key.clone(), test_msg("b"));
        assert!(!buf.has_ready());
        buf.push(key.clone(), test_msg("c"));
        assert!(buf.has_ready());
        let msgs = buf.drain(&key);
        assert_eq!(msgs.len(), 3);
        assert_eq!(buf.total_pending(), 0);
    }

    #[test]
    fn test_drain_returns_empty_for_unknown_key() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(10), 20);
        let msgs = buf.drain(&test_key("nonexistent"));
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_multiple_sources_independent() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        buf.push(test_key("group1"), test_msg("a"));
        buf.push(test_key("group2"), test_msg("b"));
        assert!(!buf.has_ready()); // Neither is at max
        buf.push(test_key("group1"), test_msg("c"));
        assert!(buf.has_ready()); // group1 is at max
        let ready = buf.ready_keys();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].source_id, "group1");
    }

    #[test]
    fn test_format_batch() {
        let key = test_key("team-chat");
        let msgs = vec![test_msg("we need supplies"), test_msg("oh wait I got them")];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        assert!(formatted.contains("team-chat"));
        assert!(formatted.contains("we need supplies"));
        assert!(formatted.contains("oh wait I got them"));
        assert!(formatted.contains("Process this batch"));
    }

    #[test]
    fn test_cleanup_empty() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        buf.drain(&key);
        assert_eq!(buf.total_pending(), 0);
        buf.cleanup_empty();
        assert!(buf.buffers.is_empty());
    }

    #[test]
    fn test_next_flush_deadline_none_when_empty() {
        let buf = PassiveBuffer::new(Duration::from_secs(10), 20);
        assert!(buf.next_flush_deadline().is_none());
    }

    #[test]
    fn test_next_flush_deadline_some_when_pending() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(10), 20);
        buf.push(test_key("group1"), test_msg("hello"));
        assert!(buf.next_flush_deadline().is_some());
    }

    // --- Multi-tenant isolation tests ---

    #[test]
    fn test_different_users_same_source_are_separate_buffers() {
        // Two users in the same WhatsApp group produce independent buffers.
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        let andrew_key = test_key_with_user("family-chat", "andrew");
        let grace_key = test_key_with_user("family-chat", "grace");

        buf.push(andrew_key.clone(), test_msg("andrew msg 1"));
        buf.push(grace_key.clone(), test_msg("grace msg 1"));
        buf.push(andrew_key.clone(), test_msg("andrew msg 2")); // andrew hits max_batch_size=2

        assert!(buf.has_ready());
        let ready = buf.ready_keys();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].user_id, "andrew");

        // Grace's buffer is untouched
        let andrew_msgs = buf.drain(&andrew_key);
        assert_eq!(andrew_msgs.len(), 2);
        assert_eq!(buf.total_pending(), 1); // grace still has 1
    }

    #[test]
    fn test_drain_does_not_leak_across_users() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 10);
        let key_a = test_key_with_user("group1", "alice");
        let key_b = test_key_with_user("group1", "bob");

        buf.push(key_a.clone(), test_msg("alice secret"));
        buf.push(key_b.clone(), test_msg("bob secret"));

        let alice_msgs = buf.drain(&key_a);
        assert_eq!(alice_msgs.len(), 1);
        assert_eq!(alice_msgs[0].content, "alice secret");

        // Bob's data is still there, not drained by Alice's drain
        assert_eq!(buf.total_pending(), 1);
        let bob_msgs = buf.drain(&key_b);
        assert_eq!(bob_msgs.len(), 1);
        assert_eq!(bob_msgs[0].content, "bob secret");
    }

    // --- Edge cases ---

    #[test]
    fn test_max_batch_size_one_flushes_immediately() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 1);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("single"));
        assert!(buf.has_ready());
        let msgs = buf.drain(&key);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_drain_resets_buffer_for_new_messages() {
        // After draining, new messages should accumulate fresh.
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        let key = test_key("group1");

        buf.push(key.clone(), test_msg("batch1-a"));
        buf.push(key.clone(), test_msg("batch1-b"));
        let batch1 = buf.drain(&key);
        assert_eq!(batch1.len(), 2);

        // Push new messages — should not include old ones
        buf.push(key.clone(), test_msg("batch2-a"));
        assert_eq!(buf.total_pending(), 1);
        let batch2 = buf.drain(&key);
        assert_eq!(batch2.len(), 1);
        assert_eq!(batch2[0].content, "batch2-a");
    }

    #[test]
    fn test_ready_keys_ordered_by_oldest_flush() {
        // Keys that were flushed least recently should come first.
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 1);
        let key_a = test_key("a");
        let key_b = test_key("b");

        // Push a first, then b — both ready immediately at batch_size=1
        buf.push(key_a.clone(), test_msg("a"));
        buf.push(key_b.clone(), test_msg("b"));

        let ready = buf.ready_keys();
        assert_eq!(ready.len(), 2);
        // Both were "last flushed" at creation time (approximately equal),
        // but key_a was inserted first so its last_flush <= key_b's.
        assert_eq!(ready[0].source_id, "a");
        assert_eq!(ready[1].source_id, "b");
    }

    #[test]
    fn test_cleanup_preserves_non_empty_buffers() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 10);
        let key_empty = test_key("empty");
        let key_full = test_key("full");

        buf.push(key_empty.clone(), test_msg("temp"));
        buf.push(key_full.clone(), test_msg("keep me"));
        buf.drain(&key_empty); // empty it

        buf.cleanup_empty();
        assert_eq!(buf.total_pending(), 1); // key_full preserved
        assert!(!buf.buffers.contains_key(&key_empty));
        assert!(buf.buffers.contains_key(&key_full));
    }

    #[test]
    fn test_format_batch_empty_messages() {
        let key = test_key("group1");
        let formatted = PassiveBuffer::format_batch(&key, &[]);
        assert!(formatted.contains("0 message(s)"));
        assert!(formatted.contains("Process this batch"));
    }

    #[test]
    fn test_format_batch_sender_none_shows_unknown() {
        let key = test_key("group1");
        let msgs = vec![test_msg_with_sender("hello", None)];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        assert!(formatted.contains("unknown: hello"));
    }

    #[test]
    fn test_format_batch_preserves_message_order() {
        let key = test_key("group1");
        let msgs = vec![
            test_msg_with_sender("first", Some("alice")),
            test_msg_with_sender("second", Some("bob")),
            test_msg_with_sender("third", Some("alice")),
        ];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        let first_pos = formatted.find("first").unwrap();
        let second_pos = formatted.find("second").unwrap();
        let third_pos = formatted.find("third").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_format_batch_includes_channel_and_source() {
        let key = BufferKey {
            channel: "telegram".to_string(),
            source_id: "chat-12345".to_string(),
            user_id: "user1".to_string(),
        };
        let msgs = vec![test_msg("hello")];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        assert!(formatted.contains("telegram/chat-12345"));
    }

    // --- Silence timeout behavior ---

    #[test]
    fn test_not_ready_before_silence_timeout() {
        // With a long timeout and small batch, messages should not be ready.
        let mut buf = PassiveBuffer::new(Duration::from_secs(3600), 100);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        assert!(!buf.has_ready());
        assert!(buf.ready_keys().is_empty());
    }

    #[test]
    fn test_zero_silence_timeout_flushes_immediately() {
        // A 0-duration timeout means any message is immediately ready.
        let mut buf = PassiveBuffer::new(Duration::from_secs(0), 100);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        // With 0 timeout, the silence gap is already met.
        assert!(buf.has_ready());
    }

    #[test]
    fn test_next_flush_deadline_reflects_silence_timeout() {
        let timeout = Duration::from_secs(30);
        let mut buf = PassiveBuffer::new(timeout, 100);
        let key = test_key("group1");

        let before = Instant::now();
        buf.push(key.clone(), test_msg("a"));
        let deadline = buf.next_flush_deadline().unwrap();

        // Deadline should be approximately now + timeout (within a few ms)
        let expected_min = before + timeout;
        assert!(
            deadline >= expected_min - Duration::from_millis(10),
            "deadline should be >= now + timeout"
        );
        assert!(
            deadline <= expected_min + Duration::from_millis(100),
            "deadline should be close to now + timeout"
        );
    }

    #[test]
    fn test_next_flush_deadline_picks_earliest() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(60), 100);
        let key_a = test_key("a");
        let key_b = test_key("b");

        buf.push(key_a.clone(), test_msg("a first"));
        // Small delay to ensure key_a has an earlier deadline
        std::thread::sleep(Duration::from_millis(5));
        buf.push(key_b.clone(), test_msg("b later"));

        let deadline = buf.next_flush_deadline().unwrap();
        // Deadline should correspond to key_a (earlier arrival)
        let key_b_buf = &buf.buffers[&key_b];
        let key_b_deadline = key_b_buf.last_arrival + Duration::from_secs(60);
        assert!(deadline < key_b_deadline);
    }

    // --- Drain after flush, refill patterns ---

    #[test]
    fn test_double_drain_returns_empty() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        buf.push(key.clone(), test_msg("b"));

        let first = buf.drain(&key);
        assert_eq!(first.len(), 2);

        let second = buf.drain(&key);
        assert!(second.is_empty());
    }

    #[test]
    fn test_message_content_preserved_through_buffer() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 10);
        let key = test_key("group1");

        let content = "Hello 👋 こんにちは emoji 🎉 and unicode: café";
        buf.push(key.clone(), test_msg(content));
        let msgs = buf.drain(&key);
        assert_eq!(msgs[0].content, content);
    }

    #[test]
    fn test_sender_preserved_through_buffer() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 10);
        let key = test_key("group1");

        buf.push(key.clone(), test_msg_with_sender("hello", Some("alice")));
        buf.push(key.clone(), test_msg_with_sender("hi", None));

        let msgs = buf.drain(&key);
        assert_eq!(msgs[0].sender.as_deref(), Some("alice"));
        assert_eq!(msgs[1].sender, None);
    }

    // --- BufferKey equality / isolation ---

    #[test]
    fn test_buffer_key_different_channels_are_separate() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 1);
        let wa_key = BufferKey {
            channel: "whatsapp".to_string(),
            source_id: "group1".to_string(),
            user_id: "user1".to_string(),
        };
        let tg_key = BufferKey {
            channel: "telegram".to_string(),
            source_id: "group1".to_string(),
            user_id: "user1".to_string(),
        };

        buf.push(wa_key.clone(), test_msg("wa msg"));
        buf.push(tg_key.clone(), test_msg("tg msg"));

        // Both ready (batch_size=1)
        let ready = buf.ready_keys();
        assert_eq!(ready.len(), 2);

        let wa_msgs = buf.drain(&wa_key);
        let tg_msgs = buf.drain(&tg_key);
        assert_eq!(wa_msgs[0].content, "wa msg");
        assert_eq!(tg_msgs[0].content, "tg msg");
    }

    // --- Large batch ---

    #[test]
    fn test_large_batch_drains_all() {
        let batch_size = 500;
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), batch_size);
        let key = test_key("group1");

        for i in 0..batch_size {
            buf.push(key.clone(), test_msg(&format!("msg {i}")));
        }
        assert!(buf.has_ready());
        let msgs = buf.drain(&key);
        assert_eq!(msgs.len(), batch_size);
        assert_eq!(buf.total_pending(), 0);
    }
}
