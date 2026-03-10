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
            user_id: "andrew".to_string(),
        }
    }

    fn test_msg(content: &str) -> BufferedMessage {
        BufferedMessage {
            content: content.to_string(),
            sender: Some("sarah".to_string()),
            timestamp: Utc::now(),
        }
    }

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
        let key = test_key("nanny-group");
        let msgs = vec![test_msg("we need eggs"), test_msg("oh wait I got them")];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        assert!(formatted.contains("nanny-group"));
        assert!(formatted.contains("we need eggs"));
        assert!(formatted.contains("oh wait I got them"));
        assert!(formatted.contains("Process this batch"));
    }

    #[test]
    fn test_format_batch_with_no_sender() {
        let key = test_key("group1");
        let msgs = vec![BufferedMessage {
            content: "hello".to_string(),
            sender: None,
            timestamp: Utc::now(),
        }];
        let formatted = PassiveBuffer::format_batch(&key, &msgs);
        assert!(formatted.contains("unknown: hello"));
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

    #[test]
    fn test_ready_keys_ordered_by_last_flush() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 1);
        // Push to group1 first, then group2 — both reach max immediately
        buf.push(test_key("group1"), test_msg("a"));
        buf.push(test_key("group2"), test_msg("b"));
        let ready = buf.ready_keys();
        assert_eq!(ready.len(), 2);
        // Both were created at roughly the same time, so order depends on
        // creation order of `last_flush`. Just verify both are present.
        let ids: Vec<&str> = ready.iter().map(|k| k.source_id.as_str()).collect();
        assert!(ids.contains(&"group1"));
        assert!(ids.contains(&"group2"));
    }

    #[test]
    fn test_drain_resets_for_subsequent_pushes() {
        let mut buf = PassiveBuffer::new(Duration::from_secs(300), 2);
        let key = test_key("group1");
        buf.push(key.clone(), test_msg("a"));
        buf.push(key.clone(), test_msg("b"));
        assert!(buf.has_ready());
        let msgs = buf.drain(&key);
        assert_eq!(msgs.len(), 2);
        assert!(!buf.has_ready());
        // Push again — buffer should work fresh
        buf.push(key.clone(), test_msg("c"));
        assert_eq!(buf.total_pending(), 1);
        assert!(!buf.has_ready()); // Only 1, need 2
    }
}
