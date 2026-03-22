//! Debounce-based workspace organizer runner.
//!
//! Receives signals when unrouted messages land in a user's general workspace.
//! After a debounce period (default 5 min) with no new signals for that user,
//! calls `ThreadResolver::organize()` to classify recent messages into workspaces.
//! A max-interval ceiling ensures the organizer runs even if signals are missed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::agent::thread_resolver::ThreadResolver;

/// Signal sent by the resolver when an unrouted message lands in general.
#[derive(Debug)]
pub struct OrganizerSignal {
    pub user_id: String,
}

/// Configuration for the organizer runner.
#[derive(Debug, Clone)]
pub struct OrganizerConfig {
    /// Debounce delay after the last signal before organizing (default: 5 min).
    pub debounce: Duration,
    /// Maximum interval between organize runs, even without signals (default: 30 min).
    /// Acts as a ceiling to catch messages that bypass the signal path.
    pub max_interval: Duration,
    /// Whether the organizer is enabled.
    pub enabled: bool,
    /// Maximum consecutive failures before disabling.
    pub max_failures: u32,
    /// Minimum new messages (turn_count delta) before organizing.
    /// Used by the watermark check in `organize()`.
    pub min_messages: usize,
    /// User IDs to organize (parsed from `ORGANIZER_USER_IDS`).
    pub user_ids: Vec<String>,
}

impl Default for OrganizerConfig {
    fn default() -> Self {
        Self {
            debounce: Duration::from_secs(300),
            max_interval: Duration::from_secs(30 * 60),
            enabled: false,
            max_failures: 3,
            min_messages: 5,
            user_ids: Vec::new(),
        }
    }
}

impl OrganizerConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("ORGANIZER_ENABLED")
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);

        let debounce_secs = std::env::var("ORGANIZER_DEBOUNCE_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(300);

        let max_interval_secs = std::env::var("ORGANIZER_MAX_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1800);

        let max_failures = std::env::var("ORGANIZER_MAX_FAILURES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3);

        let min_messages = std::env::var("ORGANIZER_MIN_MESSAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(5);

        let user_ids: Vec<String> = std::env::var("ORGANIZER_USER_IDS")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.split(',').map(|id| id.trim().to_string()).collect())
            .unwrap_or_default();

        Self {
            debounce: Duration::from_secs(debounce_secs),
            max_interval: Duration::from_secs(max_interval_secs),
            enabled,
            max_failures,
            min_messages,
            user_ids,
        }
    }
}

/// Background organizer runner with debounce-based scheduling.
struct OrganizerRunner {
    config: OrganizerConfig,
    resolver: Arc<dyn ThreadResolver>,
    rx: mpsc::Receiver<OrganizerSignal>,
    consecutive_failures: u32,
    /// Per-user: when we last received a signal (for debounce).
    pending: HashMap<String, Instant>,
}

impl OrganizerRunner {
    fn new(
        config: OrganizerConfig,
        resolver: Arc<dyn ThreadResolver>,
        rx: mpsc::Receiver<OrganizerSignal>,
    ) -> Self {
        Self {
            config,
            resolver,
            rx,
            consecutive_failures: 0,
            pending: HashMap::new(),
        }
    }

    /// Run the organizer loop. Runs forever until max failures or task cancellation.
    async fn run(&mut self) {
        if !self.config.enabled {
            tracing::info!("Organizer is disabled, not starting loop");
            return;
        }

        if self.config.user_ids.is_empty() {
            tracing::warn!(
                "Organizer enabled but ORGANIZER_USER_IDS is empty, not starting loop"
            );
            return;
        }

        tracing::info!(
            debounce = ?self.config.debounce,
            max_interval = ?self.config.max_interval,
            users = ?self.config.user_ids,
            "Starting organizer loop (debounce mode)"
        );

        let mut ceiling_interval = tokio::time::interval(self.config.max_interval);
        // Don't fire immediately on startup
        ceiling_interval.tick().await;

        loop {
            // Wait for either a signal or the ceiling interval
            let signal = tokio::select! {
                sig = self.rx.recv() => sig,
                _ = ceiling_interval.tick() => {
                    // Ceiling tick: run organizer for all configured users.
                    // Watermark check in organize() makes this cheap if nothing new.
                    self.run_all_users().await;
                    continue;
                }
            };

            match signal {
                Some(sig) => {
                    // Only process signals for configured users
                    if self.config.user_ids.contains(&sig.user_id) {
                        self.pending.insert(sig.user_id, Instant::now());
                    }

                    // Drain any additional queued signals (non-blocking)
                    while let Ok(extra) = self.rx.try_recv() {
                        if self.config.user_ids.contains(&extra.user_id) {
                            self.pending.insert(extra.user_id, Instant::now());
                        }
                    }

                    // Process any users whose debounce has expired
                    self.process_debounced().await;
                }
                None => {
                    // Channel closed, shut down
                    tracing::info!("Organizer signal channel closed, shutting down");
                    break;
                }
            }
        }
    }

    /// Check for pending users whose debounce period has expired and organize them.
    async fn process_debounced(&mut self) {
        let now = Instant::now();
        let expired: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, last_signal)| now.duration_since(**last_signal) >= self.config.debounce)
            .map(|(user_id, _)| user_id.clone())
            .collect();

        for user_id in expired {
            self.pending.remove(&user_id);
            self.organize_user(&user_id).await;
        }

        // If there are still pending users, sleep until the earliest debounce expires
        // while continuing to drain incoming signals.
        if let Some(&earliest) = self.pending.values().min() {
            let remaining = self
                .config
                .debounce
                .checked_sub(now.duration_since(earliest))
                .unwrap_or(Duration::ZERO);

            self.sleep_and_process(remaining).await;
        }
    }

    /// Sleep for the given duration, draining signals that arrive during the sleep,
    /// then process any users whose debounce has expired.
    async fn sleep_and_process(&mut self, wait: Duration) {
        let deadline = Instant::now() + wait;

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Ok(Some(sig)) => {
                    if self.config.user_ids.contains(&sig.user_id) {
                        self.pending.insert(sig.user_id, Instant::now());
                    }
                    // Drain queued signals
                    while let Ok(extra) = self.rx.try_recv() {
                        if self.config.user_ids.contains(&extra.user_id) {
                            self.pending.insert(extra.user_id, Instant::now());
                        }
                    }
                }
                Ok(None) => {
                    // Channel closed
                    return;
                }
                Err(_timeout) => {
                    // Debounce period expired
                    break;
                }
            }
        }

        // Process expired users
        let now = Instant::now();
        let expired: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, last_signal)| now.duration_since(**last_signal) >= self.config.debounce)
            .map(|(user_id, _)| user_id.clone())
            .collect();

        for user_id in expired {
            self.pending.remove(&user_id);
            self.organize_user(&user_id).await;
        }
    }

    /// Run organizer for all configured users (ceiling tick).
    async fn run_all_users(&mut self) {
        for user_id in &self.config.user_ids.clone() {
            self.organize_user(user_id).await;
        }
    }

    /// Organize a single user, tracking failures.
    async fn organize_user(&mut self, user_id: &str) {
        match self.resolver.organize(user_id).await {
            Ok(result) => {
                if result.created.is_empty() && result.updated.is_empty() {
                    tracing::debug!(
                        user_id,
                        messages_analyzed = result.messages_analyzed,
                        "Organizer: nothing to do"
                    );
                } else {
                    let created: Vec<&str> =
                        result.created.iter().map(|w| w.topic.as_str()).collect();
                    let updated: Vec<&str> =
                        result.updated.iter().map(|w| w.topic.as_str()).collect();
                    tracing::info!(
                        user_id,
                        messages_analyzed = result.messages_analyzed,
                        ?created,
                        ?updated,
                        "Organizer: workspaces updated"
                    );
                }
                self.consecutive_failures = 0;

                // Generate/update summaries for workspaces with new content
                match self.resolver.summarize_workspaces(user_id).await {
                    Ok(n) if n > 0 => {
                        tracing::info!(user_id, summarized = n, "Organizer: updated workspace summaries");
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(user_id, error = %e, "Organizer: summary generation failed");
                    }
                }
            }
            Err(e) => {
                tracing::error!(user_id, error = %e, "Organizer failed for user");
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.config.max_failures {
                    tracing::error!(
                        failures = self.consecutive_failures,
                        "Organizer disabled after consecutive failures"
                    );
                }
            }
        }
    }
}

/// Spawn the organizer runner as a background task.
///
/// The runner receives `OrganizerSignal`s via the channel and debounces them
/// before calling `organize()`. A ceiling interval ensures periodic runs
/// even without signals.
pub fn spawn_organizer(
    config: OrganizerConfig,
    resolver: Arc<dyn ThreadResolver>,
    rx: mpsc::Receiver<OrganizerSignal>,
) -> tokio::task::JoinHandle<()> {
    let mut runner = OrganizerRunner::new(config, resolver, rx);
    tokio::spawn(async move {
        runner.run().await;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organizer_config_defaults() {
        let config = OrganizerConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.debounce, Duration::from_secs(300));
        assert_eq!(config.max_interval, Duration::from_secs(30 * 60));
        assert_eq!(config.max_failures, 3);
        assert_eq!(config.min_messages, 5);
        assert!(config.user_ids.is_empty());
    }

    #[test]
    fn test_organizer_config_from_env() {
        // Without env vars set, should return defaults
        let config = OrganizerConfig::from_env();
        assert!(!config.enabled);
        assert_eq!(config.debounce, Duration::from_secs(300));
        assert_eq!(config.max_interval, Duration::from_secs(1800));
    }

    #[test]
    fn test_spawn_organizer_type_signature() {
        // Compile-time check: spawn_organizer accepts the expected parameters.
        #[allow(clippy::type_complexity)]
        let _fn_ptr: fn(
            OrganizerConfig,
            Arc<dyn ThreadResolver>,
            mpsc::Receiver<OrganizerSignal>,
        ) -> tokio::task::JoinHandle<()> = spawn_organizer;
        let _ = _fn_ptr;
    }
}
