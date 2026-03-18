//! Named tier map for model escalation.
//!
//! `TierMap` holds an ordered list of named tiers, each backed by an `LlmProvider`.
//! The agent loop uses this to swap providers mid-thread when `escalate` or
//! `de_escalate` tools are called.

use std::sync::{Arc, RwLock};

use crate::llm::provider::LlmProvider;

/// A single tier entry: a name, a provider, and whether it's the default.
pub struct TierEntry {
    /// Human-readable tier name (e.g., "local", "claude").
    pub name: String,
    /// Whether this tier is the default (used on thread start and after de-escalation).
    pub is_default: bool,
    /// The LLM provider backing this tier.
    pub provider: Arc<dyn LlmProvider>,
}

/// Named tier map with escalation/de-escalation.
///
/// Tiers are ordered by index (cheapest first). `escalate_to(None)` moves to
/// the next tier up; `de_escalate()` always reverts to the default tier.
///
/// Debug is manually implemented because `TierEntry` contains `Arc<dyn LlmProvider>`
/// which doesn't impl Debug.
pub struct TierMap {
    tiers: Vec<TierEntry>,
    current_index: RwLock<usize>,
    default_index: usize,
}

impl TierMap {
    /// Create a new TierMap from a list of tier entries.
    ///
    /// Exactly one entry must have `is_default = true`. Returns an error if
    /// no default is defined or if the entries list is empty.
    pub fn new(tiers: Vec<TierEntry>) -> Result<Self, String> {
        if tiers.is_empty() {
            return Err("At least one tier is required".into());
        }
        let default_index = tiers
            .iter()
            .position(|t| t.is_default)
            .ok_or_else(|| "No default tier defined".to_string())?;
        Ok(Self {
            tiers,
            current_index: RwLock::new(default_index),
            default_index,
        })
    }

    /// The name of the default tier.
    pub fn default_tier(&self) -> &str {
        &self.tiers[self.default_index].name
    }

    /// The name of the currently active tier.
    pub fn current_tier(&self) -> String {
        let idx = *self.current_index.read().unwrap_or_else(|e| e.into_inner());
        self.tiers[idx].name.clone()
    }

    /// The provider for the currently active tier.
    pub fn current_provider(&self) -> Arc<dyn LlmProvider> {
        let idx = *self.current_index.read().unwrap_or_else(|e| e.into_inner());
        self.tiers[idx].provider.clone()
    }

    /// Escalate to a specific tier (by name) or the next tier up (if `None`).
    ///
    /// Returns the name of the new active tier, or an error if the target
    /// tier doesn't exist or we're already at the highest tier.
    pub fn escalate_to(&self, target: Option<&str>) -> Result<String, String> {
        let mut idx = self.current_index.write().unwrap_or_else(|e| e.into_inner());
        match target {
            Some(name) => {
                let target_idx = self
                    .tiers
                    .iter()
                    .position(|t| t.name == name)
                    .ok_or_else(|| format!("Unknown tier: {}", name))?;
                *idx = target_idx;
                Ok(self.tiers[target_idx].name.clone())
            }
            None => {
                if *idx + 1 >= self.tiers.len() {
                    return Err("Already at highest tier".into());
                }
                *idx += 1;
                Ok(self.tiers[*idx].name.clone())
            }
        }
    }

    /// De-escalate to the default tier.
    ///
    /// Returns the name of the default tier. No-op if already on default.
    pub fn de_escalate(&self) -> String {
        let mut idx = self.current_index.write().unwrap_or_else(|e| e.into_inner());
        *idx = self.default_index;
        self.tiers[self.default_index].name.clone()
    }

    /// Number of configured tiers.
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// List all tier names in order.
    pub fn tier_names(&self) -> Vec<&str> {
        self.tiers.iter().map(|t| t.name.as_str()).collect()
    }
}

impl std::fmt::Debug for TierMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let idx = *self.current_index.read().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("TierMap")
            .field("tier_count", &self.tiers.len())
            .field("default_index", &self.default_index)
            .field("current_index", &idx)
            .field(
                "tier_names",
                &self
                    .tiers
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rust_decimal::Decimal;

    /// Minimal mock provider for testing.
    struct MockProvider {
        name: String,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn model_name(&self) -> &str {
            &self.name
        }

        fn cost_per_token(&self) -> (Decimal, Decimal) {
            (Decimal::ZERO, Decimal::ZERO)
        }

        async fn complete(
            &self,
            _request: crate::llm::CompletionRequest,
        ) -> Result<crate::llm::CompletionResponse, crate::llm::LlmError> {
            unimplemented!("mock")
        }

        async fn complete_with_tools(
            &self,
            _request: crate::llm::ToolCompletionRequest,
        ) -> Result<crate::llm::ToolCompletionResponse, crate::llm::LlmError> {
            unimplemented!("mock")
        }
    }

    fn mock_provider(name: &str) -> Arc<dyn LlmProvider> {
        Arc::new(MockProvider {
            name: name.to_string(),
        })
    }

    fn two_tier_map() -> TierMap {
        TierMap::new(vec![
            TierEntry {
                name: "local".into(),
                is_default: true,
                provider: mock_provider("local"),
            },
            TierEntry {
                name: "claude".into(),
                is_default: false,
                provider: mock_provider("claude"),
            },
        ])
        .unwrap()
    }

    #[test]
    fn test_tier_map_default_tier() {
        let map = two_tier_map();
        assert_eq!(map.default_tier(), "local");
        assert_eq!(map.current_tier(), "local");
    }

    #[test]
    fn test_tier_map_escalate_by_name() {
        let map = two_tier_map();
        let result = map.escalate_to(Some("claude"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "claude");
        assert_eq!(map.current_tier(), "claude");
    }

    #[test]
    fn test_tier_map_escalate_next() {
        let map = two_tier_map();
        let result = map.escalate_to(None);
        assert!(result.is_ok());
        assert_eq!(map.current_tier(), "claude");
    }

    #[test]
    fn test_tier_map_escalate_already_highest() {
        let map = two_tier_map();
        map.escalate_to(Some("claude")).unwrap();
        let result = map.escalate_to(None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("highest"));
    }

    #[test]
    fn test_tier_map_deescalate() {
        let map = two_tier_map();
        map.escalate_to(Some("claude")).unwrap();
        assert_eq!(map.current_tier(), "claude");
        map.de_escalate();
        assert_eq!(map.current_tier(), "local");
    }

    #[test]
    fn test_tier_map_deescalate_already_default() {
        let map = TierMap::new(vec![TierEntry {
            name: "local".into(),
            is_default: true,
            provider: mock_provider("local"),
        }])
        .unwrap();
        map.de_escalate(); // no-op, should not panic
        assert_eq!(map.current_tier(), "local");
    }

    #[test]
    fn test_tier_map_current_provider() {
        let map = two_tier_map();
        assert_eq!(map.current_provider().model_name(), "local");
        map.escalate_to(Some("claude")).unwrap();
        assert_eq!(map.current_provider().model_name(), "claude");
    }

    #[test]
    fn test_tier_map_requires_default() {
        let result = TierMap::new(vec![TierEntry {
            name: "local".into(),
            is_default: false,
            provider: mock_provider("local"),
        }]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No default"));
    }

    #[test]
    fn test_tier_map_empty_tiers_rejected() {
        let result = TierMap::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tier_map_escalate_unknown_tier() {
        let map = two_tier_map();
        let result = map.escalate_to(Some("nonexistent"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown tier"));
    }

    #[test]
    fn test_tier_map_tier_names() {
        let map = two_tier_map();
        assert_eq!(map.tier_names(), vec!["local", "claude"]);
    }

    /// Full escalation flow E2E test:
    /// 1. Start on "local" tier (default)
    /// 2. Escalate to "claude"
    /// 3. Verify provider swapped
    /// 4. Verify sticky (stays on claude)
    /// 5. De-escalate
    /// 6. Verify reverted to "local"
    /// 7. Verify escalate_to(None) works (next tier up)
    #[test]
    fn test_full_escalation_flow_e2e() {
        let map = TierMap::new(vec![
            TierEntry {
                name: "local".into(),
                is_default: true,
                provider: mock_provider("local-model"),
            },
            TierEntry {
                name: "claude".into(),
                is_default: false,
                provider: mock_provider("claude-sonnet"),
            },
            TierEntry {
                name: "opus".into(),
                is_default: false,
                provider: mock_provider("claude-opus"),
            },
        ])
        .unwrap();

        // 1. Starts on default tier
        assert_eq!(map.current_tier(), "local");
        assert_eq!(map.current_provider().model_name(), "local-model");

        // 2. Escalate by name to "claude"
        let new_tier = map.escalate_to(Some("claude")).unwrap();
        assert_eq!(new_tier, "claude");

        // 3. Provider swapped
        assert_eq!(map.current_provider().model_name(), "claude-sonnet");

        // 4. Sticky: stays on claude
        assert_eq!(map.current_tier(), "claude");
        assert_eq!(map.current_provider().model_name(), "claude-sonnet");

        // 4b. Escalate further to opus
        let new_tier = map.escalate_to(Some("opus")).unwrap();
        assert_eq!(new_tier, "opus");
        assert_eq!(map.current_provider().model_name(), "claude-opus");

        // 5. De-escalate reverts to default, not one step down
        let reverted = map.de_escalate();
        assert_eq!(reverted, "local");

        // 6. Back on local
        assert_eq!(map.current_tier(), "local");
        assert_eq!(map.current_provider().model_name(), "local-model");

        // 7. Escalate with None = next tier up
        let next = map.escalate_to(None).unwrap();
        assert_eq!(next, "claude");
        assert_eq!(map.current_provider().model_name(), "claude-sonnet");

        // 8. Escalate with None again = next tier up (opus)
        let next = map.escalate_to(None).unwrap();
        assert_eq!(next, "opus");
        assert_eq!(map.current_provider().model_name(), "claude-opus");

        // 9. At highest, escalate None fails
        let err = map.escalate_to(None);
        assert!(err.is_err());

        // 10. But can still de-escalate
        map.de_escalate();
        assert_eq!(map.current_tier(), "local");
    }

    /// Test that TierMap is thread-safe (can be shared across async tasks).
    #[test]
    fn test_tier_map_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TierMap>();
    }
}
