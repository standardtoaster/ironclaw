use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    tools: Vec<CachedTool>,
    cached_at: u64,
    ttl_secs: u64,
}

pub struct ServiceCache {
    dir: PathBuf,
}

impl ServiceCache {
    pub fn new(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    pub fn get(&self, service_name: &str) -> Option<Vec<CachedTool>> {
        let path = self.dir.join(format!("{}.json", service_name));
        let contents = std::fs::read_to_string(&path).ok()?;
        let entry: CacheEntry = serde_json::from_str(&contents).ok()?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now - entry.cached_at >= entry.ttl_secs {
            return None;
        }
        Some(entry.tools)
    }

    pub fn put(&self, service_name: &str, tools: &[CachedTool], ttl_secs: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let entry = CacheEntry {
            tools: tools.to_vec(),
            cached_at: now,
            ttl_secs,
        };
        let path = self.dir.join(format!("{}.json", service_name));
        if let Ok(json) = serde_json::to_string_pretty(&entry) {
            std::fs::write(&path, json).ok();
        }
    }

    /// Return cached tools regardless of TTL (for error fallback).
    pub fn get_stale(&self, service_name: &str) -> Option<Vec<CachedTool>> {
        let path = self.dir.join(format!("{}.json", service_name));
        let contents = std::fs::read_to_string(&path).ok()?;
        let entry: CacheEntry = serde_json::from_str(&contents).ok()?;
        Some(entry.tools)
    }

    pub fn invalidate(&self, service_name: &str) {
        let path = self.dir.join(format!("{}.json", service_name));
        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_miss_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools = vec![CachedTool {
            name: "ha_turn_on".to_string(),
            description: "Turn on entity".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        cache.put("home_assistant", &tools, 3600);
        let result = cache.get("home_assistant");
        assert!(result.is_some());
        let cached = result.unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].name, "ha_turn_on");
    }

    #[test]
    fn test_cache_expired_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools = vec![CachedTool {
            name: "ha_turn_on".to_string(),
            description: "Turn on entity".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        // TTL of 0 seconds -- immediately expired
        cache.put("home_assistant", &tools, 0);
        assert!(cache.get("home_assistant").is_none());
    }

    #[test]
    fn test_get_stale_returns_expired_entries() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools = vec![CachedTool {
            name: "ha_turn_on".to_string(),
            description: "Turn on entity".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        // TTL of 0 seconds -- immediately expired
        cache.put("home_assistant", &tools, 0);
        // get() returns None (expired), but get_stale() returns the data
        assert!(cache.get("home_assistant").is_none());
        let stale = cache.get_stale("home_assistant");
        assert!(stale.is_some());
        let stale = stale.unwrap();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].name, "ha_turn_on");
    }

    #[test]
    fn test_get_stale_returns_none_for_missing() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        assert!(cache.get_stale("nonexistent").is_none());
    }

    #[test]
    fn test_invalidate_removes_cache_entry() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools = vec![CachedTool {
            name: "turn_on".to_string(),
            description: "Turn on".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        cache.put("ha", &tools, 3600);
        assert!(cache.get("ha").is_some());
        cache.invalidate("ha");
        assert!(cache.get("ha").is_none());
        assert!(cache.get_stale("ha").is_none());
    }

    #[test]
    fn test_invalidate_nonexistent_is_noop() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        cache.invalidate("nonexistent");
    }

    #[test]
    fn test_corrupted_cache_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("broken.json"), "not valid json").unwrap();
        assert!(cache.get("broken").is_none());
        assert!(cache.get_stale("broken").is_none());
    }

    #[test]
    fn test_multiple_services_cached_independently() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools_ha = vec![CachedTool {
            name: "turn_on".to_string(),
            description: "Turn on".to_string(),
            input_schema: serde_json::json!({}),
        }];
        let tools_media = vec![CachedTool {
            name: "play".to_string(),
            description: "Play media".to_string(),
            input_schema: serde_json::json!({}),
        }];
        cache.put("ha", &tools_ha, 3600);
        cache.put("media", &tools_media, 3600);

        let ha = cache.get("ha").unwrap();
        assert_eq!(ha.len(), 1);
        assert_eq!(ha[0].name, "turn_on");

        let media = cache.get("media").unwrap();
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].name, "play");

        cache.invalidate("ha");
        assert!(cache.get("ha").is_none());
        assert!(cache.get("media").is_some());
    }

    #[test]
    fn test_put_overwrites_existing_cache() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools_v1 = vec![CachedTool {
            name: "v1_tool".to_string(),
            description: "Old".to_string(),
            input_schema: serde_json::json!({}),
        }];
        let tools_v2 = vec![
            CachedTool {
                name: "v2_tool_a".to_string(),
                description: "New A".to_string(),
                input_schema: serde_json::json!({}),
            },
            CachedTool {
                name: "v2_tool_b".to_string(),
                description: "New B".to_string(),
                input_schema: serde_json::json!({}),
            },
        ];
        cache.put("svc", &tools_v1, 3600);
        assert_eq!(cache.get("svc").unwrap().len(), 1);

        cache.put("svc", &tools_v2, 3600);
        let result = cache.get("svc").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "v2_tool_a");
    }

    #[test]
    fn test_empty_tools_list_cached() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        cache.put("empty_svc", &[], 3600);
        let result = cache.get("empty_svc");
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_service_name_with_special_chars() {
        let dir = TempDir::new().unwrap();
        let cache = ServiceCache::new(dir.path().to_path_buf());
        let tools = vec![CachedTool {
            name: "tool".to_string(),
            description: "test".to_string(),
            input_schema: serde_json::json!({}),
        }];
        cache.put("my-service", &tools, 3600);
        assert!(cache.get("my-service").is_some());
        cache.put("my_service_v2", &tools, 3600);
        assert!(cache.get("my_service_v2").is_some());
    }
}
