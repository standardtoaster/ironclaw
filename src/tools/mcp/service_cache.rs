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
}
