//! Types for ICS subscription tool requests and responses.

use serde::{Deserialize, Serialize};

/// Input parameters for the ICS subscription tool.
#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum IcsAction {
    /// Fetch events from an ICS subscription URL.
    FetchEvents {
        /// Direct ICS subscription URL. Either this or subscription_name is required.
        #[serde(default)]
        url: Option<String>,
        /// Name of a subscription from ICS_SUBSCRIPTIONS workspace config.
        /// Looked up to get the URL.
        #[serde(default)]
        subscription_name: Option<String>,
        /// RFC3339 start of time range filter (optional).
        #[serde(default)]
        time_min: Option<String>,
        /// RFC3339 end of time range filter (optional).
        #[serde(default)]
        time_max: Option<String>,
    },

    /// List configured ICS subscriptions from workspace.
    ListSubscriptions,
}

/// A single subscription entry from workspace config.
#[derive(Debug, Serialize, Deserialize)]
pub struct Subscription {
    /// Display name for the subscription.
    pub name: String,
    /// ICS subscription URL.
    pub url: String,
}

/// A calendar event from an ICS feed.
#[derive(Debug, Serialize)]
pub struct Event {
    /// Event UID.
    pub uid: String,
    /// Event summary/title.
    pub summary: String,
    /// Start time (RFC3339 or date string).
    pub start: String,
    /// End time (RFC3339 or date string).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
    /// Event location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Event description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Event status (CONFIRMED, TENTATIVE, CANCELLED).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Result from fetch_events.
#[derive(Debug, Serialize)]
pub struct FetchEventsResult {
    pub events: Vec<Event>,
    /// The subscription name (if resolved from config).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscription_name: Option<String>,
}

/// Result from list_subscriptions.
#[derive(Debug, Serialize)]
pub struct ListSubscriptionsResult {
    pub subscriptions: Vec<Subscription>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_fetch_events_with_url() {
        let json = r#"{"action": "fetch_events", "url": "https://example.com/cal.ics"}"#;
        let action: IcsAction = serde_json::from_str(json).unwrap();
        match action {
            IcsAction::FetchEvents { url, subscription_name, time_min, time_max } => {
                assert_eq!(url.as_deref(), Some("https://example.com/cal.ics"));
                assert!(subscription_name.is_none());
                assert!(time_min.is_none());
                assert!(time_max.is_none());
            }
            _ => panic!("Expected FetchEvents"),
        }
    }

    #[test]
    fn test_deserialize_fetch_events_with_subscription_name() {
        let json = r#"{"action": "fetch_events", "subscription_name": "Work"}"#;
        let action: IcsAction = serde_json::from_str(json).unwrap();
        match action {
            IcsAction::FetchEvents { url, subscription_name, .. } => {
                assert!(url.is_none());
                assert_eq!(subscription_name.as_deref(), Some("Work"));
            }
            _ => panic!("Expected FetchEvents"),
        }
    }

    #[test]
    fn test_deserialize_fetch_events_with_time_range() {
        let json = r#"{
            "action": "fetch_events",
            "url": "https://example.com/cal.ics",
            "time_min": "2026-03-08T00:00:00Z",
            "time_max": "2026-03-15T23:59:59Z"
        }"#;
        let action: IcsAction = serde_json::from_str(json).unwrap();
        match action {
            IcsAction::FetchEvents { time_min, time_max, .. } => {
                assert_eq!(time_min.as_deref(), Some("2026-03-08T00:00:00Z"));
                assert_eq!(time_max.as_deref(), Some("2026-03-15T23:59:59Z"));
            }
            _ => panic!("Expected FetchEvents"),
        }
    }

    #[test]
    fn test_deserialize_fetch_events_no_url_or_name() {
        // Both url and subscription_name missing — valid JSON but should fail at execute time
        let json = r#"{"action": "fetch_events"}"#;
        let action: IcsAction = serde_json::from_str(json).unwrap();
        match action {
            IcsAction::FetchEvents { url, subscription_name, .. } => {
                assert!(url.is_none());
                assert!(subscription_name.is_none());
            }
            _ => panic!("Expected FetchEvents"),
        }
    }

    #[test]
    fn test_deserialize_list_subscriptions() {
        let json = r#"{"action": "list_subscriptions"}"#;
        let action: IcsAction = serde_json::from_str(json).unwrap();
        assert!(matches!(action, IcsAction::ListSubscriptions));
    }

    #[test]
    fn test_deserialize_invalid_action() {
        let json = r#"{"action": "delete_everything"}"#;
        let result: Result<IcsAction, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_missing_action() {
        let json = r#"{"url": "https://example.com"}"#;
        let result: Result<IcsAction, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_subscription() {
        let json = r#"{"name": "Work", "url": "https://example.com/cal.ics"}"#;
        let sub: Subscription = serde_json::from_str(json).unwrap();
        assert_eq!(sub.name, "Work");
        assert_eq!(sub.url, "https://example.com/cal.ics");
    }

    #[test]
    fn test_deserialize_subscription_array() {
        let json = r#"[
            {"name": "Work", "url": "https://work.com/cal.ics"},
            {"name": "Personal", "url": "https://personal.com/cal.ics"}
        ]"#;
        let subs: Vec<Subscription> = serde_json::from_str(json).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].name, "Work");
        assert_eq!(subs[1].name, "Personal");
    }

    #[test]
    fn test_serialize_event_skips_none_fields() {
        let event = Event {
            uid: "test@example".to_string(),
            summary: "Test".to_string(),
            start: "2026-03-15T09:00:00Z".to_string(),
            end: None,
            location: None,
            description: None,
            status: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("end"));
        assert!(!json.contains("location"));
        assert!(!json.contains("description"));
        assert!(!json.contains("status"));
    }

    #[test]
    fn test_serialize_event_includes_present_fields() {
        let event = Event {
            uid: "test@example".to_string(),
            summary: "Team Sync".to_string(),
            start: "2026-03-15T09:00:00Z".to_string(),
            end: Some("2026-03-15T10:00:00Z".to_string()),
            location: Some("Room 3".to_string()),
            description: Some("Weekly sync".to_string()),
            status: Some("CONFIRMED".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"end\":\"2026-03-15T10:00:00Z\""));
        assert!(json.contains("\"location\":\"Room 3\""));
        assert!(json.contains("\"status\":\"CONFIRMED\""));
    }

    #[test]
    fn test_serialize_fetch_events_result() {
        let result = FetchEventsResult {
            events: vec![],
            subscription_name: Some("Work".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"events\":[]"));
        assert!(json.contains("\"subscription_name\":\"Work\""));
    }

    #[test]
    fn test_serialize_fetch_events_result_no_subscription() {
        let result = FetchEventsResult {
            events: vec![],
            subscription_name: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("subscription_name"));
    }

    #[test]
    fn test_serialize_list_subscriptions_result() {
        let result = ListSubscriptionsResult {
            subscriptions: vec![
                Subscription { name: "A".to_string(), url: "https://a.com".to_string() },
                Subscription { name: "B".to_string(), url: "https://b.com".to_string() },
            ],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["subscriptions"].as_array().unwrap().len(), 2);
    }
}
