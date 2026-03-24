//! Types for CalDAV tool requests and responses.

use serde::{Deserialize, Serialize};

/// Input parameters for the CalDAV tool.
#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum CalDavAction {
    /// Discover available calendars.
    ListCalendars {
        /// CalDAV server base URL (e.g., "https://caldav.icloud.com").
        /// If not provided, reads from CALDAV_CONFIG workspace file.
        #[serde(default)]
        base_url: Option<String>,
    },

    /// List events in a time range.
    ListEvents {
        /// Full URL to the calendar collection.
        /// Required unless calendar_path is provided with base_url.
        #[serde(default)]
        calendar_url: Option<String>,
        /// RFC3339 start of time range (e.g., "2026-03-08T00:00:00Z").
        time_min: String,
        /// RFC3339 end of time range (e.g., "2026-03-15T23:59:59Z").
        time_max: String,
    },

    /// Get a specific event by UID.
    GetEvent {
        /// Full URL to the calendar collection.
        calendar_url: String,
        /// The event UID.
        uid: String,
    },

    /// Query free/busy intervals for a time range.
    FreeBusy {
        /// Full URL to the calendar collection.
        calendar_url: String,
        /// RFC3339 start of time range.
        time_min: String,
        /// RFC3339 end of time range.
        time_max: String,
    },

    /// Create a new calendar event.
    CreateEvent {
        /// Full URL to the calendar collection.
        calendar_url: String,
        /// Event summary/title.
        summary: String,
        /// Start datetime (RFC3339, e.g., "2026-03-15T09:00:00Z"). For timed events.
        #[serde(default)]
        start_datetime: Option<String>,
        /// End datetime (RFC3339). For timed events.
        #[serde(default)]
        end_datetime: Option<String>,
        /// Start date (YYYY-MM-DD). For all-day events.
        #[serde(default)]
        start_date: Option<String>,
        /// End date (YYYY-MM-DD). For all-day events (exclusive, so next day).
        #[serde(default)]
        end_date: Option<String>,
        /// Event location.
        #[serde(default)]
        location: Option<String>,
        /// Event description.
        #[serde(default)]
        description: Option<String>,
        /// IANA timezone (e.g., "America/New_York"). For non-UTC timed events.
        #[serde(default)]
        timezone: Option<String>,
    },

    /// Update an existing calendar event.
    UpdateEvent {
        /// Full URL to the calendar collection.
        calendar_url: String,
        /// The event UID to update.
        uid: String,
        /// New summary/title.
        #[serde(default)]
        summary: Option<String>,
        /// New start datetime (RFC3339). For timed events.
        #[serde(default)]
        start_datetime: Option<String>,
        /// New end datetime (RFC3339). For timed events.
        #[serde(default)]
        end_datetime: Option<String>,
        /// New start date (YYYY-MM-DD). For all-day events.
        #[serde(default)]
        start_date: Option<String>,
        /// New end date (YYYY-MM-DD). For all-day events.
        #[serde(default)]
        end_date: Option<String>,
        /// New location.
        #[serde(default)]
        location: Option<String>,
        /// New description.
        #[serde(default)]
        description: Option<String>,
        /// IANA timezone for non-UTC timed events.
        #[serde(default)]
        timezone: Option<String>,
    },

    /// Delete a calendar event.
    DeleteEvent {
        /// Full URL to the calendar collection.
        calendar_url: String,
        /// The event UID to delete.
        uid: String,
    },
}

/// CalDAV configuration read from workspace.
#[derive(Debug, Deserialize)]
pub struct CalDavConfig {
    /// CalDAV server base URL.
    pub base_url: String,
    /// Username for Basic Auth (email address).
    /// Read by the host's credential injector via capabilities.json;
    /// not directly used in WASM code.
    #[allow(dead_code)]
    pub username: String,
    /// Optional: default calendar URL.
    #[serde(default)]
    pub default_calendar_url: Option<String>,
}

/// A discovered calendar.
#[derive(Debug, Serialize)]
pub struct Calendar {
    /// Calendar display name.
    pub display_name: String,
    /// Full URL to the calendar collection.
    pub url: String,
    /// Calendar color (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Calendar description (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A calendar event.
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
    /// URL to the .ics resource on the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub href: Option<String>,
}

/// A busy interval (privacy-preserving: no event details, only times).
#[derive(Debug, Serialize)]
pub struct BusyInterval {
    /// Start of busy period (RFC3339).
    pub start: String,
    /// End of busy period (RFC3339).
    pub end: String,
}

/// Result from list_calendars.
#[derive(Debug, Serialize)]
pub struct ListCalendarsResult {
    pub calendars: Vec<Calendar>,
}

/// Result from list_events.
#[derive(Debug, Serialize)]
pub struct ListEventsResult {
    pub events: Vec<Event>,
}

/// Result from get_event.
#[derive(Debug, Serialize)]
pub struct GetEventResult {
    pub event: Event,
}

/// Result from free_busy.
#[derive(Debug, Serialize)]
pub struct FreeBusyResult {
    pub busy: Vec<BusyInterval>,
}

/// Result from create_event.
#[derive(Debug, Serialize)]
pub struct CreateEventResult {
    pub event: Event,
}

/// Result from update_event.
#[derive(Debug, Serialize)]
pub struct UpdateEventResult {
    pub event: Event,
}

/// Result from delete_event.
#[derive(Debug, Serialize)]
pub struct DeleteEventResult {
    pub uid: String,
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_list_calendars_action() {
        let json = r#"{"action": "list_calendars"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        assert!(matches!(action, CalDavAction::ListCalendars { base_url: None }));
    }

    #[test]
    fn test_deserialize_list_calendars_with_base_url() {
        let json = r#"{"action": "list_calendars", "base_url": "https://caldav.icloud.com"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::ListCalendars { base_url } => {
                assert_eq!(base_url.as_deref(), Some("https://caldav.icloud.com"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_list_events_action() {
        let json = r#"{"action": "list_events", "calendar_url": "https://cal.example.com/user/cal/", "time_min": "2026-03-01T00:00:00Z", "time_max": "2026-03-31T23:59:59Z"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::ListEvents { calendar_url, time_min, time_max } => {
                assert_eq!(calendar_url.as_deref(), Some("https://cal.example.com/user/cal/"));
                assert_eq!(time_min, "2026-03-01T00:00:00Z");
                assert_eq!(time_max, "2026-03-31T23:59:59Z");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_list_events_without_calendar_url() {
        let json = r#"{"action": "list_events", "time_min": "2026-03-01T00:00:00Z", "time_max": "2026-03-31T23:59:59Z"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::ListEvents { calendar_url, .. } => {
                assert!(calendar_url.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_create_event_minimal() {
        let json = r#"{"action": "create_event", "calendar_url": "https://cal.example.com/user/cal/", "summary": "Test Event", "start_datetime": "2026-03-15T09:00:00Z"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::CreateEvent { calendar_url, summary, start_datetime, end_datetime, location, description, timezone, .. } => {
                assert_eq!(calendar_url, "https://cal.example.com/user/cal/");
                assert_eq!(summary, "Test Event");
                assert_eq!(start_datetime.as_deref(), Some("2026-03-15T09:00:00Z"));
                assert!(end_datetime.is_none());
                assert!(location.is_none());
                assert!(description.is_none());
                assert!(timezone.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_create_event_allday() {
        let json = r#"{"action": "create_event", "calendar_url": "https://cal.example.com/user/cal/", "summary": "Day Off", "start_date": "2026-03-20", "end_date": "2026-03-21"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::CreateEvent { start_date, end_date, start_datetime, .. } => {
                assert_eq!(start_date.as_deref(), Some("2026-03-20"));
                assert_eq!(end_date.as_deref(), Some("2026-03-21"));
                assert!(start_datetime.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_update_event() {
        let json = r#"{"action": "update_event", "calendar_url": "https://cal.example.com/cal/", "uid": "abc123", "summary": "Updated Title"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::UpdateEvent { uid, summary, .. } => {
                assert_eq!(uid, "abc123");
                assert_eq!(summary.as_deref(), Some("Updated Title"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_delete_event() {
        let json = r#"{"action": "delete_event", "calendar_url": "https://cal.example.com/cal/", "uid": "abc123"}"#;
        let action: CalDavAction = serde_json::from_str(json).unwrap();
        match action {
            CalDavAction::DeleteEvent { uid, .. } => {
                assert_eq!(uid, "abc123");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deserialize_unknown_action_fails() {
        let json = r#"{"action": "unknown_action"}"#;
        let result = serde_json::from_str::<CalDavAction>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_missing_action_fails() {
        let json = r#"{"calendar_url": "https://cal.example.com/cal/"}"#;
        let result = serde_json::from_str::<CalDavAction>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_caldav_config_deserialization() {
        let json = r#"{"base_url": "https://caldav.icloud.com", "username": "user@example.com"}"#;
        let config: CalDavConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.base_url, "https://caldav.icloud.com");
        assert_eq!(config.username, "user@example.com");
        assert!(config.default_calendar_url.is_none());
    }

    #[test]
    fn test_caldav_config_with_default_calendar() {
        let json = r#"{"base_url": "https://caldav.icloud.com", "username": "user@example.com", "default_calendar_url": "https://caldav.icloud.com/123/cal/"}"#;
        let config: CalDavConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.default_calendar_url.as_deref(), Some("https://caldav.icloud.com/123/cal/"));
    }

    #[test]
    fn test_event_serialization_skips_none_fields() {
        let event = Event {
            uid: "test@uid".to_string(),
            summary: "Test".to_string(),
            start: "2026-03-15T09:00:00Z".to_string(),
            end: None,
            location: None,
            description: None,
            status: None,
            href: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("location"));
        assert!(!json.contains("description"));
        assert!(!json.contains("status"));
        assert!(!json.contains("href"));
        assert!(json.contains("uid"));
        assert!(json.contains("summary"));
        assert!(json.contains("start"));
    }

    #[test]
    fn test_busy_interval_serialization() {
        let busy = BusyInterval {
            start: "2026-03-15T09:00:00Z".to_string(),
            end: "2026-03-15T10:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&busy).unwrap();
        assert!(json.contains("2026-03-15T09:00:00Z"));
        assert!(json.contains("2026-03-15T10:00:00Z"));
    }

    #[test]
    fn test_delete_event_result_serialization() {
        let result = DeleteEventResult {
            uid: "abc123".to_string(),
            deleted: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"deleted\":true"));
    }
}
