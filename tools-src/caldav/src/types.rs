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
