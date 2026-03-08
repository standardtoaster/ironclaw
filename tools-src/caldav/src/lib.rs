//! CalDAV WASM Tool for IronClaw.
//!
//! Provides CalDAV calendar integration for querying and managing calendars
//! and events via the standard CalDAV protocol (RFC 4791).
//!
//! # Capabilities Required
//!
//! - HTTP: CalDAV servers (PROPFIND, REPORT, GET, PUT, DELETE)
//! - Secrets: `caldav_password` (app-specific password, injected as Basic Auth)
//! - Workspace: `CALDAV_CONFIG` (JSON with base_url, username)
//!
//! # Supported Actions
//!
//! - `list_calendars`: Discover available calendars on the server
//! - `list_events`: List events in a time range
//! - `get_event`: Get a specific event by UID
//! - `free_busy`: Query free/busy intervals for a time range
//! - `create_event`: Create a new calendar event
//! - `update_event`: Update an existing calendar event
//! - `delete_event`: Delete a calendar event
//!
//! # Setup
//!
//! 1. Store CalDAV credentials via `ironclaw tool auth caldav`
//! 2. Write a CALDAV_CONFIG workspace file:
//!    ```json
//!    {"base_url": "https://caldav.icloud.com", "username": "your@email.com"}
//!    ```
//!
//! # Example Usage
//!
//! ```json
//! {"action": "list_events", "calendar_url": "https://caldav.icloud.com/123/cal/", "time_min": "2026-03-08T00:00:00Z", "time_max": "2026-03-15T23:59:59Z"}
//! ```

mod api;
mod ical;
mod types;

use types::CalDavAction;

wit_bindgen::generate!({
    world: "sandboxed-tool",
    path: "../../wit/tool.wit",
});

struct CalDavTool;

impl exports::near::agent::tool::Guest for CalDavTool {
    fn execute(req: exports::near::agent::tool::Request) -> exports::near::agent::tool::Response {
        match execute_inner(&req.params) {
            Ok(result) => exports::near::agent::tool::Response {
                output: Some(result),
                error: None,
            },
            Err(e) => exports::near::agent::tool::Response {
                output: None,
                error: Some(e),
            },
        }
    }

    fn schema() -> String {
        r#"{
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_calendars", "list_events", "get_event", "free_busy", "create_event", "update_event", "delete_event"],
                    "description": "The calendar operation to perform"
                },
                "base_url": {
                    "type": "string",
                    "description": "CalDAV server base URL (e.g., 'https://caldav.icloud.com'). If omitted, reads from CALDAV_CONFIG workspace file. Used by: list_calendars"
                },
                "calendar_url": {
                    "type": "string",
                    "description": "Full URL to the calendar collection. Required for: list_events, get_event, free_busy, create_event, update_event, delete_event"
                },
                "time_min": {
                    "type": "string",
                    "description": "Start of time range (RFC3339, e.g., '2026-03-08T00:00:00Z'). Required for: list_events, free_busy"
                },
                "time_max": {
                    "type": "string",
                    "description": "End of time range (RFC3339, e.g., '2026-03-15T23:59:59Z'). Required for: list_events, free_busy"
                },
                "uid": {
                    "type": "string",
                    "description": "Event UID. Required for: get_event, update_event, delete_event"
                },
                "summary": {
                    "type": "string",
                    "description": "Event title/summary. Required for: create_event. Optional for: update_event"
                },
                "start_datetime": {
                    "type": "string",
                    "description": "Start datetime (RFC3339, e.g., '2026-03-15T09:00:00Z'). For timed events. Used by: create_event, update_event"
                },
                "end_datetime": {
                    "type": "string",
                    "description": "End datetime (RFC3339). For timed events. Used by: create_event, update_event"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). For all-day events. Used by: create_event, update_event"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD, exclusive). For all-day events. Used by: create_event, update_event"
                },
                "location": {
                    "type": "string",
                    "description": "Event location. Used by: create_event, update_event"
                },
                "description": {
                    "type": "string",
                    "description": "Event description. Used by: create_event, update_event"
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone (e.g., 'America/New_York'). For non-UTC timed events. Used by: create_event, update_event"
                }
            }
        }"#
        .to_string()
    }

    fn description() -> String {
        "CalDAV calendar integration. Supports iCloud, Fastmail, and Google CalDAV. \
         Actions: list_calendars (discover calendars), list_events (events in time range), \
         get_event (by UID), free_busy (availability), create_event (new event), \
         update_event (modify existing), delete_event (remove). For timed events use \
         start_datetime/end_datetime (RFC3339); for all-day use start_date/end_date \
         (YYYY-MM-DD). Requires CALDAV_CONFIG workspace file and caldav_password secret."
            .to_string()
    }
}

fn execute_inner(params: &str) -> Result<String, String> {
    if !crate::near::agent::host::secret_exists("caldav_password") {
        return Err(
            "CalDAV password not configured. Run `ironclaw tool auth caldav` to set up \
             authentication, or set the CALDAV_PASSWORD environment variable with an \
             app-specific password."
                .to_string(),
        );
    }

    let action: CalDavAction =
        serde_json::from_str(params).map_err(|e| format!("Invalid parameters: {}", e))?;

    crate::near::agent::host::log(
        crate::near::agent::host::LogLevel::Info,
        &format!("Executing CalDAV action: {:?}", action),
    );

    let result = match action {
        CalDavAction::ListCalendars { base_url } => {
            let url = match base_url {
                Some(u) => u,
                None => {
                    let config = api::read_config()?;
                    config.base_url
                }
            };
            let result = api::list_calendars(&url)?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::ListEvents {
            calendar_url,
            time_min,
            time_max,
        } => {
            let url = match calendar_url {
                Some(u) => u,
                None => {
                    let config = api::read_config()?;
                    config.default_calendar_url.ok_or_else(|| {
                        "calendar_url is required. Use list_calendars first to find it, \
                         or set default_calendar_url in CALDAV_CONFIG."
                            .to_string()
                    })?
                }
            };
            let result = api::list_events(&url, &time_min, &time_max)?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::GetEvent { calendar_url, uid } => {
            let result = api::get_event(&calendar_url, &uid)?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::FreeBusy {
            calendar_url,
            time_min,
            time_max,
        } => {
            let result = api::free_busy(&calendar_url, &time_min, &time_max)?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::CreateEvent {
            calendar_url,
            summary,
            start_datetime,
            end_datetime,
            start_date,
            end_date,
            location,
            description,
            timezone,
        } => {
            let result = api::create_event(
                &calendar_url,
                &summary,
                start_datetime.as_deref(),
                end_datetime.as_deref(),
                start_date.as_deref(),
                end_date.as_deref(),
                location.as_deref(),
                description.as_deref(),
                timezone.as_deref(),
            )?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::UpdateEvent {
            calendar_url,
            uid,
            summary,
            start_datetime,
            end_datetime,
            start_date,
            end_date,
            location,
            description,
            timezone,
        } => {
            let result = api::update_event(
                &calendar_url,
                &uid,
                summary.as_deref(),
                start_datetime.as_deref(),
                end_datetime.as_deref(),
                start_date.as_deref(),
                end_date.as_deref(),
                location.as_deref(),
                description.as_deref(),
                timezone.as_deref(),
            )?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        CalDavAction::DeleteEvent { calendar_url, uid } => {
            let result = api::delete_event(&calendar_url, &uid)?;
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }
    };

    Ok(result)
}

export!(CalDavTool);
