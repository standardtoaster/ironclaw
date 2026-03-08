//! CalDAV WASM Tool for IronClaw.
//!
//! Provides read-only CalDAV calendar integration for querying calendars
//! and events via the standard CalDAV protocol (RFC 4791).
//!
//! # Capabilities Required
//!
//! - HTTP: CalDAV servers (PROPFIND, REPORT, GET)
//! - Secrets: `caldav_password` (app-specific password, injected as Basic Auth)
//! - Workspace: `CALDAV_CONFIG` (JSON with base_url, username)
//!
//! # Supported Actions
//!
//! - `list_calendars`: Discover available calendars on the server
//! - `list_events`: List events in a time range
//! - `get_event`: Get a specific event by UID
//! - `free_busy`: Query free/busy intervals for a time range
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
                    "enum": ["list_calendars", "list_events", "get_event", "free_busy"],
                    "description": "The calendar operation to perform"
                },
                "base_url": {
                    "type": "string",
                    "description": "CalDAV server base URL (e.g., 'https://caldav.icloud.com'). If omitted, reads from CALDAV_CONFIG workspace file. Used by: list_calendars"
                },
                "calendar_url": {
                    "type": "string",
                    "description": "Full URL to the calendar collection. Required for: list_events, get_event, free_busy"
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
                    "description": "Event UID. Required for: get_event"
                }
            }
        }"#
        .to_string()
    }

    fn description() -> String {
        "CalDAV calendar integration for reading calendars and events. Supports iCloud, \
         Fastmail, and Google CalDAV. Use list_calendars to discover available calendars, \
         list_events to see events in a time range, get_event for full event details, \
         and free_busy to check availability. Read-only (no create/update/delete). \
         Requires a CALDAV_CONFIG workspace file with base_url and username, plus a \
         caldav_password secret (app-specific password)."
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
    };

    Ok(result)
}

export!(CalDavTool);
