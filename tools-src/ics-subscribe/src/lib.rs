//! ICS Subscription WASM Tool for IronClaw.
//!
//! Provides read-only access to calendar feeds via ICS subscription URLs.
//! These are HTTPS URLs (from Office 365, Google Calendar, iCloud, etc.)
//! that return iCalendar (.ics) files. No authentication needed — the URL
//! itself contains an opaque token.
//!
//! # Capabilities Required
//!
//! - HTTP: GET to ICS feed hosts (outlook.office365.com, calendar.google.com, etc.)
//! - Workspace: `ICS_SUBSCRIPTIONS` (JSON array of {name, url} objects)
//!
//! # Supported Actions
//!
//! - `fetch_events`: GET an ICS URL, parse VEVENTs, optionally filter by time range
//! - `list_subscriptions`: Read configured subscriptions from workspace
//!
//! # Setup
//!
//! Write an ICS_SUBSCRIPTIONS workspace file:
//! ```json
//! [
//!   {"name": "Work", "url": "https://outlook.office365.com/owa/calendar/.../calendar.ics"},
//!   {"name": "Personal", "url": "https://calendar.google.com/calendar/ical/.../basic.ics"}
//! ]
//! ```
//!
//! # Example Usage
//!
//! ```json
//! {"action": "fetch_events", "subscription_name": "Work", "time_min": "2026-03-08T00:00:00Z", "time_max": "2026-03-15T23:59:59Z"}
//! ```

mod api;
#[allow(dead_code)] // Verbatim copy of caldav/src/ical.rs — write-side functions unused here
mod ical;
mod types;

use types::IcsAction;

wit_bindgen::generate!({
    world: "sandboxed-tool",
    path: "../../wit/tool.wit",
});

struct IcsSubscribeTool;

impl exports::near::agent::tool::Guest for IcsSubscribeTool {
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
                    "enum": ["fetch_events", "list_subscriptions"],
                    "description": "The operation to perform"
                },
                "url": {
                    "type": "string",
                    "description": "Direct ICS subscription URL. Used by: fetch_events (provide this or subscription_name)"
                },
                "subscription_name": {
                    "type": "string",
                    "description": "Name of a configured subscription (from ICS_SUBSCRIPTIONS workspace file). Used by: fetch_events (alternative to url)"
                },
                "time_min": {
                    "type": "string",
                    "description": "Start of time range filter (RFC3339, e.g., '2026-03-08T00:00:00Z'). Optional for: fetch_events"
                },
                "time_max": {
                    "type": "string",
                    "description": "End of time range filter (RFC3339, e.g., '2026-03-15T23:59:59Z'). Optional for: fetch_events"
                }
            }
        }"#
        .to_string()
    }

    fn description() -> String {
        "Read-only ICS calendar subscription feeds. Fetches .ics files from Office 365, \
         Google Calendar, iCloud, and other providers. Actions: fetch_events (GET an ICS \
         URL and parse events, optionally filtered by time range), list_subscriptions \
         (show configured subscription names/URLs from ICS_SUBSCRIPTIONS workspace file). \
         No authentication needed — subscription URLs are self-authenticating."
            .to_string()
    }
}

fn execute_inner(params: &str) -> Result<String, String> {
    let action: IcsAction =
        serde_json::from_str(params).map_err(|e| format!("Invalid parameters: {}", e))?;

    crate::near::agent::host::log(
        crate::near::agent::host::LogLevel::Info,
        &format!("Executing ICS action: {:?}", action),
    );

    let result = match action {
        IcsAction::FetchEvents {
            url,
            subscription_name,
            time_min,
            time_max,
        } => {
            let resolved_url = match (url, &subscription_name) {
                (Some(u), _) => u,
                (None, Some(name)) => api::resolve_subscription_url(name)?,
                (None, None) => {
                    return Err(
                        "Either 'url' or 'subscription_name' is required for fetch_events"
                            .to_string(),
                    )
                }
            };

            let events =
                api::fetch_events(&resolved_url, time_min.as_deref(), time_max.as_deref())?;

            let result = types::FetchEventsResult {
                events,
                subscription_name,
            };

            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }

        IcsAction::ListSubscriptions => {
            let subs = api::read_subscriptions()?;
            let result = types::ListSubscriptionsResult {
                subscriptions: subs,
            };
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        }
    };

    Ok(result)
}

export!(IcsSubscribeTool);
