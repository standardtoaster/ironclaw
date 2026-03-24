//! ICS subscription HTTP requests and workspace operations.
//!
//! All requests go through the host's HTTP capability. No authentication
//! is needed — ICS subscription URLs are self-authenticating via opaque tokens.

use crate::ical;
use crate::near::agent::host;
use crate::types::*;

/// Fetch events from an ICS subscription URL.
///
/// GETs the URL, parses the iCal response, optionally filters by time range,
/// and returns events sorted by start time.
pub fn fetch_events(
    url: &str,
    time_min: Option<&str>,
    time_max: Option<&str>,
) -> Result<Vec<Event>, String> {
    host::log(
        host::LogLevel::Info,
        &format!("Fetching ICS feed: {}", url),
    );

    let response = host::http_request(
        "GET",
        url,
        "{}",
        None,
        Some(60_000), // 60s timeout for large calendar feeds
    )?;

    if response.status < 200 || response.status >= 300 {
        let body_text = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "ICS server returned status {}: {}",
            response.status, body_text
        ));
    }

    let body = String::from_utf8(response.body)
        .map_err(|e| format!("Invalid UTF-8 in ICS response: {}", e))?;

    let vevents = ical::parse_vevents(&body);

    let mut events: Vec<Event> = vevents
        .into_iter()
        .filter(|ve| {
            // Filter by time range if specified.
            // After parsing, dtstart/dtend are in normalized format (RFC3339-ish or date).
            // Lexicographic comparison works for these normalized strings.
            if let Some(min) = time_min {
                // Event ends before our range starts — skip it.
                // Use dtend if available, otherwise dtstart.
                let event_end = ve.dtend.as_deref().unwrap_or(&ve.dtstart);
                if event_end < min {
                    return false;
                }
            }
            if let Some(max) = time_max {
                // Event starts after our range ends — skip it.
                if ve.dtstart.as_str() > max {
                    return false;
                }
            }
            true
        })
        .map(|ve| Event {
            uid: ve.uid,
            summary: ve.summary,
            start: ve.dtstart,
            end: ve.dtend,
            location: ve.location,
            description: ve.description,
            status: ve.status,
        })
        .collect();

    events.sort_by(|a, b| a.start.cmp(&b.start));

    host::log(
        host::LogLevel::Info,
        &format!("Parsed {} events from ICS feed", events.len()),
    );

    Ok(events)
}

/// Read ICS subscription config from workspace.
pub fn read_subscriptions() -> Result<Vec<Subscription>, String> {
    let content = host::workspace_read("ICS_SUBSCRIPTIONS").ok_or_else(|| {
        "ICS_SUBSCRIPTIONS workspace file not found. Create it with a JSON array: \
         [{\"name\": \"Work\", \"url\": \"https://outlook.office365.com/owa/calendar/.../calendar.ics\"}]"
            .to_string()
    })?;

    let subs: Vec<Subscription> =
        serde_json::from_str(&content).map_err(|e| format!("Invalid ICS_SUBSCRIPTIONS: {}", e))?;

    Ok(subs)
}

/// Look up a subscription URL by name.
pub fn resolve_subscription_url(name: &str) -> Result<String, String> {
    let subs = read_subscriptions()?;

    let sub = subs
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let available: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
            format!(
                "Subscription '{}' not found. Available: {}",
                name,
                available.join(", ")
            )
        })?;

    Ok(sub.url.clone())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_event_time_filtering_logic() {
        // Test the filtering logic with normalized datetime strings.
        // Lexicographic comparison works for RFC3339 format.
        let min = "2026-03-08T00:00:00Z";
        let max = "2026-03-15T23:59:59Z";

        // Event within range
        let start = "2026-03-10T09:00:00Z";
        let end = "2026-03-10T10:00:00Z";
        assert!(end >= min);
        assert!(start <= max);

        // Event before range — end is before our min, should be filtered out
        let end_before = "2026-03-01T10:00:00Z";
        assert!(end_before < min);

        // Event after range — start is after our max, should be filtered out
        let start_after = "2026-03-20T09:00:00Z";
        assert!(start_after > max);
    }

    #[test]
    fn test_event_spanning_range_boundary() {
        // Event that starts before range but ends within it — should be included
        let min = "2026-03-08T00:00:00Z";
        let max = "2026-03-15T23:59:59Z";

        let start = "2026-03-07T22:00:00Z"; // before min
        let end = "2026-03-08T01:00:00Z"; // after min

        // end >= min → true (event end is after range start)
        assert!(end >= min);
        // start <= max → true (event start is before range end)
        assert!(start <= max);
    }

    #[test]
    fn test_event_without_dtend_uses_dtstart_for_filtering() {
        // When dtend is None, the filter uses dtstart as event_end.
        // An event with only dtstart before range should be excluded.
        let min = "2026-03-08T00:00:00Z";

        let dtstart = "2026-03-01T09:00:00Z";
        let dtend: Option<&str> = None;
        let event_end = dtend.unwrap_or(dtstart);

        // event_end < min → should be filtered out
        assert!(event_end < min);
    }

    #[test]
    fn test_filtering_with_only_time_min() {
        // When only time_min is set, events ending before min are excluded,
        // but all events starting at any time after min are included.
        let min = "2026-03-08T00:00:00Z";

        let future_start = "2099-12-31T23:59:59Z";
        // No time_max check, so this should pass
        assert!(future_start >= min);
    }

    #[test]
    fn test_filtering_with_only_time_max() {
        // When only time_max is set, events starting after max are excluded,
        // but all events starting before max are included.
        let max = "2026-03-15T23:59:59Z";

        let past_start = "2020-01-01T00:00:00Z";
        // No time_min check, so this should pass
        assert!(past_start <= max);
    }

    #[test]
    fn test_filtering_all_day_events() {
        // All-day events use date-only format (e.g., "2026-03-10").
        // Lexicographic comparison still works: "2026-03-10" < "2026-03-10T00:00:00Z"
        let min = "2026-03-08T00:00:00Z";
        let max = "2026-03-15T23:59:59Z";

        let allday_start = "2026-03-10";
        let allday_end = "2026-03-11";

        // "2026-03-11" >= "2026-03-08T00:00:00Z" → true (lexicographic: '1' > '0' at pos 9)
        assert!(allday_end >= min);
        // "2026-03-10" <= "2026-03-15T23:59:59Z" → true
        assert!(allday_start <= max);
    }

    #[test]
    fn test_subscription_json_format() {
        // Verify the JSON format expected by read_subscriptions
        let json = r#"[
            {"name": "Work", "url": "https://outlook.office365.com/owa/calendar/abc123/calendar.ics"},
            {"name": "Personal", "url": "https://calendar.google.com/calendar/ical/user/basic.ics"}
        ]"#;
        let subs: Vec<super::Subscription> = serde_json::from_str(json).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].name, "Work");
        assert!(subs[0].url.contains("outlook.office365.com"));
    }

    #[test]
    fn test_subscription_case_insensitive_lookup() {
        // resolve_subscription_url uses eq_ignore_ascii_case
        assert!("Work".eq_ignore_ascii_case("work"));
        assert!("Work".eq_ignore_ascii_case("WORK"));
        assert!("Work".eq_ignore_ascii_case("Work"));
        assert!(!"Work".eq_ignore_ascii_case("Wor"));
    }

    #[test]
    fn test_empty_subscription_list() {
        let json = "[]";
        let subs: Vec<super::Subscription> = serde_json::from_str(json).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn test_invalid_subscription_json() {
        let json = "not valid json";
        let result: Result<Vec<super::Subscription>, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
