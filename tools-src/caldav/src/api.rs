//! CalDAV HTTP requests (PROPFIND, REPORT, GET).
//!
//! All requests go through the host's HTTP capability, which handles
//! credential injection (Basic Auth) and rate limiting. The WASM tool
//! never sees the actual password.

use crate::ical;
use crate::near::agent::host;
use crate::types::*;

/// Make a CalDAV HTTP request.
fn caldav_request(method: &str, url: &str, body: Option<&str>) -> Result<String, String> {
    let mut headers = String::from("{\"Content-Type\": \"application/xml; charset=utf-8\"");

    // PROPFIND and REPORT need Depth header
    match method {
        "PROPFIND" => {
            headers.push_str(", \"Depth\": \"1\"");
        }
        "REPORT" => {
            headers.push_str(", \"Depth\": \"1\"");
        }
        _ => {}
    }

    headers.push('}');

    let body_bytes = body.map(|b| b.as_bytes().to_vec());

    host::log(
        host::LogLevel::Debug,
        &format!("CalDAV request: {} {}", method, url),
    );

    let response = host::http_request(method, url, &headers, body_bytes.as_deref(), None)?;

    // 2xx (including 207 Multi-Status) is success for WebDAV
    if response.status < 200 || response.status >= 300 {
        let body_text = String::from_utf8_lossy(&response.body);
        return Err(format!(
            "CalDAV server returned status {}: {}",
            response.status, body_text
        ));
    }

    String::from_utf8(response.body).map_err(|e| format!("Invalid UTF-8 in response: {}", e))
}

/// Read CalDAV config from workspace.
pub fn read_config() -> Result<CalDavConfig, String> {
    let content = host::workspace_read("CALDAV_CONFIG")
        .ok_or_else(|| {
            "CALDAV_CONFIG workspace file not found. Create it with JSON content: \
             {\"base_url\": \"https://caldav.icloud.com\", \"username\": \"your@email.com\"}"
                .to_string()
        })?;

    serde_json::from_str(&content).map_err(|e| format!("Invalid CALDAV_CONFIG: {}", e))
}

/// Discover calendars via PROPFIND on the calendar home.
pub fn list_calendars(base_url: &str) -> Result<ListCalendarsResult, String> {
    // First, discover the calendar home set via PROPFIND on the principal.
    // For iCloud, the base URL is the calendar home directly.
    // For other servers, we may need to discover it.
    let body = r#"<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav" xmlns:cs="http://calendarserver.org/ns/" xmlns:ic="http://apple.com/ns/ical/">
  <d:prop>
    <d:displayname/>
    <d:resourcetype/>
    <ic:calendar-color/>
    <c:calendar-description/>
  </d:prop>
</d:propfind>"#;

    let url = base_url.trim_end_matches('/');
    let response = caldav_request("PROPFIND", &format!("{}/", url), Some(body))?;

    let mut calendars = Vec::new();

    // Parse multistatus XML response
    for response_block in xml_split_responses(&response) {
        let href = xml_extract_text(&response_block, "href").unwrap_or_default();

        // Check if this is a calendar collection
        if !xml_contains_tag(&response_block, "calendar") {
            continue;
        }

        let display_name = xml_extract_text(&response_block, "displayname")
            .unwrap_or_else(|| href_to_name(&href));

        let color = xml_extract_text(&response_block, "calendar-color");
        let description = xml_extract_text(&response_block, "calendar-description");

        // Build full URL from href
        let calendar_url = resolve_href(url, &href);

        calendars.push(Calendar {
            display_name,
            url: calendar_url,
            color,
            description,
        });
    }

    Ok(ListCalendarsResult { calendars })
}

/// List events in a time range via calendar-query REPORT.
pub fn list_events(
    calendar_url: &str,
    time_min: &str,
    time_max: &str,
) -> Result<ListEventsResult, String> {
    let start = to_ical_datetime(time_min);
    let end = to_ical_datetime(time_max);

    let body = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <d:getetag/>
    <c:calendar-data/>
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:time-range start="{}" end="{}"/>
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>"#,
        start, end
    );

    let url = calendar_url.trim_end_matches('/');
    let response = caldav_request("REPORT", &format!("{}/", url), Some(&body))?;

    let mut events = Vec::new();

    for response_block in xml_split_responses(&response) {
        let href = xml_extract_text(&response_block, "href");
        let cal_data = xml_extract_text(&response_block, "calendar-data");

        if let Some(ical_text) = cal_data {
            for vevent in ical::parse_vevents(&ical_text) {
                events.push(Event {
                    uid: vevent.uid,
                    summary: vevent.summary,
                    start: vevent.dtstart,
                    end: vevent.dtend,
                    location: vevent.location,
                    description: vevent.description,
                    status: vevent.status,
                    href: href.clone(),
                });
            }
        }
    }

    // Sort by start time
    events.sort_by(|a, b| a.start.cmp(&b.start));

    Ok(ListEventsResult { events })
}

/// Get a specific event by UID via calendar-query REPORT.
pub fn get_event(calendar_url: &str, uid: &str) -> Result<GetEventResult, String> {
    // Use a calendar-query with UID filter
    let body = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <d:getetag/>
    <c:calendar-data/>
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:prop-filter name="UID">
          <c:text-match collation="i;octet">{}</c:text-match>
        </c:prop-filter>
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>"#,
        xml_escape(uid)
    );

    let url = calendar_url.trim_end_matches('/');
    let response = caldav_request("REPORT", &format!("{}/", url), Some(&body))?;

    for response_block in xml_split_responses(&response) {
        let cal_data = xml_extract_text(&response_block, "calendar-data");
        let href = xml_extract_text(&response_block, "href");

        if let Some(ical_text) = cal_data {
            for vevent in ical::parse_vevents(&ical_text) {
                if vevent.uid == uid {
                    return Ok(GetEventResult {
                        event: Event {
                            uid: vevent.uid,
                            summary: vevent.summary,
                            start: vevent.dtstart,
                            end: vevent.dtend,
                            location: vevent.location,
                            description: vevent.description,
                            status: vevent.status,
                            href,
                        },
                    });
                }
            }
        }
    }

    Err(format!("Event with UID '{}' not found", uid))
}

/// Query free/busy intervals for a time range.
/// Falls back to listing events and extracting busy periods if the server
/// doesn't support free-busy reports (many CalDAV servers don't).
pub fn free_busy(
    calendar_url: &str,
    time_min: &str,
    time_max: &str,
) -> Result<FreeBusyResult, String> {
    // Most CalDAV servers support calendar-query better than free-busy-query,
    // so we list events and derive busy intervals from them.
    let events = list_events(calendar_url, time_min, time_max)?;

    let busy: Vec<BusyInterval> = events
        .events
        .iter()
        .filter(|e| {
            // Exclude cancelled events
            e.status.as_deref() != Some("CANCELLED")
        })
        .filter_map(|e| {
            let end = e.end.as_ref()?;
            Some(BusyInterval {
                start: e.start.clone(),
                end: end.clone(),
            })
        })
        .collect();

    Ok(FreeBusyResult { busy })
}

// ── XML helpers ──────────────────────────────────────────────────────

/// Split a multistatus XML response into individual <response> blocks.
fn xml_split_responses(xml: &str) -> Vec<String> {
    let mut responses = Vec::new();
    let mut search_from = 0;

    loop {
        // Find <d:response> or <D:response> or <response> (case-insensitive prefix)
        let start = find_tag_start(xml, "response", search_from);
        let end = find_tag_end(xml, "response", start.unwrap_or(0));

        match (start, end) {
            (Some(s), Some(e)) => {
                responses.push(xml[s..e].to_string());
                search_from = e;
            }
            _ => break,
        }
    }

    responses
}

/// Find the start of a tag (handles namespace prefixes).
fn find_tag_start(xml: &str, tag: &str, from: usize) -> Option<usize> {
    let search = &xml[from..];
    // Match <response, <d:response, <D:response, etc.
    for (i, _) in search.char_indices() {
        if search[i..].starts_with('<') {
            let rest = &search[i + 1..];
            // Check for tag directly or with namespace prefix
            if tag_name_matches(rest, tag) {
                return Some(from + i);
            }
        }
    }
    None
}

/// Find the end of a tag (closing tag position, inclusive).
fn find_tag_end(xml: &str, tag: &str, from: usize) -> Option<usize> {
    let search = &xml[from..];
    // Match </response>, </d:response>, </D:response>, etc.
    for (i, _) in search.char_indices() {
        if search[i..].starts_with("</") {
            let rest = &search[i + 2..];
            if tag_name_matches(rest, tag) {
                // Find the closing >
                if let Some(close) = search[i..].find('>') {
                    return Some(from + i + close + 1);
                }
            }
        }
    }
    None
}

/// Check if text starts with a tag name (with optional namespace prefix).
fn tag_name_matches(text: &str, tag: &str) -> bool {
    // Direct match: "response>" or "response "
    if text.starts_with(tag) {
        let after = text.as_bytes().get(tag.len());
        return matches!(after, Some(b'>') | Some(b' ') | Some(b'/') | Some(b'\n') | Some(b'\r'));
    }

    // Namespaced match: "X:response>" where X is a prefix
    if let Some(colon) = text.find(':') {
        if colon < 10 {
            // reasonable prefix length
            let after_colon = &text[colon + 1..];
            if after_colon.starts_with(tag) {
                let after = after_colon.as_bytes().get(tag.len());
                return matches!(
                    after,
                    Some(b'>') | Some(b' ') | Some(b'/') | Some(b'\n') | Some(b'\r')
                );
            }
        }
    }

    false
}

/// Extract text content of a specific element from XML.
/// Handles namespace prefixes (d:href, D:href, href, etc.).
fn xml_extract_text(xml: &str, tag: &str) -> Option<String> {
    // Find opening tag
    let start = find_tag_start(xml, tag, 0)?;
    // Find the > that closes the opening tag
    let content_start = xml[start..].find('>')? + start + 1;
    // Find closing tag
    let end = find_tag_end(xml, tag, content_start)?;
    // The closing tag starts at </...
    let close_start = xml[content_start..end].rfind("</")?;
    let text = &xml[content_start..content_start + close_start];
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(xml_unescape(trimmed))
    }
}

/// Check if an XML block contains a specific tag (e.g., <cal:calendar/> or <d:calendar>).
fn xml_contains_tag(xml: &str, tag: &str) -> bool {
    // Look for the tag as a standalone element or within resourcetype
    for (i, _) in xml.char_indices() {
        if xml[i..].starts_with('<') {
            let rest = &xml[i + 1..];
            if tag_name_matches(rest, tag) {
                return true;
            }
        }
    }
    false
}

/// Resolve a relative href against a base URL.
fn resolve_href(base_url: &str, href: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        return href.to_string();
    }

    // Extract scheme + host from base_url
    if let Some(scheme_end) = base_url.find("://") {
        let after_scheme = &base_url[scheme_end + 3..];
        let host_end = after_scheme.find('/').unwrap_or(after_scheme.len());
        let origin = &base_url[..scheme_end + 3 + host_end];

        if href.starts_with('/') {
            format!("{}{}", origin, href)
        } else {
            format!("{}/{}", base_url.trim_end_matches('/'), href)
        }
    } else {
        href.to_string()
    }
}

/// Extract a calendar name from an href path.
fn href_to_name(href: &str) -> String {
    href.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("calendar")
        .to_string()
}

/// Convert RFC3339 datetime to iCal format for time-range filters.
/// "2026-03-15T09:00:00Z" -> "20260315T090000Z"
/// Already in iCal format? Pass through.
fn to_ical_datetime(dt: &str) -> String {
    let dt = dt.trim();

    // Already in iCal format (no dashes)
    if !dt.contains('-') {
        return dt.to_string();
    }

    // Remove dashes, colons
    let mut result = String::with_capacity(dt.len());
    for c in dt.chars() {
        if c != '-' && c != ':' {
            result.push(c);
        }
    }

    // Ensure it ends with Z for UTC (required by CalDAV time-range)
    if !result.ends_with('Z') && !result.contains('+') {
        result.push('Z');
    }

    result
}

/// Escape special XML characters.
fn xml_escape(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&apos;"),
            _ => result.push(c),
        }
    }
    result
}

/// Unescape XML entities.
fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_ical_datetime() {
        assert_eq!(to_ical_datetime("2026-03-15T09:00:00Z"), "20260315T090000Z");
        assert_eq!(to_ical_datetime("2026-03-15T00:00:00Z"), "20260315T000000Z");
        assert_eq!(to_ical_datetime("20260315T090000Z"), "20260315T090000Z");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a&b<c>d"), "a&amp;b&lt;c&gt;d");
    }

    #[test]
    fn test_resolve_href() {
        assert_eq!(
            resolve_href("https://caldav.icloud.com/123", "/123/cal1/"),
            "https://caldav.icloud.com/123/cal1/"
        );
        assert_eq!(
            resolve_href("https://caldav.icloud.com", "https://other.com/cal"),
            "https://other.com/cal"
        );
    }

    #[test]
    fn test_href_to_name() {
        assert_eq!(href_to_name("/user/calendars/work/"), "work");
        assert_eq!(href_to_name("/user/calendars/personal"), "personal");
    }

    #[test]
    fn test_xml_extract_text() {
        let xml = r#"<d:response><d:href>/cal/123/</d:href><d:propstat><d:prop><d:displayname>Work</d:displayname></d:prop></d:propstat></d:response>"#;
        assert_eq!(xml_extract_text(xml, "href"), Some("/cal/123/".to_string()));
        assert_eq!(
            xml_extract_text(xml, "displayname"),
            Some("Work".to_string())
        );
    }

    #[test]
    fn test_xml_split_responses() {
        let xml = r#"<d:multistatus><d:response><d:href>/a</d:href></d:response><d:response><d:href>/b</d:href></d:response></d:multistatus>"#;
        let responses = xml_split_responses(xml);
        assert_eq!(responses.len(), 2);
        assert!(responses[0].contains("/a"));
        assert!(responses[1].contains("/b"));
    }

    #[test]
    fn test_xml_contains_tag() {
        assert!(xml_contains_tag(
            "<d:resourcetype><cal:calendar/></d:resourcetype>",
            "calendar"
        ));
        assert!(!xml_contains_tag(
            "<d:resourcetype><d:collection/></d:resourcetype>",
            "calendar"
        ));
    }

    #[test]
    fn test_tag_name_matches() {
        assert!(tag_name_matches("response>", "response"));
        assert!(tag_name_matches("d:response>", "response"));
        assert!(tag_name_matches("D:response>", "response"));
        assert!(tag_name_matches("response ", "response"));
        assert!(!tag_name_matches("responsex>", "response"));
    }

    #[test]
    fn test_xml_unescape() {
        assert_eq!(xml_unescape("a&amp;b&lt;c"), "a&b<c");
    }
}
