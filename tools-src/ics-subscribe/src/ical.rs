//! Minimal iCal VEVENT parser.
//!
//! Parses the subset of iCalendar we need: VEVENT components with
//! DTSTART, DTEND, SUMMARY, LOCATION, DESCRIPTION, UID, STATUS.
//! Does not handle RRULE (recurrence) — only individual occurrences.
//!
//! NOTE: This file is a verbatim copy of `tools-src/caldav/src/ical.rs`.
//! The write-side functions (build_vcalendar, escape_ical, etc.) are unused
//! here but kept to avoid divergence. WASM tools can't share library crates.


/// A parsed VEVENT from iCal data.
#[derive(Debug, Default)]
pub struct VEvent {
    pub uid: String,
    pub summary: String,
    pub dtstart: String,
    pub dtend: Option<String>,
    pub location: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
}

/// Parse all VEVENTs from iCal text.
pub fn parse_vevents(ical: &str) -> Vec<VEvent> {
    let mut events = Vec::new();
    let mut current: Option<VEvent> = None;
    let mut current_key: Option<String> = None;

    // Unfold lines first: lines starting with space/tab are continuations.
    let unfolded = unfold_lines(ical);

    for line in unfolded.lines() {
        let line = line.trim_end_matches('\r');

        if line == "BEGIN:VEVENT" {
            current = Some(VEvent::default());
            current_key = None;
            continue;
        }

        if line == "END:VEVENT" {
            if let Some(event) = current.take() {
                if !event.uid.is_empty() || !event.summary.is_empty() {
                    events.push(event);
                }
            }
            current_key = None;
            continue;
        }

        if current.is_none() {
            continue;
        }

        // Parse property: NAME;params:value or NAME:value
        if let Some((key, value)) = parse_property(line) {
            let event = current.as_mut().unwrap();
            current_key = Some(key.clone());
            match key.as_str() {
                "UID" => event.uid = value,
                "SUMMARY" => event.summary = value,
                "DTSTART" => event.dtstart = normalize_datetime(&value),
                "DTEND" => event.dtend = Some(normalize_datetime(&value)),
                "LOCATION" => event.location = Some(value),
                "DESCRIPTION" => event.description = Some(unescape_ical(&value)),
                "STATUS" => event.status = Some(value),
                _ => {}
            }
        } else if line.starts_with(' ') || line.starts_with('\t') {
            // Continuation line (shouldn't happen after unfolding, but be safe)
            if let Some(ref key) = current_key {
                let cont = line.trim_start();
                let event = current.as_mut().unwrap();
                match key.as_str() {
                    "DESCRIPTION" => {
                        if let Some(ref mut d) = event.description {
                            d.push_str(cont);
                        }
                    }
                    "SUMMARY" => event.summary.push_str(cont),
                    "LOCATION" => {
                        if let Some(ref mut l) = event.location {
                            l.push_str(cont);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    events
}

/// Parse a single iCal property line.
/// Returns (property_name, value), stripping any parameters.
/// e.g., "DTSTART;TZID=America/New_York:20260315T090000" -> ("DTSTART", "20260315T090000")
fn parse_property(line: &str) -> Option<(String, String)> {
    // Find the first colon that separates name(;params) from value.
    // But colons can appear in parameter values (rare for our properties).
    let colon_pos = line.find(':')?;
    let name_part = &line[..colon_pos];
    let value = &line[colon_pos + 1..];

    // Strip parameters: NAME;PARAM=VAL -> NAME
    let name = if let Some(semi) = name_part.find(';') {
        &name_part[..semi]
    } else {
        name_part
    };

    Some((name.to_uppercase(), value.to_string()))
}

/// Unfold iCal lines: join continuation lines (starting with space/tab)
/// back onto the previous line.
fn unfold_lines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut first = true;

    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if (line.starts_with(' ') || line.starts_with('\t')) && !first {
            // Continuation: append without newline, stripping leading whitespace
            result.push_str(&line[1..]);
        } else {
            if !first {
                result.push('\n');
            }
            result.push_str(line);
            first = false;
        }
    }

    result
}

/// Normalize a datetime value to a more readable format.
/// Converts "20260315T090000Z" to "2026-03-15T09:00:00Z"
/// Converts "20260315T090000" to "2026-03-15T09:00:00"
/// Passes through already-formatted strings unchanged.
fn normalize_datetime(dt: &str) -> String {
    let dt = dt.trim();

    // Already formatted with dashes? Pass through.
    if dt.contains('-') {
        return dt.to_string();
    }

    // Date-only: 20260315 -> 2026-03-15
    if dt.len() == 8 && dt.chars().all(|c| c.is_ascii_digit()) {
        return format!("{}-{}-{}", &dt[0..4], &dt[4..6], &dt[6..8]);
    }

    // DateTime: 20260315T090000Z or 20260315T090000
    if dt.len() >= 15 && dt.contains('T') {
        let has_z = dt.ends_with('Z');
        let base = if has_z { &dt[..dt.len() - 1] } else { dt };

        if base.len() >= 15 {
            let date = &base[0..8];
            let time = &base[9..15];
            let formatted = format!(
                "{}-{}-{}T{}:{}:{}",
                &date[0..4],
                &date[4..6],
                &date[6..8],
                &time[0..2],
                &time[2..4],
                &time[4..6]
            );
            return if has_z {
                format!("{}Z", formatted)
            } else {
                formatted
            };
        }
    }

    dt.to_string()
}

/// Escape text for iCal property values.
/// Reverse of unescape_ical: backslash, semicolon, comma, newline.
pub fn escape_ical(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            ';' => result.push_str("\\;"),
            ',' => result.push_str("\\,"),
            '\n' => result.push_str("\\n"),
            '\r' => {} // strip CR
            _ => result.push(c),
        }
    }
    result
}

/// Event fields for building a VCALENDAR.
pub struct EventFields {
    pub uid: String,
    pub summary: String,
    pub start_datetime: Option<String>,
    pub end_datetime: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub location: Option<String>,
    pub description: Option<String>,
    pub timezone: Option<String>,
}

/// Build a complete VCALENDAR string from event fields.
/// Uses `now_millis` for DTSTAMP.
pub fn build_vcalendar(fields: &EventFields, now_millis: u64) -> Result<String, String> {
    let dtstamp = millis_to_ical_utc(now_millis);

    // Determine DTSTART/DTEND
    let (dtstart_line, dtend_line) = if let Some(ref sd) = fields.start_date {
        // All-day event
        let start = date_to_ical(sd)?;
        let end = fields
            .end_date
            .as_deref()
            .map(date_to_ical)
            .transpose()?;
        (
            format!("DTSTART;VALUE=DATE:{}", start),
            end.map(|e| format!("DTEND;VALUE=DATE:{}", e)),
        )
    } else if let Some(ref sdt) = fields.start_datetime {
        // Timed event
        let start_ical = rfc3339_to_ical(sdt);
        let end_ical = fields.end_datetime.as_deref().map(rfc3339_to_ical);
        if let Some(ref tz) = fields.timezone {
            (
                format!("DTSTART;TZID={}:{}", tz, start_ical),
                end_ical.map(|e| format!("DTEND;TZID={}:{}", tz, e)),
            )
        } else {
            (
                format!("DTSTART:{}", start_ical),
                end_ical.map(|e| format!("DTEND:{}", e)),
            )
        }
    } else {
        return Err(
            "Either start_datetime or start_date is required to create an event".to_string(),
        );
    };

    let mut lines = vec![
        "BEGIN:VCALENDAR".to_string(),
        "VERSION:2.0".to_string(),
        "PRODID:-//IronClaw//CalDAV Tool//EN".to_string(),
        "BEGIN:VEVENT".to_string(),
        format!("UID:{}", fields.uid),
        format!("DTSTAMP:{}", dtstamp),
        dtstart_line,
    ];

    if let Some(end) = dtend_line {
        lines.push(end);
    }

    lines.push(format!("SUMMARY:{}", escape_ical(&fields.summary)));

    if let Some(ref loc) = fields.location {
        lines.push(format!("LOCATION:{}", escape_ical(loc)));
    }
    if let Some(ref desc) = fields.description {
        lines.push(format!("DESCRIPTION:{}", escape_ical(desc)));
    }

    lines.push("END:VEVENT".to_string());
    lines.push("END:VCALENDAR".to_string());

    Ok(lines.join("\r\n"))
}

/// Convert milliseconds since epoch to iCal UTC timestamp (e.g., "20260315T090000Z").
fn millis_to_ical_utc(millis: u64) -> String {
    // Convert millis to date/time components.
    // Simple implementation: days since epoch -> date, remainder -> time.
    let total_secs = millis / 1000;
    let secs_in_day: u64 = 86400;
    let mut days = total_secs / secs_in_day;
    let day_secs = total_secs % secs_in_day;

    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let seconds = day_secs % 60;

    // Days since 1970-01-01 to (year, month, day).
    // Civil calendar algorithm.
    days += 719468; // shift to 0000-03-01 epoch
    let era = days / 146097;
    let doe = days % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        y, m, d, hours, minutes, seconds
    )
}

/// Convert "YYYY-MM-DD" to iCal date "YYYYMMDD".
fn date_to_ical(date: &str) -> Result<String, String> {
    let stripped: String = date.chars().filter(|c| *c != '-').collect();
    if stripped.len() != 8 || !stripped.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("Invalid date format '{}', expected YYYY-MM-DD", date));
    }
    Ok(stripped)
}

/// Convert RFC3339 datetime to iCal format.
/// "2026-03-15T09:00:00Z" -> "20260315T090000Z"
/// Already compact? Pass through.
fn rfc3339_to_ical(dt: &str) -> String {
    let dt = dt.trim();
    if !dt.contains('-') {
        return dt.to_string();
    }
    let mut result = String::with_capacity(dt.len());
    for c in dt.chars() {
        if c != '-' && c != ':' {
            result.push(c);
        }
    }
    result
}

/// Unescape iCal text values.
/// iCal escapes: \n -> newline, \, -> comma, \; -> semicolon, \\ -> backslash
fn unescape_ical(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') | Some('N') => result.push('\n'),
                Some(',') => result.push(','),
                Some(';') => result.push(';'),
                Some('\\') => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_vevent() {
        let ical = "\
BEGIN:VCALENDAR\r
BEGIN:VEVENT\r
DTSTART:20260315T090000Z\r
DTEND:20260315T100000Z\r
SUMMARY:Team meeting\r
LOCATION:Room 3\r
UID:abc123@example.com\r
STATUS:CONFIRMED\r
END:VEVENT\r
END:VCALENDAR";

        let events = parse_vevents(ical);
        assert_eq!(events.len(), 1);
        let e = &events[0];
        assert_eq!(e.uid, "abc123@example.com");
        assert_eq!(e.summary, "Team meeting");
        assert_eq!(e.dtstart, "2026-03-15T09:00:00Z");
        assert_eq!(e.dtend.as_deref(), Some("2026-03-15T10:00:00Z"));
        assert_eq!(e.location.as_deref(), Some("Room 3"));
        assert_eq!(e.status.as_deref(), Some("CONFIRMED"));
    }

    #[test]
    fn test_parse_with_tzid() {
        let ical = "\
BEGIN:VEVENT\r
DTSTART;TZID=America/New_York:20260315T090000\r
DTEND;TZID=America/New_York:20260315T100000\r
SUMMARY:Local meeting\r
UID:tz1@example.com\r
END:VEVENT";

        let events = parse_vevents(ical);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].dtstart, "2026-03-15T09:00:00");
    }

    #[test]
    fn test_parse_all_day() {
        let ical = "\
BEGIN:VEVENT\r
DTSTART;VALUE=DATE:20260315\r
DTEND;VALUE=DATE:20260316\r
SUMMARY:All day event\r
UID:allday@example.com\r
END:VEVENT";

        let events = parse_vevents(ical);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].dtstart, "2026-03-15");
    }

    #[test]
    fn test_unfold_lines() {
        let input = "DESCRIPTION:This is a long\r\n description that spans\r\n  multiple lines";
        let result = unfold_lines(input);
        assert_eq!(
            result,
            "DESCRIPTION:This is a longdescription that spans multiple lines"
        );
    }

    #[test]
    fn test_unescape() {
        assert_eq!(unescape_ical("hello\\nworld"), "hello\nworld");
        assert_eq!(unescape_ical("a\\,b\\;c\\\\d"), "a,b;c\\d");
    }

    #[test]
    fn test_normalize_datetime() {
        assert_eq!(normalize_datetime("20260315T090000Z"), "2026-03-15T09:00:00Z");
        assert_eq!(normalize_datetime("20260315T090000"), "2026-03-15T09:00:00");
        assert_eq!(normalize_datetime("20260315"), "2026-03-15");
        assert_eq!(
            normalize_datetime("2026-03-15T09:00:00Z"),
            "2026-03-15T09:00:00Z"
        );
    }

    #[test]
    fn test_escape_ical() {
        assert_eq!(escape_ical("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_ical("a,b;c\\d"), "a\\,b\\;c\\\\d");
        assert_eq!(escape_ical("no special chars"), "no special chars");
        assert_eq!(escape_ical("line1\r\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_escape_unescape_roundtrip() {
        let original = "Meeting; with, special\nchars\\here";
        let escaped = escape_ical(original);
        let unescaped = unescape_ical(&escaped);
        assert_eq!(unescaped, original);
    }

    #[test]
    fn test_build_vcalendar_timed_event() {
        let fields = EventFields {
            uid: "test-uid-123@ironclaw".to_string(),
            summary: "Team meeting".to_string(),
            start_datetime: Some("2026-03-15T09:00:00Z".to_string()),
            end_datetime: Some("2026-03-15T10:00:00Z".to_string()),
            start_date: None,
            end_date: None,
            location: Some("Room 3".to_string()),
            description: Some("Discuss project".to_string()),
            timezone: None,
        };
        // 2026-03-15T00:00:00Z in millis
        let now = 1773619200000u64;
        let result = build_vcalendar(&fields, now).unwrap();

        assert!(result.contains("BEGIN:VCALENDAR"));
        assert!(result.contains("VERSION:2.0"));
        assert!(result.contains("PRODID:-//IronClaw//CalDAV Tool//EN"));
        assert!(result.contains("UID:test-uid-123@ironclaw"));
        assert!(result.contains("DTSTART:20260315T090000Z"));
        assert!(result.contains("DTEND:20260315T100000Z"));
        assert!(result.contains("SUMMARY:Team meeting"));
        assert!(result.contains("LOCATION:Room 3"));
        assert!(result.contains("DESCRIPTION:Discuss project"));
        assert!(result.contains("END:VEVENT"));
        assert!(result.contains("END:VCALENDAR"));
    }

    #[test]
    fn test_build_vcalendar_allday_event() {
        let fields = EventFields {
            uid: "allday-456@ironclaw".to_string(),
            summary: "Day off".to_string(),
            start_datetime: None,
            end_datetime: None,
            start_date: Some("2026-03-20".to_string()),
            end_date: Some("2026-03-21".to_string()),
            location: None,
            description: None,
            timezone: None,
        };
        let now = 1773619200000u64;
        let result = build_vcalendar(&fields, now).unwrap();

        assert!(result.contains("DTSTART;VALUE=DATE:20260320"));
        assert!(result.contains("DTEND;VALUE=DATE:20260321"));
        assert!(!result.contains("LOCATION"));
        assert!(!result.contains("DESCRIPTION"));
    }

    #[test]
    fn test_build_vcalendar_with_timezone() {
        let fields = EventFields {
            uid: "tz-789@ironclaw".to_string(),
            summary: "Local meeting".to_string(),
            start_datetime: Some("2026-03-15T09:00:00".to_string()),
            end_datetime: Some("2026-03-15T10:00:00".to_string()),
            start_date: None,
            end_date: None,
            location: None,
            description: None,
            timezone: Some("America/New_York".to_string()),
        };
        let now = 1773619200000u64;
        let result = build_vcalendar(&fields, now).unwrap();

        assert!(result.contains("DTSTART;TZID=America/New_York:20260315T090000"));
        assert!(result.contains("DTEND;TZID=America/New_York:20260315T100000"));
    }

    #[test]
    fn test_build_vcalendar_escapes_special_chars() {
        let fields = EventFields {
            uid: "esc@ironclaw".to_string(),
            summary: "Meeting; important, really".to_string(),
            start_datetime: Some("2026-03-15T09:00:00Z".to_string()),
            end_datetime: None,
            start_date: None,
            end_date: None,
            location: None,
            description: Some("Line 1\nLine 2".to_string()),
            timezone: None,
        };
        let now = 1773619200000u64;
        let result = build_vcalendar(&fields, now).unwrap();

        assert!(result.contains("SUMMARY:Meeting\\; important\\, really"));
        assert!(result.contains("DESCRIPTION:Line 1\\nLine 2"));
    }

    #[test]
    fn test_build_vcalendar_no_start_returns_error() {
        let fields = EventFields {
            uid: "no-start@ironclaw".to_string(),
            summary: "Bad event".to_string(),
            start_datetime: None,
            end_datetime: None,
            start_date: None,
            end_date: None,
            location: None,
            description: None,
            timezone: None,
        };
        let result = build_vcalendar(&fields, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_millis_to_ical_utc() {
        // 2026-03-18T14:50:45Z = 1773845445 seconds since epoch
        let millis = 1773845445000u64;
        let result = millis_to_ical_utc(millis);
        assert_eq!(result, "20260318T145045Z");
    }

    #[test]
    fn test_date_to_ical() {
        assert_eq!(date_to_ical("2026-03-15").unwrap(), "20260315");
        assert!(date_to_ical("bad-date").is_err());
        assert!(date_to_ical("2026-3-5").is_err()); // not zero-padded
    }

    #[test]
    fn test_rfc3339_to_ical() {
        assert_eq!(rfc3339_to_ical("2026-03-15T09:00:00Z"), "20260315T090000Z");
        assert_eq!(rfc3339_to_ical("20260315T090000Z"), "20260315T090000Z");
    }

    #[test]
    fn test_multiple_events() {
        let ical = "\
BEGIN:VCALENDAR
BEGIN:VEVENT
UID:1@test
SUMMARY:First
DTSTART:20260315T090000Z
END:VEVENT
BEGIN:VEVENT
UID:2@test
SUMMARY:Second
DTSTART:20260315T100000Z
END:VEVENT
END:VCALENDAR";

        let events = parse_vevents(ical);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].summary, "First");
        assert_eq!(events[1].summary, "Second");
    }
}
