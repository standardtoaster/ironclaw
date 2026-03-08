//! Minimal iCal VEVENT parser.
//!
//! Parses the subset of iCalendar we need: VEVENT components with
//! DTSTART, DTEND, SUMMARY, LOCATION, DESCRIPTION, UID, STATUS.
//! Does not handle RRULE (recurrence) — only individual occurrences.

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
