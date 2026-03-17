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
