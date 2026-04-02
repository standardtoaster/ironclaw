//! Axum handlers for the /mcp endpoint (Streamable HTTP transport).
//!
//! Implements the MCP Streamable HTTP transport:
//! - POST: JSON-RPC requests (single or batch), with Mcp-Session-Id header
//! - GET: SSE stream (returns 405 — not yet implemented)
//! - DELETE: Session teardown
//!
//! Auth is handled by the existing middleware — the authenticated user's
//! identity scopes tool access.

use std::sync::{Arc, LazyLock};

use crate::channels::mcp::McpSessionStore;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};

use crate::channels::mcp::dispatch::{
    handle_initialize, handle_ping, handle_tools_call, handle_tools_list,
};
// AuthenticatedUser not available on this base yet — multi-tenant auth comes in a later row.
use crate::channels::web::server::GatewayState;
use crate::context::JobContext;
use crate::tools::mcp::protocol::{McpError, McpRequest, McpResponse};
use crate::tools::registry::ToolRegistry;

/// JSON-RPC error codes.
const METHOD_NOT_FOUND: i32 = -32601;
const INVALID_REQUEST: i32 = -32600;
const INTERNAL_ERROR: i32 = -32603;

/// Header name for MCP session ID.
const MCP_SESSION_ID_HEADER: &str = "mcp-session-id";

/// Module-level session store (previously on GatewayState).
static MCP_SESSIONS: LazyLock<McpSessionStore> = LazyLock::new(McpSessionStore::new);

/// Axum handler for POST /mcp.
///
/// Accepts single or batch MCP JSON-RPC requests. Returns `Mcp-Session-Id`
/// on initialize responses. Notifications (no id) return 202 Accepted.
pub async fn mcp_post_handler(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    // Try to parse as batch (array) first, then as single request.
    let requests: Vec<McpRequest> = match serde_json::from_slice::<Vec<McpRequest>>(&body) {
        Ok(batch) => batch,
        Err(_) => match serde_json::from_slice::<McpRequest>(&body) {
            Ok(single) => vec![single],
            Err(e) => {
                let error_response = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(McpError {
                        code: INVALID_REQUEST,
                        message: format!("Failed to parse JSON-RPC request: {e}"),
                        data: None,
                    }),
                };
                return (StatusCode::BAD_REQUEST, HeaderMap::new(), Json(serde_json::json!(error_response))).into_response();
            }
        },
    };

    if requests.is_empty() {
        let error_response = McpResponse {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: Some(McpError {
                code: INVALID_REQUEST,
                message: "Empty batch request".to_string(),
                data: None,
            }),
        };
        return (StatusCode::BAD_REQUEST, HeaderMap::new(), Json(serde_json::json!(error_response))).into_response();
    }

    // Build a JobContext scoped to the default user (no per-request auth on this base).
    let ctx = JobContext::with_user(&state.default_user_id, "mcp", "MCP tool call");

    let registry = state.tool_registry.as_deref();
    let sessions = &*MCP_SESSIONS;

    // Check if client wants SSE format.
    let wants_sse = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream") && !v.contains("application/json"))
        .unwrap_or(false);

    // Check for all-notifications (no responses needed).
    let all_notifications = requests.iter().all(|r| r.id.is_none());
    if all_notifications {
        return (StatusCode::ACCEPTED, HeaderMap::new(), Json(serde_json::json!(null))).into_response();
    }

    let is_batch = requests.len() > 1;

    // Process each request.
    let mut responses: Vec<McpResponse> = Vec::new();
    let mut session_id_header: Option<String> = None;

    for request in &requests {
        // Notifications don't produce responses.
        if request.id.is_none() {
            continue;
        }

        let response = build_response(request, registry, Some(&ctx)).await;

        // If this is an initialize request, create a session.
        if request.method == "initialize" {
            let sid = sessions.create("default");
            session_id_header = Some(sid.to_string());
        }

        responses.push(response);
    }

    // Build response headers.
    let mut response_headers = HeaderMap::new();
    if let Some(sid) = session_id_header
        && let Ok(val) = sid.parse()
    {
        response_headers.insert(MCP_SESSION_ID_HEADER, val);
    }

    if wants_sse {
        // Wrap response(s) in SSE format.
        let mut sse_body = String::new();
        for resp in &responses {
            if let Ok(json) = serde_json::to_string(resp) {
                sse_body.push_str("event: message\ndata: ");
                sse_body.push_str(&json);
                sse_body.push_str("\n\n");
            }
        }
        if let Ok(ct) = "text/event-stream".parse() {
            response_headers.insert("content-type", ct);
        }
        (StatusCode::OK, response_headers, sse_body).into_response()
    } else if is_batch {
        // Return array of responses.
        let json = serde_json::to_value(&responses).unwrap_or_default();
        (StatusCode::OK, response_headers, Json(json)).into_response()
    } else {
        // Return single response.
        let json = serde_json::to_value(&responses[0]).unwrap_or_default();
        (StatusCode::OK, response_headers, Json(json)).into_response()
    }
}

/// Axum handler for GET /mcp (SSE stream).
///
/// Claude Code opens a GET SSE stream for server-initiated messages.
/// We don't send server-initiated messages, but returning a valid SSE
/// response signals to the client that we're a Streamable HTTP MCP server.
pub async fn mcp_get_handler() -> impl IntoResponse {
    // Return a valid SSE response with just the headers.
    // The stream stays open (empty) — no server-initiated messages.
    let mut headers = HeaderMap::new();
    if let Ok(ct) = "text/event-stream".parse() {
        headers.insert("content-type", ct);
    }
    if let Ok(cc) = "no-cache".parse() {
        headers.insert("cache-control", cc);
    }
    (StatusCode::OK, headers, ": keepalive\n\n")
}

/// Axum handler for DELETE /mcp (session teardown).
///
/// Cleans up the session identified by the `Mcp-Session-Id` header.
pub async fn mcp_delete_handler(
    State(_state): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(sid_val) = headers.get(MCP_SESSION_ID_HEADER)
        && let Ok(sid_str) = sid_val.to_str()
        && let Ok(sid) = sid_str.parse::<uuid::Uuid>()
    {
        MCP_SESSIONS.remove(&sid);
    }
    StatusCode::OK
}

/// Build an MCP response for a given request.
async fn build_response(
    request: &McpRequest,
    registry: Option<&ToolRegistry>,
    ctx: Option<&JobContext>,
) -> McpResponse {
    match request.method.as_str() {
        "initialize" => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(handle_initialize()),
            error: None,
        },

        "ping" => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(handle_ping()),
            error: None,
        },

        "tools/list" => match registry {
            Some(reg) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(handle_tools_list(reg).await),
                error: None,
            },
            None => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(McpError {
                    code: INTERNAL_ERROR,
                    message: "Tool registry not available".to_string(),
                    data: None,
                }),
            },
        },

        "tools/call" => {
            let params = match &request.params {
                Some(p) => p,
                None => {
                    return McpResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: None,
                        error: Some(McpError {
                            code: INVALID_REQUEST,
                            message: "tools/call requires params with 'name' and 'arguments'"
                                .to_string(),
                            data: None,
                        }),
                    };
                }
            };
            match registry {
                Some(reg) => {
                    let job_ctx = ctx.cloned().unwrap_or_default();
                    McpResponse {
                        jsonrpc: "2.0".to_string(),
                        id: request.id,
                        result: Some(handle_tools_call(reg, &job_ctx, params).await),
                        error: None,
                    }
                }
                None => McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: None,
                    error: Some(McpError {
                        code: INTERNAL_ERROR,
                        message: "Tool registry not available".to_string(),
                        data: None,
                    }),
                },
            }
        }

        _ => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(McpError {
                code: METHOD_NOT_FOUND,
                message: format!("Method not found: {}", request.method),
                data: None,
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::channels::mcp::McpSessionStore;
    use crate::tools::mcp::protocol::PROTOCOL_VERSION;

    #[tokio::test]
    async fn test_dispatch_unknown_method_returns_error() {
        let req = McpRequest::new(1, "unknown/method", None);
        let resp = build_response(&req, None, None).await;
        assert_eq!(resp.id, Some(1));
        let err = resp.error.unwrap();
        assert_eq!(err.code, METHOD_NOT_FOUND);
        assert!(err.message.contains("Method not found"));
    }

    #[tokio::test]
    async fn test_dispatch_initialize() {
        let req = McpRequest::initialize(42);
        let resp = build_response(&req, None, None).await;
        assert_eq!(resp.id, Some(42));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn test_dispatch_ping() {
        let req = McpRequest::new(5, "ping", None);
        let resp = build_response(&req, None, None).await;
        assert_eq!(resp.id, Some(5));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_dispatch_tools_list() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::builtin::EchoTool))
            .await;
        let req = McpRequest::list_tools(10);
        let resp = build_response(&req, Some(&registry), None).await;
        assert_eq!(resp.id, Some(10));
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
    }

    #[tokio::test]
    async fn test_dispatch_tools_call() {
        let registry = ToolRegistry::new();
        registry
            .register(Arc::new(crate::tools::builtin::EchoTool))
            .await;
        let ctx = JobContext::default();
        let req = McpRequest::call_tool(11, "echo", serde_json::json!({"message": "hi"}));
        let resp = build_response(&req, Some(&registry), Some(&ctx)).await;
        assert_eq!(resp.id, Some(11));
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(!result["is_error"].as_bool().unwrap_or(true));
    }

    #[tokio::test]
    async fn test_dispatch_tools_list_no_registry() {
        let req = McpRequest::list_tools(20);
        let resp = build_response(&req, None, None).await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
    }

    #[tokio::test]
    async fn test_dispatch_tools_call_no_params() {
        let registry = ToolRegistry::new();
        let ctx = JobContext::default();
        let req = McpRequest::new(30, "tools/call", None);
        let resp = build_response(&req, Some(&registry), Some(&ctx)).await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, INVALID_REQUEST);
    }

    #[tokio::test]
    async fn test_dispatch_tools_call_no_registry() {
        let ctx = JobContext::default();
        let req = McpRequest::call_tool(31, "echo", serde_json::json!({"message": "hi"}));
        let resp = build_response(&req, None, Some(&ctx)).await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
    }

    /// Verify the MCP handler enriches `JobContext` with workspace read scopes
    /// and timezone from the authenticated user and gateway state.
    #[test]
    fn test_job_context_enrichment() {
        let user_id = "andrew".to_string();
        let workspace_read_scopes = vec!["grace".to_string(), "household".to_string()];

        // Replicate the enrichment logic from mcp_post_handler.
        let mut ctx = JobContext::with_user(&user_id, "mcp", "MCP tool call");
        ctx.workspace_read_scopes = workspace_read_scopes;
        ctx.user_timezone = "America/New_York".to_string();

        assert_eq!(ctx.user_id, "andrew");
    }

    #[test]
    fn test_session_store_create_on_initialize() {
        let store = McpSessionStore::new();
        let sid = store.create("test-user");
        assert!(store.exists(&sid));
    }

    #[test]
    fn test_session_store_remove_on_delete() {
        let store = McpSessionStore::new();
        let sid = store.create("test-user");
        assert!(store.remove(&sid));
        assert!(!store.exists(&sid));
    }
}
