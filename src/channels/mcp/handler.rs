//! Axum handler for the POST /mcp endpoint.
//!
//! Accepts MCP JSON-RPC requests, dispatches to the appropriate method,
//! and returns MCP JSON-RPC responses. Auth is handled by the existing
//! middleware -- the authenticated user's identity scopes tool access.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::channels::web::auth::AuthenticatedUser;
use crate::channels::web::server::GatewayState;
use crate::context::JobContext;
use crate::tools::mcp::protocol::{McpError, McpRequest, McpResponse};
use crate::tools::registry::ToolRegistry;

use crate::channels::mcp::dispatch::{
    handle_initialize, handle_ping, handle_tools_call, handle_tools_list,
};

/// JSON-RPC error codes.
const METHOD_NOT_FOUND: i32 = -32601;
const INVALID_REQUEST: i32 = -32600;
const INTERNAL_ERROR: i32 = -32603;

/// Axum handler for POST /mcp.
///
/// Accepts MCP JSON-RPC requests, dispatches to the appropriate method,
/// and returns MCP JSON-RPC responses. Auth is handled by the existing
/// middleware -- the authenticated user's identity scopes tool access.
pub async fn mcp_handler(
    State(state): State<Arc<GatewayState>>,
    AuthenticatedUser(user): AuthenticatedUser,
    Json(request): Json<McpRequest>,
) -> impl IntoResponse {
    // Notifications (no id) don't get a response per JSON-RPC spec.
    if request.id.is_none() {
        return (StatusCode::NO_CONTENT, Json(None));
    }

    // Build a JobContext scoped to the authenticated user.
    let ctx = JobContext::with_user(&user.user_id, "mcp", "MCP tool call");

    let registry = state.tool_registry.as_deref();

    let response = build_response(&request, registry, Some(&ctx)).await;

    (StatusCode::OK, Json(Some(response)))
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
}
