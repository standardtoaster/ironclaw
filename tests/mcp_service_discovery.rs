//! Integration test for MCP service discovery.
//!
//! Validates the full flow: ServiceRegistry search → McpClient connection →
//! create_tools() → tool name prefixing → allow/deny filtering → scope filtering.

use std::collections::HashMap;
use std::net::SocketAddr;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use ironclaw::tools::mcp::config::McpTransportConfig;
use ironclaw::tools::mcp::service_registry::{LensOverride, ServiceAuth};
use ironclaw::tools::mcp::{McpClient, McpServerConfig, ServiceConfig, ServiceRegistry};

/// Mock MCP server handler that responds to JSON-RPC methods.
async fn mock_mcp_handler(Json(req): Json<Value>) -> Json<Value> {
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = req.get("id").and_then(|i| i.as_u64());

    match method {
        "initialize" => Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-ha", "version": "1.0"}
            }
        })),
        "notifications/initialized" => Json(json!({"jsonrpc": "2.0"})),
        "tools/list" => Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": [
                    {
                        "name": "turn_on",
                        "description": "Turn on a Home Assistant entity",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "entity_id": {"type": "string", "description": "Entity ID"}
                            },
                            "required": ["entity_id"]
                        }
                    },
                    {
                        "name": "notify",
                        "description": "Send a notification",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "message": {"type": "string"}
                            },
                            "required": ["message"]
                        }
                    },
                    {
                        "name": "delete_entity",
                        "description": "Delete an entity (destructive)",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }
        })),
        "tools/call" => {
            let tool_name = req
                .pointer("/params/name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": format!("Called {}", tool_name)}]
                }
            }))
        }
        _ => Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": "Method not found"}
        })),
    }
}

async fn start_mock_mcp_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app = Router::new().route("/mcp", post(mock_mcp_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind to ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    // Give server a moment to start accepting connections.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (addr, handle)
}

fn make_service_config(addr: &SocketAddr) -> ServiceConfig {
    ServiceConfig {
        name: "test_ha".to_string(),
        description: "Test home assistant".to_string(),
        url: format!("http://127.0.0.1:{}/mcp", addr.port()),
        auth: ServiceAuth::Bearer {
            credential: "test-token".to_string(),
        },
        scopes: vec!["andrew".to_string()],
        keywords: vec!["light".to_string(), "home".to_string()],
        allow: vec!["*".to_string()],
        deny: vec!["delete_*".to_string()],
        tier: "read".to_string(),
        lens_overrides: HashMap::new(),
    }
}

#[tokio::test]
async fn test_service_registry_keyword_search() {
    let (addr, _handle) = start_mock_mcp_server().await;
    let svc_registry = ServiceRegistry::from_configs(vec![make_service_config(&addr)]);

    // Keyword "light" should match the service.
    let results = svc_registry.search("light", "andrew");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "test_ha");

    // Keyword "home" should also match.
    let results = svc_registry.search("home", "andrew");
    assert_eq!(results.len(), 1);

    // Unrelated keyword should not match.
    let results = svc_registry.search("database", "andrew");
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_service_registry_scope_filtering() {
    let (addr, _handle) = start_mock_mcp_server().await;
    let svc_registry = ServiceRegistry::from_configs(vec![make_service_config(&addr)]);

    // andrew is in scope.
    let results = svc_registry.search("light", "andrew");
    assert_eq!(results.len(), 1);

    // unknown_user is not in scope.
    let results = svc_registry.search("light", "unknown_user");
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_mcp_client_fetches_tools_with_prefix() {
    let (addr, _handle) = start_mock_mcp_server().await;
    let svc_registry = ServiceRegistry::from_configs(vec![make_service_config(&addr)]);

    let service = svc_registry.search("light", "andrew")[0];
    let client = McpClient::new_with_config(McpServerConfig {
        name: service.name.clone(),
        url: service.url.clone(),
        transport: Some(McpTransportConfig::Http),
        headers: HashMap::from([(
            "Authorization".to_string(),
            "Bearer test-token".to_string(),
        )]),
        oauth: None,
        enabled: true,
        description: Some(service.description.clone()),
    });

    let tools = client.create_tools().await.expect("should create tools");

    // Mock returns 3 tools; create_tools prefixes them with server name.
    assert_eq!(tools.len(), 3);

    let tool_names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
    assert!(
        tool_names.contains(&"test_ha_turn_on".to_string()),
        "expected test_ha_turn_on in {:?}",
        tool_names
    );
    assert!(
        tool_names.contains(&"test_ha_notify".to_string()),
        "expected test_ha_notify in {:?}",
        tool_names
    );
    assert!(
        tool_names.contains(&"test_ha_delete_entity".to_string()),
        "expected test_ha_delete_entity in {:?}",
        tool_names
    );
}

#[tokio::test]
async fn test_allow_deny_filtering() {
    let (addr, _handle) = start_mock_mcp_server().await;
    let service = make_service_config(&addr);

    // deny = ["delete_*"], so delete_entity should be denied.
    assert!(
        !service.is_tool_allowed("delete_entity", "andrew"),
        "delete_entity should be denied by delete_* pattern"
    );
    assert!(
        service.is_tool_allowed("turn_on", "andrew"),
        "turn_on should be allowed"
    );
    assert!(
        service.is_tool_allowed("notify", "andrew"),
        "notify should be allowed"
    );
}

#[tokio::test]
async fn test_lens_override_deny() {
    let (addr, _handle) = start_mock_mcp_server().await;
    let mut service = make_service_config(&addr);
    service
        .scopes
        .push("grace".to_string());
    service.lens_overrides.insert(
        "grace".to_string(),
        LensOverride {
            allow: vec![],
            deny: vec!["automation_*".to_string()],
            tier: None,
        },
    );

    // Andrew can access automation tools (no lens override for him).
    assert!(service.is_tool_allowed("automation_create", "andrew"));

    // Grace cannot access automation tools due to her lens override.
    assert!(!service.is_tool_allowed("automation_create", "grace"));

    // Both can access non-automation tools.
    assert!(service.is_tool_allowed("turn_on", "andrew"));
    assert!(service.is_tool_allowed("turn_on", "grace"));
}

#[tokio::test]
async fn test_full_discovery_flow() {
    // End-to-end: registry search → MCP connect → tools fetched → filtering applied.
    let (addr, _handle) = start_mock_mcp_server().await;
    let mut config = make_service_config(&addr);
    config
        .scopes
        .push("grace".to_string());
    config.lens_overrides.insert(
        "grace".to_string(),
        LensOverride {
            allow: vec![],
            deny: vec!["delete_*".to_string(), "turn_*".to_string()],
            tier: None,
        },
    );

    let registry = ServiceRegistry::from_configs(vec![config]);

    // Andrew: finds service, fetches tools, filters apply.
    let services = registry.search("light", "andrew");
    assert_eq!(services.len(), 1);

    let svc = services[0];
    let client = McpClient::new_with_config(McpServerConfig {
        name: svc.name.clone(),
        url: svc.url.clone(),
        transport: Some(McpTransportConfig::Http),
        headers: HashMap::new(),
        oauth: None,
        enabled: true,
        description: None,
    });

    let tools = client.create_tools().await.expect("create tools");
    assert_eq!(tools.len(), 3);

    // Filter tools through allow/deny for Andrew.
    let andrew_tools: Vec<_> = tools
        .iter()
        .filter(|t| {
            // Strip the server prefix to get the original tool name.
            let original_name = t
                .name()
                .strip_prefix(&format!("{}_", svc.name))
                .unwrap_or(t.name());
            svc.is_tool_allowed(original_name, "andrew")
        })
        .collect();
    // Andrew: allow=["*"], deny=["delete_*"] → turn_on and notify pass.
    assert_eq!(andrew_tools.len(), 2);

    // Filter tools through allow/deny for Grace.
    let grace_tools: Vec<_> = tools
        .iter()
        .filter(|t| {
            let original_name = t
                .name()
                .strip_prefix(&format!("{}_", svc.name))
                .unwrap_or(t.name());
            svc.is_tool_allowed(original_name, "grace")
        })
        .collect();
    // Grace: base deny=["delete_*"] + lens deny=["delete_*", "turn_*"] → only notify passes.
    assert_eq!(grace_tools.len(), 1);
    assert!(grace_tools[0].name().ends_with("notify"));
}
