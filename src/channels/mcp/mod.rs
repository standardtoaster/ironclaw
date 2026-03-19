//! MCP server — exposes IronClaw's tool registry over HTTP using the MCP protocol.
//!
//! Implements the Streamable HTTP transport:
//! - POST /mcp — JSON-RPC requests (single or batch), with session management
//! - GET /mcp — 405 (SSE stream not yet supported)
//! - DELETE /mcp — Session teardown
//!
//! Auth reuses `GATEWAY_USER_TOKENS` — each MCP client authenticates as a specific lens,
//! getting scoped tool access.

pub mod dispatch;
pub mod handler;
pub mod session;
pub mod translate;

pub use handler::{mcp_delete_handler, mcp_get_handler, mcp_post_handler};
pub use session::McpSessionStore;
