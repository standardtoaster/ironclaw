//! MCP server — exposes IronClaw's tool registry over HTTP using the MCP protocol.
//!
//! Handles JSON-RPC requests (initialize, ping, tools/list, tools/call) on `POST /mcp`.
//! Auth reuses `GATEWAY_USER_TOKENS` — each MCP client authenticates as a specific lens,
//! getting scoped tool access.

pub mod dispatch;
pub mod handler;
pub mod translate;

pub use handler::mcp_handler;
