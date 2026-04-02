# Claude Container Provider — Integration Spec

## Overview

The `claude_container` LLM backend routes user messages to a local Claude Code process managed by a compute supervisor. It is activated per-user via `GATEWAY_USER_TOKENS`.

## Components

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐     ┌───────────────┐
│  IronClaw   │     │  Supervisor   │     │ Claude Code    │     │ percy-channel │
│  Gateway    │────▶│  (Python)     │────▶│ (native)       │────▶│ (MCP/Node.js) │
│             │◀────│               │     │                │◀────│               │
└─────────────┘     └───────────────┘     └────────────────┘     └───────────────┘
```

## Data Flow — Message Lifecycle

### Inbound (user → Claude)

1. **HTTP request** arrives at gateway with `Authorization: Bearer tok-andrew`
2. **Auth middleware** maps token → `user_id=andrew` via `GATEWAY_USER_TOKENS`
3. **Agent loop** calls `active_llm("andrew")` → returns `ClaudeContainerProvider`
4. **Dispatcher** calls `complete_with_tools()` on the per-user provider
5. **ClaudeContainerProvider.backend()** lazily initializes `SupervisorBackend`
6. **SupervisorBackend.get_or_create()** → `POST {supervisor_url}/claude/sessions`
   - Passes `callback_url` (IronClaw gateway URL) and `thread_id`
   - Supervisor starts Claude Code process with percy-channel MCP
   - Returns `session_id` and `channel_url` (percy-channel HTTP port)
7. **SupervisorBackend.exchange()** →
   - Inserts `(thread_id, oneshot::Sender)` into **shared** `pending_replies` map
   - `POST {channel_url}/message` with `{content, thread_id}`
   - Awaits `oneshot::Receiver` with timeout

### Outbound (Claude → user)

8. **percy-channel** receives `/message`, writes to Claude's stdin via pty
9. **Claude** processes, calls the `reply` MCP tool
10. **percy-channel** `reply` tool handler → `POST {PERCY_CALLBACK_URL}/api/claude/reply`
    - Payload: `{thread_id, content, session_id, input_tokens, output_tokens}`
11. **Gateway** `/api/claude/reply` handler:
    - Looks up `thread_id` in **shared** `pending_claude_replies` map
    - Resolves the `oneshot::Sender` with the reply content
12. **SupervisorBackend.exchange()** receives reply via oneshot, returns to provider
13. **Dispatcher** gets response, agent loop calls `channels.respond()`
14. **GatewayChannel.respond()** broadcasts SSE event with `thread_id`

## Contracts

### C1: Per-User Provider Resolution

**Invariant:** `active_llm(user_id)` returns the provider configured in `GATEWAY_USER_TOKENS` for that user, not the global default.

**Test:** Create an Agent with `user_llm_providers` containing a distinct provider for "andrew". Assert `active_llm("andrew")` returns it. Assert `active_llm("grace")` returns global.

### C2: Per-User Provider Takes Priority Over Tier Map

**Invariant:** When both `user_llm_providers` and `tier_map` are configured, per-user wins.

**Test:** Create Agent with both. Assert per-user provider for "andrew", tier map default for "grace".

### C3: Shared Pending Replies Map

**Invariant:** The `SupervisorBackend` and `GatewayState` reference the **same** `pending_replies` map. A reply inserted by the backend during `exchange()` must be resolvable by the gateway's `/api/claude/reply` handler.

**Test:** Create a shared `PendingRepliesMap`. Create `SupervisorBackend::new_with_shared_pending()` with it. Insert a pending reply via the backend. Resolve it via the shared map (simulating the gateway handler). Verify the oneshot delivers.

### C4: Non-Shared Map Isolation (Regression Guard)

**Invariant:** `SupervisorBackend::new()` (without shared map) creates an isolated map. A gateway handler using a different map will NOT find pending replies.

**Test:** Create backend with `new()` (own map). Create a separate map. Insert via backend, check separate map — should be empty.

### C5: Callback URL Propagation

**Invariant:** The `callback_url` from `ClaudeContainerProvider` config reaches percy-channel as `PERCY_CALLBACK_URL` env var. The callback URL must be the IronClaw gateway, not the supervisor.

**Test:** Verify `ContainerProviderConfig.callback_url` is set from `CLAUDE_CONTAINER_CALLBACK_URL` env var. Verify it flows through `create_session` body.

### C6: Thread ID Backfill

**Invariant:** When a client sends a message without `thread_id`, the agent loop resolves the thread internally. Before calling `channels.respond()`, `message.thread_id` must be backfilled so `GatewayChannel::respond()` can broadcast.

**Test:** Send a message with `thread_id=None`. After `handle_message`, `message.thread_id` should be `Some(uuid)`.

### C7: Graceful Shutdown

**Invariant:** On SIGINT, all per-user providers' `shutdown()` is called. For `claude_container`, this calls `SupervisorBackend.shutdown_all()` → `DELETE /claude/sessions/{id}` for each session.

**Test:** Create a provider, verify `shutdown()` calls through to `shutdown_all()`. Verify the shutdown section in main.rs fires.

### C8: `create_provider_from_user_config` — Backend Dispatch

**Invariant:** `claude_container` backend creates a `ClaudeContainerProvider`. `openai_compatible` creates a RigAdapter. Unknown backends return an error.

**Test:** Call with each backend type, verify provider type and model name.

## Wiring in main.rs

1. Gateway is built, `pending_claude_replies` map is grabbed via `gw.pending_claude_replies()`
2. User tokens are iterated, `create_provider_from_user_config_with_context(cfg, Some(pending_replies))` creates providers with shared map
3. Providers are stored in `AgentDeps.user_llm_providers`
4. A clone is kept as `user_providers_for_shutdown`
5. On shutdown, iterate providers and call `shutdown()`
