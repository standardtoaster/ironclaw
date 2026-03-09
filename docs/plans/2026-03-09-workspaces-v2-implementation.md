# Workspaces v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the three blockers from the UX review, then build topic context switching and cross-workspace search.

**Architecture:** Workspaces run through the full dispatcher path (identity, skills, conversation history), not through the bare Worker. The router sits in front of the dispatcher for transparent context switching. Cross-workspace search tools let the LLM find information across all conversations.

**Tech Stack:** Rust, PostgreSQL + pgvector, tokio async, IronClaw's existing dispatcher/agentic loop.

**Design doc:** `docs/plans/2026-03-08-workspaces-design.md` (v2)

**Prior work:** Phase 1 implementation is complete (workspace store, router, queue, delegation tool, unit + integration tests). See `docs/plans/2026-03-08-workspaces-implementation.md` for what was built.

---

## Phase 1: Fix Blockers (conversation resume must actually work)

### Task 1: Load conversation history in workspace jobs

The critical blocker. `dispatch_job_to_conversation` creates a job with `conversation_id` set, but `Worker::run` ignores it. The workspace job starts with a blank slate every time.

**Files:**
- Modify: `src/agent/worker.rs` — load conversation history when `conversation_id` is set
- Modify: `src/agent/scheduler.rs` — pass conversation history to worker, or let worker load it

**What needs to happen:**

When a Worker starts and its `JobContext.conversation_id` is `Some(id)`:
1. Load messages from DB via `store.list_conversation_messages(id)`
2. Filter to `role=user` and `role=assistant` (skip `tool_calls` metadata rows)
3. Prepend these as the initial conversation history before the current prompt
4. The current prompt (from `job_ctx.description`) should NOT be duplicated — it was already appended to the conversation by `dispatch_job_to_conversation`

**Step 1: Write a test**

In `src/agent/worker.rs` tests (or a new integration test), verify that when a Worker runs with `conversation_id` set, the LLM receives prior conversation messages.

Since Worker requires an LLM provider, this needs either a mock LLM or a test that checks the messages list construction without actually calling the LLM.

**Step 2: Modify `Worker::run`**

After loading `job_ctx`, check if `conversation_id` is set. If so, load messages and prepend them to the reasoning context.

```rust
let mut initial_messages = Vec::new();
if let Some(conv_id) = job_ctx.conversation_id {
    if let Ok(messages) = self.store().list_conversation_messages(conv_id).await {
        for msg in messages {
            match msg.role.as_str() {
                "user" => initial_messages.push(ChatMessage::user(msg.content)),
                "assistant" => initial_messages.push(ChatMessage::assistant(msg.content)),
                _ => {} // skip tool_calls metadata
            }
        }
    }
}
```

Then use `initial_messages` as the starting context instead of an empty vec.

**Step 3: Verify — run integration test**

Run: `cargo test --test workspace_routing_integration -- --test-threads=1`

**Step 4: Commit**

```
feat: load conversation history in workspace jobs
```

---

### Task 2: Workspace system prompt with identity context

The workspace Worker gets a bare "You are an autonomous agent" prompt. It needs identity docs, workspace topic context, and awareness that it's a persistent workspace.

**Files:**
- Modify: `src/agent/worker.rs` — build a richer system prompt for workspace jobs
- Modify: `src/agent/scheduler.rs` — pass workspace context (topic, turn_count) via job metadata

**What needs to happen:**

When `dispatch_job_to_conversation` creates a job for a workspace, include workspace metadata:

```rust
let metadata = serde_json::json!({
    "workspace_id": workspace_id,
    "workspace_topic": workspace_topic,
    "workspace_turn_count": workspace_turn_count,
    "delegation_depth": depth + 1,
});
```

In `Worker::run`, when workspace metadata is present, build a system prompt that includes:
1. Identity docs from the user's workspace (SOUL.md, USER.md, etc.) — requires access to the `Workspace` object or having the content pre-loaded in metadata
2. Workspace context: topic, turn count
3. Instructions about being a persistent workspace

**Pragmatic approach for now:** Load identity docs via the `Workspace` object in the Worker. The Worker already has access to `store` (which has the pool). Build a `Workspace` for the user and call `system_prompt()`.

If the `Workspace` struct isn't accessible from Worker (it's in the `workspace` module), factor out system prompt loading into a function that takes a store/pool + user_id and returns the system prompt string.

**Step 1: Add workspace identity to job metadata or Worker**

The simplest path: in `WorkspaceQueueManager::process_message`, read the user's identity docs and pass them in the job metadata. This avoids coupling Worker to the Workspace module.

**Step 2: Modify Worker system prompt construction**

When `job_ctx.metadata` contains `workspace_topic`:

```
{identity_docs}

---

## Workspace Context
You are in a persistent workspace focused on: {topic}
This conversation has {turn_count} prior turns. The full conversation history
is loaded — you can reference anything discussed previously.

When you finish a task, provide a clear summary of what was done and any
decisions made, since the caller may only see your final response.
```

**Step 3: Test — verify system prompt includes identity**

**Step 4: Commit**

```
feat: workspace jobs get identity docs and topic context
```

---

### Task 3: Auto-set workspace topic on creation

Currently the workspace topic stays empty unless the LLM calls `set_workspace_topic`. This means the workspace can never be routed to again.

**Files:**
- Modify: `src/tools/builtin/delegate_workspace.rs` — set topic immediately on creation
- Modify: `src/agent/workspace_queue.rs` — optionally update topic after each turn

**What needs to happen:**

When `delegate_to_workspace` creates a new workspace (the `Ok(None)` branch in routing):
1. Use the `prompt` + `workspace_hint` as the initial topic text
2. Embed it immediately
3. Call `update_agent_workspace_topic(ws.id, topic_text, &embedding)`

This ensures the workspace is routable from its first creation.

**Step 1: Modify the creation branch**

After `create_agent_workspace`, embed and set the topic:

```rust
let topic_text = if let Some(hint) = workspace_hint {
    format!("{}: {}", hint, prompt)
} else {
    prompt.to_string()
};
// Truncate to reasonable length for embedding
let topic_for_embedding = &topic_text[..topic_text.len().min(500)];
let embedding = self.router.embed(topic_for_embedding).await?;
self.db.update_agent_workspace_topic(ws.id, topic_for_embedding, &embedding).await?;
```

Note: `WorkspaceRouter` needs a public `embed()` method (currently it embeds internally in `route`). Add a thin wrapper.

**Step 2: Add `embed` method to WorkspaceRouter**

```rust
impl WorkspaceRouter {
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, RouterError> {
        self.embedder.embed(text).await.map_err(RouterError::Embedding)
    }
}
```

**Step 3: Test — verify new workspaces have topic set**

Add to integration tests: create via delegate, then verify workspace has non-empty topic and is routable.

**Step 4: Commit**

```
feat: auto-set workspace topic on creation
```

---

## Phase 2: Topic Context Switching (Feature 1)

### Task 4: Pre-dispatcher routing hook

Route incoming messages to the right workspace BEFORE the dispatcher runs. The workspace conversation loads via the existing `maybe_hydrate_thread()` → `run_agentic_loop()` path.

**Files:**
- Modify: `src/agent/agent_loop.rs` — add routing step in `handle_message`
- Possibly modify: `src/agent/session_manager.rs` — ensure workspace conversations are tracked

**What needs to happen:**

In `Agent::handle_message()`, after initial message parsing but before `process_user_input()`:

1. If the message doesn't already have a `thread_id` (new conversation):
   a. Embed the message content
   b. Query the workspace router for a match
   c. If match found: set `message.thread_id = Some(workspace.conversation_id.to_string())`
   d. Touch the workspace (update `last_accessed`, increment `turn_count`)
2. If the message has a `thread_id`: don't reroute (respect explicit thread selection)

The existing `maybe_hydrate_thread()` handles loading DB history when a `thread_id` is present. So the routing is just: pick the right `thread_id`.

**Considerations:**
- The router needs access to the embedding provider and DB — these need to be on the `Agent` struct (or passed in)
- Should be behind a feature flag initially (`WORKSPACE_ROUTING_ENABLED=true`)
- What if routing picks wrong? The user needs an escape hatch (explicit workspace selection)
- Performance: embedding + DB query on every message. Should be fast (< 100ms) but measure.

**Step 1: Add router to Agent struct**

Store `Option<Arc<WorkspaceRouter>>` on `Agent`. Initialize during startup if embeddings are enabled.

**Step 2: Add routing step in handle_message**

```rust
// After message parsing, before process_user_input
if message.thread_id.is_none() {
    if let Some(router) = &self.workspace_router {
        if let Ok(Some(ws)) = router.route(&message.user_id, &message.content).await {
            message.thread_id = Some(ws.conversation_id.to_string());
            // Touch workspace
            let _ = self.store().touch_agent_workspace(ws.id).await;
        }
    }
}
```

**Step 3: Test — send messages and verify routing**

**Step 4: Commit**

```
feat: pre-dispatcher workspace routing for automatic context switching
```

---

### Task 5: Workspace context injection in dispatcher

When the dispatcher runs for a workspace conversation, inject workspace-specific context into the system prompt.

**Files:**
- Modify: `src/agent/dispatcher.rs` — detect workspace context, inject topic info
- Possibly: lookup workspace by conversation_id

**What needs to happen:**

In `run_agentic_loop`, after building the system prompt:
1. Check if the current `thread_id` corresponds to a workspace
2. If so, append workspace context (topic, turn count) to the system prompt
3. Make `set_workspace_topic` and `search_workspace_history` available as tools

This requires a way to look up a workspace by `conversation_id`. Add:

```rust
async fn get_agent_workspace_by_conversation(&self, conversation_id: Uuid)
    -> Result<Option<AgentWorkspace>, DatabaseError>;
```

**Step 1: Add DB method**

**Step 2: Inject context in dispatcher**

**Step 3: Test**

**Step 4: Commit**

```
feat: inject workspace context into dispatcher system prompt
```

---

## Phase 3: Cross-Workspace Intelligence (Feature 3)

### Task 6: `search_workspace_history` tool

Search across all workspace conversation histories. This is the key tool that makes the single-surface model work.

**Files:**
- Create: `src/tools/builtin/search_workspace_history.rs`
- Modify: `src/tools/registry.rs` — register the tool

**Tool schema:**

```json
{
  "query": "string — what to search for",
  "workspace_id": "uuid? — limit to specific workspace",
  "limit": "integer? — max results (default 10)"
}
```

**Implementation:**

Search conversation messages across all workspaces for the user:

```sql
SELECT m.content, m.role, m.created_at, w.topic, w.id as workspace_id
FROM conversation_messages m
JOIN agent_workspaces w ON w.conversation_id = m.conversation_id
WHERE w.user_id = $1
  AND m.role IN ('user', 'assistant')
  AND m.content ILIKE '%' || $2 || '%'
ORDER BY m.created_at DESC
LIMIT $3
```

For v1, simple text search (ILIKE) is fine. Later, add FTS and/or vector search on conversation messages.

**Output format:**

```
[Workspace: home automation (3 days ago)]
User: set up porch light automation with motion sensor
Assistant: I've configured the Hue motion sensor with 3-minute timeout...

[Workspace: grocery (1 day ago)]
User: add motion sensor batteries to the list
Assistant: Added "motion sensor batteries (CR2450)" to your grocery list.
```

**Step 1: Implement tool**

**Step 2: Add to registry**

**Step 3: Test with integration test**

**Step 4: Commit**

```
feat: search_workspace_history tool for cross-workspace search
```

---

### Task 7: `workspace_summary` tool

Generate a condensed summary of a workspace's conversation history.

**Files:**
- Create: `src/tools/builtin/workspace_summary.rs`
- Modify: `src/tools/registry.rs`

**Implementation:** Load all messages from the workspace's conversation, format them as a readable timeline, truncate to a reasonable length.

For v1, don't use an LLM to summarize — just return the last N turns as formatted text. LLM summarization can be added later.

---

## Phase 4: Fix Delegation UX

### Task 8: Stream workspace events during delegation

During `delegate_to_workspace`, the user sees silence for minutes. Stream the workspace's tool events and thinking to the caller's SSE connection.

**Files:**
- Modify: `src/agent/workspace_queue.rs` — pipe workspace job events to caller
- Modify: `src/channels/web/types.rs` — add workspace-specific SSE event types

**What needs to happen:**

When the workspace job runs (via `start_processing`), its events should be forwarded to the SSE broadcaster with a `workspace_id` prefix. The web UI can then show "Workspace (home automation): calling memory_search..." etc.

This is a nice-to-have — functional correctness comes first.

---

### Task 9: Fix timeout consistency

Remove the dual timeout (300s inner / 600s outer). Use a single configurable timeout.

**Files:**
- Modify: `src/tools/builtin/delegate_workspace.rs` — remove or align timeouts
- Modify: `src/agent/workspace_queue.rs` — use consistent timeout

---

## Phase 5: Validation

### Task 10: Conversation resumption test harness

Build a test that validates the core promise: workspace conversations actually remember prior context.

**Approach:** Multi-turn scripted test against a running IronClaw with Postgres.

```
Turn 1: "My porch light is a Philips Hue model 123. Set up motion detection."
  → Workspace created, conversation saved

Turn 2: "What model is my porch light?"
  → Should route to same workspace
  → Response should reference "Philips Hue model 123" from Turn 1
  → This proves conversation history was loaded
```

**Files:**
- Create/modify: `tests/test_workspaces_e2e.py` — add resumption tests

**Signals to measure:**
- Does the response contain the specific detail from the prior turn?
- Was the same workspace routed to (check via `list_workspaces` turn count)?
- How long did the response take (latency of hydration + routing)?
- Did the system prompt include workspace context?

---

### Task 11: UX review scenario tests

Test the full user experience for common scenarios:

1. **Topic switch:** Talk about groceries, then home automation, then back to groceries. Does the context switch seamlessly?
2. **Cross-reference:** Ask a question that spans workspaces ("did I already add motion sensor batteries to the grocery list?"). Does `search_workspace_history` find it?
3. **New topic detection:** Say something the router has never seen. Does it create a new workspace or land in the main thread?
4. **Wrong routing recovery:** The router picks the wrong workspace. Can the user correct it?

---

## Dependency graph

```
Task 1 (load history) ──┐
Task 2 (system prompt) ──┼── Task 4 (pre-dispatcher routing) ── Task 5 (context injection)
Task 3 (auto-topic) ────┘                                            │
                                                                      │
Task 6 (search history) ──── Task 7 (workspace summary)              │
                                                                      │
Task 8 (stream events) ── Task 9 (fix timeouts)                      │
                                                                      │
                                                   Task 10 (resumption tests)
                                                   Task 11 (UX scenarios)
```

Tasks 1-3 are blockers — everything else depends on them.
Tasks 4-5 are Feature 1 (context switching).
Tasks 6-7 are Feature 3 (cross-workspace intelligence).
Tasks 8-9 are delegation UX polish.
Tasks 10-11 are validation.
