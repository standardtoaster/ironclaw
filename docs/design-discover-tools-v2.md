# Design: Tool Discovery & Management — North Star

**Author:** Row 6 (discover-tools), synthesizing row-11 (skill-tool), row-13 (MCP + unified pipeline)
**Date:** 2026-03-24
**Status:** Architectural proposal — for lead review

## The Problem: 4 Turns vs 0

Row-13 measured the cost of using a configured MCP service tool in IronClaw today:

| Turn | IronClaw | Claude Code |
|------|----------|-------------|
| 1 | User asks for something | User asks for something |
| 2 | LLM calls discover_tools(query="notion") | LLM calls notion_create_page(...) |
| 3 | LLM calls discover_tools(query="notion", load=true) | **Done.** |
| 4 | mark_discovered is a no-op — tool still invisible | — |
| 5 | LLM gives up or hallucinates | — |

**4 turns and failure vs 0 turns and success.** This is the core design failure. Everything else flows from it.

Claude Code's model: configured tools are **auto-registered at startup**, schemas are **deferred until first use** (via ToolSearch), and access control happens **at invoke time**. The model never "discovers" a tool it should already know about.

## Reference Model: Claude Code

Observed directly from running inside Claude Code:

**1. Auto-registration.** All tools — builtins, MCP servers, deferred tools — are registered at startup. No search-and-load flow. The model's first turn can use any tool.

**2. Deferred schemas.** Claude Code has ~40+ tools but most start as **names only** in a system reminder. When the model needs one, it calls `ToolSearch` to lazy-load the full JSON schema. After that, the tool is callable for the rest of the session.
- Compact token footprint: name is ~10 tokens vs ~200 for full schema
- Zero round-trips wasted on "search" — the model knows the tool exists
- Schema loaded on demand, persists for the session
- The model knows WHAT exists without paying the cost of HOW to call it

**3. "When NOT to use" guidance.** Bash says: "Do NOT use to run cat, grep when dedicated tools exist. Prefer Read over cat, Edit over sed." Anti-pattern guidance is more useful than positive descriptions.

**4. Permissions gate execution, not visibility.** The user approves/denies at call time. All tools visible. Visibility is never the security boundary (except for the deferred/loaded distinction).

**5. Context-aware activation via hooks.** Skills fire from file pattern and bash pattern matching. No explicit "search for skills" — the system injects context automatically.

### The Three Concerns Claude Code Separates

| Concern | Claude Code | IronClaw Today |
|---------|------------|----------------|
| **Awareness** (tool exists) | Always — names in prompt | CORE_TOOLS filters visibility |
| **Readiness** (schema available) | Deferred via ToolSearch | All-or-nothing |
| **Permission** (allowed to run) | Invoke-time approval | Visibility-time removal |

IronClaw uses visibility as both a security boundary (attenuation — correct) and a UX lever (CORE_TOOLS — wrong). This conflation creates the discover_tools problem.

## What We Have Today

### Three Independent Filtering Mechanisms

1. **CORE_TOOLS env var** — static whitelist at startup. Only named tools in LLM context. Applied first.
2. **Skill attenuation** — Installed (untrusted) skill active → drops to 8 read-only tools. Applied second.
3. **discover_tools** — keyword search over registry. `load=true` calls `mark_discovered()` — **a no-op stub**.

### Dead Code and Design Conflicts

- **mark_discovered** — empty body, no `discovered` field in ToolRegistry, `tool_definitions_core()` doesn't consult one
- **tools_prefix** — proposed for skill manifests but never implemented. Skills can't declare tool needs.
- **MCP tools swallowed by CORE_TOOLS** — `notion_create_page` needs exact name match; `CORE_TOOLS=notion` doesn't match
- **Three access control paths** — no shared interface, no composability

## Proposed Architecture

### Principle: Auto-Register → Deferred Schema → Invoke-Time Control

Replace "search → load → use" with "tools are there → model fetches schema when needed → policy checks at call time."

### Registration: Everything Auto-Registers at Startup

**Builtins** — registered as today (echo, time, json, http, memory_*, shell, etc.)

**MCP services** — auto-connect at startup for all configured servers. Tool names registered immediately. No discover_tools needed. This is how Claude Code handles MCP — tools exist from turn 1.

```toml
# percy.toml
[[mcp]]
name = "notion"
url = "http://localhost:3456"
auto_connect = true  # default: connect at startup, register tools immediately

[[mcp]]
name = "experimental-api"
auto_connect = false  # browsable via tool catalog, not auto-registered
```

**WASM extensions** — registered from filesystem/DB as today.

**Skill-provided tools** — when a skill activates, its `requires_tools` get promoted to schema-ready tier.

### Schema Delivery: Two Tiers

**Core tier** (full schema every turn): Frequently-used tools configured per-lens. The LLM can call these instantly — no extra round-trip.

**Deferred tier** (name + description only): Everything else. Listed in system prompt so the model knows they exist. Model calls `tool_info(name="X", detail="schema")` to get the full parameter schema, then calls the tool. One extra turn, same as Claude Code's ToolSearch.

```toml
# percy.toml — per-lens schema tiers
[[lens]]
name = "andrew"
core_tools = ["memory_search", "memory_write", "shell", "message", "create_job"]
# Full schemas every turn. Everything else deferred.

[[lens]]
name = "grace"
core_tools = ["memory_search", "memory_write", "message", "time"]
# Simpler set. All tools visible, fewer instant-callable.
```

**Key difference from current CORE_TOOLS:** Core tier controls **schema readiness**, not **visibility**. All tools are always visible. Grace sees the same tool names as Andrew — she just gets prompted to fetch schemas for the ones she rarely uses.

**Empty core_tools** → all tools get full schemas (backward compatible, current default).

**Env var CORE_TOOLS** → deployment-level override, merges with lens config.

### Schema Fetch = The New "Discovery"

When the model needs a deferred tool:

```
Turn 1: User asks "schedule a reminder for Thursday"
Turn 2: LLM sees "routine_create" in deferred list → calls tool_info(name="routine_create", detail="schema")
Turn 3: LLM calls routine_create({...}) with the now-known schema
```

**2 turns for a deferred tool. 1 turn for a core tool. 0 turns for search.**

Compare to today: discover_tools(query) → discover_tools(query, load=true) → mark_discovered no-op → failure. **4 turns, no result.**

The existing `tool_info` tool already supports schema lookup. It just needs to be the primary path for deferred tools, not a secondary "help" feature.

### Skill-Declared Tool Requirements

Skills promote tools to core tier when they activate:

```yaml
# SKILL.md metadata
activation:
  keywords: [schedule, routine, cron]
  requires_tools: [routine_create, routine_list, routine_update]
```

When a skill activates:
1. Its `requires_tools` are added to core tier for this turn (full schemas)
2. Subject to trust ceiling — Installed skills can only promote read-only tools
3. Tools contract back to deferred tier when skill deactivates

This is row-11's per-turn expansion recommendation. The model doesn't search for tools — skills bring their tools with them.

### Access Policy: Unified, Invoke-Time

Row-13's `ToolAccessPolicy` trait with pluggable backends:

```rust
#[async_trait]
trait ToolAccessPolicy: Send + Sync {
    /// Can this tool be called right now?
    async fn check(&self, tool_name: &str, ctx: &PolicyContext) -> Result<(), PolicyDenial>;
}

struct PolicyContext {
    user_id: String,
    active_skills: Vec<LoadedSkill>,
    tool_provenance: ToolProvenance,  // Builtin, Wasm, Mcp, Skill
}

enum PolicyDenial {
    TrustCeiling { min_trust: SkillTrust, tool: String },
    AdminBlocked { reason: String },
    ApprovalRequired,
    McpServerUnhealthy { server: String },
}
```

**Backends:**
- **SkillAttenuationPolicy** — security boundary. The ONE policy that removes tools from visibility entirely. Untrusted skills can't see dangerous tool schemas. This stays as-is — it's correct and conservative.
- **AdminPolicy** — hard blocks for specific deployments ("this instance cannot run shell"). Replaces CORE_TOOLS-as-deny-list use case.
- **ApprovalPolicy** — existing per-invocation user approval. Gates execution, not visibility.
- **HealthPolicy** — MCP server health check before execution. Stale cache fallback for home network resilience.

**Key principle:** Only SkillAttenuationPolicy controls visibility. Everything else is invoke-time.

### What Happens to Each Existing Piece

| Current | Becomes |
|---------|---------|
| **discover_tools** (search+load) | **Retired for configured services.** Auto-registration makes it unnecessary. Repurposed as `tool_info(query="...")` search mode for browsing internal tools, and potentially as an unconfigured-service catalog browser. |
| **mark_discovered** (no-op) | **Fixed as correctness bug** (15 lines), but superseded by the auto-register + deferred schema model. Per-conversation semantics while it exists. |
| **CORE_TOOLS env var** | **Repurposed:** controls schema readiness tier (core vs deferred), not visibility. Per-lens in percy.toml. Env var is deployment-level override. |
| **tool_info** | **Promoted:** becomes the primary schema-fetch mechanism for deferred tools. Adds `query` parameter for search mode (absorbs discover_tools search). |
| **Skill attenuation** | **Preserved as-is.** Correct security model. Only policy that controls visibility. |
| **MCP tool registration** | **Auto-connect at startup** for configured servers. No discover → load flow. Tools available from turn 1. |

### Percy-Specific Additions (Claude Code Doesn't Need These)

**Per-lens scoping.** Claude Code has one user. Percy has multiple lenses with different needs. The per-lens core_tools config is Percy-specific — it determines which tools get full schemas for each lens.

**Stale cache fallback.** Percy runs on a home network where MCP servers may be intermittently unavailable. Deferred schemas should cache the last-known-good schema so a brief server outage doesn't break tool calling. Claude Code runs in Anthropic's infrastructure and doesn't need this.

**ServiceRegistry config.** Percy's percy.toml already defines MCP services. The auto-connect model reads from this config. Claude Code uses `claude mcp add` CLI for persistent config.

## Flow Comparison

### Today: Configured MCP Tool (4+ turns, fails)

```
User: "Create a Notion page for the grocery list"
LLM:  [doesn't know notion_create_page exists — not in CORE_TOOLS]
LLM:  calls discover_tools(query="notion")
      → returns: "Found 1 tool: notion_create_page: ..."
LLM:  calls discover_tools(query="notion", load=true)
      → returns: "Loaded 1 tool" [LIE — mark_discovered is no-op]
LLM:  tries to call notion_create_page(...)
      → ERROR: tool not in definitions [still hidden by CORE_TOOLS]
LLM:  "I'm unable to create the page directly..."
```

### Proposed: Configured MCP Tool (0-1 turns, succeeds)

**If notion_create_page is in core tier (0 turns):**
```
User: "Create a Notion page for the grocery list"
LLM:  calls notion_create_page({title: "Grocery List", ...})
      → policy check passes → executes → success
```

**If notion_create_page is in deferred tier (1 turn):**
```
User: "Create a Notion page for the grocery list"
LLM:  sees "notion_create_page" in deferred tools list
LLM:  calls tool_info(name="notion_create_page", detail="schema")
      → returns full parameter schema
LLM:  calls notion_create_page({title: "Grocery List", ...})
      → policy check passes → executes → success
```

### Proposed: Skill-Activated Tool (0 turns)

```
User: "Set up a weekly reminder to check the garden"
      [skill selector scores routine-advisor → activates]
      [routine-advisor declares requires_tools: [routine_create]]
      [routine_create promoted to core tier for this turn]
LLM:  calls routine_create({trigger_type: "cron", schedule: "0 9 * * 1", ...})
      → success, zero discovery overhead
```

## Implementation Priority

### Phase 1: Fix What's Broken (this branch, now)

1. **Fix mark_discovered** — 15 lines, per-conversation. Tests written. Correctness bug even though long-term plan supersedes it.
2. **Add search mode to tool_info** — `tool_info(query="schedule")` returns matching tools. This is the first step toward absorbing discover_tools.
3. **Add `requires_tools` to SkillManifest** — parse from SKILL.md. Wire into dispatcher: when skill activates, its tools get full schemas.

### Phase 2: Auto-Register + Deferred Schemas

4. **MCP auto-connect at startup** — configured servers connect and register tools immediately. No discover flow needed.
5. **Implement schema tiers** — `tool_definitions_core` returns full schemas for core tier, name+description stubs for deferred tier. `tool_info` fetches full schema on demand.
6. **Per-lens core_tools in percy.toml** — lens config for schema tiers.
7. **Retire discover_tools as primary flow** — keep as `tool_info` search mode for browsing. Remove as separate tool.

### Phase 3: Unified Access Policy

8. **ToolAccessPolicy trait** — shared interface for attenuation, admin, approval, health.
9. **Invoke-time policy checks** — all policies except attenuation move to execution time.
10. **Tool provenance tracking** — tag tools with source (builtin/wasm/mcp/skill).
11. **MCP health-aware execution** — stale cache fallback for intermittent servers.

### Phase 4: Quality

12. **Tool descriptions** — "when NOT to use" guidance, disambiguation, approval hints.
13. **Discovery summaries** for complex tools (shell, http, create_job, memory_write, message).
14. **Tool categories in system prompt** — semantic grouping of flat list.

## What This Should NOT Become

- **Not a search-and-load system.** The 4-turn failure is the anti-pattern. Auto-register + deferred schema eliminates it.
- **Not a visibility-based security system** (except attenuation). Invoke-time policy is the right model.
- **Not a recommendation engine.** Good descriptions + skill requirements > "you might also like."
- **Not over-engineered for 40 tools.** ~~Deferred schemas matter at 100+. At 40, send everything.~~ **CORRECTED by benchmark data:** local models degrade significantly starting at 15 tools. Schema tiering matters now, not at 100+.
- **Not a marketplace browser.** Runtime tool discovery (tool_info) is separate from extension installation (registry catalog). Don't merge them.

## Benchmark Data: Tool Count Degradation

Empirical data from `tests/bench-tool-count.py` measuring tool selection accuracy across three model variants at increasing tool counts. Each tier was tested with 3 runs per task across 10-13 scenarios (simple selection, disambiguation, no-tool-needed, MCP tools).

### Accuracy by Tool Count

| Tier | Base Qwen3-30B-A3B | overlay-d (light fine-tune) | percy-instruct (full fine-tune) |
|------|-------------------|---------------------------|-------------------------------|
| 5    | 100.0%            | 100.0%                    | 100.0%                        |
| 10   | 100.0%            | 100.0%                    | 100.0%                        |
| 15   | 90.0%             | 90.0%                     | 90.0%                         |
| 20   | **100.0%**        | 80.0%                     | 80.0%                         |
| 30   | **90.0%**         | 86.7%                     | 60.0%                         |
| 40   | **90.9%**         | 81.8%                     | 81.8%                         |
| 50   | **75.0%**         | 55.6%                     | 47.2%                         |
| 80   | **84.6%**         | 76.9%                     | 71.8%                         |

### MCP Tool Accuracy (unfamiliar tool names)

| Tier | Base Qwen | overlay-d | percy-instruct |
|------|-----------|-----------|----------------|
| 40   | 100%      | 0%        | 0%             |
| 50   | 50%       | 0%        | 0%             |
| 80   | 100%      | 67%       | 33%            |

### Key Findings

1. **All models are perfect at ≤10 tools.** The 10-tool core tier is empirically validated.

2. **First failures at 15 tools, cliff at 50.** Breakpoints detected at 15 (-10%), 30 (-10 to -30%), and 50 (-16 to -35%) depending on model.

3. **Fine-tuning hurts tool generalization.** Base Qwen outperforms both fine-tuned variants at every tier ≥20. The more fine-tuning, the worse the accuracy with unfamiliar tools. percy-instruct drops to 47% at 50 tools where base Qwen holds 75%.

4. **MCP tools expose the overfitting most clearly.** percy-instruct scores 0% on MCP tasks at 40 and 50 tools. Base Qwen scores 100% and 50%. The fine-tuning overfit to core IronClaw tool names, making the model rigid with unknown schemas.

5. **Zero hallucinations across all models.** No model invented tool names — failures are wrong-tool selection or failure-to-call, not fabrication.

6. **no_tool tasks are bulletproof.** 100% across all tiers and models — none spuriously call tools.

### The Fine-Tuning Trade-Off: Generalization vs Personality

percy-instruct was fine-tuned for Percy-specific behaviors: memory interaction patterns, conversational tone, IronClaw-specific tool idioms. This improves the *conversational* experience but degrades *tool generalization* — the model becomes rigid, preferring familiar tool names over unfamiliar ones.

This creates a tension: we want Percy's personality (percy-instruct) but need general tool-calling ability (base Qwen) when MCP tools or extensions enter the picture.

**Row-12 model-escalation solves this.** The tier system (local → escalation → cloud) can route by *task type*, not just complexity:

- **Conversational turns** (memory recall, chat, personality) → percy-instruct (fine-tuned personality, works perfectly with ≤10 core tools)
- **Tool-heavy turns** (MCP tools, many schemas, unfamiliar tools) → base Qwen or escalation model (better generalization at high tool counts)
- **Complex reasoning** → cloud tier (Claude, etc.)

This eliminates the trade-off entirely: the fine-tuned model handles what it's good at (personality + core tools), and routing shifts tool-heavy work to models that generalize better. The deferred schema approach amplifies this — core tier stays at ≤10 tools where all models are perfect, and the rare deferred-schema path can trigger escalation awareness.

### Design Validation

The benchmark confirms:

- **CORE_TOOLS filtering is necessary**, not just a token optimization — accuracy drops materially above 10 tools
- **Deferred schemas are the right abstraction** — keep core tier small, expand on demand
- **"Not over-engineered for 40 tools"** was wrong — the data shows significant degradation starting at 15 tools for local models, making schema tiering important even at current scale
- **MCP auto-registration needs schema tiering** — blindly adding MCP tool schemas to every turn would degrade tool selection for the exact tools we're trying to enable

## Open Questions

1. **Attenuation: visibility-time vs invoke-time?** Currently removes tools entirely from schema (stronger security — LLM can't attempt to call what it can't see). But the model can't tell the user "I'd need shell access but my permissions don't allow it." Trade-off: security vs transparency. **Recommendation: keep visibility-time removal for attenuation.** It's the right call for untrusted code.

2. **Schema fetch: auto-expand or explicit?** When the model calls `tool_info` for a deferred tool's schema, should the schema automatically become available for the rest of the conversation (like Claude Code's ToolSearch), or should the model include the schema in its call? **Recommendation: auto-expand.** Once fetched, the tool moves to core tier for the rest of the conversation. Less friction.

3. **discover_tools retirement timeline?** Can we absorb it into tool_info in Phase 1, or does backward compat require keeping both? **Recommendation: add search to tool_info in Phase 1, deprecate discover_tools in Phase 2, remove in Phase 3.**

## Summary

The north star is **0-1 turns to use any configured tool**, matching Claude Code. The path:

1. **Auto-register** configured services at startup (not discover → load)
2. **Deferred schemas** for token efficiency (name + description always visible, full schema on demand via tool_info)
3. **Invoke-time access control** via unified ToolAccessPolicy (except attenuation, which stays visibility-time for security)
4. **Skill-declared requirements** promote tools to core tier automatically when skills activate

discover_tools becomes a niche browsing tool, not the primary path. CORE_TOOLS becomes a schema tier selector, not a visibility filter. The model always knows what exists — it just fetches schemas when needed.
