# Skill Discovery and Scoping Design Proposal

**Author:** row-11 (skill-activation)
**Date:** 2026-03-24
**Status:** Draft
**Informs:** skill-activation (row 11), discover-tools (row 6), Percy deployment config

---

## 1. Problem Statement

Percy's skill system works but has fundamental design gaps compared to proven patterns
in Claude Code. The current system:

- **Loads eagerly, activates by keyword matching.** Every registered skill is scored
  against every message. The LLM has no say in activation.
- **Bloats context.** Up to 3 skills × 4000 tokens = 12K tokens injected into the
  system prompt every turn, regardless of relevance.
- **No lazy loading.** Skills are fully loaded at boot. There's no "I know this exists
  but haven't read it yet" state.
- **tools_prefix is a dead feature.** The field parses and round-trips but nothing
  reads it at runtime.
- **No hook-based activation.** Skills can't fire based on what the agent is *doing*
  (which tool it's calling, which file it's reading) — only on what the user *said*.

Meanwhile, Percy's multi-lens architecture creates scoping requirements that Claude Code
doesn't face: member vs group lenses, consent boundaries, and per-user tool surfaces.

---

## 2. How Claude Code Does It

Observations from running inside Claude Code (this proposal was written by an agent
running in Claude Code, so the patterns are directly observable):

### 2.1 Skill Catalog as System Metadata

Claude Code lists all available skills in a system-reminder block with one-line
trigger descriptions:

```
- simplify: Review changed code for reuse, quality, and efficiency...
- ship: Run the full Rust quality gate (fmt, clippy, tests)...
- trace: Trace a data flow or bug through the IronClaw codebase end-to-end
```

The model sees ~50 skills listed by name and trigger. Total cost: ~2K tokens of
metadata, not 12K of injected content. The model can browse this list and decide
which skills are relevant — it's not pre-decided by keyword scoring.

### 2.2 On-Demand Loading via Skill Tool

Skills are loaded by the model calling `Skill("ship")`. The full skill content is
only injected when explicitly requested. This is the key architectural difference:

| Aspect | Percy | Claude Code |
|--------|-------|-------------|
| Discovery | Keyword match at message time | Model reads catalog, decides |
| Loading | All matching skills injected | Skill content loaded on-demand |
| Context cost | 3 × 4000 = 12K tokens always | 0 tokens until invoked |
| Model agency | None (deterministic filter) | Full (model decides what to load) |

### 2.3 Hook-Based Activation

Claude Code's PreToolUse hooks fire when the model is about to use a tool. The hook
matches the tool's target (file path or command) against skill patterns:

- Editing `*.tsx` → injects `react-best-practices` skill
- Running `npm` → injects `vercel-cli` guidance
- Reading a Next.js file → injects `nextjs` patterns

This is *contextual* activation — skills fire based on what the agent is doing, not
what the user asked for. The hook system:

1. Matches file/bash patterns from skill frontmatter
2. Injects matched skill content as `additionalContext` on the tool response
3. Deduplicates per session (each skill injected at most once)
4. Caps at 3 skills per hook invocation, sorted by priority

### 2.4 Deferred Tools (Lazy Loading)

Claude Code's `ToolSearch` tool provides lazy tool loading. Tools are listed by name
in system reminders but their schemas aren't loaded until the model calls ToolSearch.
This keeps the base context small while making the full tool surface discoverable.

### 2.5 Skill Types and Composition

Skills self-declare as rigid or flexible:
- **Rigid** (TDD, debugging): follow exactly, don't adapt
- **Flexible** (patterns, architecture): adapt principles to context

Multiple skills compose per task with explicit priority ordering:
1. Process skills first (brainstorming, debugging)
2. Implementation skills second (frontend-design, mcp-builder)

### 2.6 Skills Change Behavior, Not Just Add Tools

Claude Code skills inject detailed behavioral instructions — not just tool schemas.
A skill like `systematic-debugging` changes the model's entire approach to the task:
stop guessing, reproduce first, form hypotheses, verify. This is fundamentally
different from "here are some extra tools."

---

## 3. What Percy Should Adopt

### 3.1 Skill Catalog in System Prompt (Adopt)

**Current:** Skills are invisible until they activate. The LLM doesn't know what
skills exist.

**Proposed:** Inject a lightweight skill catalog into the system prompt:

```xml
<available-skills>
  <skill name="deploy-helper" trigger="deployment, release, ship" scope="andrew"/>
  <skill name="grocery-tracker" trigger="grocery, shopping list" scope="grace"/>
  <skill name="household-planner" trigger="schedule, calendar, plan" />
</available-skills>
```

Cost: ~50-100 tokens for Percy's 5-10 skills. The LLM can see what's available
without loading full content.

**Implementation:** Generate the catalog block in `build_system_prompt_with_tools()`
from the skill registry. Include name, trigger keywords, and scope (so the LLM
understands which skills are available to the current user).

### 3.2 On-Demand Skill Loading (Adopt with Adaptation)

**Current:** `prefilter_skills()` scores and injects skills every turn.

**Proposed:** Two activation paths:

1. **Model-driven:** The LLM calls a `activate_skill` tool to load a skill by name.
   The skill's content is injected into the next system prompt rebuild. This replaces
   keyword-based injection for interactive sessions.

2. **Auto-activation (keep, but demote):** Keyword matching remains as a fallback for
   cases where the model fails to activate the right skill, and for non-interactive
   contexts (heartbeat, routines, background jobs) where the LLM doesn't get a
   browsing step. But auto-activation should be secondary, not primary.

**Why both:** Percy's LLMs (local Qwen, MLX models) are less capable than Claude at
self-directed tool use. They may not reliably decide to invoke `activate_skill` when
appropriate. Keyword matching provides a safety net until model quality improves. But
for Anthropic/OpenAI backends, model-driven activation should be preferred.

### 3.3 Hook-Based Activation (Adopt, Percy-Specific)

**Current:** No equivalent.

**Proposed:** Add activation hooks to the IronClaw Hook system (which already has
6 hook points: BeforeInbound, BeforeToolCall, BeforeOutbound, OnSessionStart,
OnSessionEnd, TransformResponse).

The `BeforeToolCall` hook is the natural integration point:

```rust
// In hook execution, before a tool runs:
if let Some(skill_registry) = &skill_registry {
    let matching_skills = skill_registry.match_tool_patterns(tool_name, tool_args);
    for skill in matching_skills {
        // Inject skill guidance as additional context for this tool call
        inject_skill_context(skill, &mut reasoning_context);
    }
}
```

**Percy-specific patterns:**
- `memory_write` tool call → inject "memory organization" skill
- `http` tool call to external API → inject "API safety" skill
- `shell` tool with deployment commands → inject "deploy-helper" skill
- WASM tool from a specific collection → inject collection-specific guidance

This requires adding `tool_patterns` to `ActivationCriteria` (alongside the existing
`keywords` and `patterns` which match message content).

### 3.4 Deferred Tool Loading (Adopt)

**Current:** CORE_TOOLS is a static whitelist. tools_prefix is unwired.

**Proposed:** Replace the current tool visibility model:

```
Boot: Register all tools
       ↓
Turn start: CORE_TOOLS whitelist (static)
       ↓
Skill activation: tools_prefix expands the set (ephemeral, per-turn)
       ↓
Attenuation: Trust ceiling applied
       ↓
LLM sees final tool set
```

The `discover_tools` tool already exists and lets the LLM search for tools. The
missing piece is `mark_discovered` actually doing something. But rather than
persistent discovery, use **per-session discovery state**:

```rust
struct SessionToolState {
    /// Tools explicitly promoted by skill activation this turn
    skill_promoted: HashSet<String>,
    /// Tools the LLM has discovered via discover_tools (session-scoped)
    user_discovered: HashSet<String>,
}
```

The final tool set for a turn is:
```
CORE_TOOLS ∪ skill_promoted ∪ user_discovered
```
All subject to attenuation.

### 3.5 Skill Types (Adopt Partially)

**Current:** Skills have `trust` (Trusted/Installed) which controls tool access.

**Proposed:** Add a `mode` field separate from trust:

```yaml
name: systematic-debugging
mode: rigid          # or "flexible" (default)
trust: trusted       # (existing: controls tool ceiling)
```

- **Rigid skills** inject a `⚠ Follow these steps exactly` prefix
- **Flexible skills** inject `Adapt these principles to the situation`

This is a small change (one field, one conditional in the XML builder) but gives
skill authors explicit control over how strictly the LLM should follow instructions.

---

## 4. What Percy Should NOT Adopt

### 4.1 Full Model-Driven Activation as Primary Path (Not Yet)

Claude Code runs on Claude — a highly capable model that reliably decides when to
invoke skills. Percy often runs on local models (Qwen 32B, MLX) that are less
reliable at self-directed tool invocation. Making model-driven activation the
*primary* path would degrade skill coverage for the most common deployment.

**Timeline:** Adopt model-driven as primary when Percy's default model can reliably
use a skill catalog. Until then, keyword matching + hook activation as primary,
model-driven as enhancement.

### 4.2 Session Deduplication (Not Needed at Scale)

Claude Code deduplicates skill injection per session to avoid re-injecting the
same skill content. This matters when there are 50+ skills and hooks fire on
every tool call. Percy has 5-10 skills. The dedup machinery isn't worth the
complexity.

### 4.3 Priority-Based Skill Ranking with Budget (Partially Exists)

Claude Code caps at 3 skills per hook invocation sorted by priority. Percy already
has `max_active_skills = 3` and scores by keyword/tag/regex. The current scoring
is sufficient — the gap is in activation trigger quality, not selection ranking.

---

## 5. Percy-Specific Design: Multi-Lens Skill Scoping

Claude Code has one user per session. Percy has multiple lenses (users) sharing
one IronClaw instance. This creates scoping requirements Claude Code doesn't face.

### 5.1 Current Model: SkillScope

```rust
enum SkillScope {
    Single(String),      // scope: "andrew"
    Multiple(Vec<String>), // scope: ["andrew", "grace"]
}
```

Skills without a scope activate for all users. Skills with a scope activate only
for matching user_ids. This works today.

### 5.2 Proposed Extension: Typed Scopes

For Percy's lens architecture, add a tagged variant:

```rust
#[serde(untagged)]
enum SkillScope {
    Single(String),
    Multiple(Vec<String>),
    Typed { lens_type: LensType },
}

enum LensType {
    Member,    // All member lenses (andrew, grace, ...)
    Group,     // All group lenses (household, ...)
    Any,       // Explicit wildcard
}
```

YAML usage:
```yaml
# Personal skill: only member lenses
scope:
  lens_type: member

# Group coordination skill: only group lenses
scope:
  lens_type: group
```

**Why:** When a new family member is added, they automatically get all `member`
skills without updating every skill's scope list.

**When:** Not now. Percy has 3 lenses. Build this when there are 5+.

### 5.3 Consent-Aware Skills

Some skills might need to know about other users (e.g., "schedule coordinator"
needs to check Andrew's and Grace's calendars). The current WASM tool consent
model handles this at the tool level, not the skill level. Skills should not
bypass consent — they inject instructions, and the instructions direct the LLM
to use consent-governed tools.

**Design rule:** Skills never get direct access to other users' data. Skills
instruct the LLM to use tools, and tools enforce consent boundaries. This is
already the case and should remain so.

### 5.4 Per-Lens Skill Directories

Currently, skills are loaded from `~/.ironclaw/skills/` (user) and
`workspace/skills/` (workspace). For Percy's multi-lens deployment:

```
skills/
├── shared/           # All lenses: universal skills
├── members/          # Member lenses only
│   ├── andrew/       # Andrew-specific skills
│   └── grace/        # Grace-specific skills
└── groups/           # Group lenses only
    └── household/    # Household-specific skills
```

The `generate.py` config tool could map these directories to skill scopes
automatically, so operators don't need to add `scope:` to every SKILL.md.

---

## 6. Proposed Implementation Phases

### Phase 1: Lightweight Catalog + Skill Types (Low effort, high value)

1. Add skill catalog block to system prompt (~50-100 tokens listing all
   available skills with trigger descriptions)
2. Add `mode: rigid|flexible` field to `SkillManifest`
3. Adjust injected skill XML to include mode guidance prefix

**Result:** LLM knows what skills exist. Skill authors can control strictness.

### Phase 2: Wire tools_prefix with Ephemeral Per-Turn Injection (Medium effort)

1. In `dispatcher.rs`, after selecting active skills, read `tools_prefix`
2. Search tool registry for matches
3. Merge into turn's tool definitions (before attenuation)
4. Implement `SessionToolState` for `discover_tools` persistence

**Result:** Skills bring their tools. CORE_TOOLS + skill expansion + attenuation
compose correctly.

### Phase 3: Hook-Based Activation (Medium effort)

1. Add `tool_patterns` field to `ActivationCriteria`
2. In `BeforeToolCall` hook, match tool name/args against skill patterns
3. Inject matching skill content into reasoning context
4. Respect existing max_active_skills budget

**Result:** Skills activate based on what the agent does, not just what the user
says. Memory writes get organization guidance. Shell commands get safety guidance.

### Phase 4: Model-Driven Activation (Medium effort, model-dependent)

1. Add `activate_skill` built-in tool
2. LLM can browse the catalog (from Phase 1) and activate skills by name
3. Activated skill content injected on next prompt rebuild
4. Keep keyword matching as fallback for weaker models

**Result:** Capable models get full agency over skill activation. Weak models
still get keyword-based activation.

### Phase 5: Typed Scopes for Lens Architecture (Low effort, deferred)

1. Add `LensType` variant to `SkillScope`
2. Pass lens metadata (type, name) into `prefilter_skills`
3. Add per-lens skill directory convention to Percy's generate.py

**Result:** New family members automatically get appropriate skills. No need to
update every skill's scope list when adding a lens.

---

## 7. Key Design Decisions

### D1: Keyword Matching — Keep as Fallback, Not Primary

Keyword matching is brittle (false positives, false negatives) but works without
model cooperation. Keep it, but design the system so it's the *fallback* path:

```
Activation priority:
1. Model explicitly calls activate_skill → highest confidence
2. Hook fires on tool call pattern → contextual, reliable
3. Keyword match on user message → broad fallback
```

### D2: tools_prefix — Ephemeral, Not Persistent

When a skill activates with `tools_prefix: "collection_"`, the matching tools are
available for *that turn only*. They don't persist to subsequent turns where the
skill isn't active. This prevents tool set creep and keeps the context predictable.

### D3: Skill Content — Lazy Load, Not Eager Inject

The catalog (skill names + triggers) is always in the system prompt. Full skill
content is injected only when:
- The model calls `activate_skill`
- A hook matches a tool call pattern
- Keyword matching selects the skill (fallback)

### D4: Attenuation — Always Final Gate

Regardless of how tools enter the turn's tool set (CORE_TOOLS, tools_prefix,
discover_tools), attenuation is the final filter. INSTALLED skills can never
promote dangerous tools. This is a security invariant that must never be violated.

### D5: Skills Inject Instructions, Not Capabilities

Skills change *how* the model behaves, not *what* it can do. Tool access is
controlled by CORE_TOOLS, discovery, and attenuation. Skills provide guidance
on when and how to use those tools. This separation of concerns is critical
for security — a malicious INSTALLED skill can't escalate tool access.

---

## 8. What We Can Steal Directly from Claude Code

| Pattern | Steal? | Adaptation |
|---------|--------|------------|
| Skill catalog in system prompt | Yes | Smaller scale, include scope info |
| On-demand Skill tool | Yes (Phase 4) | Keep keyword fallback for weak models |
| PreToolUse hook injection | Yes (Phase 3) | Map to IronClaw's BeforeToolCall hook |
| ToolSearch / deferred tools | Partially | `discover_tools` already exists, needs `mark_discovered` to work |
| Session dedup | No | Not needed at Percy's scale (5-10 skills) |
| Rigid vs flexible types | Yes (Phase 1) | Direct adoption, one field |
| Skill priority ordering | No | Existing score-based ranking is sufficient |
| 18KB byte budget per injection | No | 4000 token budget is equivalent |

---

## 9. Open Questions

1. **Should activation scripts compose with hook-based activation?** If a skill has
   both an activation script and a tool_pattern, should the script run on hook match
   or only on keyword/model activation?

2. **Should the skill catalog include scope information?** Showing `scope: "andrew"`
   in the catalog means the LLM knows which skills belong to which user. This could
   be useful (the LLM avoids trying to activate out-of-scope skills) or a privacy
   leak (if the LLM mentions other users' skills in conversation).

3. **How should skill activation interact with compaction?** When the context is
   compacted (long conversations), should active skill content be preserved or
   re-injected? Currently it's in the system prompt (always preserved), but
   hook-injected content would be in tool responses (compactable).

4. **Should Percy support skill composition rules?** Claude Code has explicit
   ordering (process skills before implementation skills). Percy's scoring naturally
   ranks by relevance, but doesn't have semantic categories.
