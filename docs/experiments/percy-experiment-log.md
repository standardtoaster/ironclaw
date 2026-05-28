# Percy: Experiment Log for a Multi-Tenant Family Assistant

Percy is a self-hosted AI assistant for a whole household. The hard, novel part is not "an
assistant that remembers things" — it is doing that for *multiple people who share a home* without
leaking private context between them. Off-the-shelf assistants are single-user; Percy is
**multi-tenant and safe by construction**, supporting both a private per-member context and a shared
family context in one running process.

It is built on [IronClaw](https://github.com/nearai/ironclaw), a Rust agent framework, with a fork
that adds the capabilities a household assistant needs. This log records three experiments. The
first has measured results and is the flagship. The other two are designed and implemented, with
unit tests, but not yet benchmarked — I keep that distinction explicit throughout.

---

## Architecture (brief)

Source: the project's architecture doc, `docs/percy-overview.md` (written to be public and
implementation-general).

**One entity, many lenses.** A single IronClaw process serves every family member. Each auth token
maps to a user ID, and memory, workspaces, collections, and conversation history are all scoped by
user ID *at the database level*. There is no code path that returns one member's data to another
without passing through an explicit consent layer. Isolation is a property of the data model, not a
prompt instruction.

**Private vs shared context.** A member's lens can read shared household memory (the family grocery
list, a shared calendar) while keeping its own private memory invisible to others. A deliberate
rule keeps identity files (personality, user profile, agent definitions) bound to the primary scope
only, so reading shared data never causes one lens's personality or instructions to bleed into
another's.

**Consent for cross-member queries.** "Is the other parent free Saturday?" is answered through a
WASM-sandboxed tool that calls the target member's lens; the response is limited to free/busy
status, never calendar detail. The sandbox restricts the tool to allow-listed endpoints and injects
credentials so the model never sees raw tokens.

**Memory model.** Memory is structured rather than a flat log: a compact always-loaded index plus
on-demand reads, with a freeform journal for the unstructured. On top of this sit two higher-level
primitives that the experiments below concern — **structured collections** (typed, queryable data)
and **workspaces** (topic-scoped context buckets, auto-organized). Storage is PostgreSQL +
pgvector; the default local model is Qwen3-30B-A3B served via MLX on Apple Silicon, with cloud
Claude models (Haiku / Sonnet / Opus) available via tiered routing.

---

## Experiment 1 — Structured collections vs flat memory docs (measured)

**Hypothesis.** For data-tracking tasks (lists, logs, ledgers), giving the model *schema-validated
CRUD tools over a typed collection* beats letting it read and write flat memory documents — and this
holds regardless of model size.

**Why it matters.** A household assistant spends much of its time on structured data: childcare
shift logs, grocery lists, todos, payment/transaction records. The naive approach is to keep these
as memory documents the model reads and edits as text. The alternative is to define a typed
collection with a schema and auto-generate per-collection tools (`<name>_add`, `_update`, `_remove`,
`_query`, `_summary`), plus an auto-generated tool document so the model knows the contract.

**Method.** A 2×2 factorial over the same task set:

- **Models (2):** local Qwen3-30B-A3B (MLX, 8-bit) and cloud Claude Haiku 4.5.
- **Data mode (2):** *structured collections* (typed CRUD tools) vs *flat memory docs* (read/write
  over markdown/JSON files via memory tools).
- **Tasks:** 28 scenarios spanning four categories — grocery, nanny/childcare logs, todos, and
  transactions — covering reads, writes, filtered queries, and aggregation. Questions and expected
  answers are identical across the two data modes; only the seeding and the tool surface differ, so
  the comparison isolates the storage/tool design rather than the question difficulty.

Scoring is per-scenario (0–1, higher is better) and averaged.

**Results.**

| Model | Flat memory docs | Structured collections | Δ |
|-------|:---:|:---:|:---:|
| Qwen3-30B-A3B (local) | 0.37 | 0.65 | **+0.28** |
| Claude Haiku 4.5 (cloud) | 0.26 | 0.70 | **+0.44** |

Findings:

1. **Collections win for both models**, and the improvement is consistent across all four
   categories — there is no category where flat docs were better.
2. **Biggest gains on writes and aggregation** — the operations where free-text editing is most
   error-prone and where a typed query/summary tool does the arithmetic deterministically instead of
   asking the model to do it in prose.
3. **Collections largely equalize the two models.** With flat docs the cloud and local models were
   far apart and both poor; with collections they land close together (0.70 vs 0.65). The structured
   tool surface lifts the cheaper local model to roughly the cloud model's level — directly relevant
   to running a household assistant mostly on free local inference.

The hypothesis is **confirmed**: structured collections outperform flat memory docs for
data-tracking tasks regardless of model size.

**Engineering footprint and rigor.** The collections feature is the larger of the fork's two big
features: roughly 10k lines across ~26 files, landed over ~11 commits. It ships schema definition
and validation, per-collection dynamic tool generation, a REST API, an event system for mutations,
and per-user scoping. Test coverage is **126 tests** (79 unit / 35 integration / 12 tool-level),
passing against **both PostgreSQL and libSQL** backends — the backend abstraction is exercised, not
assumed.

> Note: an earlier, separate tool-calling study (`docs/experiments/structured-collections-tool-calling.md`)
> looked at *whether* small local models can reliably call dynamically-generated collection tools at
> all. It is methodologically interesting — it includes a writeup of a test-harness bug (a bash
> subshell variable-scoping error) that invalidated an entire result batch, the diagnosis, and the
> re-run — but its numbers are not the 2×2 results above and are not mixed in here.

---

## Experiment 2 — Dynamic tool discovery (designed + implemented; not benchmarked)

Source: the design doc `docs/design-discover-tools-v2.md` (distinct from the model-tier routing
spec). Implementation status from `docs/features/row-06-discover-tools.md`.

**Problem.** Percy has dozens of tools across lenses (calendar, collections, consent queries,
escalation, search, MCP services). Putting every tool's full schema in every prompt wastes context
and measurably degrades tool selection on smaller models. But the first attempt — a `discover_tools`
"search then load" meta-tool — was worse: it cost multiple round-trips and could still fail, because
the "load" step was a no-op and the tool stayed hidden behind a static visibility filter.

**Design.** The v2 design reframes the problem around three concerns that were previously conflated:
*awareness* (the model knows a tool exists), *readiness* (the model has the tool's schema), and
*permission* (the model is allowed to run it). The target behavior, modeled on how Claude Code
itself handles tools:

1. **Auto-register at startup** — all configured tools, including MCP services, register their
   *names* immediately. The model never "discovers" a tool it should already know exists.
2. **Deferred schemas in two tiers** — a small per-lens *core tier* carries full schemas every turn;
   everything else is *deferred*, present as name + description only, with the full parameter schema
   fetched on demand (`tool_info`) and then promoted for the rest of the conversation. This turns the
   old "search → load → fail" path into "0 turns for a core tool, 1 turn for a deferred tool."
3. **Invoke-time access control** — a unified policy trait checks permission at call time. Only the
   untrusted-skill attenuation policy removes a tool from visibility; everything else gates execution,
   not awareness.
4. **Skill-declared tool needs** — when a skill activates, the tools it declares are promoted to the
   core tier for that turn, so the model gets the right schemas without searching.

A notable rigor point: the design is corrected *by its own benchmark data*. An earlier assumption
("don't bother tiering schemas below ~100 tools") is struck through and replaced after a tool-count
sweep showed local-model tool-selection accuracy starting to drop at ~15 tools and falling off a
cliff near 50 — and showed that fine-tuning for Percy's personality actually *hurt* generalization to
unfamiliar (MCP) tools. That measured tension feeds directly into the routing design: route
conversational turns to the personality-tuned model and tool-heavy turns to the base model.

**Status.** The current code implements the `discover_tools` meta-tool with keyword search and
`CORE_TOOLS` filtering, with unit tests for the filtering behavior; the full auto-register /
deferred-schema / invoke-time-policy architecture is the documented next step. No end-to-end
selection benchmark has been run on the v2 design, so I make no accuracy claim for it here.

---

## Experiment 3 — Automated workspace selection (designed + implemented; not benchmarked)

Source: workspace code under `src/workspace` and `src/agent/workspace_router.rs`, the design doc
`docs/plans/2026-03-10-workspace-auto-organization-design.md`, and commit `b118f130` ("feat:
automatic workspace organization with topic detection").

**Problem.** A workspace is a topic-scoped context bucket (e.g. a trip, a renovation, tax prep).
Relying on the model to call `create_workspace` when it notices a new topic failed in testing
roughly 80% of the time — models optimize for answering, not for filing — and neither a stronger
model nor prompt engineering fixed it. So organization has to be automatic, and routing a new
message to the right existing workspace has to handle collisions between similar topics.

**Design.** Three layers:

- A generic `ThreadResolver` trait in the agent loop — IronClaw knows only that *something* may
  redirect a message to a different thread and inject context; it knows nothing about workspaces.
- A `WorkspaceRouter` that does embedding-similarity routing with explicit **confidence tiers**, and
  a `WorkspaceOrganizer` that uses a cheap local model to group recent messages into project-level
  workspaces (deliberately an LLM, not pure embedding clustering, because clustering fragments a
  project — "what camera for the trip?" and "weather in April?" are far apart in embedding space but
  belong together).
- A routine that triggers the organizer on a schedule / on demand, sharing the same code path as the
  test harness.

The router's confidence tiers, as implemented in `workspace_router.rs`:

| Tier | Condition | Action |
|------|-----------|--------|
| **Strong** | top score ≥ 0.75 | route to the matching workspace |
| **Moderate (confident)** | score ≥ 0.5 and clear gap to the runner-up | route confidently |
| **Ambiguous** | score ≥ 0.5 but the top two are within a small gap | don't guess — fall to default with a disambiguation note, broken by recency/"stickiness" momentum |
| **No match** | top score < 0.5 | stay in the default workspace (correct for one-offs) |

Layered on top is conversation **stickiness**: short, anaphoric, follow-up messages ("what about near
the station?", "make it two nights") stay in the active workspace unless a strong match to a
*different* workspace breaks the momentum, or too much time has passed.

**Status: implemented and unit-tested.** The router, resolver, and organizer are real code with
substantial unit coverage — on the order of 17 tests in the router, 32 in the thread resolver, and 11
in the organizer runner — covering strong/moderate/ambiguous/no-match routing, stickiness
continuation, topic-switch breaks, and idempotency of re-running the organizer. Per the design doc,
collision-aware routing refinements and full end-to-end validation are still open, and **no routing
or organization accuracy benchmark has been run** — this is design + implementation, not a measured
result.

---

## Summary

| | Status | Evidence |
|---|---|---|
| Multi-tenant isolation + consent | Running | architecture doc; DB-level user scoping; WASM consent tool |
| **Structured collections** | **Measured** | 2×2 factorial, 28 scenarios; collections beat flat docs for both models; 126 tests on two backends |
| Dynamic tool discovery (v2) | Designed; partially implemented | design doc + tool-count benchmark informing it; `CORE_TOOLS` + `discover_tools` shipped with unit tests |
| Automated workspace selection | Designed + implemented; unit-tested | router/resolver/organizer code, confidence tiers, ~60 unit tests; not yet benchmarked |

The throughline is treating the assistant as a system to be measured, not just built: a controlled
factorial for the storage question, a tool-count sweep that overturned a prior design assumption, and
honest labeling of what is measured versus what is designed.
