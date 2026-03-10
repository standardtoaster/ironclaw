#!/usr/bin/env python3
"""Workspace A/B validation harness.

Sends scripted conversation sequences derived from real Claude.ai conversation
patterns and measures whether workspace routing improves response quality vs
a single general thread with memory_search.

Two modes:
  --mode workspaces   (default) Full workspace routing — LLM can create_workspace
  --mode baseline     Single thread, no workspace routing — LLM relies on memory_search

Requires a running IronClaw instance with PostgreSQL + pgvector.

## Cost estimate (Haiku 4.5, 36 messages)

  System prompt:  ~2000 tokens (identity + tools + SOUL.md)
  Avg context:    ~800 tokens/msg (accumulates within sequence, resets between)
  Avg response:   ~300 tokens/msg
  Embedding:      ~36 calls × $0.00002 = ~$0.001

  Input:   36 msgs × ~2800 avg tokens = ~100K tokens → $0.10
  Output:  36 msgs × ~300 tokens = ~10.8K tokens → $0.05
  Tools:   ~15 tool calls × ~500 tokens = ~7.5K tokens → $0.01
  Total:   ~$0.16/run on Haiku, ~$0.65/run on Sonnet

  With --mode both: double those numbers (~$0.32 Haiku, ~$1.30 Sonnet).

## Signal map (what each sequence tells us)

  Seq 1: home_assistant_recurring (5 msgs, 12 checks)
    - CREATE: Does LLM create workspace for recurring smart home topic?
    - ROUTE:  Do follow-up HA messages route back after a topic switch?
    - RECALL: Can it reference prior Zigbee/Hue context after an interruption?
    - JUDGE:  Does the hollandaise question stay in general (no workspace)?

  Seq 2: financial_recurring (4 msgs, 10 checks)
    - CREATE: Does LLM create workspace for complex tax discussion?
    - ROUTE:  Does finance question route back after a UniFi question?
    - RECALL: Does it remember the US/UK expat + rental context?
    - REASK:  Does it avoid re-asking which country the user lives in?
    - JUDGE:  Does the UniFi camera question stay in general?

  Seq 3: one_off_questions (4 msgs, 8 checks)  ** MOST IMPORTANT **
    - JUDGE:  Pure workspace restraint test — 4 unrelated one-off questions.
             ALL should stay general. ANY workspace creation = over-eager.
             This is the critical "don't over-create" signal.

  Seq 4: travel_planning (4 msgs, 7 checks)
    - CREATE: Does LLM create workspace for multi-step trip planning?
    - RECALL: Does it remember Budapest context when comparing to Rome?
    - JUDGE:  Do continuation messages stay in the trip workspace (no new ws)?

  Seq 5: interleaved_topics (5 msgs, 12 checks)  ** DEPENDS ON SEQ 1,2,4 **
    - ROUTE:  The hard test. Rapidly switches HA → travel → HA → finance → general.
             Each message should route to the correct existing workspace.
    - RECALL: Can it pull context from the right workspace when switching?
    - JUDGE:  Does the egg question stay in general?
    - NOTE:   This sequence MUST run after 1, 2, and 4 (needs existing workspaces).

  Seq 6: passive_then_query (4 msgs, 4 checks)
    - RECALL: Tests whether information stated as facts (not questions) is retained
             and retrievable. Simulates WhatsApp-style passive ingestion.
    - This is a pure context recall test — no routing or creation checks.

  Seq 7: camera_troubleshooting (3 msgs, 6 checks)
    - CREATE: Does LLM create workspace for a recurring tech problem?
    - RECALL: Does it carry Nanit context into the UniFi replacement discussion?
    - JUDGE:  Do follow-up messages stay in the camera workspace?

  Seq 8: property_home (3 msgs, 5 checks)
    - CREATE: Does LLM create workspace for property investment discussion?
    - JUDGE:  Do related-but-different property messages stay together?

  Seq 9: hobby_topics (4 msgs, 7 checks)
    - JUDGE:  Tests workspace restraint for hobby topics that SEEM like they
             could be recurring but are actually one-off. The limoncello
             follow-up is deliberately ambiguous (tests LLM judgment edge case).

  TOTALS: 36 messages, 71 assertion checks across 4 metrics:
    - Topic routing:      25 checks  (does it go to the right workspace?)
    - Creation judgment:  31 checks  (create when should, don't when shouldn't)
    - Context recall:     12 checks  (can it reference prior conversation?)
    - Re-ask avoidance:    3 checks  (does it avoid redundant questions?)

Usage:
    python tests/test_workspace_switching.py --mode workspaces
    python tests/test_workspace_switching.py --mode baseline
    python tests/test_workspace_switching.py --mode both
    python tests/test_workspace_switching.py --sequences home_assistant_recurring one_off_questions
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_TIMEOUT = 120  # seconds per message
INTER_MESSAGE_DELAY = 5  # seconds between messages (reduced from 20)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Result of sending a message and observing routing."""
    message: str
    response_text: str
    routed_workspace_id: Optional[str] = None
    routed_topic: Optional[str] = None
    is_new_workspace: bool = False
    tools_called: list[str] = field(default_factory=list)
    elapsed_secs: float = 0.0
    error: Optional[str] = None


@dataclass
class Message:
    """A single message in a conversation sequence."""
    content: str
    # What we check in the response
    expect_topic_keywords: Optional[list[str]] = None   # any keyword match in routed topic = pass
    expect_workspace_created: Optional[bool] = None      # None = don't check, True/False = assert
    expect_recall_any: Optional[list[str]] = None        # any keyword in response = pass
    expect_no_reask: Optional[list[str]] = None          # NONE of these should appear in response
    description: str = ""                                 # what this message tests


@dataclass
class ConversationSequence:
    """A multi-message conversation that tests topic handling."""
    name: str
    description: str
    messages: list[Message]
    depends_on: list[str] = field(default_factory=list)  # sequences that must run first
    lens: str = "andrew"


@dataclass
class MessageResult:
    """Combined scenario + result for a single message."""
    message: Message
    routing: RoutingResult
    topic_correct: Optional[bool] = None
    context_recalled: Optional[bool] = None
    avoided_reask: Optional[bool] = None
    workspace_created_correct: Optional[bool] = None


@dataclass
class SequenceResult:
    """Results for an entire conversation sequence."""
    sequence: ConversationSequence
    message_results: list[MessageResult]

    @property
    def context_recall_rate(self) -> Optional[float]:
        checks = [r.context_recalled for r in self.message_results if r.context_recalled is not None]
        return sum(checks) / len(checks) if checks else None

    @property
    def reask_avoidance_rate(self) -> Optional[float]:
        checks = [r.avoided_reask for r in self.message_results if r.avoided_reask is not None]
        return sum(checks) / len(checks) if checks else None


# ---------------------------------------------------------------------------
# Topic matching
# ---------------------------------------------------------------------------

def topic_matches(routed_topic: Optional[str], keywords: list[str]) -> bool:
    """Fuzzy topic match: any keyword as substring of the routed topic.

    Handles None/empty as "general" — if keywords contain "general" that matches.
    Examples:
        topic_matches("home assistant setup", ["home", "smart", "ha"]) → True
        topic_matches("financial planning", ["financ", "tax", "budget"]) → True
        topic_matches(None, ["general"]) → True
        topic_matches("general", ["general"]) → True
    """
    effective = (routed_topic or "general").lower()
    return any(kw.lower() in effective for kw in keywords)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def http_get(base_url: str, path: str, headers: dict, timeout: int = 10) -> tuple[int, str]:
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def http_post_json(base_url: str, path: str, headers: dict, data: dict, timeout: int = 10) -> tuple[int, str]:
    url = f"{base_url}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SSE chat with workspace routing observation
# ---------------------------------------------------------------------------

def send_and_observe(base_url: str, token: str, message: str,
                     thread_id: Optional[str] = None,
                     timeout: int = CHAT_TIMEOUT) -> RoutingResult:
    """Send a chat message and collect SSE events including workspace_routed."""
    headers = {"Authorization": f"Bearer {token}"}
    start_time = time.monotonic()

    holder = {
        "chunks": [],
        "done": False,
        "error": None,
        "tool_events": [],
        "workspace_routed": None,
    }

    def sse_listener():
        try:
            req = urllib.request.Request(
                f"{base_url}/api/chat/events",
                headers={**headers, "Accept": "text/event-stream"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                current_event = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        current_event = None
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        etype = current_event or event.get("type", "")
                        event["type"] = etype
                        # Debug: log all SSE event types
                        if etype not in ("stream_chunk", "heartbeat"):
                            print(f"    [SSE] {etype}: {json.dumps(event)[:200]}", flush=True)

                        if etype == "stream_chunk":
                            holder["chunks"].append(event.get("content", ""))
                        elif etype == "response":
                            holder["chunks"] = [event.get("content", "")]
                            holder["done"] = True
                            return
                        elif etype == "workspace_routed":
                            holder["workspace_routed"] = event
                        elif etype in ("tool_started", "tool_completed", "tool_result"):
                            holder["tool_events"].append(event)
                        elif etype == "error":
                            holder["error"] = event.get("message", "unknown SSE error")
                            holder["done"] = True
                            return

                    if holder["done"]:
                        return
        except Exception as e:
            holder["error"] = str(e)

    listener = threading.Thread(target=sse_listener, daemon=True)
    listener.start()
    time.sleep(0.5)

    body = {"content": message}
    if thread_id:
        body["thread_id"] = thread_id

    try:
        status, resp_body = http_post_json(
            base_url, "/api/chat/send", headers, body
        )
        if status != 202:
            return RoutingResult(
                message=message, response_text="",
                elapsed_secs=time.monotonic() - start_time,
                error=f"chat/send returned {status}: {resp_body}",
            )
    except Exception as e:
        return RoutingResult(
            message=message, response_text="",
            elapsed_secs=time.monotonic() - start_time,
            error=f"Failed to send message: {e}",
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if holder["done"] or holder["error"]:
            break
        time.sleep(0.5)

    elapsed = time.monotonic() - start_time
    routed = holder["workspace_routed"]
    tools_called = [
        e.get("name", "?")
        for e in holder["tool_events"]
        if e.get("type") == "tool_started"
    ]

    return RoutingResult(
        message=message,
        response_text="".join(holder["chunks"]),
        routed_workspace_id=routed.get("workspace_id") if routed else None,
        routed_topic=routed.get("topic") if routed else None,
        is_new_workspace=routed.get("is_new", False) if routed else False,
        tools_called=tools_called,
        elapsed_secs=elapsed,
        error=holder.get("error"),
    )


# ---------------------------------------------------------------------------
# Scenarios derived from real conversation data
# ---------------------------------------------------------------------------

# These mirror actual topics from 282 Claude.ai conversations and 246
# Claude Code sessions analysed on 2026-03-09.
#
# IMPORTANT: Sequences 1-4 must run before sequence 5 (interleaved_topics)
# because seq 5 expects workspaces created by earlier sequences to exist.
# The runner enforces this via depends_on.

SEQUENCES: list[ConversationSequence] = [

    # ── 1. Home Assistant (7 real conversations, 123 total messages) ──────
    # Signal: workspace creation for recurring topic, routing persistence,
    #         context recall across topic switches
    ConversationSequence(
        name="home_assistant_recurring",
        description="Home Assistant setup — recurring topic across multiple sessions",
        messages=[
            Message(
                content="I just got a SLZB-06 Zigbee coordinator. How do I set it up with Home Assistant?",
                expect_workspace_created=True,
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                description="First HA message — should create workspace",
            ),
            Message(
                content="Got it working. Now I want to connect my Hue bulbs directly via Zigbee instead of the Hue bridge. Is that possible?",
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                expect_workspace_created=False,
                description="Continuation — should route to HA workspace, not create new",
            ),
            Message(
                content="What's a good recipe for hollandaise sauce for eggs benedict?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="Topic switch — one-off cooking, should NOT create workspace",
            ),
            Message(
                content="Back to the smart home — I'm trying to get my thermostat into Home Assistant but it only supports Matter. Do I need a Matter hub?",
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                expect_workspace_created=False,
                expect_recall_any=["zigbee", "slzb", "coordinator", "hue"],
                description="Return to HA — should route back, recall prior context",
            ),
            Message(
                content="The Hue bulbs keep dropping off Zigbee. They reconnect after a few minutes but it's annoying. Any ideas?",
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                expect_workspace_created=False,
                expect_recall_any=["hue", "bulb", "bridge", "zigbee"],
                description="Deep continuation — should recall Hue bulb discussion",
            ),
        ],
    ),

    # ── 2. Financial planning (35 real conversations, 520 total messages) ─
    # Signal: workspace creation for complex multi-jurisdiction tax,
    #         context retention of specific details (country, rental income),
    #         avoidance of redundant clarifying questions
    ConversationSequence(
        name="financial_recurring",
        description="Financial planning — most recurring real topic domain",
        messages=[
            Message(
                content="I need to file US taxes as an expat in the UK. I have rental income from a property in New York and contractor income from a UK company. What forms do I need?",
                expect_workspace_created=True,
                expect_topic_keywords=["financ", "tax", "budget", "money", "expat"],
                description="Tax discussion — should create financial workspace",
            ),
            Message(
                content="For the UK side, do I need to register for Making Tax Digital? I'm above the threshold.",
                expect_topic_keywords=["financ", "tax", "budget", "money", "expat"],
                expect_workspace_created=False,
                description="Continuation — same financial workspace",
            ),
            Message(
                content="Can you help me figure out the best UniFi camera for a nursery? I need a fixed wide angle mount.",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="Topic switch to tech — should stay general",
            ),
            Message(
                content="Going back to finances — should I pay down the mortgage or invest in uranium stocks? The mortgage is at 4.5% and uranium miners are up 40% this year.",
                expect_topic_keywords=["financ", "tax", "budget", "money", "expat"],
                expect_workspace_created=False,
                expect_recall_any=["rental", "new york", "expat", "uk", "contractor"],
                expect_no_reask=["what country", "where do you live", "which country", "where are you based"],
                description="Return to finance — should recall US/UK expat context, not re-ask location",
            ),
        ],
    ),

    # ── 3. One-off questions (should NOT create workspaces) ──────────────
    # Signal: pure workspace restraint — the MOST IMPORTANT sequence.
    #         Over-creation is the #1 risk with workspace tools. Every message
    #         here should stay in general. Any workspace creation = failure.
    ConversationSequence(
        name="one_off_questions",
        description="Quick questions that should stay in general workspace — no workspaces created",
        messages=[
            Message(
                content="What temperature should I roast a frozen chicken at without thawing it first?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="One-off cooking question — no workspace",
            ),
            Message(
                content="How do I clean mold off a baby stroller before a trip?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="One-off cleaning question — no workspace",
            ),
            Message(
                content="What's the tipping etiquette at a Four Seasons hotel in Athens?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="One-off travel etiquette — no workspace",
            ),
            Message(
                content="Can I use a UK Dyson Airwrap in Greece without a voltage converter?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="One-off voltage question — no workspace",
            ),
        ],
    ),

    # ── 4. Travel planning (10 real conversations, episodic) ─────────────
    # Signal: workspace creation for multi-step planning, context retention
    #         of trip details (destination, child age, hotel preferences)
    ConversationSequence(
        name="travel_planning",
        description="Vacation planning — multi-step, should accumulate context",
        messages=[
            Message(
                content="We're thinking about a family trip to Budapest in April. Gordon is 2. What are good family-friendly hotels with a pool?",
                expect_workspace_created=True,
                expect_topic_keywords=["travel", "trip", "vacation", "budapest", "holiday"],
                description="Trip planning starts — should create workspace",
            ),
            Message(
                content="What's the rate at the Four Seasons Budapest in April? Also what's the weather like?",
                expect_workspace_created=False,
                expect_topic_keywords=["travel", "trip", "vacation", "budapest", "holiday"],
                description="Continuation — same trip workspace, no new creation",
            ),
            Message(
                content="Actually can we compare Budapest vs Rome for a toddler-friendly trip? We went to Rome before but it was pre-kid.",
                expect_workspace_created=False,
                expect_recall_any=["budapest", "gordon", "pool", "family", "april", "four seasons"],
                description="Expanding discussion — should recall Budapest context",
            ),
            Message(
                content="What time zone is Budapest in? I need to figure out the flight times from London.",
                expect_workspace_created=False,
                description="Quick sub-question — should stay in trip workspace",
            ),
        ],
    ),

    # ── 5. Interleaved topics (the hard test) ────────────────────────────
    # Signal: routing accuracy when rapidly switching between EXISTING workspaces.
    #         This is the primary differentiator for workspace routing vs baseline.
    #         In baseline mode, the LLM must memory_search for context; with
    #         workspaces, context is pre-loaded via routing.
    ConversationSequence(
        name="interleaved_topics",
        description="Rapidly switching between established topics — requires seqs 1,2,4",
        depends_on=["home_assistant_recurring", "financial_recurring", "travel_planning"],
        messages=[
            Message(
                content="I set up a new Zigbee network with the SLZB-06. Can I run both Zigbee and Thread on it simultaneously?",
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                expect_workspace_created=False,
                description="HA topic — should route to existing HA workspace",
            ),
            Message(
                content="Oh also, did we decide on Budapest or Rome for April?",
                expect_topic_keywords=["travel", "trip", "vacation", "budapest", "holiday"],
                expect_workspace_created=False,
                expect_recall_any=["toddler", "gordon", "family", "pool", "budapest", "rome"],
                description="Trip check — should route to travel workspace, recall discussion",
            ),
            Message(
                content="Back to the smart home — I want to automate the heating. When the last person leaves, turn it to 18C. When someone arrives, back to 21C.",
                expect_topic_keywords=["home", "smart", "ha", "assistant", "zigbee", "automation"],
                expect_workspace_created=False,
                description="Back to HA — should route correctly, not create new workspace",
            ),
            Message(
                content="For the tax filing, can I deduct the flights to New York to check on the rental property?",
                expect_topic_keywords=["financ", "tax", "budget", "money", "expat"],
                expect_workspace_created=False,
                expect_recall_any=["rental", "new york", "expat", "uk", "property"],
                description="Finance question — should route to financial workspace",
            ),
            Message(
                content="How long does it take to hard boil an egg?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="Random one-off — should go to general, no workspace",
            ),
        ],
    ),

    # ── 6. Passive ingestion then query ──────────────────────────────────
    # Signal: can the system retain facts stated as context (not questions)?
    #         Simulates WhatsApp bridge ingestion where messages arrive as
    #         statements. The LLM must store and retrieve these on query.
    #         No routing or creation checks — pure context recall.
    ConversationSequence(
        name="passive_then_query",
        description="Ingested context messages followed by queries — tests recall only",
        messages=[
            Message(
                content="Grace said she's taking Gordon to the pediatrician on Thursday for his 2-year checkup. They're going through WPA insurance.",
                description="Passive ingestion — family health context",
            ),
            Message(
                content="The nanny Leticia confirmed she can do Tuesday and Thursday this week, but not Friday.",
                description="Passive ingestion — nanny schedule",
            ),
            Message(
                content="Is the pediatrician appointment this week? Which day?",
                expect_recall_any=["thursday", "thurs"],
                expect_no_reask=["when is", "do you have", "i don't have", "i'm not sure when"],
                description="Query — should recall Thursday from ingested context",
            ),
            Message(
                content="When is Leticia available this week?",
                expect_recall_any=["tuesday", "thursday", "tues", "thurs"],
                expect_no_reask=["who is leticia", "i don't know", "could you tell me"],
                description="Query — should recall nanny schedule",
            ),
        ],
    ),

    # ── 7. Camera/tech troubleshooting (5 real Nanit conversations) ──────
    # Signal: workspace creation for a recurring debugging topic, context
    #         continuity as the problem evolves (Nanit → replacement options)
    ConversationSequence(
        name="camera_troubleshooting",
        description="Nanit baby camera issues — recurring debugging topic",
        messages=[
            Message(
                content="The Nanit camera in Gordon's room keeps disconnecting. It shows a solid red light then reconnects after a few minutes. This is the third time this week.",
                expect_workspace_created=True,
                expect_topic_keywords=["camera", "nanit", "baby", "monitor", "tech"],
                description="Tech issue — should create workspace for recurring problem",
            ),
            Message(
                content="I found the receipt — bought it 14 months ago from Amazon. Is it still under warranty?",
                expect_workspace_created=False,
                expect_recall_any=["nanit", "disconnect", "red light", "camera"],
                description="Continuation — same camera issue, should recall the problem",
            ),
            Message(
                content="If the Nanit is unreliable, what UniFi camera would work as a baby monitor? I already have a UniFi network.",
                expect_workspace_created=False,
                expect_recall_any=["nanit", "disconnect", "warranty", "red light", "14 month"],
                description="Evolving — from Nanit fix to UniFi replacement, should carry context",
            ),
        ],
    ),

    # ── 8. Property/home (13 real conversations) ─────────────────────────
    # Signal: workspace creation for property investment, topic coherence
    #         across related-but-different sub-topics (investment, fibre, electrical)
    ConversationSequence(
        name="property_home",
        description="St John's Wood apartment — ongoing property discussions",
        messages=[
            Message(
                content="I'm looking at investing in a studio apartment in St John's Wood. The asking price is 450k. What's the rental yield like in that area?",
                expect_workspace_created=True,
                expect_topic_keywords=["property", "home", "apartment", "real estate", "st john"],
                description="Property investment — should create workspace",
            ),
            Message(
                content="The building needs an MDU fibre installation from Openreach. I've been trying to get them to schedule it for months. Any tips on escalating?",
                expect_workspace_created=False,
                expect_recall_any=["st john", "apartment", "studio", "building", "450"],
                description="Property infrastructure — same workspace, should recall property details",
            ),
            Message(
                content="The breaker in the kitchen keeps tripping after we installed the new EV charger. It's a 32A circuit on a 40A breaker. Is that enough headroom?",
                expect_workspace_created=False,
                description="Electrical at property — should stay in property workspace",
            ),
        ],
    ),

    # ── 9. Limoncello and hobby topics ───────────────────────────────────
    # Signal: workspace restraint for hobby topics. Most hobbies are one-off.
    #         The limoncello follow-up (msg 2) is deliberately ambiguous —
    #         it COULD warrant a workspace but probably shouldn't. Acceptable
    #         either way, but creating workspaces for shoes or camera parts = fail.
    ConversationSequence(
        name="hobby_topics",
        description="Hobby discussions — tests workspace restraint for one-off-ish topics",
        messages=[
            Message(
                content="I want to make limoncello. What kind of lemons should I use and how long does the infusion take?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="Hobby question — probably one-off, no workspace",
            ),
            Message(
                content="Actually I might want to do a bigger batch for Christmas gifts. Can you help me scale the recipe for 12 bottles?",
                # This is deliberately ambiguous — creating a workspace here is acceptable
                # but not required. We don't assert on workspace_created.
                expect_recall_any=["lemon", "infus", "limoncello", "zest", "grain alcohol", "vodka"],
                description="Follow-up — might warrant workspace (edge case), should recall recipe",
            ),
            Message(
                content="My suede Chelsea boots got water stains. What's the best way to clean them without damaging the nap?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="One-off shoe care — no workspace",
            ),
            Message(
                content="I'm looking for a replacement light meter for a Hasselblad 500C/M. Where can I find vintage camera parts in London?",
                expect_topic_keywords=["general"],
                expect_workspace_created=False,
                description="Camera hobby question — likely one-off, no workspace",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_sequence(base_url: str, token: str, seq: ConversationSequence,
                 mode: str, timeout: int) -> SequenceResult:
    """Run a single conversation sequence and evaluate results."""
    print(f"\n{'─' * 60}")
    print(f"Sequence: {seq.name}")
    print(f"  {seq.description}")
    print(f"  Mode: {mode} | Messages: {len(seq.messages)}")
    if seq.depends_on:
        print(f"  Depends on: {', '.join(seq.depends_on)}")
    print(f"{'─' * 60}")

    results = []
    for i, msg in enumerate(seq.messages):
        label = msg.description or msg.content[:60]
        print(f"\n  [{i+1}/{len(seq.messages)}] {label}")
        print(f"  >>> {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")

        routing = send_and_observe(base_url, token, msg.content, timeout=timeout)

        if routing.error:
            print(f"  ERROR: {routing.error}")
            results.append(MessageResult(message=msg, routing=routing))
            continue

        mr = MessageResult(message=msg, routing=routing)

        # Topic routing check (fuzzy keyword match)
        if msg.expect_topic_keywords is not None:
            mr.topic_correct = topic_matches(routing.routed_topic, msg.expect_topic_keywords)
            status = "OK" if mr.topic_correct else "MISS"
            matched_label = f"keywords={msg.expect_topic_keywords}"
            print(f"  [{status}] topic: got='{routing.routed_topic or '(none/general)'}' {matched_label}")

        # Workspace creation check
        if msg.expect_workspace_created is not None:
            created = "create_workspace" in routing.tools_called
            mr.workspace_created_correct = (created == msg.expect_workspace_created)
            if msg.expect_workspace_created:
                status = "OK" if created else "MISS"
                print(f"  [{status}] workspace creation: {'created' if created else 'not created'}")
            elif created:
                print(f"  [OVER] unexpected workspace creation")
            else:
                print(f"  [ OK ] no workspace created (correct)")

        # Context recall check (any keyword match)
        if msg.expect_recall_any is not None:
            response_lower = routing.response_text.lower()
            matched = [kw for kw in msg.expect_recall_any if kw.lower() in response_lower]
            mr.context_recalled = len(matched) > 0
            if matched:
                print(f"  [ OK ] context recall: matched {matched}")
            else:
                print(f"  [MISS] context recall: none of {msg.expect_recall_any} found")

        # Re-ask avoidance check (none of these should appear)
        if msg.expect_no_reask is not None:
            response_lower = routing.response_text.lower()
            found = [phrase for phrase in msg.expect_no_reask if phrase.lower() in response_lower]
            mr.avoided_reask = len(found) == 0
            if found:
                print(f"  [FAIL] re-asked: found {found}")
            else:
                print(f"  [ OK ] no re-asking detected")

        # Tools and timing
        if routing.tools_called:
            print(f"  tools: {', '.join(routing.tools_called)}")
        response_preview = routing.response_text[:150].replace("\n", " ")
        print(f"  response: {response_preview}{'...' if len(routing.response_text) > 150 else ''}")
        print(f"  time: {routing.elapsed_secs:.1f}s")

        results.append(mr)
        time.sleep(INTER_MESSAGE_DELAY)

    return SequenceResult(sequence=seq, message_results=results)


def print_summary(all_results: list[SequenceResult], mode: str) -> dict:
    """Print aggregate summary and return metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — mode: {mode}")
    print(f"{'=' * 60}")

    total_msgs = 0
    total_topic_checks = 0
    topic_correct = 0
    total_recall_checks = 0
    recall_correct = 0
    total_reask_checks = 0
    reask_avoided = 0
    total_creation_checks = 0
    creation_correct = 0
    total_time = 0.0
    errors = 0

    for sr in all_results:
        for mr in sr.message_results:
            total_msgs += 1
            total_time += mr.routing.elapsed_secs
            if mr.routing.error:
                errors += 1
            if mr.topic_correct is not None:
                total_topic_checks += 1
                if mr.topic_correct:
                    topic_correct += 1
            if mr.context_recalled is not None:
                total_recall_checks += 1
                if mr.context_recalled:
                    recall_correct += 1
            if mr.avoided_reask is not None:
                total_reask_checks += 1
                if mr.avoided_reask:
                    reask_avoided += 1
            if mr.workspace_created_correct is not None:
                total_creation_checks += 1
                if mr.workspace_created_correct:
                    creation_correct += 1

    print(f"\nMessages sent:       {total_msgs}")
    print(f"Errors:              {errors}")
    print(f"Avg response time:   {total_time/max(total_msgs,1):.1f}s")
    print(f"Total time:          {total_time:.0f}s")
    print()

    if total_topic_checks:
        pct = topic_correct / total_topic_checks * 100
        print(f"Topic routing:       {topic_correct}/{total_topic_checks} ({pct:.0f}%)")

    if total_creation_checks:
        pct = creation_correct / total_creation_checks * 100
        print(f"Creation judgment:   {creation_correct}/{total_creation_checks} ({pct:.0f}%)")

    if total_recall_checks:
        pct = recall_correct / total_recall_checks * 100
        print(f"Context recall:      {recall_correct}/{total_recall_checks} ({pct:.0f}%)")

    if total_reask_checks:
        pct = reask_avoided / total_reask_checks * 100
        print(f"Avoided re-asking:   {reask_avoided}/{total_reask_checks} ({pct:.0f}%)")

    # Per-sequence breakdown
    print(f"\nPer-sequence breakdown:")
    for sr in all_results:
        name = sr.sequence.name
        checks = []
        for mr in sr.message_results:
            if mr.topic_correct is not None:
                checks.append(("T", mr.topic_correct))
            if mr.workspace_created_correct is not None:
                checks.append(("C", mr.workspace_created_correct))
            if mr.context_recalled is not None:
                checks.append(("R", mr.context_recalled))
            if mr.avoided_reask is not None:
                checks.append(("A", mr.avoided_reask))

        if checks:
            passed = sum(1 for _, v in checks if v)
            total = len(checks)
            detail = " ".join(f"{label}:{'ok' if v else 'X'}" for label, v in checks)
            print(f"  {name:30s}  {passed}/{total}  {detail}")
        else:
            print(f"  {name:30s}  (no checks)")

    return {
        "mode": mode,
        "total_msgs": total_msgs,
        "errors": errors,
        "topic_routing": f"{topic_correct}/{total_topic_checks}" if total_topic_checks else "n/a",
        "creation_judgment": f"{creation_correct}/{total_creation_checks}" if total_creation_checks else "n/a",
        "context_recall": f"{recall_correct}/{total_recall_checks}" if total_recall_checks else "n/a",
        "reask_avoidance": f"{reask_avoided}/{total_reask_checks}" if total_reask_checks else "n/a",
        "avg_time": f"{total_time/max(total_msgs,1):.1f}s",
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_health(base_url: str, token: str) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        status, _ = http_get(base_url, "/api/health", headers)
        return status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Workspace A/B validation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cost estimate per run:
  Haiku 4.5:    ~$0.16  (36 messages, ~100K input + ~10K output tokens)
  Sonnet 4:     ~$0.65
  --mode both:  double the above

Signal map:
  T=topic routing  C=creation judgment  R=context recall  A=re-ask avoidance
        """,
    )
    parser.add_argument("--port", type=int, default=3003)
    parser.add_argument("--token", default="andrew-dev-token")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", type=int, default=CHAT_TIMEOUT)
    parser.add_argument("--mode", choices=["workspaces", "baseline", "both"], default="workspaces",
                        help="workspaces: full routing, baseline: single thread, both: run both and compare")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Run only named sequences (e.g. --sequences home_assistant_recurring one_off_questions)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("Workspace A/B Validation Harness")
    print(f"Target: {base_url}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    if not check_health(base_url, args.token):
        print(f"\nERROR: Cannot reach IronClaw at {base_url}")
        print("Start IronClaw first, then re-run this harness.")
        return 1

    print("Health check: OK")

    # Filter sequences if requested, respecting dependencies
    sequences = SEQUENCES
    if args.sequences:
        # Collect requested + their dependencies
        requested = set(args.sequences)
        all_names = {s.name for s in SEQUENCES}
        to_run = set()
        queue = list(requested)
        while queue:
            name = queue.pop()
            if name in to_run:
                continue
            to_run.add(name)
            seq = next((s for s in SEQUENCES if s.name == name), None)
            if seq:
                for dep in seq.depends_on:
                    if dep not in to_run:
                        queue.append(dep)

        # Preserve original order
        sequences = [s for s in SEQUENCES if s.name in to_run]
        added_deps = to_run - requested
        if added_deps:
            print(f"Auto-added dependencies: {', '.join(sorted(added_deps))}")

        if not sequences:
            print(f"No matching sequences. Available: {[s.name for s in SEQUENCES]}")
            return 1

    # Count checks
    total_checks = 0
    check_counts = {"topic": 0, "creation": 0, "recall": 0, "reask": 0}
    for seq in sequences:
        for msg in seq.messages:
            if msg.expect_topic_keywords is not None:
                check_counts["topic"] += 1
                total_checks += 1
            if msg.expect_workspace_created is not None:
                check_counts["creation"] += 1
                total_checks += 1
            if msg.expect_recall_any is not None:
                check_counts["recall"] += 1
                total_checks += 1
            if msg.expect_no_reask is not None:
                check_counts["reask"] += 1
                total_checks += 1

    msg_count = sum(len(s.messages) for s in sequences)
    print(f"Sequences: {len(sequences)} ({msg_count} messages, {total_checks} checks)")
    print(f"  T(topic)={check_counts['topic']}  C(creation)={check_counts['creation']}  "
          f"R(recall)={check_counts['recall']}  A(reask)={check_counts['reask']}")

    modes = ["workspaces", "baseline"] if args.mode == "both" else [args.mode]

    all_summaries = []
    for mode in modes:
        print(f"\n{'#' * 60}")
        print(f"# Running in {mode.upper()} mode")
        print(f"{'#' * 60}")

        if mode == "baseline":
            print("\nBASELINE MODE: All messages go to a single thread.")
            print("The LLM must rely on memory_search for context recall.")
            print("(In practice: send with a fixed thread_id, no workspace routing)")
            # TODO: For baseline mode we need to either:
            # 1. Disable workspace routing server-side (env var)
            # 2. Send all messages with a fixed thread_id to bypass routing
            # For now we just note this and run normally
            print("NOTE: baseline mode not yet implemented — running with routing enabled")

        results = []
        for seq in sequences:
            sr = run_sequence(base_url, args.token, seq, mode, args.timeout)
            results.append(sr)

        summary = print_summary(results, mode)
        all_summaries.append(summary)

    # Comparison if both modes ran
    if len(all_summaries) == 2:
        print(f"\n{'=' * 60}")
        print("A/B COMPARISON")
        print(f"{'=' * 60}")
        for key in ["context_recall", "reask_avoidance", "creation_judgment", "topic_routing", "avg_time"]:
            a = all_summaries[0].get(key, "n/a")
            b = all_summaries[1].get(key, "n/a")
            print(f"  {key:25s}  workspaces={a:12s}  baseline={b:12s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
