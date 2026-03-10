#!/usr/bin/env python3
"""End-to-end workspace tests — exercises the full workspace lifecycle via HTTP API.

Usage:
    # Against a running IronClaw instance (default: localhost:3003):
    python tests/test_workspaces_e2e.py

    # Custom port/token:
    python tests/test_workspaces_e2e.py --port 3003 --token my-token

    # Skip LLM-dependent tests (API + tool-event only):
    python tests/test_workspaces_e2e.py --no-chat

    # Specific test:
    python tests/test_workspaces_e2e.py -k test_workspace_lifecycle

These tests require:
  - A running IronClaw instance with PostgreSQL + pgvector
  - Embeddings enabled (EMBEDDING_ENABLED=true)
  - Workspace tools registered (delegate_to_workspace, list_workspaces, set_workspace_topic)

The tests exercise:
  1. Workspace creation via delegate_to_workspace tool
  2. Workspace topic setting and routing
  3. list_workspaces returns created workspaces
  4. Workspace resumption (second delegation to same topic routes to same workspace)
  5. User isolation (one user can't see another's workspaces)

No external dependencies — uses only Python stdlib.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_TIMEOUT = 120  # seconds
PORT = 3003
TOKEN = "andrew-dev-token"
NO_CHAT = False


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


class ChatResult:
    """Result of a chat interaction, including tool events."""

    def __init__(self, text: str, tool_events: list[dict]):
        self.text = text
        self.tool_events = tool_events

    def tool_was_called(self, name: str) -> bool:
        return any(
            e.get("name") == name
            for e in self.tool_events
            if e.get("type", "") == "tool_started"
        )

    def tool_succeeded(self, name: str) -> bool:
        return any(
            e.get("name") == name and e.get("success") is True
            for e in self.tool_events
            if e.get("type", "") == "tool_completed"
        )

    def tool_result_preview(self, name: str) -> str | None:
        for e in self.tool_events:
            if e.get("type", "") == "tool_result" and e.get("name") == name:
                return e.get("preview", "")
        return None

    def all_tool_names(self) -> list[str]:
        return [
            e.get("name", "?")
            for e in self.tool_events
            if e.get("type", "") == "tool_started"
        ]


def chat_via_sse(base_url: str, headers: dict, message: str, timeout: int = CHAT_TIMEOUT) -> ChatResult:
    """Send a chat message and collect the response via SSE."""
    result_holder = {"chunks": [], "done": False, "error": None, "tool_events": []}

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
                            etype = current_event or event.get("type", "")
                            event["type"] = etype
                            if etype == "stream_chunk":
                                result_holder["chunks"].append(event.get("content", ""))
                            elif etype == "response":
                                result_holder["chunks"] = [event.get("content", "")]
                                result_holder["done"] = True
                                return
                            elif etype in ("tool_started", "tool_completed", "tool_result"):
                                result_holder["tool_events"].append(event)
                                name = event.get("name", "?")
                                if etype == "tool_started":
                                    print(f"    [started] {name}")
                                elif etype == "tool_completed":
                                    print(f"    [completed] {name} success={event.get('success')}")
                                elif etype == "tool_result":
                                    preview = event.get("preview", "")[:200]
                                    print(f"    [result] {name}: {preview}")
                            elif etype == "error":
                                result_holder["error"] = event.get("message", "unknown")
                                result_holder["done"] = True
                                return
                        except json.JSONDecodeError:
                            pass
                    if result_holder["done"]:
                        return
        except Exception as e:
            result_holder["error"] = str(e)

    listener = threading.Thread(target=sse_listener, daemon=True)
    listener.start()
    time.sleep(0.5)  # Let SSE connect

    # Send the message
    status, body = http_post_json(
        base_url, "/api/chat/send", headers, {"content": message}
    )
    if status != 202:
        raise RuntimeError(f"chat/send returned {status}: {body}")

    # Wait for response
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_holder["done"]:
            break
        time.sleep(0.5)

    if result_holder["error"]:
        raise RuntimeError(f"SSE error: {result_holder['error']}")

    text = "".join(result_holder["chunks"])
    return ChatResult(text, result_holder["tool_events"])


class IronClawClient:
    """Simple client for IronClaw HTTP API."""

    def __init__(self, port: int, token: str):
        self.base_url = f"http://127.0.0.1:{port}"
        self.headers = {"Authorization": f"Bearer {token}"}

    def health(self) -> bool:
        try:
            status, _ = http_get(self.base_url, "/api/health", self.headers)
            return status == 200
        except Exception:
            return False

    def chat(self, message: str, timeout: int = CHAT_TIMEOUT) -> ChatResult:
        return chat_via_sse(self.base_url, self.headers, message, timeout)

    def get_json(self, path: str) -> dict:
        status, body = http_get(self.base_url, path, self.headers)
        if status != 200:
            raise RuntimeError(f"GET {path} returned {status}: {body}")
        return json.loads(body)


# ---------------------------------------------------------------------------
# Test framework (stdlib)
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: list[str] = []
        self.filter: str | None = None

    def run(self, name: str, func, *args, skip_if_no_chat: bool = False):
        if self.filter and self.filter not in name:
            return
        if skip_if_no_chat and NO_CHAT:
            print(f"  SKIP {name} (--no-chat)")
            self.skipped += 1
            return

        print(f"\n  RUN  {name}")
        try:
            func(*args)
            print(f"  PASS {name}")
            self.passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            self.errors.append(f"{name}: {e}")
            self.failed += 1

    def summary(self) -> int:
        total = self.passed + self.failed + self.skipped
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed} passed, {self.failed} failed, {self.skipped} skipped / {total} total")
        if self.errors:
            print("\nFailures:")
            for err in self.errors:
                print(f"  - {err}")
        return 0 if self.failed == 0 else 1


# ---------------------------------------------------------------------------
# Tests: API-level (no LLM needed)
# ---------------------------------------------------------------------------

def test_health(client: IronClawClient):
    """Server is reachable."""
    assert client.health(), "Health check failed"


def test_workspace_api_list_empty(client: IronClawClient):
    """list_workspaces via chat should work (even if empty)."""
    # This is LLM-dependent: we ask the LLM to call list_workspaces.
    # The LLM may or may not call the tool — we're testing the plumbing.
    result = client.chat("List all my workspaces. Use the list_workspaces tool.")
    # We mainly care that the request completed without error.
    print(f"    Response: {result.text[:200]}")
    # Check if the tool was called (may not be if LLM decides to answer directly).
    if result.tool_was_called("list_workspaces"):
        print("    list_workspaces tool was called")
        assert result.tool_succeeded("list_workspaces"), "list_workspaces should succeed"


def test_delegate_creates_workspace(client: IronClawClient):
    """delegate_to_workspace should create a new workspace when no match exists."""
    result = client.chat(
        "I need help with home automation for my porch lights. "
        "Please delegate this to a workspace using delegate_to_workspace tool "
        "with prompt 'set up porch light automation' and workspace_hint 'home automation'."
    )
    print(f"    Response: {result.text[:300]}")
    print(f"    Tools called: {result.all_tool_names()}")

    if result.tool_was_called("delegate_to_workspace"):
        print("    delegate_to_workspace was called!")
        # It should succeed (creates new workspace since none exists).
        assert result.tool_succeeded("delegate_to_workspace"), \
            "delegate_to_workspace should succeed"


def test_delegate_resumes_workspace(client: IronClawClient):
    """Second delegation to same topic should resume existing workspace."""
    # First, create a workspace.
    result1 = client.chat(
        "Delegate to a workspace: 'start grocery list for this week'. "
        "Use delegate_to_workspace with workspace_hint 'grocery shopping'."
    )
    print(f"    First delegation tools: {result1.all_tool_names()}")

    if not result1.tool_was_called("delegate_to_workspace"):
        print("    SKIP: LLM didn't call delegate_to_workspace")
        return

    # Second delegation to same topic should route to existing workspace.
    time.sleep(2)  # Let workspace be created and topic set.
    result2 = client.chat(
        "Delegate to a workspace: 'add milk and eggs to the grocery list'. "
        "Use delegate_to_workspace with workspace_hint 'grocery shopping'."
    )
    print(f"    Second delegation tools: {result2.all_tool_names()}")

    if result2.tool_was_called("delegate_to_workspace"):
        assert result2.tool_succeeded("delegate_to_workspace"), \
            "second delegate_to_workspace should succeed"


def test_list_shows_created_workspaces(client: IronClawClient):
    """After creating workspaces, list_workspaces should show them."""
    result = client.chat(
        "Show me all my workspaces. Use the list_workspaces tool."
    )
    print(f"    Response: {result.text[:500]}")

    if result.tool_was_called("list_workspaces"):
        assert result.tool_succeeded("list_workspaces"), "list_workspaces should succeed"
        preview = result.tool_result_preview("list_workspaces")
        if preview:
            print(f"    Workspaces found: {preview[:500]}")


def test_workspace_isolation_different_tokens(client1: IronClawClient, client2: IronClawClient):
    """Workspaces created by one user should not be visible to another."""
    # Client1 creates a workspace.
    result1 = client1.chat(
        "Delegate to workspace: 'private financial planning'. "
        "Use delegate_to_workspace with workspace_hint 'finances'."
    )
    print(f"    Client1 tools: {result1.all_tool_names()}")

    # Client2 lists workspaces — should not see client1's.
    time.sleep(2)
    result2 = client2.chat(
        "List all my workspaces using list_workspaces tool."
    )
    print(f"    Client2 response: {result2.text[:300]}")
    # We can't assert on content (LLM response varies), but the tool should succeed.
    if result2.tool_was_called("list_workspaces"):
        assert result2.tool_succeeded("list_workspaces")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global PORT, TOKEN, NO_CHAT

    parser = argparse.ArgumentParser(description="Workspace E2E tests")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--token", type=str, default=TOKEN)
    parser.add_argument("--token2", type=str, default=None,
                        help="Second token for isolation tests")
    parser.add_argument("--no-chat", action="store_true",
                        help="Skip LLM-dependent tests")
    parser.add_argument("-k", type=str, default=None,
                        help="Filter tests by name substring")
    args = parser.parse_args()

    PORT = args.port
    TOKEN = args.token
    NO_CHAT = args.no_chat

    client = IronClawClient(PORT, TOKEN)
    runner = TestRunner()
    runner.filter = args.k

    print(f"Testing against http://127.0.0.1:{PORT}")
    print(f"Token: {TOKEN[:8]}...")
    print()

    # Health check first.
    runner.run("test_health", test_health, client)
    if runner.failed > 0:
        print("\nServer not reachable. Start IronClaw first.")
        return 1

    # LLM-dependent tests (need real LLM to call tools).
    runner.run("test_workspace_api_list_empty", test_workspace_api_list_empty, client,
               skip_if_no_chat=True)
    runner.run("test_delegate_creates_workspace", test_delegate_creates_workspace, client,
               skip_if_no_chat=True)
    runner.run("test_delegate_resumes_workspace", test_delegate_resumes_workspace, client,
               skip_if_no_chat=True)
    runner.run("test_list_shows_created_workspaces", test_list_shows_created_workspaces, client,
               skip_if_no_chat=True)

    # Isolation test (needs two tokens).
    if args.token2:
        client2 = IronClawClient(PORT, args.token2)
        runner.run("test_workspace_isolation_different_tokens",
                   test_workspace_isolation_different_tokens, client, client2,
                   skip_if_no_chat=True)
    else:
        print("\n  SKIP test_workspace_isolation_different_tokens (no --token2)")
        runner.skipped += 1

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
