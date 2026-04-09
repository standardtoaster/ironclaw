#!/usr/bin/env bash
#
# verify_session_continuity.sh — before/after test for session continuity fix
#
# Demonstrates the behavioral difference:
#   BEFORE: each message spawns a new thread (different thread_ids)
#   AFTER:  follow-up messages resume the same thread (same thread_id)
#
# Usage:
#   ./tests/verify_session_continuity.sh [base_url] [token]
#
# Defaults: http://localhost:8080, e2e-test-token
# Requires: curl, jq, timeout (coreutils)

set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
TOKEN="${2:-e2e-test-token}"
TIMEOUT=30

log()  { printf '\033[1m%s\033[0m\n' "$*"; }
pass() { printf '\033[32m✓ %s\033[0m\n' "$*"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$*"; exit 1; }

# Extract thread_id from SSE response event.
# Opens an SSE stream, sends a message, waits for the "response" event,
# returns the thread_id from its JSON payload.
send_and_get_thread_id() {
    local message="$1"
    local tmp_events
    tmp_events=$(mktemp)

    # Start SSE listener in background — capture events until we see a response
    curl -sN "${BASE_URL}/api/chat/events?token=${TOKEN}" \
        -H "Accept: text/event-stream" \
        > "$tmp_events" 2>/dev/null &
    local sse_pid=$!

    # Give SSE a moment to connect
    sleep 1

    # Send the message
    local send_result
    send_result=$(curl -sf -X POST "${BASE_URL}/api/chat/send" \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"${message}\"}" 2>&1) || {
        kill "$sse_pid" 2>/dev/null; rm -f "$tmp_events"
        fail "Failed to send message: ${send_result}"
    }

    # Wait for a response event (up to TIMEOUT seconds)
    local elapsed=0
    local thread_id=""
    while [ $elapsed -lt $TIMEOUT ]; do
        # Look for a response event in the captured SSE stream
        thread_id=$(grep -A1 'event: response' "$tmp_events" 2>/dev/null \
            | grep '^data:' \
            | head -1 \
            | sed 's/^data: *//' \
            | jq -r '.thread_id // empty' 2>/dev/null || true)

        if [ -n "$thread_id" ]; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    kill "$sse_pid" 2>/dev/null || true
    rm -f "$tmp_events"

    if [ -z "$thread_id" ]; then
        fail "No response event within ${TIMEOUT}s for message: ${message}"
    fi

    echo "$thread_id"
}

# ── Main ────────────────────────────────────────────────────

log "Session continuity verification"
log "Target: ${BASE_URL}"
echo

# First, create a fresh thread so we start clean
log "Creating fresh thread..."
curl -sf -X POST "${BASE_URL}/api/chat/thread/new" \
    -H "Authorization: Bearer ${TOKEN}" > /dev/null 2>&1 || true

# Turn 1: send first message
log "Turn 1: sending message..."
THREAD_1=$(send_and_get_thread_id "Hello, what is your name?")
log "  thread_id = ${THREAD_1}"

# Brief pause between turns
sleep 2

# Turn 2: send follow-up message to same conversation
log "Turn 2: sending follow-up..."
THREAD_2=$(send_and_get_thread_id "What did I just ask you?")
log "  thread_id = ${THREAD_2}"

echo
# ── Verdict ─────────────────────────────────────────────────

if [ "$THREAD_1" = "$THREAD_2" ]; then
    pass "PASS: Same thread_id across turns — session continuity works"
    echo "  Both turns used thread: ${THREAD_1}"
else
    fail "FAIL: Different thread_ids — new thread spawned per message (bug)
  Turn 1: ${THREAD_1}
  Turn 2: ${THREAD_2}"
fi
