#!/usr/bin/env bash
# Run the workspace A/B test harness.
#
# Usage:
#   ./tests/run-ws-test.sh                    # workspaces mode (default)
#   ./tests/run-ws-test.sh --mode baseline    # baseline mode
#   ./tests/run-ws-test.sh --mode both        # A/B comparison
#   ./tests/run-ws-test.sh --sequences one_off_questions interleaved_topics
#
# Prerequisites:
#   - PostgreSQL running on localhost:5434 (Docker: percy-postgres)
#   - Ollama running with nomic-embed-text model
#   - Binary built: cargo build --release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$SCRIPT_DIR/.env.ws-test"
BINARY="$ROOT_DIR/target/release/ironclaw"
PORT=3003
TOKEN="andrew-dev-token"
SOUL_MD="$ROOT_DIR/../shared/SOUL.md"

# Check prerequisites
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run: cargo build --release"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Test env not found at $ENV_FILE"
    exit 1
fi

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama not running. Start it: ollama serve"
    exit 1
fi

if ! docker exec percy-postgres psql -U percy -d percy_ws_test -c "SELECT 1;" > /dev/null 2>&1; then
    echo "ERROR: percy_ws_test database not found."
    echo "Create it: docker exec percy-postgres psql -U percy -c 'CREATE DATABASE percy_ws_test;'"
    echo "Then: docker exec percy-postgres psql -U percy -d percy_ws_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
    exit 1
fi

# Option: reset the database for a clean run
if [ "${RESET_DB:-0}" = "1" ]; then
    echo "Resetting percy_ws_test database..."
    docker exec percy-postgres psql -U percy -c "DROP DATABASE IF EXISTS percy_ws_test;"
    docker exec percy-postgres psql -U percy -c "CREATE DATABASE percy_ws_test;"
    docker exec percy-postgres psql -U percy -d percy_ws_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
    echo "Database reset."
fi

# Kill any existing IronClaw on this port
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "Stopping existing process on port $PORT..."
    kill $(lsof -ti:$PORT) 2>/dev/null || true
    sleep 1
fi

# Start IronClaw with test env
# Use a FIFO for stdin so the REPL doesn't get EOF and quit
FIFO="/tmp/ironclaw-ws-stdin"
rm -f "$FIFO"
mkfifo "$FIFO"
echo "Starting IronClaw (workspace branch) on port $PORT..."
env $(grep -v '^#' "$ENV_FILE" | xargs) "$BINARY" < "$FIFO" > /tmp/ironclaw-ws-test.log 2>&1 &
IC_PID=$!
# Keep the FIFO open with a background cat that never writes
exec 3>"$FIFO"
echo "PID: $IC_PID (log: /tmp/ironclaw-ws-test.log)"

# Wait for health
echo -n "Waiting for health check..."
for i in $(seq 1 30); do
    if curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/api/health" > /dev/null 2>&1; then
        echo " OK (${i}s)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo " FAILED after 30s"
        echo "Last log lines:"
        tail -20 /tmp/ironclaw-ws-test.log
        kill $IC_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Seed SOUL.md (workspace guidance for the LLM)
if [ -f "$SOUL_MD" ]; then
    echo "Seeding SOUL.md..."
    python3 -c "
import json, urllib.request
with open('$SOUL_MD') as f:
    content = f.read()
body = json.dumps({'path': 'SOUL.md', 'content': content}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:$PORT/api/memory/write',
    data=body,
    headers={'Authorization': 'Bearer $TOKEN', 'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=10) as resp:
    print(f'  SOUL.md seeded ({resp.status})')
"
else
    echo "WARNING: SOUL.md not found at $SOUL_MD — LLM won't have workspace guidance"
fi

# Run the test harness
echo ""
echo "Running test harness..."
echo "================================================"
python3 "$SCRIPT_DIR/test_workspace_switching.py" \
    --port $PORT --token $TOKEN "$@"
TEST_EXIT=$?

# Cleanup
echo ""
echo "Stopping IronClaw (PID $IC_PID)..."
exec 3>&-  # close FIFO writer
rm -f "$FIFO"
kill $IC_PID 2>/dev/null || true
wait $IC_PID 2>/dev/null || true

exit $TEST_EXIT
