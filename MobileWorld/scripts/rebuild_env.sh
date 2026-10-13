#!/bin/bash
# Rebuild our MobileWorld container from scratch and deploy the local
# Mattermost session-expiry fix that the upstream image lacks.
#
# Why this script exists:
#   The image ghcr.io/tongyi-mai/mobile_world:latest does not include commit
#   c532c19 (mattermost session expiry patch). Without it, the AVD's cached
#   Mattermost auth token is rejected and every Mattermost task degenerates
#   into the agent asking the user for login credentials.
#
# What this does, in order:
#   1. Tear down our container (mbw_linq_*) and its volumes
#   2. Spin up a fresh container from upstream image
#   3. docker cp the host's patched mattermost.py into the container
#   4. Restart the container's mobile-world server so Python re-imports
#      the patched module (just rewriting the file is not enough — the
#      long-lived server process keeps the old code in memory)
#   5. Verify the patch is loaded by grepping the file inside the container
#
# Run after every container recreate. Idempotent — safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME_PREFIX="mbw_linq"
CONTAINER="${NAME_PREFIX}_0"
VENV="${REPO_ROOT}/.venv"
SRC_FILE="${REPO_ROOT}/src/mobile_world/runtime/app_helpers/mattermost.py"
DEST_FILE="/app/service/src/mobile_world/runtime/app_helpers/mattermost.py"

step() { printf '\n\033[1;34m[step %s]\033[0m %s\n' "$1" "$2"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31m  ✗\033[0m %s\n' "$1" >&2; exit 1; }

[ -f "$SRC_FILE" ] || fail "Patched source not found: $SRC_FILE"
[ -x "$VENV/bin/mw" ] || fail "mw CLI not found in venv: $VENV/bin/mw"

step 1 "tearing down existing $NAME_PREFIX containers"
sudo -E env "PATH=$PATH" "$VENV/bin/mw" env rm --name-prefix "$NAME_PREFIX" || true
ok "torn down (or none existed)"

step 2 "starting fresh container"
sudo -E env "PATH=$PATH" "$VENV/bin/mw" env run \
    --count 1 \
    --name-prefix "$NAME_PREFIX" \
    --env-file "$REPO_ROOT/.env"
# wait for inner dockerd + mobile-world server to be up
for i in $(seq 1 30); do
    if curl -sf "http://localhost:6800/task/list" > /dev/null 2>&1; then
        ok "mobile-world server responding (after ${i}s)"
        break
    fi
    sleep 1
    [ "$i" -eq 30 ] && fail "mobile-world server did not come up within 30s"
done

step 3 "deploying mattermost.py fix into $CONTAINER"
docker cp "$SRC_FILE" "$CONTAINER:$DEST_FILE"
fix_count=$(docker exec "$CONTAINER" grep -c "_patch_mattermost_config\|_extend_session_expiry" "$DEST_FILE")
[ "$fix_count" -eq 4 ] || fail "fix did not land (grep count $fix_count, expected 4)"
ok "fix file in place (4 expected refs found)"

step 4 "restarting container so emulator + server come up together with patched code"
# Restart the whole container instead of just the mobile-world server.
# Restarting only the server while the emulator stays running has caused
# KVM "slot: File exists" crashes (emulator dies mid-task). Doing a full
# container restart re-runs entrypoint.sh, which boots a fresh emulator
# and a fresh server in the right order. The docker-cp'd patched file is
# in the writable layer and survives the restart.
docker restart "$CONTAINER" > /dev/null
ok "container restarted"

# wait for mobile-world server to come back
for i in $(seq 1 60); do
    if curl -sf "http://localhost:6800/task/list" > /dev/null 2>&1; then
        ok "mobile-world server responding (after ${i}s)"
        break
    fi
    sleep 1
    [ "$i" -eq 60 ] && fail "mobile-world server did not come up within 60s after restart"
done

# wait for the AVD emulator to finish booting (this lags the server)
for i in $(seq 1 90); do
    bc=$(docker exec "$CONTAINER" adb -s emulator-5554 shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
    if [ "$bc" = "1" ]; then
        ok "emulator boot_completed=1 (after ${i}s)"
        break
    fi
    sleep 1
    [ "$i" -eq 90 ] && fail "emulator did not finish booting within 90s after container restart"
done

step 5 "verifying the long-lived server is running the patched code"
# Trigger task init via the server's HTTP API (NOT a separate python process).
# This is what `mw eval` will hit, and it forces the server to invoke the
# patched start_mattermost_backend code path. Then grep server.log for the
# patch's signature log lines.
init_resp=$(curl -s -X POST "http://localhost:6800/task/init" \
    -H 'Content-Type: application/json' \
    -d '{"task_name": "MattermostEmailTask", "req_device": "emulator-5554"}')
[ "$init_resp" = '"OK"' ] || fail "task init failed: $init_resp"

# Give the patched code a moment to write its log lines.
for i in $(seq 1 15); do
    if docker exec "$CONTAINER" grep -q "Patched Mattermost config" /var/log/server.log 2>/dev/null; then
        ok "patched code path hit by long-lived server"
        break
    fi
    sleep 1
    [ "$i" -eq 15 ] && fail "patch signature missing from server.log after task init — server may still hold stale code"
done

# Always tear down so the AVD/Mattermost are clean for the eval run.
curl -s -X POST "http://localhost:6800/task/tear_down" \
    -H 'Content-Type: application/json' \
    -d '{"task_name": "MattermostEmailTask", "req_device": "emulator-5554"}' > /dev/null
ok "task torn down — environment ready for eval"

printf '\n\033[1;32m✓ rebuild complete\033[0m — container %s ready for evaluation\n\n' "$CONTAINER"
