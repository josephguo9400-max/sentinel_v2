#!/usr/bin/env bash
# Milestone cascade-and-shadow checkpoint measurements. Run from the repo root under the shared lock:
#   flock /home/claude/measure.lock bash reports/cascade-and-shadow/run_checkpoint.sh
# BEFORE = the gateway-mvp gateway (worktree at $BEFORE_DIR); AFTER = this branch with registry_integration.json.
set -euo pipefail
OUT=reports/cascade-and-shadow
BEFORE_DIR=${BEFORE_DIR:-/home/claude/wt-before}
export NO_PROXY='*' no_proxy='*'
wait_up() { for i in $(seq 1 120); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway $1 did not start" >&2; return 1; }
uptime > "$OUT/uptime_start.txt"

( cd "$BEFORE_DIR" && exec python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8001 --log-level warning ) &
BPID=$!
wait_up http://127.0.0.1:8001
python3 -m redteam.probe_runner --gateway-url http://127.0.0.1:8001 --label "before: gateway-mvp" --out "$OUT/probes_before.json" > /dev/null
kill $BPID; wait $BPID 2>/dev/null || true

SENTINEL_REGISTRY_PATH="$PWD/$OUT/registry_integration.json" python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
wait_up http://127.0.0.1:8000
curl -s --noproxy '*' http://127.0.0.1:8000/health > "$OUT/health_after.json"
python3 -m redteam.probe_runner --gateway-url http://127.0.0.1:8000 --label "after: cascade-and-shadow" --out "$OUT/probes_after.json" > /dev/null
python3 -m training.replay_runner --url http://127.0.0.1:8000 --rps 2 --mode open --max-inflight 8 --out-dir "$OUT/replay_after" > "$OUT/replay_after_stdout.txt"
curl -s --noproxy '*' http://127.0.0.1:8000/metrics > "$OUT/metrics_after.txt"
kill $APID; wait $APID 2>/dev/null || true
uptime > "$OUT/uptime_end.txt"
echo done
