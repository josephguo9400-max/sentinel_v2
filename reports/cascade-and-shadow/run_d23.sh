#!/usr/bin/env bash
# D23 default deployment (artifacts/registry.json: DeBERTa decides, Stage 1 observe-only): probes v2 + live replay.
#   flock /home/claude/measure.lock bash reports/cascade-and-shadow/run_d23.sh
set -euo pipefail
OUT=reports/cascade-and-shadow/d23
mkdir -p "$OUT"
export NO_PROXY='*' no_proxy='*'
wait_up() { for i in $(seq 1 120); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway $1 did not start" >&2; return 1; }
uptime > "$OUT/uptime_start.txt"
SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
wait_up http://127.0.0.1:8000
curl -s --noproxy '*' http://127.0.0.1:8000/health > "$OUT/health.json"
python3 -m redteam.probe_runner --gateway-url http://127.0.0.1:8000 --label "D23: DeBERTa decides, Stage 1 observe" --full --out "$OUT/probes.json" > /dev/null
python3 -m training.replay_runner --url http://127.0.0.1:8000 --rps 2 --mode open --max-inflight 8 --out-dir "$OUT/replay" > "$OUT/replay_stdout.txt"
curl -s --noproxy '*' http://127.0.0.1:8000/metrics > "$OUT/metrics.txt"
uptime > "$OUT/uptime_end.txt"
echo done
