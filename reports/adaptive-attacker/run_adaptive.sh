#!/usr/bin/env bash
# Pilot adaptive mutation search against the served D23 bundle (dev split, small budget). No API key:
# the search is black-box over the gateway's local DeBERTa (the LLM only pre-wrote the benign carriers).
#
#   flock /home/claude/measure.lock bash reports/adaptive-attacker/run_adaptive.sh [BUDGET]
set -euo pipefail
export NO_PROXY='*' no_proxy='*'
BUDGET="${1:-20}"
wait_up() { for i in $(seq 1 120); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway $1 did not start" >&2; return 1; }

SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" \
  python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
wait_up http://127.0.0.1:8000
python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
  --goal-split dev --budget "$BUDGET" --out-dir reports/adaptive-attacker/pilot
echo "done"
