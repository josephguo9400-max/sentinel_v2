#!/usr/bin/env bash
# Adaptive mutation search against the served D23 bundle: black-box decision-only (headline) then the
# score-feedback ablation. The search needs NO API key (it is black-box over the gateway's local
# DeBERTa; the LLM only pre-wrote the committed benign carriers). Run under the measure lock.
#
#   flock /home/claude/measure.lock bash reports/adaptive-attacker/run_eval.sh [BUDGET] [SPLIT]
set -euo pipefail
export NO_PROXY='*' no_proxy='*'   # localhost gateway only; this run makes no outbound API call
BUDGET="${1:-100}"
SPLIT="${2:-eval}"
wait_up() { for i in $(seq 1 180); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway did not start" >&2; return 1; }

SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" \
  python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
wait_up http://127.0.0.1:8000

python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
  --goal-split "$SPLIT" --budget "$BUDGET" --out-dir reports/adaptive-attacker/eval_decision
python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
  --goal-split "$SPLIT" --budget "$BUDGET" --score-feedback \
  --out-dir reports/adaptive-attacker/eval_score
echo "done"
