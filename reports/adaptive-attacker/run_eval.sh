#!/usr/bin/env bash
# Eval-goal headline runs against the served D23 bundle: black-box decision-only, then the
# score-feedback ablation. One gateway for both phases (the model loads once). The API key must
# already be in the environment; it is never printed.
#
#   set -a; . /root/.sentinel/.env; set +a
#   flock /home/claude/measure.lock bash reports/adaptive-attacker/run_eval.sh [BUDGET]
set -euo pipefail
export NO_PROXY='*' no_proxy='*'
BUDGET="${1:-20}"
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then echo "ANTHROPIC_API_KEY not set (load /root/.sentinel/.env)" >&2; exit 2; fi
wait_up() { for i in $(seq 1 120); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway did not start" >&2; return 1; }

SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" \
  python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
wait_up http://127.0.0.1:8000

python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
  --goal-split eval --budget "$BUDGET" --max-usd 15 \
  --out-dir reports/adaptive-attacker/eval_decision
python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
  --goal-split eval --budget "$BUDGET" --max-usd 15 --score-feedback \
  --out-dir reports/adaptive-attacker/eval_score
echo "done"
