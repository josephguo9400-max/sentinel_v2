#!/usr/bin/env bash
# Adaptive attacker against the served D23 bundle (artifacts/registry.json, bundle 2ebc7d9a…).
# Starts the gateway, waits for health, then runs `python -m redteam.adaptive.run attack` with the
# arguments passed after `--`. The API key must already be in the environment; it is never printed.
#
# Hold the shared CPU lock for the whole run (DeBERTa scoring competes with the frozen-test gate job):
#   set -a; . /root/.sentinel/.env; set +a
#   flock /home/claude/measure.lock bash reports/adaptive-attacker/run_adaptive.sh -- \
#       --gateway-url http://127.0.0.1:8000 --goal-split dev --budget 8 --max-usd 1 --pilot \
#       --out-dir reports/adaptive-attacker/pilot
set -euo pipefail
export NO_PROXY='*' no_proxy='*'
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then echo "ANTHROPIC_API_KEY not set (load /root/.sentinel/.env)" >&2; exit 2; fi
# Split args: everything after `--` goes to the attacker CLI.
ATTACK_ARGS=()
seen=0
for a in "$@"; do if [[ "$a" == "--" ]]; then seen=1; continue; fi; if [[ $seen == 1 ]]; then ATTACK_ARGS+=("$a"); fi; done
if [[ ${#ATTACK_ARGS[@]} -eq 0 ]]; then echo "usage: run_adaptive.sh -- <attacker args>" >&2; exit 2; fi

wait_up() { for i in $(seq 1 120); do curl -s --noproxy '*' "$1/health" >/dev/null && return 0; sleep 1; done; echo "gateway $1 did not start" >&2; return 1; }

SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" \
  python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
wait_up http://127.0.0.1:8000
python3 -m redteam.adaptive.run attack "${ATTACK_ARGS[@]}"
echo "done"
