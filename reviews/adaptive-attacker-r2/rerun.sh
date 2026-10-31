#!/usr/bin/env bash
# Reviewer reruns (scratch outputs only): reproduce eval_decision seed 0, seeds 1-2, and matched swap controls.
set -euo pipefail
export NO_PROXY='*' no_proxy='*'
S=${S:?set S to a scratch dir holding raw/<run>.jsonl copied from the implementer's gitignored queries.jsonl}
cd /home/claude/wt-verify-att2
SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!
trap 'kill $APID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -s --noproxy '*' http://127.0.0.1:8000/health >/dev/null && break; sleep 1; done
curl -s --noproxy '*' http://127.0.0.1:8000/health; echo
python3 $S/swap_ctrl.py $S/raw/eval_decision.jsonl $S/swap_decision.json
python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 --goal-split eval --budget 100 --out-dir $S/rep_seed0
for sd in 1 2; do
python3 -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 --goal-split eval --budget 100 --seed $sd --no-controls --out-dir $S/rep_seed$sd
done
echo done
