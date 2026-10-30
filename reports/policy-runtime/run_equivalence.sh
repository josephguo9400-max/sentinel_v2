#!/usr/bin/env bash
# Equivalence runs of the policy engine (T3, T4, T5; docs/design/capability-policy.md §8).
#   bash reports/policy-runtime/run_equivalence.sh
# T3 is resumable: it keeps a per-item progress file next to its output and a row-score cache in a scratch
# directory, and each invocation stops after --max-seconds, so it never holds /home/claude/measure.lock for long.
set -euo pipefail
OUT=reports/policy-runtime
CACHE="${CACHE:-${TMPDIR:-/tmp}/t3.rowcache.jsonl}"
mkdir -p "$OUT"

# T3: all 2,000 replay items through the legacy service and the policy engine, real models, timeouts disabled.
until flock /home/claude/measure.lock -c \
    "python -m training.replay_policy_equivalence t3 --out $OUT/t3_d23_equivalence.json --cache '$CACHE' \
     --max-seconds 240" | tee /dev/stderr | grep -q '"complete": true'; do
  sleep 2
done

# T4: the route policy against the committed serving-faithful counterfactual (no DeBERTa inference).
flock /home/claude/measure.lock -c \
  "python -m training.replay_policy_equivalence t4 --out $OUT/t4_cascade_equivalence.json"

# T5: the T3 decisions against the live D23 run (no models).
python -m training.replay_policy_equivalence t5 --t3 $OUT/t3_d23_equivalence.json \
  --out $OUT/t5_live_comparison.json
echo done

# The release gate on the development split, for the bundle the registry serves (resumable; never promotes).
# The frozen test is NOT run here: the owner decides when the final split is read
#   (python -m training.gate evaluate --split frozen_test --frozen-test-approved-by NAME).
until flock /home/claude/measure.lock -c \
    "python -m training.gate evaluate --registry artifacts/registry.json --split val \
     --out $OUT/gate_val_d23.json --max-seconds 240" | tee /dev/stderr | grep -q '"complete": true'; do
  sleep 2
done
