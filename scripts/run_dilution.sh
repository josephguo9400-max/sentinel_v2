#!/usr/bin/env bash
# Fill the dilution-curve score cache (training/dilution_curve.py), one bounded chunk per lock hold.
#   scripts/run_dilution.sh "510 128" 300
set -euo pipefail
cd "$(dirname "$0")/.."
WINDOWS=${1:-"510"}
CHUNK=${2:-300}
for w in $WINDOWS; do
  while true; do
    rc=0
    flock /home/claude/measure.lock python3 -m training.dilution_curve score --window "$w" --max-items "$CHUNK" || rc=$?
    [ "$rc" -eq 10 ] && continue
    [ "$rc" -eq 0 ] && break
    exit "$rc"
  done
done
