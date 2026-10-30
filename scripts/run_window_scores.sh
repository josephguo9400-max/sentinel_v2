#!/usr/bin/env bash
# Fill the window-score caches for the dilution study (training/DILUTION.md), one bounded chunk per lock hold so
# the shared CPU lock is released between chunks. Resumable: rerun after an interruption.
#   scripts/run_window_scores.sh "val replay probes" "256 128 64" 300
set -euo pipefail
cd "$(dirname "$0")/.."
SETS=${1:-"val replay probes"}
WINDOWS=${2:-"510 256 128 64"}
CHUNK=${3:-300}
for set in $SETS; do
  for w in $WINDOWS; do
    while true; do
      rc=0
      flock /home/claude/measure.lock python3 -m training.window_scores --set "$set" --window "$w" --max-items "$CHUNK" || rc=$?
      [ "$rc" -eq 10 ] && continue
      [ "$rc" -eq 0 ] && break
      exit "$rc"
    done
  done
done
