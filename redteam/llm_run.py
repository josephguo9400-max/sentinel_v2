"""CLI for the LLM defender: record a live run (needs ANTHROPIC_API_KEY) or replay a recorded one.

  # Live run (fails with a clear message if the key is absent):
  python -m redteam.llm_run --scenario dev_daily_digest --mode A --model claude-3-5-haiku-20241022

  # Replay a recorded transcript (no key needed):
  python -m redteam.llm_run --replay fixtures/recorded_runs/<run>.transcript.json --scenario dev_daily_digest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .judge import judge_attempt
from .llm_defender import (DEFAULT_MODEL, MISSING_KEY_MESSAGE, AnthropicLLMClient, LLMBudget,
                           replay_recorded, run_recorded)
from .scenario import find_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True)
    p.add_argument("--mode", choices=["A", "B", "C"], default="A")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--replay", type=Path, help="replay this transcript instead of a live run (no key)")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=20_000)
    p.add_argument("--wall-time", type=float, default=25.0)
    p.add_argument("--out-dir", type=Path, default=None, help="where to write the trace (replay only)")
    args = p.parse_args(argv)

    scenario = find_scenario(args.scenario)
    if args.mode in ("B", "C"):
        p.error("this CLI wires the defender only; modes B/C need a classifier (use the probe/run tooling)")
    budget = LLMBudget(max_steps=args.max_steps, max_tokens=args.max_tokens, wall_time_s=args.wall_time)

    if args.replay:
        res, trace = replay_recorded(args.replay, scenario, out_dir=args.out_dir)
        v = judge_attempt(trace.attempt_events(res.attempt_id), scenario)
        print(json.dumps({"replay": str(args.replay), "status": res.status, "source": "recorded",
                          "verdict": {k: v.to_dict()[k] for k in ("status", "attack_success",
                                                                   "legitimate_task_success", "read_only")}}))
        return 0

    try:
        client = AnthropicLLMClient(model=args.model)
    except RuntimeError as exc:
        # No key / no SDK: fail with a clear message, not a traceback.
        print(str(exc), file=sys.stderr)
        return 2
    res, transcript_path = run_recorded(scenario, args.mode, client, model=args.model, budget=budget)
    print(json.dumps({"recorded": str(transcript_path), "status": res.status, "source": "recorded"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
