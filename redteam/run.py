"""CLI: run scripted defenders over scenarios in modes A/B/C, write one JSONL trace per run, print verdicts.

Examples:
  python -m redteam.run --split dev --modes A,B,C --defender gullible --classifier keyword
  python -m redteam.run --scenario dev_daily_digest --modes B --classifier gateway --gateway-url http://127.0.0.1:8000
  python -m redteam.run --judge runs/<run_id>.jsonl      # re-judge a stored trace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classifier import Classifier, FixedClassifier, KeywordClassifier
from .defenders import DEFENDERS
from .gateway_client import GatewayClient
from .judge import judge_attempt, summarize
from .sandbox import Limits, run_attempt
from .scenario import find_scenario, load_split
from .trace import DEFAULT_RUNS_DIR, RunTrace, group_by_attempt, read_jsonl


def build_classifier(args: argparse.Namespace) -> Classifier:
    if args.classifier == "keyword":
        return KeywordClassifier()
    if args.classifier == "allow-all":
        return FixedClassifier("allow")
    if args.classifier == "gateway":
        if not args.gateway_url:
            raise SystemExit("--classifier gateway requires --gateway-url")
        return GatewayClient(args.gateway_url, timeout_s=args.gateway_timeout, on_unavailable=args.on_unavailable)
    raise SystemExit(f"unknown classifier {args.classifier!r}")


def rejudge(path: Path) -> int:
    events = read_jsonl(path)
    verdicts = []
    for attempt_id, evs in group_by_attempt(events).items():
        sc_id = (evs[0].detail or {}).get("scenario_id")
        v = judge_attempt(evs, find_scenario(sc_id))
        verdicts.append(v)
        print(json.dumps(v.to_dict()))
    print(json.dumps({"summary": summarize(verdicts)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["dev", "heldout"], help="run every scenario of a split")
    p.add_argument("--scenario", action="append", default=[], help="scenario_id (repeatable)")
    p.add_argument("--modes", default="A,B,C")
    p.add_argument("--defender", choices=sorted(DEFENDERS), default="gullible")
    p.add_argument("--classifier", choices=["keyword", "allow-all", "gateway"], default="keyword")
    p.add_argument("--gateway-url")
    p.add_argument("--gateway-timeout", type=float, default=2.0, help="seconds per /classify call")
    p.add_argument("--on-unavailable", choices=["flag", "allow", "error"], default="flag")
    p.add_argument("--attacker-body-file", type=Path, help="replace the attacker email body with this file")
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--run-id")
    p.add_argument("--deadline", type=float, default=30.0)
    p.add_argument("--judge", type=Path, help="re-judge an existing trace file and exit")
    args = p.parse_args(argv)

    if args.judge:
        return rejudge(args.judge)
    scenarios = list(load_split(args.split)) if args.split else []
    scenarios += [find_scenario(s) for s in args.scenario]
    if not scenarios:
        p.error("give --split or --scenario")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if any(m not in ("A", "B", "C") for m in modes):
        p.error("--modes must be a comma list of A, B, C")
    attacker_body = args.attacker_body_file.read_text(encoding="utf-8") if args.attacker_body_file else None

    trace = RunTrace(run_id=args.run_id, out_dir=args.runs_dir)
    classifier = build_classifier(args) if any(m in ("B", "C") for m in modes) else None
    verdicts = []
    try:
        for sc in scenarios:
            for mode in modes:
                res = run_attempt(sc, mode, DEFENDERS[args.defender](), trace,
                                  classifier=None if mode == "A" else classifier,
                                  attacker_body=attacker_body, limits=Limits(deadline_s=args.deadline))
                v = judge_attempt(trace.attempt_events(res.attempt_id), sc)
                verdicts.append(v)
                print(json.dumps({k: v.to_dict()[k] for k in (
                    "scenario_id", "mode", "status", "unsafe_attempt", "blocked", "attack_success",
                    "legitimate_task_success", "quarantined", "facts_missing")}))
    finally:
        if isinstance(classifier, GatewayClient):
            classifier.close()
    print(json.dumps({"run_id": trace.run_id, "trace": str(trace.path), "source": "scripted",
                      "summary": summarize(verdicts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
