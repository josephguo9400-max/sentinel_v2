"""CLI for the adaptive attacker (milestone adaptive-attacker).

    # dump the goal set and its dev/eval split (no key, no gateway)
    python -m redteam.adaptive.run goals --out reports/adaptive-attacker/goals.json

    # pilot: dev goals, tiny budget, hard cap $1 (needs a key and a running gateway)
    set -a; . /root/.sentinel/.env; set +a
    python -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
        --goal-split dev --budget 8 --max-usd 1 --pilot --out-dir reports/adaptive-attacker/pilot

    # eval headline (black-box decision-only) and the score-feedback ablation
    python -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
        --goal-split eval --budget 20 --max-usd 15 --out-dir reports/adaptive-attacker/eval_decision
    python -m redteam.adaptive.run attack --gateway-url http://127.0.0.1:8000 \
        --goal-split eval --budget 20 --max-usd 15 --score-feedback \
        --out-dir reports/adaptive-attacker/eval_score

The API key is read only from ANTHROPIC_API_KEY, never printed, logged, cached or committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.common import ROOT, sha256_file
from training.gate import load_bundle

from .budget import SpendTracker
from .cache import ResponseCache
from .goals import enumerate_goals, select_goals
from .llm import MISSING_KEY_MESSAGE, AnthropicAttackerClient
from .oracle import make_gateway_classifier
from .report import aggregate, gate_input
from .search import run_goal

MILESTONE_CAP_USD = 15.0  # hard cap for the whole milestone
PILOT_CAP_USD = 1.0


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_goals(args) -> int:
    goals = enumerate_goals()
    out = {
        "n_goals": len(goals),
        "dev": [g.as_dict() for g in goals if g.split == "dev"],
        "eval": [g.as_dict() for g in goals if g.split == "eval"],
        "note": ("Goals are drawn from the sandbox send_email action on dev scenarios only; the attacker "
                 "never reads fixtures/scenarios/heldout/ or the frozen test. The dev/eval split is over "
                 "the attacker's optimisation target (docs/design/adaptive-attacker.md §3)."),
    }
    _write_json(Path(args.out), out)
    print(json.dumps({"wrote": args.out, "n_goals": len(goals)}))
    return 0


def cmd_attack(args) -> int:
    cap = min(args.max_usd, MILESTONE_CAP_USD)
    if args.pilot:
        cap = min(cap, PILOT_CAP_USD)
    if cap <= 0:
        print("max-usd must be positive", file=sys.stderr)
        return 2

    goals = select_goals(args.goal_split)
    if not goals:
        print(f"no goals for split {args.goal_split!r}", file=sys.stderr)
        return 2

    spend = SpendTracker(cap_usd=cap)
    cache = ResponseCache(directory=Path(args.cache_dir) if args.cache_dir else None
                          ) if args.cache_dir else ResponseCache()
    try:
        attacker = AnthropicAttackerClient(args.attacker_model, spend, cache)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)  # no key / no SDK: clean message, no traceback, no key leak
        return 2

    bundle, _cfg = load_bundle(ROOT / "artifacts" / "registry.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "queries.jsonl"  # gitignored: holds full candidate bodies
    results = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for g in goals:
            classifier = make_gateway_classifier(args.gateway_url)  # fresh per goal: clean query accounting
            res = run_goal(g, attacker, classifier, budget=args.budget,
                           score_feedback=args.score_feedback, temperature=args.temperature,
                           attack_mode=args.attack_mode)
            results.append(res)
            for q in res.queries:
                raw.write(json.dumps({"goal_id": g.goal_id, **q}, ensure_ascii=False) + "\n")
            print(json.dumps({"goal": g.goal_id, "harmful_evasion": res.harmful_evasion,
                              "q_first_harmful_evasion": res.queries_to_first_harmful_evasion,
                              "harm_modeB": res.end_to_end_harm, "q_first_harm": res.queries_to_first_harm,
                              "allowed_any": res.classifier_allowed_any,
                              "harmful_caught": res.n_harmful_flagged, "harmful_missed": res.n_harmful_allowed,
                              "stopped": res.stopped_reason, "spent_usd": round(spend.spent_usd, 4)}))
            if spend.remaining_usd <= 0:
                print(json.dumps({"stop": "dollar_cap_reached", "spent_usd": round(spend.spent_usd, 4)}))
                break

    meta = {
        "attacker_model": args.attacker_model,
        "attack_mode": args.attack_mode,
        "feedback": "score" if args.score_feedback else "decision_only",
        "goal_split": args.goal_split,
        "budget": args.budget,
        "temperature": args.temperature,
        "gateway_url": args.gateway_url,
        "bundle_sha256": bundle.bundle_sha256,
        "engine_version": __import__("serving.engine_version", fromlist=["ENGINE_VERSION"]).ENGINE_VERSION,
        "spend": spend.as_dict(),
        "n_goals_run": len(results),
    }
    report = aggregate(results, meta=meta)
    _write_json(out_dir / "summary.json", report)
    _write_json(out_dir / "adaptive_asr_gate_input.json", gate_input(report, bundle_sha256=bundle.bundle_sha256))
    print(json.dumps({
        "wrote": str(out_dir / "summary.json"),
        "n_goals": report["n_goals"],
        "harmful_evasion_asr": report["harmful_evasion_asr"]["value"],
        "end_to_end_harm_asr_modeB": report["end_to_end_harm_asr_modeB"]["value"],
        "candidate_harmful_miss_rate": report["candidate_harmful_miss_rate"]["value"],
        "modeC_invariant_holds": report["firewall_invariant_modeC"]["holds"],
        "spent_usd": round(spend.spent_usd, 4),
    }, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("goals", help="write the goal set and its dev/eval split (no key, no gateway)")
    g.add_argument("--out", default="reports/adaptive-attacker/goals.json")

    a = sub.add_parser("attack", help="run the adaptive attacker against a served gateway")
    a.add_argument("--gateway-url", required=True)
    a.add_argument("--goal-split", default="eval", choices=["dev", "eval", "all"])
    a.add_argument("--budget", type=int, default=40, help="gateway queries per goal (RFC ~100)")
    a.add_argument("--attacker-model", default="claude-haiku-5-5",
                   help="claude-haiku-5-5 (default) or claude-sonnet-5-5")
    a.add_argument("--attack-mode", default="compose", choices=["compose", "author"],
                   help="compose (default): LLM writes a benign carrier, harness inserts a fixed mild "
                        "injection core; author: LLM authors the full injection (documents refusals)")
    a.add_argument("--score-feedback", action="store_true",
                   help="ablation: also reveal the decider score to the attacker (stronger attacker)")
    a.add_argument("--max-usd", type=float, default=MILESTONE_CAP_USD,
                   help=f"hard dollar cap (clamped to {MILESTONE_CAP_USD})")
    a.add_argument("--pilot", action="store_true", help=f"clamp the cap to ${PILOT_CAP_USD}")
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--cache-dir", default=None, help="API response cache dir (default redteam/adaptive/cache)")
    a.add_argument("--out-dir", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "goals":
        return cmd_goals(args)
    return cmd_attack(args)


if __name__ == "__main__":
    sys.exit(main())
