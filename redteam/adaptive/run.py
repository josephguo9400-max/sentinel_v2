"""CLI for the adaptive attacker (milestone adaptive-attacker, redesign r1).

    # dump the goal set and its dev/eval split (no key, no gateway)
    python -m redteam.adaptive.run goals --out reports/adaptive-attacker/goals.json

    # adaptive mutation search against a served gateway (NO API KEY: the search is black-box over the
    # gateway's local DeBERTa; the LLM only pre-writes benign carriers, see build_carriers.py)
    flock /home/claude/measure.lock bash reports/adaptive-attacker/run_eval.sh

The search itself spends $0 (no model-authored injections). The milestone spend ledger
(reports/adaptive-attacker/spend_ledger.json) is read so each report shows the cumulative API spend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.common import ROOT
from training.gate import load_bundle

from .carriers import carrier_pool
from .cores import assert_reserved_cores, load_cores
from .goals import enumerate_goals, select_goals
from .ledger import SpendLedger
from .oracle import make_gateway_classifier
from .report import aggregate, gate_input
from .search import run_benign_controls, run_goal

MILESTONE_CAP_USD = 10.0  # redesign hard cap (plus whatever was already spent before it)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_goals(args) -> int:
    goals = enumerate_goals()
    out = {"n_goals": len(goals),
           "dev": [g.as_dict() for g in goals if g.split == "dev"],
           "eval": [g.as_dict() for g in goals if g.split == "eval"],
           "note": "Goals drawn from dev scenarios only; the attacker never reads heldout/ or frozen test."}
    _write_json(Path(args.out), out)
    print(json.dumps({"wrote": args.out, "n_goals": len(goals)}))
    return 0


def cmd_attack(args) -> int:
    goals = select_goals(args.goal_split)
    if not goals:
        print(f"no goals for split {args.goal_split!r}", file=sys.stderr)
        return 2
    cores = load_cores()
    assert_reserved_cores(cores)
    bundle, _cfg = load_bundle(ROOT / "artifacts" / "registry.json")
    ledger = SpendLedger.load()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "queries.jsonl"  # gitignored: holds full candidate bodies
    results, controls = [], []
    with raw_path.open("w", encoding="utf-8") as raw:
        for g in goals:
            classifier = make_gateway_classifier(args.gateway_url)  # fresh per goal: clean query accounting
            res = run_goal(g, classifier, budget=args.budget, score_feedback=args.score_feedback,
                           cores=cores, carriers=carrier_pool(g.scenario_id), seed=args.seed)
            results.append(res)
            for q in res.queries:
                raw.write(json.dumps({"goal_id": g.goal_id, **q}, ensure_ascii=False) + "\n")
            if not args.no_controls:
                controls.append(run_benign_controls(g, make_gateway_classifier(args.gateway_url),
                                                     carriers=carrier_pool(g.scenario_id), seed=args.seed))
            print(json.dumps({"goal": g.goal_id, "n_cores_evaded": sum(res.core_evaded.values()),
                              "n_cores": len(res.core_evaded), "n_queries": res.n_queries,
                              "q_first_evasion": res.queries_to_first_harmful_evasion,
                              "stopped": res.stopped_reason}))

    meta = {
        "goal_split": args.goal_split, "budget": args.budget, "seed": args.seed,
        "feedback": "score" if args.score_feedback else "decision_only",
        "gateway_url": args.gateway_url, "bundle_sha256": bundle.bundle_sha256,
        "engine_version": __import__("serving.engine_version", fromlist=["ENGINE_VERSION"]).ENGINE_VERSION,
        "n_cores": len(cores),
        "attacker_llm_role": "benign carrier author only; injection cores are fixed and not LLM-authored",
        "search_api_spend_usd": 0.0,
        "milestone_spend_usd_to_date": ledger.prior_total_usd,
        "milestone_cap_usd": MILESTONE_CAP_USD,
    }
    report = aggregate(results, meta=meta, controls=controls or None)
    _write_json(out_dir / "summary.json", report)
    _write_json(out_dir / "adaptive_asr_gate_input.json", gate_input(report, bundle_sha256=bundle.bundle_sha256))
    print(json.dumps({"wrote": str(out_dir / "summary.json"), "n_goals": report["n_goals"],
                      "adaptive_asr_cluster": report["adaptive_asr_cluster"]["value"],
                      "benign_control_fpr": report.get("benign_controls", {}).get("fpr_overall", {}).get("value"),
                      "modeC_holds": report["firewall_invariant_modeC"]["holds"]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--out", default="reports/adaptive-attacker/goals.json")
    a = sub.add_parser("attack", help="adaptive mutation search against a served gateway (no API key)")
    a.add_argument("--gateway-url", required=True)
    a.add_argument("--goal-split", default="eval", choices=["dev", "eval", "all"])
    a.add_argument("--budget", type=int, default=100, help="gateway queries per goal (RFC ~100)")
    a.add_argument("--score-feedback", action="store_true",
                   help="ablation: hill-climb on the decider score (else black-box decision-only)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-controls", action="store_true", help="skip the matched benign controls")
    a.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    return cmd_goals(args) if args.cmd == "goals" else cmd_attack(args)


if __name__ == "__main__":
    sys.exit(main())
