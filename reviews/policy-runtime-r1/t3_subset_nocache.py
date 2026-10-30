"""Reviewer re-run of T3 on a subset, WITHOUT the shared row-score cache (reviews/policy-runtime-r1.md, T3 section).

Why: the committed T3 run (training/replay_policy_equivalence.py) memoizes DeBERTa scores by row content across both
engines and was produced in resumable chunks whose code version is not recorded per item. This re-run uses the
committed code at HEAD, scores every row once per engine (no cache), and compares each item with the same
`differences()` function plus the committed per-item record.

T3 cannot be launched as committed at HEAD (artifacts/registry.json is now v2; see finding M1), so this driver points
the script's registry_state() at reports/policy-runtime/registry_v1_d23.json, the v1 state D23 served.

Subset: the first 100 items by id, every item the committed run decided by the coverage policy, and every item whose
Stage 2 scored >= 3 rows (the items where chunk selection matters most).

    flock /home/claude/measure.lock python reviews/policy-runtime-r1/t3_subset_nocache.py \
        reviews/policy-runtime-r1/t3_subset_nocache.json      # 193 items, 343 s on the 2-core VM
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training import replay_policy_equivalence as eq  # noqa: E402
from training.common import read_jsonl  # noqa: E402


def main(out: Path) -> int:
    committed = json.loads((ROOT / "reports/policy-runtime/t3_d23_equivalence.json").read_text())
    citems = {it["id"]: it for it in committed["items"]}
    ids = sorted(citems)
    pick = set(ids[:100])
    pick |= {i for i, it in citems.items() if it["coverage_policy_applied"] is not None}
    pick |= {i for i, it in citems.items() if it["n_rows"] >= 3}
    items = [it for it in sorted(read_jsonl(eq.REPLAY), key=lambda x: x["id"]) if it["id"] in pick]

    pointers = json.loads((ROOT / "reports/policy-runtime/registry_v1_d23.json").read_text())
    config = eq.no_timeout_config()
    specs = eq.load_specs(pointers)
    legacy, policy_svc, recs, policy, bundle = eq.build_engines(pointers, config, None, specs)
    diff = eq.timeout_only_diff(eq.committed_policy(), policy)

    rows_out = []

    async def go():
        await legacy.start()
        await policy_svc.start()
        try:
            for it in items:
                views = {}
                for name, svc in (("legacy", legacy), ("policy", policy_svc)):
                    recs[name].rows.clear()
                    o = await svc.classify(it["text"], f"rv-{it['id']}")
                    views[name] = eq.item_view(o, recs[name].rows)
                d = eq.differences(views["legacy"], views["policy"])
                c = citems[it["id"]]
                vs_committed = [k for k in ("decision", "decided_by", "stages_run", "policy_applied",
                                            "coverage_policy_applied", "path", "observe_outcome")
                                if views["policy"][k] != c[k]]
                if len(views["policy"]["rows"]) != c["n_rows"]:
                    vs_committed.append("n_rows")
                max_score_gap = max([abs(a["score"] - b["score"]) for a, b in
                                     zip(views["legacy"]["active"], views["policy"]["active"])
                                     if a["score"] is not None and b["score"] is not None] or [0.0])
                rows_out.append({"id": it["id"], "diffs": d, "vs_committed": vs_committed,
                                 "n_rows": len(views["legacy"]["rows"]), "max_active_score_gap": max_score_gap})
        finally:
            await legacy.stop()
            await policy_svc.stop()

    t0 = time.perf_counter()
    asyncio.run(go())
    rep = {
        "what": "reviewer re-run of T3 on a subset, no row cache, code at HEAD",
        "registry": "reports/policy-runtime/registry_v1_d23.json",
        "run_policy_differs_from_committed_in": diff,
        "n_items": len(rows_out), "n_rows_per_engine": sum(r["n_rows"] for r in rows_out),
        "n_items_with_engine_differences": sum(1 for r in rows_out if r["diffs"]),
        "n_items_differing_from_committed_report": sum(1 for r in rows_out if r["vs_committed"]),
        "max_active_score_gap_between_engines": max(r["max_active_score_gap"] for r in rows_out),
        "wall_s": round(time.perf_counter() - t0, 1),
        "items": rows_out,
    }
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, indent=1))
    return 0 if not rep["n_items_with_engine_differences"] and not rep["n_items_differing_from_committed_report"] \
        else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
