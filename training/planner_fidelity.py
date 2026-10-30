"""How close is the dilution study's W = 510 proxy to the served Stage 2 planner? (review r2 m-d). No model.

    python -m training.planner_fidelity     # -> training/results/planner_fidelity.json

The window study (training/window_scores.py, training/window_eval.py) scores W = 510 with the original sliding-window
rule (training.cascade_scores.window_starts, the rule of serving/encoding.py chunk_ids: one token sequence, windows
every 384 tokens plus one ending at the tail). The gateway now plans Stage 2 chunks with the structure-aware planner
(serving/structure.py build_chunks / plan_windows: per-segment tokenization, paragraph boundaries, kinds kept apart).

Inputs (both committed): the W = 510 rows of training/results/window_scores_replay.jsonl, and the serving-faithful
counterfactual (reports/cascade-and-shadow/replay_counterfactual_served_faithful.json), whose items[].stage2_chunks
hold the real DeBERTa score of every chunk the served planner sent to Stage 2. Compared on the escalated replay items
whose every planned chunk was scored (len(stage2_chunks) == n_chunks; items over the max_chunks budget are left out
because part of their text was never scored): served score = max over the scored chunks; proxy score = max over the
W = 510 windows. Both files round scores to 6 places, and the two were computed in separate runs (serving inference
vs the offline cache; a fresh rescoring of cached rows differs by up to ~5e-7), so a score that is the same model
output can differ in the 6th place. "identical_max_score" counts equality at 5 decimal places (the headline);
"identical_max_score_6dp" counts equality at the stored 6 places.

Also reported, descriptive: the chosen two-pass setting's replay gain (window_eval.json two_pass.chosen) restricted
to items the served planner splits into at most max_chunks chunks (items over the cap are coverage-flagged under
the D23 configuration anyway).
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from training.common import ROOT, read_jsonl, sha256_file, write_json
from training.metrics import rate

THRESHOLD = 0.5
SERVED = ROOT / "reports" / "cascade-and-shadow" / "replay_counterfactual_served_faithful.json"
WINDOWS = ROOT / "training" / "results" / "window_scores_replay.jsonl"
WINDOW_EVAL = ROOT / "training" / "results" / "window_eval.json"
OUT = ROOT / "training" / "results" / "planner_fidelity.json"


def window_scores(path=WINDOWS) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for d in read_jsonl(path):
        out.setdefault(d["id"], {})[d["W"]] = d["scores"]
    return out


def compare(items: list[dict], scores: dict[str, dict[int, list[float]]]) -> dict:
    full = [it for it in items if it["escalated"] and it["stage2_chunks"] and len(it["stage2_chunks"]) == it["n_chunks"]]
    same_score = same_6dp = same_count = 0
    flips: Counter = Counter()
    for it in full:
        served = max(c["deberta"] for c in it["stage2_chunks"])
        proxy = scores[it["id"]][510]
        same_score += round(served, 5) == round(max(proxy), 5)
        same_6dp += served == max(proxy)
        same_count += len(proxy) == it["n_chunks"]
        if (served >= THRESHOLD) != (max(proxy) >= THRESHOLD):
            lab = it["label"]
            flips[f"label={'none' if lab is None else lab}:" + ("served_only" if served >= THRESHOLD else "proxy_only")] += 1
    return {"n_escalated": sum(it["escalated"] for it in items), "n_fully_scored": len(full),
            "identical_max_score": same_score, "identical_max_score_6dp": same_6dp,
            "identical_window_count": same_count,
            "decision_flips_at_0.5": sum(flips.values()), "flips_by_label": dict(sorted(flips.items()))}


def two_pass_within_cap(items: list[dict], scores: dict[str, dict[int, list[float]]], chosen: dict,
                        max_chunks: int) -> dict:
    w2, n_gate, t2 = chosen["W2"], chosen["N"], chosen["t2"]
    n_tok = {d["id"]: d["n_tokens"] for d in read_jsonl(WINDOWS) if d["W"] == 510}
    out = {}
    for name, keep in (("all", lambda it: True), (f"n_chunks<={max_chunks}", lambda it: it["n_chunks"] <= max_chunks),
                       (f"n_chunks>{max_chunks}", lambda it: it["n_chunks"] > max_chunks)):
        sel = [it for it in items if keep(it) and it["label"] in (0, 1)]
        res = {}
        for lab, key in ((1, "recall"), (0, "fpr")):
            g = [it for it in sel if it["label"] == lab]
            base = [max(scores[it["id"]][510]) >= THRESHOLD for it in g]
            tp = [b or (n_tok[it["id"]] > n_gate and max(scores[it["id"]][w2]) >= t2) for b, it in zip(base, g)]
            res[key] = {"deberta_510": rate(sum(base), len(g)) if g else None,
                        "two_pass": rate(sum(tp), len(g)) if g else None,
                        "gained": sum(t and not b for t, b in zip(tp, base)),
                        "lost": sum(b and not t for t, b in zip(tp, base))}
        out[name] = res
    return {"setting": {"W2": w2, "N": n_gate, "t2": t2}, "by_served_chunk_count": out}


def build() -> dict:
    sf = json.loads(SERVED.read_text(encoding="utf-8"))
    scores = window_scores()
    chosen = json.loads(WINDOW_EVAL.read_text(encoding="utf-8"))["two_pass"]["chosen"]
    max_chunks = sf["inputs"]["config"]["max_chunks"]
    return {"inputs": {"served": str(SERVED.relative_to(ROOT)), "served_sha256": sha256_file(SERVED),
                       "windows": str(WINDOWS.relative_to(ROOT)), "windows_sha256": sha256_file(WINDOWS),
                       "window_eval": str(WINDOW_EVAL.relative_to(ROOT)), "max_chunks": max_chunks},
            "threshold": THRESHOLD,
            "w510_proxy_vs_served_planner": compare(sf["items"], scores),
            "two_pass_gain_by_served_chunk_count": two_pass_within_cap(sf["items"], scores, chosen, max_chunks)}


def main(argv=None) -> int:
    rep = build()
    write_json(OUT, rep)
    print(json.dumps(rep["w510_proxy_vs_served_planner"], indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
