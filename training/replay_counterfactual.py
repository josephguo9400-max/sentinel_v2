"""Cascade vs DeBERTa alone on the replay set, the traffic mix the kill criterion is judged on (RFC, D22).

    python -m training.replay_counterfactual \
        --served reports/cascade-and-shadow/replay_after/requests.jsonl \
        --out reports/cascade-and-shadow/replay_counterfactual.json

Inputs
  training/results/cascade_scores_replay.jsonl   offline per-item scores (training/cascade_scores.py --split replay)
  data/replay/replay-v1.jsonl                     ground-truth labels (label None = unlabeled, excluded from rates)
  --served                                        optional: the gateway's per-request log from training/replay_runner

Three views, each scored against ALL ground-truth labels (not only the labels revealed during the replay):
  deberta_only     flag iff max DeBERTa window score >= 0.5, every window scored (the reference)
  cascade_offline  cascade_sweep.decide() at the served gray zone (Stage 1 finalizes allow only, in-scope languages)
  served           the decisions the gateway actually returned (timeouts, coverage policy and fail-closed included)

Savings = share of requests Stage 2 never sees. The kill criterion asks for >= 30% savings at equal recall, so the
report also gives the largest savings whose cascade recall equals DeBERTa's (no DeBERTa-caught injection finalized
as allow by Stage 1). The low-threshold sweep here is descriptive only: the replay set is an evaluation set and no
threshold is chosen from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from training.cascade_sweep import DEBERTA_THRESHOLD, KILL_MIN_SAVED, decide, load, newcombe_paired
from training.common import ROOT, read_jsonl, sha256_file, write_json
from training.metrics import rate

CACHE = ROOT / "training" / "results" / "cascade_scores_replay.jsonl"
REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"
SERVED_GRAY_ZONE = (0.505068, 1.0)  # artifacts/candidates/stage1-tfidf-v2.1/manifest.json
LOW_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.505068]


def _rates(items: list[tuple[int, int]]) -> dict:
    """items: (label, flag) for labeled items."""
    pos = [f for y, f in items if y == 1]
    neg = [f for y, f in items if y == 0]
    return {"recall": rate(sum(pos), len(pos)), "fpr": rate(sum(neg), len(neg))}


def _paired(rows, a_key, b_key, label):
    a = b = c = d = 0
    for r in rows:
        if r["label"] != label:
            continue
        x, y = r[a_key], r[b_key]
        a += x and y
        b += x and not y
        c += y and not x
        d += not x and not y
    out = newcombe_paired(a, b, c, d)
    return None if out is None else {"diff": out[0], "ci95": [out[1], out[2]], "only_first": b, "only_second": c}


def build(cache: Path, served_path: Path | None) -> dict:
    rows = load(cache)
    truth = {r["id"]: r for r in read_jsonl(REPLAY)}
    if {r["id"] for r in rows} != set(truth):
        raise SystemExit(f"{cache} does not cover the replay set ({len(rows)} of {len(truth)} items)")
    low, high = SERVED_GRAY_ZONE
    for r in rows:
        r["deb_flag"] = int(r["deb"] >= DEBERTA_THRESHOLD)
        dd = decide(r, low, high)
        r["cas_flag"], r["cas_zone"] = dd["flag"], dd["zone"]
    served = {}
    if served_path:
        for s in read_jsonl(served_path):
            if s.get("error") is None:
                served[s["id"]] = s
    for r in rows:
        s = served.get(r["id"])
        r["srv_flag"] = None if s is None else int(s["decision"] == "flag")
        r["srv_s2"] = None if s is None else int("stage2" in s["stages_run"])

    labeled = [r for r in rows if r["label"] in (0, 1)]
    views = {
        "deberta_only": _rates([(r["label"], r["deb_flag"]) for r in labeled]),
        "cascade_offline": _rates([(r["label"], r["cas_flag"]) for r in labeled]),
    }
    views["cascade_offline"]["stage2_share"] = rate(sum(r["cas_zone"] == "stage2" for r in rows), len(rows))
    if served:
        sv = [r for r in labeled if r["srv_flag"] is not None]
        views["served"] = _rates([(r["label"], r["srv_flag"]) for r in sv])
        got = [r for r in rows if r["srv_s2"] is not None]
        views["served"]["stage2_share"] = rate(sum(r["srv_s2"] for r in got), len(got))
        views["served"]["n_responses"] = len(got)
        # served vs offline cascade: should agree except for timeouts, coverage policy and float noise
        both = [r for r in got]
        views["served"]["disagrees_with_offline_cascade"] = sum(r["srv_flag"] != r["cas_flag"] for r in both)

    paired = {"cascade_minus_deberta_recall": _paired(labeled, "cas_flag", "deb_flag", 1),
              "cascade_minus_deberta_fpr": _paired(labeled, "cas_flag", "deb_flag", 0)}

    # Injections Stage 1 finalized as allow: would DeBERTa have caught them?
    s1_allowed_pos = [r for r in labeled if r["label"] == 1 and r["cas_zone"] == "low"]
    lost = [r for r in s1_allowed_pos if r["deb_flag"]]
    stage1_misses = {
        "injections_finalized_allow_by_stage1": len(s1_allowed_pos),
        "of_which_deberta_flags": len(lost),
        "by_source": dict(Counter(r["source"] for r in lost)),
        "s1_score_of_lost": sorted(round(r["s1"], 4) for r in lost)[:10] + (["..."] if len(lost) > 10 else []),
    }

    slices = defaultdict(list)
    for r in labeled:
        slices[f"source={r['source']}"].append(r)
        slices["lang=en" if r.get("lang") == "en" else "lang=non_en"].append(r)
    by_slice = {k: {"deberta_only": _rates([(r["label"], r["deb_flag"]) for r in v]),
                    "cascade_offline": _rates([(r["label"], r["cas_flag"]) for r in v]),
                    "stage2_share": rate(sum(r["cas_zone"] == "stage2" for r in v), len(v))}
                for k, v in sorted(slices.items())}

    sweep = []
    for lo in LOW_GRID:
        flags = {r["id"]: decide(r, lo, 1.0) for r in rows}
        n_s2 = sum(f["zone"] == "stage2" for f in flags.values())
        lost_lo = sum(1 for r in labeled if r["label"] == 1 and r["deb_flag"] and flags[r["id"]]["zone"] == "low")
        rec = rate(sum(flags[r["id"]]["flag"] for r in labeled if r["label"] == 1),
                   sum(1 for r in labeled if r["label"] == 1))
        sweep.append({"low": lo, "saved": 1 - n_s2 / len(rows), "recall": rec["value"],
                      "deberta_caught_injections_lost": lost_lo})
    equal = [s for s in sweep if s["deberta_caught_injections_lost"] == 0]
    best_equal = max(equal, key=lambda s: s["saved"]) if equal else None
    served_saved = 1 - views["cascade_offline"]["stage2_share"]["value"]

    return {
        "inputs": {"cache": str(cache.relative_to(ROOT)), "cache_sha256": sha256_file(cache),
                   "replay_sha256": sha256_file(REPLAY),
                   "served": str(served_path) if served_path else None,
                   "n_items": len(rows), "n_labeled": len(labeled),
                   "n_positive": sum(r["label"] == 1 for r in labeled)},
        "served_gray_zone": list(SERVED_GRAY_ZONE),
        "views": views,
        "paired": paired,
        "stage1_misses": stage1_misses,
        "by_slice": by_slice,
        "low_sweep_descriptive": sweep,
        "kill_criterion": {
            "min_saved": KILL_MIN_SAVED,
            "served_setting_saved": served_saved,
            "served_setting_recall_equal": stage1_misses["of_which_deberta_flags"] == 0,
            "best_equal_recall_setting": best_equal,
            "passes_at_equal_recall": bool(best_equal and best_equal["saved"] >= KILL_MIN_SAVED),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--served", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = build(a.cache, a.served)
    write_json(a.out, rep)
    print(json.dumps({k: rep[k] for k in ("views", "paired", "stage1_misses", "kill_criterion")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
