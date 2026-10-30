"""Cascade gray-zone sweep on val, from the cached per-item scores (training/cascade_scores.py).

    python -m training.cascade_sweep            # -> training/results/cascade_sweep_val.json

Served decision for a request, given gray_zone (low, high), Stage 1 score s1 and DeBERTa score deb:

  1. lang in INPUT_LANGUAGES and s1 < low   -> allow, decided by Stage 1 (Stage 2 never runs)
  2. lang in INPUT_LANGUAGES and s1 > high  -> flag, decided by Stage 1 (Stage 2 never runs)
  3. otherwise                              -> escalate: flag iff deb >= DEBERTA_THRESHOLD

Served view used for selection (the milestone's definition): Stage 1 may finalize ALLOW only (below low, in-scope
language only); everything else takes the DeBERTa decision. In manifest terms that is high = 1.0: rule 2 never
fires because the Stage 1 ONNX probability is < 1.0 (max on val 0.9999987) and the contract says "above high".
The contract also permits Stage 1 to finalize flags above high < 1.0; that variant is swept and reported
(`flag_finalizing_variant`) but never selected: on val it "saves" more and even raises recall only because Stage 1
was trained and thresholded on the same sources, which says nothing about held-out traffic (frozen-test Stage 1
recall is 83.5% vs 97.9% on val, DATA.md section 6).

DeBERTa alone is the reference: flag iff deb >= 0.5, every window scored (no chunk cap).

Chunk-level variant (RFC milestone 3): Stage 1 also scores each DeBERTa window's character span. A request
that escalates sends to DeBERTa only the windows with Stage 1 span score >= low, plus the first and the last
window; a non-English request sends all its windows (Stage 1 is not validated there). Decision: flag iff the
max DeBERTa score over the scored windows >= 0.5. Rules 1 and 2 use the request-level s1 as above.

Selection rule (allow-only served view, high = 1.0; low ranges over a grid up to the Stage 1 threshold, since the
manifest requires low <= threshold):
  maximize the fraction of requests NOT escalated (val overall), subject to
  (a) 95% CI lower bound of (cascade recall - DeBERTa recall), paired Newcombe interval, >= -0.02 (RFC gate check);
  (b) Wilson 95% lower bound of cascade recall >= DeBERTa recall - 0.02 (the milestone's stated rule);
  (c) paired 95% CI upper bound of (cascade FPR - DeBERTa FPR) <= +0.02 and (d) benign_hard false positives not
      above DeBERTa's. (c) and (d) hold by construction under allow-only (Stage 1 can only turn flags into
      allows) and are checked anyway. Ties: smaller low.
Kill criterion (RFC): the cascade fails if it saves < 30% of Stage 2 requests at equal recall. Checked at two
settings: the chosen one (recall within the 2-point margin) and the best setting with strictly equal recall
(no positive that DeBERTa flags is finalized as allow by Stage 1).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.common import ROOT, read_jsonl, sha256_file, write_json
from training.metrics import Z95, rate, wilson

DEBERTA_THRESHOLD = 0.5  # serving/config.py default threshold
STAGE1_THRESHOLD = 0.505068  # stage1-tfidf-v2 manifest threshold
INPUT_LANGUAGES = ("en",)
PROVISIONAL_GRAY_ZONE = (0.282715, 0.764614)  # stage1-tfidf-v2 manifest
KILL_MIN_SAVED = 0.30
MAX_RECALL_DROP = 0.02
MAX_FPR_RISE = 0.02
LOW_GRID = [round(x, 3) for x in np.arange(0.01, 0.505, 0.01)] + [PROVISIONAL_GRAY_ZONE[0], STAGE1_THRESHOLD]
HIGH_GRID = [PROVISIONAL_GRAY_ZONE[1], 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
CACHE = ROOT / "training" / "results" / "cascade_scores_val.jsonl"
OUT = ROOT / "training" / "results" / "cascade_sweep_val.json"


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float, float] | None:
    """Difference p1 - p2 of two paired proportions with Newcombe's hybrid score interval (method 10).

    a: both 1, b: only the first is 1, c: only the second is 1, d: both 0.
    Returns (difference, lower, upper) or None when n == 0.
    """
    n = a + b + c + d
    if n == 0:
        return None
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0
    if den > 0:
        num = a * d - b * c
        phi = (max(num - n / 2, 0) if num > 0 else num) / math.sqrt(den)
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    diff = p1 - p2
    return diff, max(-1.0, diff - dl), min(1.0, diff + du)


def load(path: Path = CACHE) -> list[dict]:
    rows = list(read_jsonl(path))
    for r in rows:
        r["in_scope"] = r.get("lang") in INPUT_LANGUAGES
    return rows


def slices_for(r: dict) -> list[str]:
    s = ["overall", f"source={r['source']}", "lang=en" if r.get("lang") == "en" else "lang=non_en"]
    if r["source"] == "benign_hard":
        s += ["benign_hard", f"benign_hard_family={r['family']}"]
    return s


def route(r: dict, low: float, high: float) -> str:
    """'low' (Stage 1 allow), 'high' (Stage 1 flag) or 'stage2'."""
    if r["in_scope"]:
        if r["s1"] < low:
            return "low"
        if r["s1"] > high:
            return "high"
    return "stage2"


def chunk_escalation(r: dict, low: float) -> list[int]:
    """Indices of DeBERTa windows sent to Stage 2 in the chunk-level variant (request already escalated)."""
    ch = r["chunks"]
    if len(ch) <= 2 or not r["in_scope"]:
        return list(range(len(ch)))
    return [i for i, c in enumerate(ch) if i in (0, len(ch) - 1) or c[3] >= low]


def decide(r: dict, low: float, high: float) -> dict:
    z = route(r, low, high)
    n_win = len(r["chunks"])
    if z == "low":
        return {"zone": z, "flag": 0, "flag_chunk": 0, "win": 0, "win_chunk": 0}
    if z == "high":
        return {"zone": z, "flag": 1, "flag_chunk": 1, "win": 0, "win_chunk": 0}
    idx = chunk_escalation(r, low)
    return {"zone": z, "flag": int(r["deb"] >= DEBERTA_THRESHOLD),
            "flag_chunk": int(max(r["chunks"][i][2] for i in idx) >= DEBERTA_THRESHOLD),
            "win": n_win, "win_chunk": len(idx)}


def _paired(ys, cas, ref, positive_label: int):
    """2x2 counts over items with label == positive_label: a both flag, b cascade only, c reference only."""
    a = b = c = d = 0
    for y, p, q in zip(ys, cas, ref):
        if y != positive_label:
            continue
        if p and q:
            a += 1
        elif p:
            b += 1
        elif q:
            c += 1
        else:
            d += 1
    return a, b, c, d


def _diff(cnt) -> dict | None:
    r = newcombe_paired(*cnt)
    if r is None:
        return None
    return {"value": r[0], "ci95": [r[1], r[2]], "ci_method": "newcombe_paired_hybrid_score",
            "discordant": {"cascade_only": cnt[1], "deberta_only": cnt[2]}}


def metrics(rows: list[dict], low: float, high: float) -> dict:
    """Overall and per-slice metrics for one gray zone, plus DeBERTa alone on the same rows."""
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        dec = decide(r, low, high)
        for s in slices_for(r):
            groups[s].append((r, dec))
    out = {}
    for name in sorted(groups, key=lambda k: (k != "overall", k)):
        g = groups[name]
        ys = [r["label"] for r, _ in g]
        cas = [d["flag"] for _, d in g]
        cas_ch = [d["flag_chunk"] for _, d in g]
        ref = [int(r["deb"] >= DEBERTA_THRESHOLD) for r, _ in g]
        n = len(g)
        pos = sum(ys)
        neg = n - pos
        esc = sum(d["zone"] == "stage2" for _, d in g)
        win_total = sum(len(r["chunks"]) for r, _ in g)
        win_req = sum(d["win"] for _, d in g)
        win_chunk = sum(d["win_chunk"] for _, d in g)

        def rec(p):
            return rate(sum(1 for y, x in zip(ys, p) if y == 1 and x), pos)

        def fpr(p):
            return rate(sum(1 for y, x in zip(ys, p) if y == 0 and x), neg)

        out[name] = {
            "n": n, "positives": pos, "negatives": neg,
            "zones": {z: sum(d["zone"] == z for _, d in g) for z in ("low", "stage2", "high")},
            "escalated": rate(esc, n),
            "saved_requests": rate(n - esc, n),
            "deberta_windows_escalated_request_level": rate(win_req, win_total),
            "deberta_windows_escalated_chunk_level": rate(win_chunk, win_total),
            "cascade": {"recall": rec(cas), "fpr": fpr(cas),
                        "recall_minus_deberta": _diff(_paired(ys, cas, ref, 1)),
                        "fpr_minus_deberta": _diff(_paired(ys, cas, ref, 0))},
            "cascade_chunk_level": {"recall": rec(cas_ch), "fpr": fpr(cas_ch),
                                    "recall_minus_deberta": _diff(_paired(ys, cas_ch, ref, 1))},
            "deberta_alone": {"recall": rec(ref), "fpr": fpr(ref)},
            "stage1_finalized_errors": {
                "positives_allowed_by_stage1": sum(1 for (r, d) in g if d["zone"] == "low" and r["label"] == 1),
                "of_which_deberta_would_flag": sum(1 for (r, d) in g if d["zone"] == "low" and r["label"] == 1
                                                   and r["deb"] >= DEBERTA_THRESHOLD),
                "negatives_flagged_by_stage1": sum(1 for (r, d) in g if d["zone"] == "high" and r["label"] == 0),
            },
        }
    return out


def constraints(o: dict, bh: dict | None) -> dict:
    c = o["cascade"]
    rd, fd = c["recall_minus_deberta"], c["fpr_minus_deberta"]
    checks = {
        "a_recall_diff_ci_lower_ge_-0.02": rd is not None and rd["ci95"][0] >= -MAX_RECALL_DROP,
        "b_recall_wilson_lower_ge_deberta_minus_0.02":
            c["recall"]["ci95"][0] >= o["deberta_alone"]["recall"]["value"] - MAX_RECALL_DROP,
        "c_fpr_diff_ci_upper_le_+0.02": fd is not None and fd["ci95"][1] <= MAX_FPR_RISE,
        "d_benign_hard_fp_not_above_deberta": bh is None or (
            bh["cascade"]["fpr"]["numerator"] <= bh["deberta_alone"]["fpr"]["numerator"]),
    }
    checks["all"] = all(checks.values())
    return checks


def compact(low: float, high: float, m: dict) -> dict:
    o, bh = m["overall"], m.get("benign_hard")
    ne = m.get("lang=non_en")
    c = o["cascade"]
    return {
        "low": low, "high": high,
        "saved_requests": o["saved_requests"]["value"],
        "escalated": [o["escalated"]["numerator"], o["escalated"]["denominator"]],
        "escalated_ci95": o["escalated"]["ci95"],
        "deberta_windows_escalated_request_level": o["deberta_windows_escalated_request_level"]["value"],
        "deberta_windows_escalated_chunk_level": o["deberta_windows_escalated_chunk_level"]["value"],
        "recall": [c["recall"]["numerator"], c["recall"]["denominator"]], "recall_ci95": c["recall"]["ci95"],
        "fpr": [c["fpr"]["numerator"], c["fpr"]["denominator"]], "fpr_ci95": c["fpr"]["ci95"],
        "recall_minus_deberta_ci95": c["recall_minus_deberta"]["ci95"],
        "fpr_minus_deberta_ci95": c["fpr_minus_deberta"]["ci95"],
        "chunk_level_recall": [o["cascade_chunk_level"]["recall"]["numerator"],
                               o["cascade_chunk_level"]["recall"]["denominator"]],
        "benign_hard_fp": bh["cascade"]["fpr"]["numerator"] if bh else None,
        "non_en_escalated": ne["escalated"]["value"] if ne else None,
        "constraints": constraints(o, bh),
    }


def sweep(rows: list[dict]) -> dict:
    grid = []
    for high in HIGH_GRID:
        for low in sorted(set(LOW_GRID)):
            if not low <= STAGE1_THRESHOLD <= high:
                continue
            grid.append(compact(low, high, metrics(rows, low, high)))
    allow_only = [g for g in grid if g["high"] == 1.0]
    feasible = [g for g in allow_only if g["constraints"]["all"]]
    chosen = max(feasible, key=lambda g: (g["saved_requests"], -g["low"])) if feasible else None
    # Strictly equal recall: no positive that DeBERTa flags is finalized as allow by Stage 1 (under allow-only,
    # Stage 1 can never rescue a positive DeBERTa misses, so the paired difference is then exactly 0).
    strict_eq = [g for g in allow_only if g["recall"][0] == metrics(rows, 0.0, 1.0)["overall"]["deberta_alone"][
        "recall"]["numerator"]]
    best_strict = max(strict_eq, key=lambda g: (g["saved_requests"], -g["low"])) if strict_eq else None
    flag_var = [g for g in grid if g["high"] < 1.0 and g["constraints"]["all"]]
    best_flag = max(flag_var, key=lambda g: (g["saved_requests"], g["high"], -g["low"])) if flag_var else None
    return {"grid": grid, "chosen": chosen, "strictly_equal_recall_best": best_strict,
            "flag_finalizing_variant_best": best_flag}


BASE_RATES = (0.01, 0.05, 0.10, 0.30)


def reweighted_savings(o_neg: dict, o_pos: dict) -> list[dict]:
    """Saved fraction if traffic had injection base rate pi: (1 - pi) * saved_neg + pi * saved_pos.

    Derived, not measured: it assumes per-class Stage 1 score distributions equal to val's. The bounds combine
    the per-class Wilson 95% bounds and are conservative for the combination, not a joint 95% interval.
    """
    sn, sp = o_neg["saved_requests"], o_pos["saved_requests"]
    out = []
    for pi in BASE_RATES:
        out.append({"injection_base_rate": pi,
                    "saved_requests": (1 - pi) * sn["value"] + pi * sp["value"],
                    "bounds_from_per_class_wilson": [(1 - pi) * sn["ci95"][0] + pi * sp["ci95"][0],
                                                     (1 - pi) * sn["ci95"][1] + pi * sp["ci95"][1]]})
    return out


def build(cache: Path = CACHE) -> dict:
    rows = load(cache)
    s = sweep(rows)
    ch = s["chosen"]
    detail = {}
    named = {"deberta_alone_reference": (0.0, 1.0),  # low 0 and high 1.0 -> everything escalates
             "provisional_stage1_tfidf_v2": PROVISIONAL_GRAY_ZONE}
    if ch:
        named["chosen"] = (ch["low"], ch["high"])
    for key in ("strictly_equal_recall_best", "flag_finalizing_variant_best"):
        if s[key]:
            named[key] = (s[key]["low"], s[key]["high"])
    for k, (lo, hi) in named.items():
        detail[k] = {"low": lo, "high": hi, "slices": metrics(rows, lo, hi)}
        neg = metrics([r for r in rows if r["label"] == 0], lo, hi)["overall"]
        pos = metrics([r for r in rows if r["label"] == 1], lo, hi)["overall"]
        detail[k]["saved_by_label"] = {"benign": neg["saved_requests"], "injection": pos["saved_requests"]}
        detail[k]["saved_reweighted_by_base_rate"] = reweighted_savings(neg, pos)
    o = detail["chosen"]["slices"]["overall"] if ch else None
    saved = o["saved_requests"]["value"] if o else 0.0
    multi = [r for r in rows if len(r["chunks"]) > 1]
    return {
        "split": "val",
        "n_items": len(rows),
        "scores_file": str(cache.relative_to(ROOT)),
        "scores_file_sha256": sha256_file(cache),
        "definitions": {
            "deberta": "protectai/deberta-v3-base-prompt-injection-v2 ONNX fp32 (sha256 pinned in "
                       "training/cascade_scores.py), [CLS]+510 content tokens+[SEP] windows, stride 384, max over "
                       f"ALL windows (no chunk cap), flag iff score >= {DEBERTA_THRESHOLD}",
            "stage1": f"stage1-tfidf-v2 model.onnx (same bytes as stage1-tfidf-v2.1), threshold {STAGE1_THRESHOLD}",
            "routing": "lang in input_languages and s1 < low -> Stage 1 allow; lang in input_languages and s1 > high "
                       "-> Stage 1 flag; else DeBERTa decides. high = 1.0 means Stage 1 never flags.",
            "input_languages": list(INPUT_LANGUAGES),
            "lang": "langdetect tag stored in data/splits/val.jsonl; the gateway must use its own detector",
            "saved_requests": "fraction of requests that never reach Stage 2",
            "deberta_windows_escalated_request_level": "DeBERTa windows computed / DeBERTa windows needed by "
                                                       "DeBERTa alone, request-level cascade",
            "deberta_windows_escalated_chunk_level": "same, chunk-level variant (suspicious + first + last windows)",
            "intervals": "Wilson 95% for proportions; Newcombe paired hybrid score 95% for differences vs DeBERTa alone",
        },
        "selection_rule": {
            "objective": "maximize saved_requests (overall, val)",
            "constraints": {
                "a": f"paired 95% CI lower bound of recall(cascade) - recall(DeBERTa) >= -{MAX_RECALL_DROP}",
                "b": f"Wilson 95% lower bound of recall(cascade) >= recall(DeBERTa) - {MAX_RECALL_DROP}",
                "c": f"paired 95% CI upper bound of FPR(cascade) - FPR(DeBERTa) <= +{MAX_FPR_RISE}",
                "d": "benign_hard false positives(cascade) <= benign_hard false positives(DeBERTa)",
            },
            "routing": "allow-only (high = 1.0); flag-finalizing settings (high < 1.0) are reported, never chosen",
            "tie_break": "smaller low",
            "grid": {"low": sorted(set(LOW_GRID)), "high": HIGH_GRID},
        },
        "chosen_gray_zone": [ch["low"], ch["high"]] if ch else None,
        "kill_criterion": {
            "rule": f"RFC: saved_requests < {KILL_MIN_SAVED} at equal recall -> fall back to the single model and "
                    "report the negative result",
            "at_chosen": {"low": ch["low"] if ch else None, "saved_requests": saved,
                          "triggered": saved < KILL_MIN_SAVED},
            "at_strictly_equal_recall": {
                "low": s["strictly_equal_recall_best"]["low"] if s["strictly_equal_recall_best"] else None,
                "saved_requests": s["strictly_equal_recall_best"]["saved_requests"]
                if s["strictly_equal_recall_best"] else 0.0,
                "triggered": (s["strictly_equal_recall_best"]["saved_requests"]
                              if s["strictly_equal_recall_best"] else 0.0) < KILL_MIN_SAVED},
            "triggered": saved < KILL_MIN_SAVED,
            "why": "under allow-only routing Stage 1 can save a request only by allowing it, so savings are bounded "
                   f"by the benign share of the traffic; val is {sum(r['label'] == 0 for r in rows)}/{len(rows)} "
                   "benign. See named_settings.*.saved_by_label and saved_reweighted_by_base_rate.",
        },
        "val_caveats": [
            "val is 82% spml (in-distribution for Stage 1); savings and recall on held-out sources will differ",
            "jackhhao is confirmed Stage 2 training data, so DeBERTa's val numbers on it are inflated",
            "val has few non-English items (lang=non_en slice); Stage 1 never finalizes them by construction",
            f"{len(multi)} of {len(rows)} val items need more than one DeBERTa window, so the chunk-level variant "
            "has little room to differ from the request-level cascade on val",
            f"{sum(1 for r in rows if len(r['chunks']) > 4)} val items need more than 4 DeBERTa windows (the gateway's "
            "default max_chunks); this sweep scores all windows",
        ],
        "named_settings": detail,
        "grid": s["grid"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = build(a.cache)
    write_json(a.out, res)
    ch = res["named_settings"].get("chosen")
    print(f"chosen gray_zone {res['chosen_gray_zone']}; kill: {json.dumps(res['kill_criterion']['at_chosen'])} "
          f"strict: {json.dumps(res['kill_criterion']['at_strictly_equal_recall'])}")
    if ch:
        o = ch["slices"]["overall"]
        print(f"saved {o['saved_requests']['numerator']}/{o['n']}; recall {o['cascade']['recall']['numerator']}/"
              f"{o['positives']}; fpr {o['cascade']['fpr']['numerator']}/{o['negatives']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
