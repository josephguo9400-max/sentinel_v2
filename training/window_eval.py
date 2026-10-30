"""Window-size study and the two-pass design, offline from the cached window scores (no model).

    python -m training.window_eval      # -> training/results/window_eval.json

Inputs: training/results/window_scores_{val,replay,probes}.jsonl (training/window_scores.py) and, when present,
training/results/window_timing_{val,replay}.json (training/window_timing.py, measured under the CPU lock).

Single-pass views, one per W in WINDOWS: flag iff max over all W-token windows >= 0.5 (W = 510 is DeBERTa as the
gateway runs it, minus the chunk cap). Per set: recall, FPR, per-source and per-family rates, rates by length
bucket, paired differences vs W = 510 (Newcombe hybrid score 95%), and cost (windows and model tokens incl.
[CLS]/[SEP]; measured ms per request from the timing sample).

Two-pass design (W2, N, t2): flag iff s510 >= 0.5, or n_tokens > N and s_W2 >= t2. Pass 2 runs only for requests
with more than N content tokens (N >= W2: for n <= W2 the W2 score equals the 510 score). Every parameter is chosen
on VAL ONLY by SELECTION_RULE; replay and the probes are evaluated at the chosen setting and never used to choose.
Replay label None (mosscap) is excluded from every rate. The frozen test is not read.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from training.cascade_sweep import MAX_FPR_RISE, newcombe_paired
from training.common import ROOT, sha256_file, write_json
from training.metrics import rate
from training.window_scores import WINDOWS, load_set, out_path, stride_for, window_starts

THRESHOLD = 0.5
BASE = 510
SMALL = tuple(w for w in WINDOWS if w != BASE)
N_FACTORS = (1, 2)  # N = factor * W2, plus N = 510 (pass 2 only for requests that already need > 1 window)
T2_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
LENGTH_BUCKETS = ((0, 64), (65, 128), (129, 256), (257, 510), (511, 10**9))
OUT = ROOT / "training" / "results" / "window_eval.json"
SELECTION_RULE = (
    "On val only, over W2 in {256, 128, 64}, N in {W2, 2*W2, 510} and t2 in T2_GRID: keep settings whose paired 95% "
    "upper bound of FPR(two-pass) - FPR(510) is <= +2 points and that add no benign_hard false positive over "
    "DeBERTa-510; among them maximize val recall, then minimize pass-2 model tokens, then prefer the larger t2.")


# ----------------------------------------------------------------------------------------------- pure logic

def model_tokens(n_tokens: int, window: int) -> int:
    """Model input tokens (incl. [CLS] and [SEP]) to score an n-token request with windows of `window`."""
    k = len(window_starts(n_tokens, window, stride_for(window)))
    return k * (min(n_tokens, window) + 2)


def two_pass_flag(s_base: float, s_small: float, n_tokens: int, n_gate: int, t2: float,
                  t1: float = THRESHOLD) -> int:
    return int(s_base >= t1 or (n_tokens > n_gate and s_small >= t2))


def paired(x: list[int], y: list[int]) -> dict | None:
    """Newcombe paired difference mean(x) - mean(y) over the same items."""
    a = sum(1 for i, j in zip(x, y) if i and j)
    b = sum(1 for i, j in zip(x, y) if i and not j)
    c = sum(1 for i, j in zip(x, y) if j and not i)
    d = len(x) - a - b - c
    out = newcombe_paired(a, b, c, d)
    if out is None:
        return None
    return {"diff": out[0], "ci95": [out[1], out[2]], "only_first": b, "only_second": c, "n": len(x)}


def bucket(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"
    raise ValueError(n)


# ----------------------------------------------------------------------------------------------- loading

def load(name: str, path: Path | None = None) -> list[dict]:
    """One row per item: id, label, source, family, n, s{W} for every W, from the window-score cache."""
    meta = {r["id"]: r for r in load_set(name)}
    by: dict[str, dict] = {}
    for line in (path or out_path(name)).open(encoding="utf-8"):
        d = json.loads(line)
        m = meta[d["id"]]
        r = by.setdefault(d["id"], {"id": d["id"], "label": m["label"], "source": m["source"],
                                    "family": m.get("family"), "n": d["n_tokens"]})
        if r["n"] != d["n_tokens"]:
            raise SystemExit(f"{d['id']}: token count differs between windows")
        r[f"s{d['W']}"] = max(d["scores"])
        r[f"k{d['W']}"] = len(d["scores"])
    rows = list(by.values())
    for w in WINDOWS:
        missing = [r["id"] for r in rows if f"s{w}" not in r]
        if missing or len(rows) != len(meta):
            raise SystemExit(f"{name}: W={w} incomplete ({len(missing)} missing, {len(rows)}/{len(meta)} items)")
    return sorted(rows, key=lambda r: r["id"])


def slices(name: str, r: dict) -> list[str]:
    s = ["overall", f"source={r['source']}", f"len={bucket(r['n'])}"]
    if r["family"]:
        s.append(f"family={r['family']}")
    if name == "val" and r["source"] == "benign_hard":
        s.append("benign_hard")
    if name == "replay" and r["source"] == "replay_email":
        s.append("replay_email_long" if r["family"] and r["family"].endswith("_long") else "replay_email_short")
    return s


def rates(name: str, rows: list[dict], flag_key: str, ref_key: str | None = None) -> dict:
    """Recall/FPR per slice for flags in r[flag_key]; paired diffs vs r[ref_key] when given."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for s in slices(name, r):
            groups[s].append(r)
    out = {}
    for s in sorted(groups):
        g = groups[s]
        pos = [r for r in g if r["label"] == 1]
        neg = [r for r in g if r["label"] == 0]
        o = {"n": len(g), "unlabeled": sum(r["label"] is None for r in g),
             "recall": rate(sum(r[flag_key] for r in pos), len(pos)) if pos else None,
             "fpr": rate(sum(r[flag_key] for r in neg), len(neg)) if neg else None}
        if ref_key:
            o["recall_minus_ref"] = paired([r[flag_key] for r in pos], [r[ref_key] for r in pos]) if pos else None
            o["fpr_minus_ref"] = paired([r[flag_key] for r in neg], [r[ref_key] for r in neg]) if neg else None
        out[s] = o
    return out


def cost(rows: list[dict], windows_of) -> dict:
    """windows_of(r) -> list of W whose windows are computed for r."""
    win = sum(r[f"k{w}"] for r in rows for w in windows_of(r))
    tok = sum(model_tokens(r["n"], w) for r in rows for w in windows_of(r))
    base_tok = sum(model_tokens(r["n"], BASE) for r in rows)
    return {"requests": len(rows), "windows": win, "model_tokens": tok, "model_tokens_vs_510": round(tok / base_tok, 4),
            "requests_with_pass2": sum(1 for r in rows if len(windows_of(r)) > 1)}


def timing(name: str) -> dict | None:
    p = ROOT / "training" / "results" / f"window_timing_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms_summary(vals: list[float]) -> dict:
    v = sorted(vals)
    return {"n": len(v), "mean": round(statistics.fmean(v), 1), "p50": round(v[len(v) // 2], 1),
            "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 1)}


# ----------------------------------------------------------------------------------------------- views

def single_pass(name: str, rows: list[dict]) -> dict:
    for r in rows:
        for w in WINDOWS:
            r[f"f{w}"] = int(r[f"s{w}"] >= THRESHOLD)
    out = {}
    for w in WINDOWS:
        out[str(w)] = {"stride": stride_for(w), "rates": rates(name, rows, f"f{w}", None if w == BASE else f"f{BASE}"),
                       "cost": cost(rows, lambda r, w=w: [w])}
    return out


def apply_two_pass(rows: list[dict], w2: int, n_gate: int, t2: float, key: str = "tp") -> None:
    for r in rows:
        r[key] = two_pass_flag(r[f"s{BASE}"], r[f"s{w2}"], r["n"], n_gate, t2)


def n_gates(w2: int) -> list[int]:
    return sorted({f * w2 for f in N_FACTORS} | {BASE})


def grid(rows: list[dict]) -> list[dict]:
    """Every two-pass setting on `rows` (val), with what the selection rule needs."""
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    bh_base = sum(r[f"f{BASE}"] for r in rows if r["source"] == "benign_hard")
    out = []
    for w2 in SMALL:
        for n_gate in n_gates(w2):
            for t2 in T2_GRID:
                apply_two_pass(rows, w2, n_gate, t2)
                fd = paired([r["tp"] for r in neg], [r[f"f{BASE}"] for r in neg])
                rd = paired([r["tp"] for r in pos], [r[f"f{BASE}"] for r in pos])
                bh = sum(r["tp"] for r in rows if r["source"] == "benign_hard")
                c = cost(rows, lambda r, w2=w2, n_gate=n_gate: [BASE, w2] if r["n"] > n_gate else [BASE])
                ok = fd["ci95"][1] <= MAX_FPR_RISE and bh <= bh_base
                out.append({"W2": w2, "N": n_gate, "t2": t2, "recall": rate(sum(r["tp"] for r in pos), len(pos)),
                            "fpr": rate(sum(r["tp"] for r in neg), len(neg)), "recall_minus_510": rd,
                            "fpr_minus_510": fd, "benign_hard_fp": bh, "benign_hard_fp_510": bh_base,
                            "pass2_model_tokens": c["model_tokens"] - sum(model_tokens(r["n"], BASE) for r in rows),
                            "cost": c, "admissible": ok})
    return out


def select(g: list[dict]) -> dict | None:
    ok = [x for x in g if x["admissible"]]
    if not ok:
        return None
    return min(ok, key=lambda x: (-x["recall"]["numerator"], x["pass2_model_tokens"], -x["t2"]))


def two_pass_ms(t: dict | None, w2: int, n_gate: int) -> dict | None:
    """ms per request on the timing sample: measured 510 time plus measured W2 time for requests with n > N."""
    if t is None:
        return None
    out = {}
    for mode in ("seq", "batched"):
        base = [x["ms"][mode][str(BASE)] for x in t["rows"]]
        tp = [x["ms"][mode][str(BASE)] + (x["ms"][mode][str(w2)] if x["n_tokens"] > n_gate else 0.0)
              for x in t["rows"]]
        out[mode] = {"deberta_510": ms_summary(base), "two_pass": ms_summary(tp)}
    out["n_with_pass2"] = sum(1 for x in t["rows"] if x["n_tokens"] > n_gate)
    out["note"] = "derived: sum of the per-request measurements of each pass on the timing sample"
    return out


def build() -> dict:
    data = {name: load(name) for name in ("val", "replay", "probes")}
    res = {"threshold": THRESHOLD, "windows": list(WINDOWS), "strides": {str(w): stride_for(w) for w in WINDOWS},
           "inputs": {name: {"file": str(out_path(name).relative_to(ROOT)), "sha256": sha256_file(out_path(name)),
                             "items": len(rows), "labeled": sum(r["label"] is not None for r in rows)}
                      for name, rows in data.items()},
           "single_pass": {}, "timing": {}}
    for name, rows in data.items():
        res["single_pass"][name] = single_pass(name, rows)
    for name in ("val", "replay"):
        t = timing(name)
        if t:
            res["timing"][name] = {"n": t["n"], "threads": t["threads"], "onnxruntime": t["onnxruntime"],
                                   "summary_ms": t["summary_ms"],
                                   "mean_tokens": round(statistics.fmean(x["n_tokens"] for x in t["rows"]), 1)}
    g = grid(data["val"])
    chosen = select(g)
    res["two_pass"] = {"selection_rule": SELECTION_RULE, "chosen_on": "val", "grid_val": g, "chosen": None}
    if chosen:
        w2, n_gate, t2 = chosen["W2"], chosen["N"], chosen["t2"]
        ev = {}
        for name, rows in data.items():
            apply_two_pass(rows, w2, n_gate, t2)
            ev[name] = {"rates": rates(name, rows, "tp", f"f{BASE}"),
                        "cost": cost(rows, lambda r: [BASE, w2] if r["n"] > n_gate else [BASE]),
                        "ms": two_pass_ms(timing(name), w2, n_gate) if name != "probes" else None}
        res["two_pass"]["chosen"] = {"W2": w2, "N": n_gate, "t2": t2, "evaluated": ev}
        # Descriptive only (not chosen anywhere): the same W2 and N with t2 = 0.5 on every set.
        desc = {}
        for name, rows in data.items():
            apply_two_pass(rows, w2, n_gate, 0.5, key="tp05")
            desc[name] = {"rates": {k: v for k, v in rates(name, rows, "tp05", f"f{BASE}").items()
                                    if k in ("overall", "benign_hard", "replay_email_long", "replay_email_short")}}
        res["two_pass"]["descriptive_t2_0.5"] = desc
    return res


def main(argv=None) -> int:
    write_json(OUT, build())
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
