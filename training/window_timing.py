"""Measured DeBERTa cost per request for each window size, on a fixed seeded sample (run under the CPU lock).

    flock /home/claude/measure.lock python -m training.window_timing --set replay --n 60
    flock /home/claude/measure.lock python -m training.window_timing --set val --n 60

For each sampled request and each W in training.window_scores.WINDOWS (in a per-request rotated order, so drift
does not favor one W) the wall time of tokenize + build windows + score every window is measured in two modes:
  seq      one session.run per window (batch size 1), as training/window_scores.py scores
  batched  one session.run per request with all of its windows stacked; every window of a request has the same
           length (W + 2, or n + 2 for a single window), so this adds no padding
Five warm-up requests (not recorded) run first. Sample: the n requests with the smallest seeded hash of their id
(labels and unlabeled items alike: this is a cost sample, not an accuracy sample). Output:
training/results/window_timing_<set>.json with per-request rows {id, n_tokens, ms: {mode: {W: ms}}, windows: {W}}
plus host, threads and onnxruntime version. ms values are single measurements on a shared 2-core VM.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import statistics
import sys
import time

import numpy as np

from training.common import ROOT, write_json
from training.window_scores import WINDOWS, load_set, stride_for, window_rows

SEED = "window-timing-v1:20261030"


def sample(items: list[dict], n: int) -> list[dict]:
    return sorted(items, key=lambda r: hashlib.sha256(f"{SEED}|{r['id']}".encode()).hexdigest())[:n]


def _run_batch(deb, rows: list[list[int]]) -> list[float]:
    ids = np.asarray(rows, dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
    logits = deb.sess.run(None, {k: v for k, v in feed.items() if k in deb.inputs})[0].astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return list(e[:, deb.inj] / e.sum(axis=1))


def time_request(deb, text: str, window: int, mode: str) -> tuple[float, int, float]:
    t0 = time.perf_counter()
    ids = deb.tok.encode(text, add_special_tokens=False).ids
    rows = window_rows(ids, window, stride_for(window), deb.cls_id, deb.sep_id)
    if mode == "seq":
        s = max(deb.infer_one(r) for r in rows)
    else:
        s = max(_run_batch(deb, rows))
    return (time.perf_counter() - t0) * 1000.0, len(rows), s


def summary(vals: list[float]) -> dict:
    v = sorted(vals)
    return {"n": len(v), "mean": round(statistics.fmean(v), 1), "p50": round(v[len(v) // 2], 1),
            "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 1), "max": round(v[-1], 1)}


def run(name: str, n: int, threads: int) -> dict:
    import onnxruntime as ort

    from training.cascade_scores import DebertaScorer

    deb = DebertaScorer(threads=threads)
    items = sample(load_set(name), n)
    for r in items[:5]:
        for w in WINDOWS:
            for mode in ("seq", "batched"):
                time_request(deb, r["text"], w, mode)
    rows = []
    for i, r in enumerate(items):
        order = WINDOWS[i % len(WINDOWS):] + WINDOWS[: i % len(WINDOWS)]
        ms = {"seq": {}, "batched": {}}
        nwin = {}
        for w in order:
            for mode in ("seq", "batched"):
                t, k, _ = time_request(deb, r["text"], w, mode)
                ms[mode][str(w)] = round(t, 2)
                nwin[str(w)] = k
        rows.append({"id": r["id"], "n_tokens": len(deb.tok.encode(r["text"], add_special_tokens=False).ids),
                     "ms": ms, "windows": nwin})
    return {"set": name, "seed": SEED, "n": len(rows), "threads": threads, "onnxruntime": ort.__version__,
            "host": {"cpus": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version()},
            "summary_ms": {mode: {str(w): summary([x["ms"][mode][str(w)] for x in rows]) for w in WINDOWS}
                           for mode in ("seq", "batched")},
            "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["val", "replay"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    out = ROOT / "training" / "results" / f"window_timing_{a.set}.json"
    write_json(out, run(a.set, a.n, a.threads))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
