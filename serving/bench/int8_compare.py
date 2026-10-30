"""fp32 vs int8 Stage 2: latency and score agreement. Measurement only; the release decision is the gate's.

    flock /home/claude/measure.lock -c 'python -m serving.bench.int8_compare --out serving/bench_results/int8_compare.json'

Both detectors are loaded through the registry verifier (manifest, sha256, runtime) from
artifacts/baseline/deberta-pi-v2 and artifacts/candidates/deberta-pi-v2-int8, default onnxruntime threads.

Latency: single-row ONNX calls (batch 1) on synthetic benign prose (serving/loadtest.make_text) cut to exactly L
content tokens + [CLS]/[SEP], for L in --lengths (default 64, 128, 256, 510). Per repeat and length: 5 warmup calls,
then --calls timed calls of `infer` (tokenization excluded). fp32 and int8 alternate per repeat (fp32, int8, fp32,
...) so drift in host load hits both. 3 repeats by default.

Agreement: the first --n-val val texts (data/splits/val.jsonl, file order), each scored as the gateway's Stage 2
would score it alone (structure-aware windows, max over all windows, no chunk cap). Reported: decision agreement at
each manifest threshold, with counts by label, and absolute score differences (mean, p50, p95, p99, max).
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from serving.bench.common import ROOT, host_info, lat_summary, now, read_val, require_lock, uptime, write_result
from serving.loadtest import make_text
from serving.registry import load_detector
from serving.shadow import score_with_spec

FP32 = "baseline/deberta-pi-v2"
INT8 = "candidates/deberta-pi-v2-int8"


def exact_rows(det, length: int, seed: int) -> list[int]:
    ids = det.tokenize(make_text(int(length * 1.5) + 20, seed))
    assert len(ids) >= length, (len(ids), length)
    return [det.cls_id, *ids[:length], det.sep_id]


def time_calls(det, row: list[int], calls: int, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        det.infer([row])
    out = []
    for _ in range(calls):
        t0 = time.perf_counter()
        det.infer([row])
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "serving" / "bench_results" / "int8_compare.json")
    ap.add_argument("--lengths", default="64,128,256,510")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-agreement", action="store_true")
    ap.add_argument("--no-lock-check", action="store_true")
    args = ap.parse_args(argv)
    require_lock(args.no_lock_check)
    root = ROOT / "artifacts"
    specs = {"fp32": load_detector(root, "stage2", FP32), "int8": load_detector(root, "shadow", INT8)}
    out = {"generated_by": "python -m serving.bench.int8_compare", "at": now(), "host": host_info(),
           "uptime_before": uptime(),
           "artifacts": {k: {"dir": v.artifact_dir, "detector_version": v.detector_version, "sha256": v.sha256,
                             "threshold": v.threshold} for k, v in specs.items()},
           "model_bytes": {k: (root / v.artifact_dir / "onnx" / "model.onnx").stat().st_size
                           for k, v in specs.items()}}
    if not args.skip_latency:
        lengths = [int(x) for x in args.lengths.split(",")]
        lat = {"conditions": {"batch": 1, "lengths_content_tokens": lengths, "calls_per_length": args.calls,
                              "warmup_calls": 5, "repeats": args.repeats, "intra_op_threads": "onnxruntime default",
                              "timed": "OnnxDebertaDetector.infer only (no tokenization, no HTTP)",
                              "order": "fp32 then int8 within each repeat"},
               "per_repeat": [], "pooled": {}}
        pooled = {(k, L): [] for k in specs for L in lengths}
        for rep in range(args.repeats):
            entry = {"repeat": rep + 1, "uptime_start": uptime()}
            for k, sp in specs.items():
                for L in lengths:
                    ms = time_calls(sp.detector, exact_rows(sp.detector, L, seed=L), args.calls)
                    pooled[(k, L)] += ms
                    entry[f"{k}_{L}"] = lat_summary(ms)
                    print(rep + 1, k, L, entry[f"{k}_{L}"]["p50_ms"], entry[f"{k}_{L}"]["p95_ms"], flush=True)
            lat["per_repeat"].append(entry)
        for (k, L), ms in pooled.items():
            lat["pooled"][f"{k}_{L}"] = lat_summary(ms)
        for L in lengths:
            f, i = lat["pooled"][f"fp32_{L}"], lat["pooled"][f"int8_{L}"]
            lat["pooled"][f"speedup_p50_{L}"] = round(f["p50_ms"] / i["p50_ms"], 3)
        out["latency"] = lat
    if not args.skip_agreement:
        rows = read_val(args.n_val)
        res = {k: [] for k in specs}
        t0 = time.perf_counter()
        for r in rows:
            for k, sp in specs.items():
                s, n_tok, n_chunks, trunc = score_with_spec(sp, r["text"], max_chunks=10 ** 6)
                assert not trunc
                res[k].append(s)
        diffs = sorted(abs(a - b) for a, b in zip(res["fp32"], res["int8"]))
        th_f, th_i = specs["fp32"].threshold, specs["int8"].threshold
        dec_f = [s >= th_f for s in res["fp32"]]
        dec_i = [s >= th_i for s in res["int8"]]
        by_label = {}
        for lab in (0, 1):
            idx = [j for j, r in enumerate(rows) if r["label"] == lab]
            by_label[str(lab)] = {"n": len(idx), "agree": sum(dec_f[j] == dec_i[j] for j in idx),
                                  "fp32_flag": sum(dec_f[j] for j in idx), "int8_flag": sum(dec_i[j] for j in idx)}
        from serving.bench.common import percentile
        out["agreement"] = {
            "data": f"data/splits/val.jsonl, first {len(rows)} rows in file order", "n": len(rows),
            "sources": {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})},
            "decision_agree": sum(a == b for a, b in zip(dec_f, dec_i)),
            "flip_fp32_flag_int8_allow": sum(a and not b for a, b in zip(dec_f, dec_i)),
            "flip_fp32_allow_int8_flag": sum(b and not a for a, b in zip(dec_f, dec_i)),
            "by_label": by_label,
            "abs_score_diff": {"mean": statistics.fmean(diffs), "p50": percentile(diffs, 0.5),
                               "p95": percentile(diffs, 0.95), "p99": percentile(diffs, 0.99), "max": diffs[-1]},
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "scoring": "structure-aware windows (510 content tokens, stride 384), max over all windows, no cap",
        }
        print(out["agreement"], flush=True)
    out["uptime_after"] = uptime()
    write_result(args.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
