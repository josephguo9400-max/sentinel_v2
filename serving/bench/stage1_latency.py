"""Stage 1 latency vs input length: `ClassifierService._run_stage1` on one prepared chunk plan, in-process.

    flock /home/claude/measure.lock -c 'python -m serving.bench.stage1_latency --out serving/bench_results/stage1_latency.json'

What is timed: `_run_stage1(cs)` only (per-chunk language identification, Stage 1 windows, segment pieces and
64/32 sub-windows, Stage 1 ONNX inference), on a ChunkSet built once by `_prepare` (Stage 2 tokenization is not
timed). No event loop, no thread pool, no HTTP: this is the work the Stage 1 thread does for one request, without
queueing. Also timed separately, for the breakdown: the language identifier alone over the same chunk texts.

When the service supports a cooperative Stage 1 deadline (`_run_stage1(cs, deadline=...)`), the script also runs
it with the effective Stage 1 budget and reports how many chunks Stage 1 scored inside the budget.

Inputs (deterministic):
- `val_benign_en`: label-0, lang=en rows of data/splits/val.jsonl in file order, joined as paragraphs
  ("\\n\\n"), cut at exactly N Stage 2 (DeBERTa) content tokens, for N in --lengths.
- `probe_middle_overlong`: the 12 bodies of fixtures/probes/middle_overlong.json (2,600 DeBERTa tokens each), the
  inputs behind the Stage 1 timeouts in reports/cascade-and-shadow/probes_after.json.

Per input: --warmup untimed calls, then --repeats x --calls timed calls; percentiles are pooled (nearest rank).
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

from serving.bench.common import ROOT, host_info, lat_summary, now, read_val, require_lock, uptime, write_result
from serving.config import GatewayConfig
from serving.registry import load_registry
from serving.service import ClassifierService

PROBES = ROOT / "fixtures" / "probes" / "middle_overlong.json"


def benign_corpus() -> str:
    return "\n\n".join(r["text"].strip() for r in read_val() if r["label"] == 0 and r.get("lang") == "en")


def cut_tokens(det, text: str, n: int) -> str:
    ids, offsets = det.tokenize_with_offsets(text)
    if len(ids) < n:
        raise SystemExit(f"corpus has only {len(ids)} tokens, need {n}")
    return text[: offsets[n - 1][1]]


def time_calls(fn, warmup: int, repeats: int, calls: int) -> tuple[list[float], list[dict]]:
    for _ in range(warmup):
        fn()
    pooled, per_repeat = [], []
    for _ in range(repeats):
        ms = []
        for _ in range(calls):
            t = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t) * 1000)
        pooled += ms
        per_repeat.append(lat_summary(ms))
    return pooled, per_repeat


def measure(svc: ClassifierService, texts: list[str], args, budget_ms: float, has_deadline: bool) -> dict:
    sets = [svc._prepare(t) for t in texts]
    i = {"k": 0}

    def next_cs():
        cs = sets[i["k"] % len(sets)]
        i["k"] += 1
        return cs

    pooled, per_repeat = time_calls(lambda: svc._run_stage1(next_cs()), args.warmup, args.repeats, args.calls)
    lang_ms = None
    if svc.language is not None and svc.stage1_languages is not None:
        lang, _ = time_calls(lambda: [svc.language.detect(ch.text) for ch in next_cs().chunks if ch.text is not None],
                             args.warmup, args.repeats, args.calls)
        lang_ms = lat_summary(lang)
    cs0 = sets[0]
    out = {"n_inputs": len(texts), "stage2_tokens": cs0.n_tokens, "chunks": len(cs0.chunks),
           "stage1_regex_tokens": svc._run_stage1(cs0).n_tokens,
           "run_stage1": lat_summary(pooled), "run_stage1_per_repeat": per_repeat,
           "language_only": lang_ms,
           "over_budget": {"k": sum(x > budget_ms for x in pooled), "n": len(pooled), "budget_ms": budget_ms}}
    if has_deadline:
        scored, lat = [], []
        for _ in range(args.repeats * args.calls):
            cs = next_cs()
            t = time.perf_counter()
            res = svc._run_stage1(cs, deadline=t + svc.s1_soft_budget_s)
            lat.append((time.perf_counter() - t) * 1000)
            scored.append(len(res.scores))
        out["with_deadline"] = {"soft_budget_ms": round(svc.s1_soft_budget_s * 1000, 3),
                                "run_stage1": lat_summary(lat),
                                "over_budget": {"k": sum(x > budget_ms for x in lat), "n": len(lat)},
                                "chunks_scored_min": min(scored), "chunks_scored_max": max(scored),
                                "all_chunks_scored": {"k": sum(s == len(cs0.chunks) for s in scored),
                                                      "n": len(scored)}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "serving" / "bench_results" / "stage1_latency.json")
    ap.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry.json")
    ap.add_argument("--lengths", default="64,256,510,1000,2600,5000")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--no-lock-check", action="store_true")
    args = ap.parse_args(argv)
    require_lock(args.no_lock_check)
    cfg = GatewayConfig(registry_path=str(args.registry), shadow_log_path="")
    loaded = load_registry(args.registry, Path(cfg.artifacts_root))
    if "stage1" not in loaded.specs or "stage2" not in loaded.specs:
        raise SystemExit(f"registry {args.registry} needs a loadable stage1 and stage2: {loaded.rejections}")
    s1, s2 = loaded.specs["stage1"], loaded.specs["stage2"]
    svc = ClassifierService(s2, cfg, stage1=s1)
    has_deadline = "deadline" in inspect.signature(svc._run_stage1).parameters
    budget_ms = svc.s1_timeout_s * 1000
    before = uptime()
    corpus = benign_corpus()
    out = {"generated_by": "python -m serving.bench.stage1_latency " + " ".join(argv or []), "at": now(),
           "host": host_info(), "uptime_before": before,
           "stage1": {"pointer": loaded.pointers["stage1"], "detector_version": s1.detector_version,
                      "manifest_timeout_ms": s1.timeout_ms, "effective_budget_ms": budget_ms,
                      "input_languages": svc.stage1_languages},
           "config": {"stage1_subwindow_tokens": cfg.stage1_subwindow_tokens,
                      "stage1_subwindow_stride": cfg.stage1_subwindow_stride, "lang_max_chars": cfg.lang_max_chars,
                      "stride": cfg.stride, "structure_aware": cfg.structure_aware},
           "protocol": {"warmup": args.warmup, "repeats": args.repeats, "calls": args.calls,
                        "cooperative_deadline": has_deadline},
           "inputs": {}}
    for n in [int(x) for x in args.lengths.split(",")]:
        out["inputs"][f"val_benign_en_{n}"] = measure(svc, [cut_tokens(s2.detector, corpus, n)], args, budget_ms,
                                                      has_deadline)
    probes = [p["body"] for p in json.loads(PROBES.read_text())["probes"]]
    out["inputs"]["probe_middle_overlong"] = measure(svc, probes, args, budget_ms, has_deadline)
    out["uptime_after"] = uptime()
    write_result(args.out, out)
    for k, v in out["inputs"].items():
        r = v["run_stage1"]
        line = (f"{k:28s} tok={v['stage2_tokens']:5d} chunks={v['chunks']:2d} p50={r['p50_ms']:7.2f} "
                f"p95={r['p95_ms']:7.2f} over={v['over_budget']['k']}/{v['over_budget']['n']}")
        if v.get("language_only"):
            line += f" lang_p50={v['language_only']['p50_ms']:6.2f}"
        if "with_deadline" in v:
            w = v["with_deadline"]
            line += (f" | deadline p95={w['run_stage1']['p95_ms']:6.2f} over={w['over_budget']['k']}/"
                     f"{w['over_budget']['n']} all_scored={w['all_chunks_scored']['k']}/{w['all_chunks_scored']['n']}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
