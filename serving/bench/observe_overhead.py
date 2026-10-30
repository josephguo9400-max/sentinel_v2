"""Active-path effect of observe-only Stage 1 (D23) under concurrency: observe mode vs no Stage 1.

    flock /home/claude/measure.lock -c 'python -m serving.bench.observe_overhead --out serving/bench_results/observe_overhead.json'

Two ClassifierService instances share the same loaded detectors from the registry (Stage 2 DeBERTa fp32, Stage 1
stage1-tfidf-v2.1): `none` (no Stage 1) and `observe` (stage1_mode=observe, default stage1_observe_concurrency).
They are driven in-process (no HTTP, no client process), closed loop at --concurrency workers over the first --n
replay items (data/replay/replay-v1.jsonl, file order: the traffic mix the D22/D23 decisions are judged on), modes
interleaved per repeat. Default config otherwise (Stage 2 timeout from its manifest, 5,000 ms; max_inflight 64).

Per mode, pooled over repeats:
- response latency (what a caller waits for; in observe mode it includes that request's own Stage 1 run),
- decision latency = the Stage 2 result's latency_ms (request start -> Stage 2 done, queueing included), which is
  what observe work could stretch for OTHER requests through CPU contention,
- Stage 2 failures (timeout / overloaded) and failure-policy decisions,
- decisions that differ between the two modes for the same item and repeat,
- observe outcomes (scored / dropped because the observe slot was busy).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from serving.bench.common import ROOT, host_info, lat_summary, now, require_lock, uptime, write_result
from serving.config import GatewayConfig
from serving.registry import load_registry
from serving.service import ClassifierService

REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"


def read_replay(n: int) -> list[dict]:
    rows = []
    with REPLAY.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


async def drive(svc: ClassifierService, items: list[dict], concurrency: int, tag: str) -> list[dict]:
    queue = list(enumerate(items))
    recs: list[dict | None] = [None] * len(items)

    async def worker():
        while queue:
            i, it = queue.pop(0)
            t0 = time.perf_counter()
            o = await svc.classify(it["text"], f"{tag}-{i}")
            wall = (time.perf_counter() - t0) * 1000
            s2 = next(r for r in o.results if r.role == "active" and r.stage != "stage1")
            recs[i] = {"wall_ms": wall, "decision_ms": s2.latency_ms, "decision": o.decision,
                       "decided_by": o.decided_by, "policy_applied": o.policy_applied,
                       "s2_status": s2.status, "s2_reason": s2.failure_reason,
                       "observe": o.info.get("stage1_observe", {}).get("outcome")}

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return recs


async def run(args, out: dict) -> None:
    cfg = GatewayConfig(shadow_log_path="")
    loaded = load_registry(Path(cfg.registry_path), Path(cfg.artifacts_root))
    if loaded.rejections or "stage1" not in loaded.specs:
        raise SystemExit(f"registry must load stage1 and stage2 cleanly: {loaded.rejections}")
    s1, s2 = loaded.specs["stage1"], loaded.specs["stage2"]
    services = {"none": ClassifierService(s2, cfg), "observe": ClassifierService(s2, cfg, stage1=s1,
                                                                                    stage1_mode="observe")}
    for svc in services.values():
        await svc.start()
    items = read_replay(args.n)
    out["n_items"] = len(items)
    pooled = {m: [] for m in services}
    try:
        for svc in services.values():  # warmup, not recorded
            await drive(svc, items[:4], 2, "warm")
        for rep in range(args.repeats):
            per = {}
            for m, svc in services.items():
                recs = await drive(svc, items, args.concurrency, f"{m}-{rep}")
                per[m] = recs
                pooled[m] += recs
                out["modes"][m]["per_repeat"].append({
                    "repeat": rep + 1, "uptime_after": uptime(),
                    "response": lat_summary([r["wall_ms"] for r in recs]),
                    "decision": lat_summary([r["decision_ms"] for r in recs])})
                print(m, rep + 1, out["modes"][m]["per_repeat"][-1]["decision"], flush=True)
            diff = [i for i, (a, b) in enumerate(zip(per["none"], per["observe"])) if a["decision"] != b["decision"]]
            out["decision_differences"].append({
                "repeat": rep + 1, "n": len(diff),
                "items": [{"index": i, "none": [per["none"][i][k] for k in ("decision", "decided_by", "s2_status")],
                           "observe": [per["observe"][i][k] for k in ("decision", "decided_by", "s2_status")]}
                          for i in diff]})
    finally:
        for svc in services.values():
            await svc.stop()
    for m, recs in pooled.items():
        out["modes"][m]["all_repeats"] = {
            "response": lat_summary([r["wall_ms"] for r in recs]),
            "decision": lat_summary([r["decision_ms"] for r in recs]),
            "stage2_status": dict(Counter(f"{r['s2_status']}/{r['s2_reason']}" for r in recs)),
            "policy_applied": sum(r["policy_applied"] is not None for r in recs),
            "observe_outcomes": dict(Counter(r["observe"] for r in recs if r["observe"] is not None)),
            "n": len(recs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "serving" / "bench_results" / "observe_overhead.json")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-lock-check", action="store_true")
    args = ap.parse_args(argv)
    require_lock(args.no_lock_check)
    cfg = GatewayConfig()
    out = {"generated_by": "python -m serving.bench.observe_overhead " + " ".join(argv or []), "at": now(),
           "host": host_info(), "uptime_before": uptime(), "registry": cfg.registry_path,
           "items": f"data/replay/replay-v1.jsonl, first {args.n} in file order", "concurrency": args.concurrency,
           "repeats": args.repeats, "transport": "in-process ClassifierService.classify (no HTTP)",
           "config": {"stage1_observe_concurrency": cfg.stage1_observe_concurrency,
                      "stage1_deadline_margin_ms": cfg.stage1_deadline_margin_ms, "max_chunks": cfg.max_chunks,
                      "max_inflight": cfg.max_inflight, "workers": cfg.workers, "max_batch": cfg.max_batch},
           "modes": {"none": {"per_repeat": []}, "observe": {"per_repeat": []}}, "decision_differences": []}
    asyncio.run(run(args, out))
    out["uptime_after"] = uptime()
    write_result(args.out, out)
    for m, v in out["modes"].items():
        a = v["all_repeats"]
        print(m, "response", a["response"], "decision", a["decision"], a["stage2_status"], a["observe_outcomes"])
    print("decision differences per repeat", [d["n"] for d in out["decision_differences"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
