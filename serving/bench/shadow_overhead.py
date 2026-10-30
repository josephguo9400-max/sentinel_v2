"""Active-path latency overhead of the async shadow mirror.

    flock /home/claude/measure.lock -c 'python -m serving.bench.shadow_overhead --out serving/bench_results/shadow_overhead.json'

Runs the real registry cascade (Stage 1 TF-IDF + Stage 2 DeBERTa fp32, as artifacts/registry.json says) in-process,
loaded once, through the ASGI app (httpx.ASGITransport: no sockets, so no client process competing for the 2 cores), sequentially
(concurrency 1), on a fixed list of val texts. Shadow modes, interleaved per repeat:

- none:    no shadow configured (modes are swapped on the running service; each shadow drains before the next mode)
- crash:   a shadow that raises immediately (exercises queue + log + metrics, no CPU)
- sleep:   a shadow that sleeps 2 s per item with a 200 ms shadow timeout (hangs one thread; the rest are shed)
- deberta: the same DeBERTa fp32 detector as a shadow with 1 intra-op thread (real CPU contention)

Reported per mode: active request latency p50/p95/p99 over all repeats and per repeat, plus the decisions, which
must be identical across modes. The shadow's own latency and errors come from /health counters.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from serving.bench.common import ROOT, host_info, lat_summary, now, read_val, require_lock, uptime, write_result
from serving.config import GatewayConfig
from serving.registry import DetectorSpec, read_registry


class CrashShadow:
    detector_id, detector_version, threshold, max_len, cls_id, sep_id = "shadow-crash", "0", 0.5, 512, 1, 2

    def tokenize(self, text):
        raise RuntimeError("shadow crash (benchmark)")

    def tokenize_with_offsets(self, text):
        raise RuntimeError("shadow crash (benchmark)")

    def infer(self, batch):
        raise RuntimeError("shadow crash (benchmark)")


class SleepShadow(CrashShadow):
    detector_id = "shadow-sleep"

    def tokenize_with_offsets(self, text):
        time.sleep(2.0)
        return [5], [(0, 1)]

    def infer(self, batch):
        return [0.0 for _ in batch]


def spec(det, timeout_ms) -> DetectorSpec:
    return DetectorSpec(detector=det, detector_id=det.detector_id, detector_version=det.detector_version,
                        threshold=det.threshold, kind="token", role="stage2", failure_policy="fail_open",
                        timeout_ms=timeout_ms, max_input_tokens=512)


def make_shadow(mode: str, base_cfg: GatewayConfig, cache: dict) -> DetectorSpec | None:
    from serving.registry import load_detector

    if mode == "crash":
        return spec(CrashShadow(), 200)
    if mode == "sleep":
        return spec(SleepShadow(), 200)
    if mode == "deberta":
        if "deberta" not in cache:  # loaded once (a second ~740 MB session), reused across repeats
            rel = read_registry(Path(base_cfg.registry_path))["stage2"]
            sh = load_detector(Path(base_cfg.artifacts_root), "shadow", rel, intra_op_threads=1)
            sh.detector_id = sh.detector_id + "-shadow"
            sh.timeout_ms = 10000
            cache["deberta"] = sh
        return cache["deberta"]
    return None


async def run_modes(modes: list[str], repeats: int, texts: list[str], base_cfg: GatewayConfig, out: dict) -> list:
    """One app (one Stage 1 + Stage 2 load); the shadow is swapped on the running service per mode."""
    import httpx

    from serving.app import create_app
    from serving.shadow import ShadowMirror

    app = create_app(base_cfg)
    cache: dict = {}
    all_lat = {m: [] for m in modes}
    ref = None
    async with app.router.lifespan_context(app):
        svc = app.state.gateway["service"]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=60) as c:
            for i, t in enumerate(texts[:3]):  # warmup, not recorded
                await c.post("/classify", json={"text": t, "request_id": f"w{i}"})
            for rep in range(repeats):
                for m in modes:
                    sh = make_shadow(m, base_cfg, cache)
                    svc.shadow = None
                    if sh is not None:
                        svc.shadow = ShadowMirror(sh, queue_size=base_cfg.shadow_queue, timeout_ms=sh.timeout_ms,
                                                  max_chunks=base_cfg.shadow_max_chunks, stride=base_cfg.stride,
                                                  structure_aware=True, metrics=svc.metrics, log_path=None)
                        await svc.shadow.start()
                    lat, decisions = [], []
                    for i, t in enumerate(texts):
                        t0 = time.perf_counter()
                        r = await c.post("/classify", json={"text": t, "request_id": f"{m}-{rep}-{i}"})
                        lat.append((time.perf_counter() - t0) * 1000)
                        b = r.json()
                        decisions.append((b["decision"], tuple(b["stages_run"])))
                    shadow_info = svc.shadow.describe() if svc.shadow else None
                    if svc.shadow is not None:
                        # let the shadow drain (bounded) so its CPU use does not leak into the next mode
                        deadline = time.monotonic() + 120
                        while (svc.shadow.queue_depth or svc.shadow._busy) and time.monotonic() < deadline:
                            await asyncio.sleep(0.1)
                        shadow_info = svc.shadow.describe()
                        await svc.shadow.stop()
                        svc.shadow = None
                    all_lat[m] += lat
                    out["modes"][m]["per_repeat"].append({"repeat": rep + 1, "uptime_after": uptime(),
                                                          **lat_summary(lat),
                                                          "shadow_counts": (shadow_info or {}).get("counts"),
                                                          "shadow_agreement_rate": (shadow_info or {}).get(
                                                              "agreement_rate")})
                    ref = ref or decisions
                    out["modes"][m].setdefault("decisions_identical_to_first_run", True)
                    if decisions != ref:
                        out["modes"][m]["decisions_identical_to_first_run"] = False
                    print(m, rep + 1, lat_summary(lat), flush=True)
    for m in modes:
        out["modes"][m]["all_repeats"] = lat_summary(all_lat[m])
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "serving" / "bench_results" / "shadow_overhead.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default="none,crash,sleep,deberta")
    ap.add_argument("--no-lock-check", action="store_true")
    args = ap.parse_args(argv)
    require_lock(args.no_lock_check)
    # Every 20th val text in file order, so sources and lengths mix.
    texts = [r["text"] for r in read_val()][::20][: args.n]
    cfg = GatewayConfig(shadow_log_path="", stage2_timeout_ms=30000)
    modes = args.modes.split(",")
    out = {"generated_by": "python -m serving.bench.shadow_overhead", "at": now(), "host": host_info(),
           "uptime_before": uptime(), "n_texts": len(texts), "repeats": args.repeats,
           "texts": "data/splits/val.jsonl, every 20th row in file order", "transport": "httpx.ASGITransport",
           "concurrency": 1, "modes": {m: {"per_repeat": []} for m in modes}}
    ref = asyncio.run(run_modes(modes, args.repeats, texts, cfg, out))
    out["escalated_fraction"] = sum(1 for _, s in ref if "stage2" in s) / len(ref)
    out["uptime_after"] = uptime()
    write_result(args.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
