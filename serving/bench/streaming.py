"""Streaming guard measurement: synthetic token streams through serving/streaming.py and the served D23 bundle.

    flock /home/claude/measure.lock -c 'python -m serving.bench.streaming run'      # scores cache misses
    python -m serving.bench.streaming run --cache-only                               # no model: re-derive the JSON
    flock /home/claude/measure.lock -c 'python -m serving.bench.streaming live'     # real-time cross-check

**Streams (248).** Built with training/dilution_curve.py `build_inputs()` (val split and the probe generator; the
frozen test is never read): its 62 short injections (2 redteam probes + 60 val injections, <= 80 DeBERTa tokens)
and, per injection, one 510-token benign context from each of its two pools (`newsletter`: the probe sentence
pool; `val_benign`: val benign items DeBERTa alone scores < 0.5). An **injection stream** is
`ctx[:128] + injection + ctx[128:192]` (the injection starts at token 128); a **benign stream** is `ctx[:256]`.
So 124 injection streams and 124 benign streams. A stream token is one DeBERTa token piece (`▁` -> space); the
guard joins pieces into text and the service re-tokenizes it, as it would an LLM's detokenized output.

**Classifier.** Each check is one `ClassifierService.classify` call on the served bundle
(`artifacts/registry.json`, D23: DeBERTa fp32 decides, Stage 1 observe-only), through `serving.streaming.service_check`
semantics. The service runs on a normal event loop in a worker thread; the guard and the token source run on the
virtual-time loop (serving/bench/vclock.py), and every check is charged the wall latency the service measured for
that exact text (x `slowdown`). Results are cached by the text's sha256 in `check_cache.jsonl` (decision, score,
latency), so a re-run with `--cache-only` reproduces every number without the model. Latencies are from one
stream at a time on the idle 2-core host; a busy gateway would be slower, which is what `slowdown` stands for.

**Configurations.** (1) The B x k x W sweep at 50 tok/s, stall mode (FAIL_CLOSED, no deadline, no lag budget):
B in {0, 8, 16, 32, 64}, k in {8, 16, 32, 64}, W in {full prefix, 128, 64}. (2) Token rate 20/50/100 tok/s at
B = k = 16. (3) Backpressure at 100 tok/s, B = k = 16: slowdown 1 and 4 for stall, FAIL_CLOSED and FAIL_OPEN with
a 64-token lag budget.

**Metrics.** Injection streams: stopped (Wilson 95%), stopped before the injection started (a benign-prefix flag),
detection delay = tokens generated after the injection start when the stream stopped, leak = injection tokens
released before the stop (any leak: Wilson; mean leaked tokens), prevented = stopped with zero injection tokens
released. Benign streams: false stops (Wilson). All streams: first-visible-token time and its extra over an
unguarded stream (1/rate), per-token visible delay (release - arrival), checks per 1,000 generated tokens,
unchecked tokens released, classifier busy fraction.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

from serving.bench.common import ROOT, host_info, now, percentile, require_lock, uptime, write_result
from serving.bench.vclock import run_virtual
from serving.streaming import CheckResult, Stop, StreamGuardConfig, StreamStats, collect

OUT_DIR = ROOT / "serving" / "bench_results" / "streaming"
CACHE = OUT_DIR / "check_cache.jsonl"
PRE, POST, BENIGN_LEN = 128, 64, 256
WINDOWS = (None, 128, 64)
BS = (0, 8, 16, 32, 64)
KS = (8, 16, 32, 64)


def wilson(k: int, n: int) -> list[float] | None:
    from training.metrics import wilson as w
    iv = w(k, n)
    return None if iv is None else [round(iv[0], 4), round(iv[1], 4)]


def prop(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


# ---------------------------------------------------------------------- streams

def build_streams() -> tuple[list[dict], dict]:
    from training.dilution_curve import _tok, build_inputs

    injections, contexts, prov = build_inputs()
    tok = _tok()

    def pieces(ids):
        return [tok.id_to_token(i).replace("▁", " ") for i in ids]

    streams = []
    for inj in injections:
        for pool in ("newsletter", "val_benign"):
            ctx = contexts[(inj["id"], pool)]
            ids = ctx[:PRE] + inj["ids"] + ctx[PRE:PRE + POST]
            streams.append({"id": f"inj|{inj['id']}|{pool}", "kind": "injection", "pool": pool, "group": inj["group"],
                            "tokens": pieces(ids), "span": [PRE, PRE + len(inj["ids"])]})
            streams.append({"id": f"ben|{inj['id']}|{pool}", "kind": "benign", "pool": pool, "group": inj["group"],
                            "tokens": pieces(ctx[:BENIGN_LEN]), "span": None})
    meta = {"n_injection_streams": sum(s["kind"] == "injection" for s in streams),
            "n_benign_streams": sum(s["kind"] == "benign" for s in streams),
            "injection_start": PRE, "post_tokens": POST, "benign_len": BENIGN_LEN,
            "dilution_inputs": {k: prov[k] for k in ("n_injections", "by_group", "val_benign_pool_size")},
            "injection_tokens": {"min": min(prov["injection_tokens"].values()),
                                 "median": statistics.median(prov["injection_tokens"].values()),
                                 "max": max(prov["injection_tokens"].values())}}
    return streams, meta


# ---------------------------------------------------------------------- the classifier, cached

class CachedScorer:
    """Blocking text -> {decision, failed, score, latency_ms}; a miss runs the served bundle in a worker thread."""

    def __init__(self, cache: Path, cache_only: bool, skip_lock: bool):
        self.path, self.cache_only, self.skip_lock = cache, cache_only, skip_lock
        self.rows: dict[str, dict] = {}
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.rows[r["sha"]] = r
        self.misses = 0
        self._loop = self._svc = None
        self._fh = None

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _start(self):
        if self.cache_only:
            raise SystemExit("cache miss with --cache-only: run `python -m serving.bench.streaming run` under the lock")
        require_lock(self.skip_lock)
        from serving.service import ClassifierService
        from training.gate import load_bundle

        bundle, config = load_bundle(ROOT / "artifacts" / "registry.json")
        self.bundle_sha256 = bundle.bundle_sha256
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._svc = ClassifierService(bundle=bundle, config=config)
        asyncio.run_coroutine_threadsafe(self._svc.start(), self._loop).result()
        for _ in range(3):  # warm-up (not cached): the first ONNX runs are slower than steady state
            asyncio.run_coroutine_threadsafe(self._svc.classify("warm-up text " * 40), self._loop).result()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def get(self, text: str) -> dict:
        k = self.key(text)
        if k in self.rows:
            return self.rows[k]
        if self._svc is None:
            self._start()

        async def one():
            t0 = time.perf_counter()
            out = await self._svc.classify(text)
            return out, (time.perf_counter() - t0) * 1000

        out, ms = asyncio.run_coroutine_threadsafe(one(), self._loop).result()
        active = [r for r in out.results if r.role == "active" and r.stage != "stage1"]
        score = max((r.score for r in active if r.score is not None), default=None)
        row = {"sha": k, "n_chars": len(text), "decision": out.decision, "failed": out.policy_applied is not None,
               "coverage": out.coverage_policy_applied, "score": None if score is None else round(score, 6),
               "latency_ms": round(ms, 2), "bundle": out.bundle_sha256[:12]}
        self.rows[k] = row
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()
        self.misses += 1
        return row

    def close(self):
        if self._svc is not None:
            asyncio.run_coroutine_threadsafe(self._svc.stop(), self._loop).result()
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._fh is not None:
            self._fh.close()


# ---------------------------------------------------------------------- one stream on the virtual clock

async def _source(tokens, rate):
    dt = 1.0 / rate
    for t in tokens:
        await asyncio.sleep(dt)
        yield t


def simulate(stream: dict, cfg: StreamGuardConfig, rate: float, scorer: CachedScorer, slowdown: float = 1.0) -> dict:
    async def check(text: str) -> CheckResult:
        row = scorer.get(text)  # blocks on a miss; the virtual clock does not move meanwhile
        await asyncio.sleep(row["latency_ms"] / 1000 * slowdown)
        return CheckResult(row["decision"], failed=row["failed"], score=row["score"])

    events, st = run_virtual(collect(_source(stream["tokens"], rate), check, cfg))
    return summarize_stream(stream, events, st, rate)


def summarize_stream(stream, events, st: StreamStats, rate: float) -> dict:
    last = events[-1]
    rel_idx = [i for i, t in enumerate(st.release_t) if t is not None]
    span = stream["span"]
    leaked = sum(1 for i in rel_idx if span and span[0] <= i < span[1])
    delays = [st.release_t[i] - st.arrival_t[i] for i in rel_idx]
    busy = sum((c.t_end or st.end_t) - c.t_start for c in st.checks)
    dur = (st.end_t or 0.0) - st.t0
    rec = {"id": stream["id"], "kind": stream["kind"], "pool": stream["pool"], "n_tokens": len(stream["tokens"]),
           "generated": st.generated, "released": st.released, "released_unchecked": st.released_unchecked,
           "checks": len(st.checks), "stopped": isinstance(last, Stop),
           "stop_reason": last.reason if isinstance(last, Stop) else None,
           "stop_check_start": last.check_span[0] if isinstance(last, Stop) and last.check_span else None,
           "stop_check_end": last.check_span[1] if isinstance(last, Stop) and last.check_span else None,
           "first_visible_s": (st.release_t[rel_idx[0]] - st.t0) if rel_idx else None,
           "delay_p50_s": statistics.median(delays) if delays else None, "delay_max_s": max(delays) if delays else None,
           "delays": delays, "busy_frac": busy / dur if dur > 0 else None,
           "tail_s": (st.end_t - st.arrival_t[-1]) if st.arrival_t and not isinstance(last, Stop) else None}
    if span:
        rec.update(inj_start=span[0], inj_end=span[1], inj_len=span[1] - span[0], leaked=leaked,
                   delay_tokens=(last.generated_tokens - span[0]) if isinstance(last, Stop) else None)
    return rec


# ---------------------------------------------------------------------- aggregation

def _q(vals, p):
    s = sorted(vals)
    return round(percentile(s, p), 4) if s else None


def _caught(r) -> bool:
    """Stopped by a flag from a check whose window contains at least one injection token."""
    return (r["stop_reason"] == "flag" and r["stop_check_end"] is not None
            and r["stop_check_start"] < r["inj_end"] and r["stop_check_end"] > r["inj_start"])


def aggregate(recs: list[dict], rate: float) -> dict:
    inj = [r for r in recs if r["kind"] == "injection"]
    ben = [r for r in recs if r["kind"] == "benign"]
    caught = [r for r in inj if _caught(r)]
    flag_other = [r for r in inj if r["stop_reason"] == "flag" and not _caught(r)]
    failed = [r for r in inj if r["stop_reason"] in ("lag", "classifier_failure")]
    dtok = [r["delay_tokens"] for r in caught]
    all_delays = [d for r in recs for d in r["delays"]]
    fv = [r["first_visible_s"] for r in recs if r["first_visible_s"] is not None]
    gen = sum(r["generated"] for r in recs)
    rel = sum(r["released"] for r in recs)
    reasons = lambda rs: {x: sum(r["stop_reason"] == x for r in rs) for x in  # noqa: E731
                          sorted({r["stop_reason"] for r in rs if r["stop_reason"]})}
    out = {
        "injection": {
            "caught": prop(len(caught), len(inj)),
            "caught_zero_leak": prop(sum(r["leaked"] == 0 for r in caught), len(inj)),
            "stopped_by_flag_outside_injection": prop(len(flag_other), len(inj)),
            "stopped_by_failure_or_lag": prop(len(failed), len(inj)),
            "injection_never_shown": prop(sum(r["leaked"] == 0 for r in inj), len(inj)),
            "any_leak": prop(sum(r["leaked"] > 0 for r in inj), len(inj)),
            "any_leak_among_caught": prop(sum(r["leaked"] > 0 for r in caught), len(caught)),
            "leaked_tokens_mean_caught": round(statistics.fmean(r["leaked"] for r in caught), 2) if caught else None,
            "leaked_fraction_mean_caught": round(statistics.fmean(r["leaked"] / r["inj_len"] for r in caught), 4)
            if caught else None,
            "delay_tokens_caught": {"n": len(dtok), "p50": _q(dtok, 0.5), "p90": _q(dtok, 0.9),
                                    "max": max(dtok) if dtok else None},
            "stop_reasons": reasons(inj),
        },
        "benign": {"false_stop": prop(sum(r["stopped"] for r in ben), len(ben)),
                   "false_stop_by_flag": prop(sum(r["stop_reason"] == "flag" for r in ben), len(ben)),
                   "stop_reasons": reasons(ben)},
        "latency": {"first_visible_p50_s": _q(fv, 0.5), "first_visible_p95_s": _q(fv, 0.95),
                    "first_visible_extra_p50_s": None if not fv else round(_q(fv, 0.5) - 1.0 / rate, 4),
                    "token_delay_p50_s": _q(all_delays, 0.5), "token_delay_p95_s": _q(all_delays, 0.95),
                    "token_delay_max_s": round(max(all_delays), 4) if all_delays else None,
                    "tail_p50_s": _q([r["tail_s"] for r in recs if r["tail_s"] is not None], 0.5)},
        "compute": {"checks_per_1k_generated_tokens": round(1000 * sum(r["checks"] for r in recs) / gen, 1),
                    "busy_frac_p50": _q([r["busy_frac"] for r in recs if r["busy_frac"] is not None], 0.5)},
        "unchecked": {"released_unchecked_tokens": sum(r["released_unchecked"] for r in recs),
                      "per_1k_released": round(1000 * sum(r["released_unchecked"] for r in recs) / rel, 1) if rel else None,
                      "streams_with_unchecked": sum(r["released_unchecked"] > 0 for r in recs)},
    }
    by_pool = {}
    for pool in ("newsletter", "val_benign"):
        pi = [r for r in inj if r["pool"] == pool]
        pb = [r for r in ben if r["pool"] == pool]
        by_pool[pool] = {"caught": prop(sum(_caught(r) for r in pi), len(pi)),
                         "caught_zero_leak": prop(sum(_caught(r) and r["leaked"] == 0 for r in pi), len(pi)),
                         "benign_false_stop": prop(sum(r["stopped"] for r in pb), len(pb))}
    out["by_pool"] = by_pool
    return out


PER_STREAM_FIELDS = ("id", "stop_reason", "stop_check_start", "stop_check_end", "generated", "released",
                     "released_unchecked", "checks", "leaked", "delay_tokens")


def run_config(streams, scorer, *, B, k, W, rate, mode="FAIL_CLOSED", lag=None, slowdown=1.0) -> dict:
    cfg = StreamGuardConfig(holdback_tokens=B, check_every=k, window_tokens=W, failure_mode=mode,
                            max_unchecked_tokens=lag)
    recs = [simulate(s, cfg, rate, scorer, slowdown) for s in streams]
    return {"config": {"B": B, "k": k, "W": "full" if W is None else W, "rate_tok_s": rate, "failure_mode": mode,
                       "max_unchecked_tokens": lag, "slowdown": slowdown, "stalls": cfg.stalls},
            "metrics": aggregate(recs, rate),
            "per_stream": [[r.get(f) for f in PER_STREAM_FIELDS] for r in recs]}


def cmd_run(a) -> int:
    streams, meta = build_streams()
    if a.limit:
        streams = streams[: a.limit]
    scorer = CachedScorer(CACHE, a.cache_only, a.no_lock_check)
    t0 = time.perf_counter()
    host_before = uptime()
    try:
        sweep = []
        for W in WINDOWS:
            for k in KS:
                for B in BS:
                    sweep.append(run_config(streams, scorer, B=B, k=k, W=W, rate=50.0))
                    print(f"sweep W={W} k={k} B={B} misses={scorer.misses} {time.perf_counter() - t0:.0f}s", flush=True)
        rates = [run_config(streams, scorer, B=16, k=16, W=W, rate=r) for W in (None, 64) for r in (20.0, 50.0, 100.0)]
        # not streaming: buffer the whole response and check it once at the end (k and B beyond any stream)
        whole = [run_config(streams, scorer, B=10**6, k=10**6, W=None, rate=50.0)]
        back = []
        for W in (None, 64):
            for slow in (1.0, 4.0):
                for mode, lag in (("FAIL_CLOSED", None), ("FAIL_CLOSED", 64), ("FAIL_OPEN", 64)):
                    back.append(run_config(streams, scorer, B=16, k=16, W=W, rate=100.0, mode=mode, lag=lag,
                                           slowdown=slow))
            print(f"backpressure W={W} misses={scorer.misses} {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        scorer.close()
    lat_all = sorted(r["latency_ms"] for r in scorer.rows.values())
    bundles = sorted({r["bundle"] for r in scorer.rows.values()})
    result = {"generated_at": now(), "host": host_info(), "uptime_before": host_before, "uptime_after": uptime(),
              "streams": meta, "limit": a.limit, "cache": {"path": str(CACHE.relative_to(ROOT)),
                                                          "rows": len(scorer.rows), "misses_this_run": scorer.misses,
                                                          "bundle_sha256_prefixes": bundles},
              "check_latency_ms_all_cached": {"n": len(lat_all), "p50": _q(lat_all, 0.5), "p95": _q(lat_all, 0.95),
                                              "max": lat_all[-1] if lat_all else None},
              "check_latency_ms_by_chars": _lat_by_chars(scorer.rows.values()),
              "sweep": [_strip(x) for x in sweep], "rates": [_strip(x) for x in rates],
              "backpressure": [_strip(x) for x in back], "whole_response_once": [_strip(x) for x in whole],
              "wall_s": round(time.perf_counter() - t0, 1)}
    write_result(a.out, result)
    per = {"sweep": sweep, "rates": rates, "backpressure": back, "whole_response_once": whole}
    pth = a.out.with_name(a.out.stem + "_per_stream.json")
    pth.write_text(json.dumps({"fields": PER_STREAM_FIELDS, "configs": {
        name: [{"config": x["config"], "rows": x["per_stream"]} for x in v] for name, v in per.items()}},
        separators=(",", ":"), sort_keys=True) + "\n")
    if scorer.misses:  # the scoring run's own record survives later --cache-only re-derivations
        with (OUT_DIR / "scoring_runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"generated_at": result["generated_at"], "misses": scorer.misses,
                                 "rows_after": len(scorer.rows), "wall_s": result["wall_s"],
                                 "uptime_before": host_before, "uptime_after": result["uptime_after"],
                                 "host": result["host"]}, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(scorer.rows), "misses": scorer.misses, "wall_s": result["wall_s"]}))
    return 0


def _spread(v):
    return {"n": len(v), "min": round(min(v), 4), "p50": round(statistics.median(v), 4), "max": round(max(v), 4)} \
        if v else None


def _strip(x):
    return {"config": x["config"], "metrics": x["metrics"]}


def _lat_by_chars(rows):
    buckets = {"<300": [], "300-700": [], "700-1200": [], ">=1200": []}
    for r in rows:
        n = r["n_chars"]
        b = "<300" if n < 300 else "300-700" if n < 700 else "700-1200" if n < 1200 else ">=1200"
        buckets[b].append(r["latency_ms"])
    return {b: {"n": len(v), "p50": _q(v, 0.5), "p95": _q(v, 0.95)} for b, v in buckets.items()}


# ---------------------------------------------------------------------- real-time cross-check

def cmd_live(a) -> int:
    """The guard on a real event loop against the real service (no cache, wall time), on a few streams, compared
    with the virtual-clock simulation of the same streams from the cache."""
    require_lock(a.no_lock_check)
    from serving.service import ClassifierService
    from serving.streaming import service_check
    from training.gate import load_bundle

    streams, _ = build_streams()
    pick = [s for s in streams if s["kind"] == "injection"][: a.n // 2] + \
        [s for s in streams if s["kind"] == "benign"][: a.n // 2]
    cfg = StreamGuardConfig(holdback_tokens=16, check_every=16, window_tokens=a.window)
    bundle, config = load_bundle(ROOT / "artifacts" / "registry.json")

    async def main():
        svc = ClassifierService(bundle=bundle, config=config)
        await svc.start()
        out = []
        try:
            for s in pick:
                events, st = await collect(_source(s["tokens"], a.rate), service_check(svc), cfg)
                out.append(summarize_stream(s, events, st, a.rate))
        finally:
            await svc.stop()
        return out

    t0 = time.perf_counter()
    live = asyncio.run(main())
    scorer = CachedScorer(CACHE, True, True)
    sim = [simulate(s, cfg, a.rate, scorer) for s in pick]
    keys = ("stopped", "stop_reason", "stop_check_end", "released", "leaked")
    rows = []
    for lv, sm in zip(live, sim):
        rows.append({"id": lv["id"], "live": {k: lv.get(k) for k in keys + ("checks", "first_visible_s")},
                     "sim": {k: sm.get(k) for k in keys + ("checks", "first_visible_s")},
                     "same_outcome": all(lv.get(k) == sm.get(k) for k in ("stopped", "stop_reason"))})
    result = {"generated_at": now(), "host": host_info(), "bundle_sha256": bundle.bundle_sha256,
              "config": {"B": 16, "k": 16, "W": a.window or "full", "rate_tok_s": a.rate, "n": len(pick)},
              "same_outcome": prop(sum(r["same_outcome"] for r in rows), len(rows)),
              "same_stop_check_end": sum(r["live"]["stop_check_end"] == r["sim"]["stop_check_end"] for r in rows),
              "same_released_tokens": sum(r["live"]["released"] == r["sim"]["released"] for r in rows),
              "same_check_count": sum(r["live"]["checks"] == r["sim"]["checks"] for r in rows),
              "first_visible_live_minus_sim_s": _spread([r["live"]["first_visible_s"] - r["sim"]["first_visible_s"]
                                                         for r in rows if r["live"]["first_visible_s"] is not None
                                                         and r["sim"]["first_visible_s"] is not None]),
              "rows": rows, "wall_s": round(time.perf_counter() - t0, 1)}
    write_result(a.out_live, result)
    print(json.dumps({k: result[k] for k in ("same_outcome", "same_stop_check_end", "same_released_tokens",
                                             "first_visible_live_minus_sim_s", "wall_s")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache-only", action="store_true")
    r.add_argument("--limit", type=int, default=None, help="first N streams only (smoke)")
    r.add_argument("--out", type=Path, default=OUT_DIR / "streaming_sweep.json")
    r.add_argument("--no-lock-check", action="store_true")
    lv = sub.add_parser("live")
    lv.add_argument("--n", type=int, default=12)
    lv.add_argument("--rate", type=float, default=50.0)
    lv.add_argument("--window", type=int, default=64)
    lv.add_argument("--out-live", type=Path, default=OUT_DIR / "live_check.json")
    lv.add_argument("--no-lock-check", action="store_true")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_live(a)


if __name__ == "__main__":
    raise SystemExit(main())
