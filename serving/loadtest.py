"""Closed-loop load generator for the Sentinel gateway.

`--concurrency` clients each send one request, wait for the response, and send the next, until `--n`
requests have been sent in total. Texts are benign synthetic prose of roughly `--lengths` content
tokens (cycled), each one distinct. Measured client-side latency includes HTTP and JSON overhead.

    python -m serving.loadtest --n 1000 --concurrency 4 --url http://127.0.0.1:8000 --lengths 64,256

Prints a JSON summary and writes it to --out. Every number in it is measured in this run.
"""
import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SENTENCES = [
    "The quarterly review covered the budget and the hiring plan.",
    "Please send the updated slides before Thursday's meeting.",
    "Our team finished the migration ahead of schedule.",
    "The vendor confirmed delivery for the second week of next month.",
    "Lunch will be provided in the large conference room.",
    "Remember to submit expense reports by the end of the week.",
    "The client asked for a short summary of the open issues.",
    "We moved the design review to the afternoon session.",
]
TOKENS_PER_WORD = 1.3  # rough DeBERTa-v3 ratio for this prose; actual counts come back in the responses


def make_text(approx_tokens: int, seed: int) -> str:
    n_words = max(1, int(approx_tokens / TOKENS_PER_WORD))
    out: list[str] = [f"Ref {seed}:"]
    i = seed
    while sum(len(s.split()) for s in out) < n_words:
        out.append(SENTENCES[i % len(SENTENCES)])
        i += 3
    return " ".join(" ".join(out).split()[:n_words])


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    k = max(0, min(len(sorted_vals) - 1, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[k]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def load_average() -> list[float] | None:
    """1, 5 and 15 minute load averages (as `uptime` prints them), or None where unavailable."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def metric_value(metrics_text: str, name: str) -> float | None:
    """Value of an unlabelled sample in Prometheus text, e.g. `sentinel_inference_real_tokens_total`."""
    for ln in metrics_text.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0] == name:
            return float(parts[1])
    return None


def server_counters(metrics_text: str) -> dict:
    return {k: metric_value(metrics_text, f"sentinel_{k}") for k in
            ("inference_batch_size_count", "inference_batch_size_sum", "inference_real_tokens_total",
             "inference_computed_tokens_total")}


def counter_delta(before: dict, after: dict) -> dict:
    """Server counters for this run only (after minus before, so warmup and earlier runs are excluded)."""
    d = {k: None if before.get(k) is None or after.get(k) is None else after[k] - before[k] for k in after}
    calls, chunks = d["inference_batch_size_count"], d["inference_batch_size_sum"]
    real, computed = d["inference_real_tokens_total"], d["inference_computed_tokens_total"]
    return {"onnx_calls": calls, "chunks": chunks,
            "mean_chunks_per_call": round(chunks / calls, 3) if calls else None,
            "real_tokens": real, "computed_tokens": computed,
            "computed_per_real_token": round(computed / real, 4) if real else None}


def summarize(lat_ms: list[float]) -> dict:
    if not lat_ms:
        return {"count": 0}
    s = sorted(lat_ms)
    small = len(s) < 100  # with fewer than 100 samples, nearest-rank p99 is the maximum
    return {"count": len(s), "p50_ms": round(percentile(s, 0.50), 1), "p95_ms": round(percentile(s, 0.95), 1),
            "p99_ms": round(s[-1] if small else percentile(s, 0.99), 1), "max_ms": round(s[-1], 1),
            "mean_ms": round(statistics.fmean(s), 1),
            "p99_is_max": small or percentile(s, 0.99) == s[-1]}


async def run(url: str, n: int, concurrency: int, lengths: list[int], timeout_s: float, warmup: int) -> dict:
    texts = [make_text(lengths[i % len(lengths)], i) for i in range(n)]
    records: list[dict] = []
    counter = iter(range(n))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=timeout_s, limits=limits, trust_env=False) as client:
        health = (await client.get("/health")).json()
        for i in range(warmup):
            await client.post("/classify", json={"text": make_text(lengths[i % len(lengths)], 10_000 + i)})

        async def worker() -> None:
            for i in counter:
                rec = {"i": i, "approx_tokens": lengths[i % len(lengths)]}
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/classify", json={"text": texts[i], "request_id": f"lt-{i}"})
                    rec["latency_ms"] = (time.perf_counter() - t0) * 1000
                    rec["http_status"] = resp.status_code
                    if resp.status_code == 200:
                        body = resp.json()
                        r = body["results"][0]
                        rec.update(decision=body["decision"], status=r["status"], input_tokens=r["input_tokens"],
                                   n_chunks=r["n_chunks"], detector_latency_ms=r["latency_ms"])
                except httpx.HTTPError as exc:
                    rec["latency_ms"] = (time.perf_counter() - t0) * 1000
                    rec["transport_error"] = type(exc).__name__
                records.append(rec)

        before = server_counters((await client.get("/metrics")).text)
        load_before = load_average()
        wall0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall_s = time.perf_counter() - wall0
        load_after = load_average()
        metrics_text = (await client.get("/metrics")).text
        server_run = counter_delta(before, server_counters(metrics_text))

    ok = [r for r in records if r.get("status") == "ok"]
    transport = sum(1 for r in records if "transport_error" in r)
    http_err = sum(1 for r in records if r.get("http_status") not in (None, 200))
    det_fail: dict[str, int] = {}
    for r in records:
        if r.get("status") not in (None, "ok"):
            det_fail[r["status"]] = det_fail.get(r["status"], 0) + 1
    by_len = {}
    for length in lengths:
        sub = [r for r in ok if r["approx_tokens"] == length]
        toks = sorted(r["input_tokens"] for r in sub)
        by_len[str(length)] = {**summarize([r["latency_ms"] for r in sub]),
                               "input_tokens_median": toks[len(toks) // 2] if toks else None}
    batch_lines = [ln for ln in metrics_text.splitlines() if ln.startswith("sentinel_inference_batch_size_")
                   and ("_sum" in ln or "_count" in ln)]
    try:
        import onnxruntime
        ort_version = onnxruntime.__version__
    except ImportError:
        ort_version = None
    return {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {"cpu": cpu_model(), "logical_cores": os.cpu_count(), "gpu": None,
                     "note": "client and server on the same machine",
                     "load_average_1_5_15": {"before": load_before, "after": load_after}},
        "versions": {"python": platform.python_version(), "httpx": httpx.__version__, "onnxruntime": ort_version},
        "server": {"detector": health.get("detector"), "config": health.get("config")},
        "load": {"mode": "closed-loop", "concurrency": concurrency, "n": n, "warmup": warmup,
                 "lengths_approx_tokens": lengths, "client_timeout_s": timeout_s},
        "errors": {"transport": transport, "http_non_200": http_err, "detector_status_not_ok": det_fail,
                   "total": transport + http_err + sum(det_fail.values())},
        "latency_ok_requests": summarize([r["latency_ms"] for r in ok]),
        "latency_by_length": by_len,
        "throughput_rps": round(len(records) / wall_s, 2),
        "throughput_ok_rps": round(len(ok) / wall_s, 2),
        "wall_s": round(wall_s, 2),
        "server_batch_size_metric": batch_lines,
        "server_this_run": server_run,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--lengths", default="64,256", help="comma-separated approximate content-token lengths")
    ap.add_argument("--timeout", type=float, default=60.0, help="client timeout in seconds")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default="loadtest_result.json")
    ap.add_argument("--label", default=None, help="free-form label stored in the output")
    a = ap.parse_args(argv)
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    if a.n < 1 or a.concurrency < 1 or not lengths:
        ap.error("need --n >= 1, --concurrency >= 1 and at least one length")
    result = asyncio.run(run(a.url, a.n, a.concurrency, lengths, a.timeout, a.warmup))
    if a.label:
        result["label"] = a.label
    text = json.dumps(result, indent=2)
    print(text)
    Path(a.out).write_text(text + "\n")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0 if result["errors"]["total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
