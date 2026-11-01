"""Shared helpers for the serving benchmarks (serving/bench/*.py).

Every benchmark records host conditions (CPU, cores, load average before/after, versions) next to its numbers and
refuses to run unless it holds the shared measurement lock, so no latency number is produced without it:

    flock /home/claude/measure.lock -c 'python -m serving.bench.<name> ...'

`flock` exports nothing, so the scripts check for the lock themselves: they try a non-blocking exclusive flock on
the same file and require it to FAIL (someone, i.e. the wrapping flock, holds it). Pass --no-lock-check only for
smoke runs whose numbers are not reported.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from serving.loadtest import cpu_model

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = "/home/claude/measure.lock"
VAL_PATH = ROOT / "data" / "splits" / "val.jsonl"


def lock_held_by_someone(path: str = LOCK_PATH) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def require_lock(skip: bool) -> None:
    if skip:
        return
    if not lock_held_by_someone():
        raise SystemExit(f"run me under the shared lock: flock {LOCK_PATH} -c '<this command>' "
                         f"(or pass --no-lock-check for an unreported smoke run)")


def uptime() -> str:
    try:
        return subprocess.run(["uptime"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def host_info() -> dict:
    import onnxruntime
    import tokenizers

    return {"cpu": cpu_model(), "logical_cores": os.cpu_count(), "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__, "tokenizers": tokenizers.__version__,
            "platform": platform.platform()}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile (same rule as serving/loadtest.py)."""
    k = max(0, min(len(sorted_vals) - 1, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[k]


def lat_summary(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"n": len(s), "p50_ms": round(percentile(s, 0.5), 2), "p95_ms": round(percentile(s, 0.95), 2),
            "p99_ms": round(percentile(s, 0.99), 2), "max_ms": round(s[-1], 2),
            "mean_ms": round(statistics.fmean(s), 2), "p99_is_max": len(s) < 100 or percentile(s, 0.99) == s[-1]}


def read_val(limit: int | None = None) -> list[dict]:
    """data/splits/val.jsonl in file order. The frozen test split is never read by serving benchmarks."""
    rows = []
    with VAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_result(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
