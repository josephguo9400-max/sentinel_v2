"""Asynchronous shadow mirror.

After the active path has computed its response, the gateway offers the same input to the shadow detector with
`ShadowMirror.offer()`, a non-blocking `put_nowait` into a bounded asyncio queue. The active request never awaits
the shadow; when the queue is full the item is dropped and counted (`sentinel_shadow_dropped_total`).

One consumer task takes items in order and runs the whole shadow pipeline (segmentation, tokenization, inference)
as one job in a dedicated single-thread executor, under its own timeout. Outcomes:

- ok                         a score; compared with the active decision
- timeout (timeout)          the job exceeded the shadow timeout. The thread cannot be killed and keeps running.
- error   (exception)        the shadow raised
- error   (overloaded)       the previous, abandoned job still holds the shadow thread; the item is not started
                             (so a hung shadow costs at most one thread, and the queue drains instead of piling up)

Every outcome becomes a contracts.DetectorResult with role="shadow", is appended to the JSONL log (no raw text,
only input_sha256), and is counted in metrics. Agreement is computed only when both sides made a model decision:
the active response had no policy_applied and no coverage_policy_applied, and the shadow status is ok. The shadow
decision is flag iff score >= its manifest threshold (also for a stage1-role shadow).

Bounded active-path overhead: building the item and one put_nowait; everything else runs in the consumer task or
the shadow thread. The shadow thread competes for CPU with the active path; `shadow_intra_op_threads` (default 1)
limits it to one core's worth of onnxruntime threads.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import DetectorResult

from .detector import tokenize_with_offsets
from .metrics import GatewayMetrics
from .registry import DetectorSpec
from .structure import Segment, build_chunks, segment

log = logging.getLogger("sentinel.shadow")


@dataclass
class ShadowItem:
    request_id: str
    text: str
    active: dict[str, Any]  # decision, decided_by, policy_applied, coverage_policy_applied, stages_run
    enqueued_at: float


def select_first_last(n: int, budget: int) -> list[int]:
    """First (budget - 1) indices plus the last one."""
    if n <= budget:
        return list(range(n))
    if budget == 1:
        return [0]
    return [*range(budget - 1), n - 1]


def score_with_spec(spec: DetectorSpec, text: str, *, max_chunks: int, stride: int = 384,
                    structure_aware: bool = True) -> tuple[float, int, int, bool]:
    """Synchronous single-detector scoring: (max score, input tokens, chunks scored, truncated)."""
    det = spec.detector
    if spec.kind == "text":
        windows, n_tokens = det.windows(text)
        idx = select_first_last(len(windows), max_chunks)
        scores = det.infer([windows[i] for i in idx])
        return max(scores), n_tokens, len(idx), len(idx) < len(windows)
    window = det.max_len - 2
    segs = segment(text) if structure_aware else ([Segment("body", text)] if text.strip() else [])
    cs = build_chunks(segs, lambda t: tokenize_with_offsets(det, t), window=window, stride=min(stride, window),
                      structure_aware=structure_aware)
    idx = select_first_last(len(cs.chunks), max_chunks)
    scores = [det.infer([[det.cls_id, *cs.chunks[i].ids, det.sep_id]])[0] for i in idx]
    return max(scores), cs.n_tokens, len(idx), not cs.covered(idx)


class ShadowMirror:
    def __init__(self, spec: DetectorSpec, *, queue_size: int, timeout_ms: int, max_chunks: int, stride: int,
                 structure_aware: bool, metrics: GatewayMetrics, log_path: str | None):
        self.spec = spec
        self.timeout_s = timeout_ms / 1000
        self.max_chunks = max_chunks
        self.stride = stride
        self.structure_aware = structure_aware
        self.metrics = metrics
        self.log_path = Path(log_path) if log_path else None
        self.queue_size = queue_size
        self._queue: asyncio.Queue[ShadowItem] | None = None
        self._task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._busy = 0  # shadow jobs holding the thread, including abandoned ones
        self._busy_lock = threading.Lock()
        self.counts = {"offered": 0, "dropped": 0, "ok": 0, "timeout": 0, "exception": 0, "overloaded": 0,
                       "agree": 0, "disagree": 0, "not_comparable": 0}

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.queue_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-shadow")
        self._task = asyncio.create_task(self._run(), name="sentinel-shadow")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def queue_depth(self) -> int:
        return 0 if self._queue is None else self._queue.qsize()

    def describe(self) -> dict:
        c = self.counts
        compared = c["agree"] + c["disagree"]
        return {"detector": self.spec.describe(), "queue_depth": self.queue_depth, "queue_size": self.queue_size,
                "timeout_ms": int(self.timeout_s * 1000), "busy": self._busy, "counts": dict(c),
                "agreement_rate": None if compared == 0 else c["agree"] / compared, "n_compared": compared,
                "log_path": str(self.log_path) if self.log_path else None}

    # ------------------------------------------------------------------ active side (must stay cheap)

    def offer(self, item: ShadowItem) -> bool:
        """Non-blocking. Never raises into the active path."""
        try:
            self.counts["offered"] += 1
            if self._queue is None or self._task is None or self._task.done():
                raise asyncio.QueueFull
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.counts["dropped"] += 1
            self.metrics.shadow_dropped.inc()
            return False
        except Exception:  # pragma: no cover - defensive
            log.exception("shadow offer failed")
            return False

    # ------------------------------------------------------------------ consumer

    def _job_done(self, _fut) -> None:
        with self._busy_lock:
            self._busy -= 1

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._process(item)
            except asyncio.CancelledError:
                raise
            except Exception:  # the consumer must survive anything the shadow does
                log.exception("shadow consumer error on request %s", item.request_id)

    async def _process(self, item: ShadowItem) -> None:
        spec = self.spec
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        status, reason, score, n_tokens, n_chunks, truncated = "ok", None, None, 0, 1, False
        with self._busy_lock:
            busy = self._busy >= 1
            if not busy:
                self._busy += 1
        if busy:
            status, reason = "error", "overloaded"
        else:
            fut = loop.run_in_executor(self._executor, lambda: score_with_spec(
                spec, item.text, max_chunks=self.max_chunks, stride=self.stride,
                structure_aware=self.structure_aware))
            fut.add_done_callback(self._job_done)
            try:
                score, n_tokens, n_chunks, truncated = await asyncio.wait_for(asyncio.shield(fut), self.timeout_s)
            except TimeoutError:
                status, reason = "timeout", "timeout"
                fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # retrieve late exceptions
            except Exception:
                log.exception("shadow detector %s failed on request %s", spec.detector_id, item.request_id)
                status, reason = "error", "exception"
        latency_ms = (time.perf_counter() - t0) * 1000
        if status != "ok":
            truncated, score = True, None
        result = DetectorResult(
            request_id=item.request_id, detector_id=spec.detector_id, detector_version=spec.detector_version,
            status=status, score=score, threshold=spec.threshold,
            predicted_injection=None if score is None else bool(score >= spec.threshold), latency_ms=latency_ms,
            input_sha256=hashlib.sha256(item.text.encode("utf-8")).hexdigest(), input_tokens=n_tokens,
            n_chunks=max(1, n_chunks), truncated=truncated, aggregation="max", stage=spec.role, role="shadow",
            failure_reason=reason)
        a = item.active
        comparable = status == "ok" and a["policy_applied"] is None and a["coverage_policy_applied"] is None
        agree = None if not comparable else (result.predicted_injection == (a["decision"] == "flag"))
        m = self.metrics
        self.counts["ok" if status == "ok" else reason] += 1
        m.shadow_results.labels(spec.detector_id, status, reason or "none").inc()
        m.shadow_latency.labels(spec.detector_id, status).observe(latency_ms / 1000)
        self.counts["not_comparable" if agree is None else ("agree" if agree else "disagree")] += 1
        m.shadow_comparisons.labels(spec.detector_id, "not_comparable" if agree is None else
                                    ("agree" if agree else "disagree")).inc()
        if self.log_path is not None:
            rec = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                   "request_id": item.request_id, "queue_wait_ms": (t0 - item.enqueued_at) * 1000,
                   "active": a, "shadow": result.model_dump(mode="json"), "agreement": agree}
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
            except OSError:
                log.exception("cannot write shadow log %s", self.log_path)
