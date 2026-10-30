"""Classification service: chunking, batching, timeout and failure policy for the single active detector.

The decision rule for this milestone (one active detector, stage="single"):
- Aggregation is always reported as "max": it is the rule this gateway applies (a single chunk is the
max over one score), and for failed results it is the rule that would have applied.

- status ok      -> decision = "flag" if predicted_injection else "allow", policy_applied = None
- status != ok   -> the detector's failure policy: fail_closed -> "flag", fail_open -> "allow",
                    policy_applied = the policy name

Overload is fail-fast at two points, both with status "error", reason "overloaded" and the failure policy:
1. Admission, before any work: at most `max_inflight` unfinished requests holding at most
   `max_inflight_chars` characters in total. Tokenization runs in a pool of `tokenize_workers` threads,
   so a burst of large bodies cannot fan out into an unbounded number of tokenizer threads.
2. The chunk queue (`max_queue`), checked after tokenization.
"""
import asyncio
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from contracts import DetectorResult

from .batcher import MicroBatcher, Overloaded
from .config import GatewayConfig
from .detector import Detector
from .encoding import plan_chunks
from .metrics import GatewayMetrics

log = logging.getLogger("sentinel.serving")


@dataclass
class ClassifyOutcome:
    request_id: str
    decision: Literal["allow", "flag"]
    decided_by: str
    policy_applied: Literal["fail_open", "fail_closed"] | None
    results: list[DetectorResult] = field(default_factory=list)

    def as_response(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "decision": self.decision, "decided_by": self.decided_by,
                "policy_applied": self.policy_applied, "results": [r.model_dump() for r in self.results]}


class ClassifierService:
    def __init__(self, detector: Detector, config: GatewayConfig, metrics: GatewayMetrics | None = None):
        if detector.max_len != config.max_len:
            raise ValueError(f"detector max_len {detector.max_len} != config max_len {config.max_len}")
        self.detector = detector
        self.config = config
        self.metrics = metrics or GatewayMetrics()
        self.batcher = MicroBatcher(detector.infer, max_batch=config.max_batch, max_wait_ms=config.max_wait_ms,
                                    workers=config.workers, max_queue=config.max_queue,
                                    on_batch=self.metrics.batch_size.observe)
        self.inflight = 0
        self.inflight_chars = 0
        self._tokenizer_pool: ThreadPoolExecutor | None = None
        self.started = False

    async def start(self) -> None:
        self._tokenizer_pool = ThreadPoolExecutor(max_workers=self.config.tokenize_workers,
                                                  thread_name_prefix="sentinel-tokenize")
        await self.batcher.start()
        self.started = True

    async def stop(self) -> None:
        self.started = False
        await self.batcher.stop()
        if self._tokenizer_pool is not None:
            self._tokenizer_pool.shutdown(wait=False, cancel_futures=True)

    def _admit(self, n_chars: int) -> bool:
        """Reserve capacity for one request, or return False (overloaded). Runs on the event loop, so
        check-and-reserve is atomic without a lock."""
        c = self.config
        if self.inflight >= c.max_inflight or self.inflight_chars + n_chars > c.max_inflight_chars:
            return False
        self.inflight += 1
        self.inflight_chars += n_chars
        self.metrics.inflight.set(self.inflight)
        return True

    def _release(self, n_chars: int) -> None:
        self.inflight -= 1
        self.inflight_chars -= n_chars
        self.metrics.inflight.set(self.inflight)

    async def _score(self, text: str, progress: dict[str, Any]) -> tuple[float, int, int, bool]:
        """(max score, content tokens, chunks scored, truncated). Raises on detector error/overload.

        `progress` records tokens and planned chunks as soon as they are known, so a result for a
        request that later times out still reports them.
        """
        d, c = self.detector, self.config
        # Bounded pool: if this request times out while its tokenization is still queued, cancelling the
        # asyncio future cancels the pool job too, so it never runs.
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(self._tokenizer_pool, d.tokenize, text)
        progress["input_tokens"] = len(content)
        plan = plan_chunks(content, d.cls_id, d.sep_id, c.max_len, c.stride, c.max_chunks)
        progress["n_chunks"] = len(plan.chunks)
        futures = self.batcher.submit(plan.chunks)
        try:
            scores = await asyncio.gather(*futures)
        except BaseException:
            for f in futures:  # withdraw queued chunks of this request; running ones cannot be stopped
                if not f.done():
                    f.cancel()
                elif not f.cancelled():
                    f.exception()  # mark retrieved
            raise
        return max(scores), len(content), len(plan.chunks), plan.truncated

    async def classify(self, text: str, request_id: str | None = None) -> ClassifyOutcome:
        d, c, m = self.detector, self.config, self.metrics
        request_id = request_id or uuid.uuid4().hex
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        t0 = time.perf_counter()
        status, reason = "ok", ""
        score = None
        n_tokens, n_chunks, truncated = 0, 1, False
        progress: dict[str, Any] = {}
        if not self._admit(len(text)):
            status, reason = "error", "overloaded"
        else:
            try:
                score, n_tokens, n_chunks, truncated = await asyncio.wait_for(self._score(text, progress),
                                                                              c.timeout_ms / 1000)
            except TimeoutError:
                status, reason = "timeout", "timeout"
            except Overloaded:
                status, reason = "error", "overloaded"
            except Exception:
                log.exception("detector %s failed on request %s", d.detector_id, request_id)
                status, reason = "error", "exception"
            finally:
                self._release(len(text))
        latency_ms = (time.perf_counter() - t0) * 1000

        if status != "ok":
            # Nothing was scored. Report what we know honestly: content was not scored -> truncated=True.
            truncated = True
            n_tokens = progress.get("input_tokens", 0)
            n_chunks = progress.get("n_chunks", 1)
        result = DetectorResult(
            request_id=request_id, detector_id=d.detector_id, detector_version=d.detector_version, status=status,
            score=score, threshold=d.threshold,
            predicted_injection=None if score is None else bool(score >= d.threshold),
            latency_ms=latency_ms, input_sha256=sha, input_tokens=n_tokens, n_chunks=n_chunks,
            truncated=truncated, aggregation="max", stage="single", role="active")

        if status == "ok":
            decision = "flag" if result.predicted_injection else "allow"
            policy = None
            m.chunks.observe(n_chunks)
        else:
            policy = c.failure_policy
            decision = "flag" if policy == "fail_closed" else "allow"
            m.detector_failures.labels(d.detector_id, status, reason).inc()
        if truncated:
            m.truncated.labels(d.detector_id).inc()
        m.detector_latency.labels(d.detector_id, status).observe(latency_ms / 1000)
        m.requests.labels(decision, policy or "none").inc()
        return ClassifyOutcome(request_id=request_id, decision=decision, decided_by=d.detector_id,
                               policy_applied=policy, results=[result])
