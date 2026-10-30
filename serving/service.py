"""Cascade classification service: structure-aware chunking, chunk-level Stage 1 -> Stage 2 routing, coverage
policy, timeouts, failure policies, admission control and the shadow hand-off.

Request flow (all CPU work runs in bounded thread pools, never on the event loop):

1. Admission (event loop): at most `max_inflight` requests holding `max_inflight_chars` characters; otherwise every
   detector result is status=error, failure_reason=overloaded, and Stage 2's failure policy decides.
2. Prepare (tokenizer pool): segment the input (serving/structure.py) and build one chunk plan in Stage 2
   tokenizer units (windows of max_len - 2 content ids). Both stages see the same chunks.
3. Stage 1 (stage-1 pool, Stage 1 timeout): for each of the first `stage1_max_chunks` chunks, the language of the
   chunk text (serving/language.py) and the Stage 1 score (max over Stage 1's own <=512-regex-token windows of the
   chunk text). A chunk is CLEARED iff Stage 1 scored it, score < gray_zone low, and its language is in Stage 1's
   input_languages (when declared). Stage 1 failure clears nothing: Stage 1's failure policy is never applied while
   a Stage 2 exists, every chunk escalates instead.
4. Stage 1 finalizes ALLOW iff every chunk is cleared. With `stage1_finalize_flag` (default off) it finalizes FLAG
   when any chunk scores >= gray_zone high. Otherwise the request escalates.
5. Stage 2 (micro-batcher, rest of the Stage 2 budget) scores at most `max_chunks` chunks: first and last chunk,
   then uncleared chunks by descending Stage 1 score (chunks Stage 1 did not score first), ties by position.
6. Decision:
   - any Stage 2 chunk score >= threshold                    -> flag, decided by Stage 2
   - Stage 2 failed (timeout / exception / overloaded)       -> Stage 2's failure policy (policy_applied)
   - every chunk cleared by Stage 1 or scored by Stage 2      -> allow, decided by Stage 2
   - otherwise some content was scored by neither stage       -> the coverage policy (coverage_policy_applied):
       flag | quarantine (decision "flag"; the wire contract has no quarantine decision) | allow (explicit) |
       deep_scan: Stage 2 scores the remaining uncovered chunks under `deep_scan_max_chunks` and
       `deep_scan_timeout_ms`; a positive chunk flags, full coverage allows, anything still unscored (budget,
       timeout, error) falls back to `deep_scan_fallback`.
   A flag backed by a positive score is never overridden: under max aggregation unscored content cannot lower the
   request score, so only an outcome that would otherwise be "allow" depends on the coverage policy.

Result semantics: one DetectorResult per stage that ran, in order. Each result's `truncated` is about that detector:
Stage 2's is True whenever Stage 2 did not score every chunk, even if Stage 1 cleared the rest. Whether content was
scored by *neither* stage is visible as coverage_policy_applied (and in metrics and logs).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from contracts import ClassifyResponse, DetectorResult

from .batcher import MicroBatcher, Overloaded
from .config import GatewayConfig
from .detector import TFIDF_V1_TOKEN_RE, tokenize_with_offsets
from .language import UNKNOWN
from .metrics import GatewayMetrics
from .registry import DetectorSpec
from .shadow import ShadowItem, ShadowMirror
from .structure import ChunkSet, Segment, build_chunks, segment

log = logging.getLogger("sentinel.serving")

Decision = Literal["allow", "flag"]


@dataclass
class ClassifyOutcome:
    request_id: str
    decision: Decision
    decided_by: str
    policy_applied: Literal["fail_open", "fail_closed"] | None
    results: list[DetectorResult] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    coverage_policy_applied: Literal["flag", "allow", "quarantine", "deep_scan"] | None = None
    # Not on the wire: diagnostics for metrics, logs and tests.
    info: dict[str, Any] = field(default_factory=dict)

    def to_response(self) -> ClassifyResponse:
        """The validated wire object (contracts.ClassifyResponse v1.2). Raises if inconsistent."""
        return ClassifyResponse(request_id=self.request_id, decision=self.decision, decided_by=self.decided_by,
                                policy_applied=self.policy_applied,
                                coverage_policy_applied=self.coverage_policy_applied,
                                stages_run=list(self.stages_run), results=list(self.results))

    def as_response(self) -> dict[str, Any]:
        return self.to_response().model_dump(mode="json")


@dataclass
class _Stage1Out:
    scores: dict[int, float]  # chunk index -> Stage 1 score (only scored chunks)
    langs: dict[int, str]
    n_tokens: int  # Stage 1 tokenizer units (regex tokens over all segments)


def stage2_order(n: int, cleared: set[int], s1: dict[int, float]) -> list[int]:
    """Stage 2 priority: first, last, then uncleared chunks (Stage 1-unscored first, then by descending score)."""
    head = [0] if n == 1 else [0, n - 1]
    rest = [i for i in range(1, n - 1) if i not in cleared]
    rest.sort(key=lambda i: (-(s1.get(i, 2.0)), i))  # unscored by Stage 1 -> 2.0 -> first
    return head + rest


class ClassifierService:
    """Serves one active configuration: optional Stage 1 (text model), required Stage 2 (token model), optional
    shadow. Built by serving/app.py from the registry (or from an injected detector in tests)."""

    def __init__(self, stage2: DetectorSpec, config: GatewayConfig, metrics: GatewayMetrics | None = None, *,
                 stage1: DetectorSpec | None = None, shadow: DetectorSpec | None = None, language=None):
        if not isinstance(stage2, DetectorSpec):  # a bare detector object (tests)
            stage2 = DetectorSpec(detector=stage2, detector_id=stage2.detector_id,
                                  detector_version=stage2.detector_version, threshold=stage2.threshold, kind="token",
                                  role="single", failure_policy=config.failure_policy, timeout_ms=config.timeout_ms,
                                  max_input_tokens=stage2.max_len)
        det = stage2.detector
        if det.max_len != config.max_len and stage2.artifact_dir is None:
            raise ValueError(f"detector max_len {det.max_len} != config max_len {config.max_len}")
        if stage1 is not None and stage1.gray_zone is None:
            raise ValueError("stage1 needs a gray_zone")
        self.stage1 = stage1
        self.stage2 = stage2
        self.shadow_spec = shadow
        self.detector = det  # kept for /health and older callers
        self.config = config
        self.metrics = metrics or GatewayMetrics()
        self.window = det.max_len - 2
        self.stride = min(config.stride, self.window)
        self.stage1_languages = None
        if stage1 is not None:
            self.stage1_languages = stage1.input_languages if stage1.input_languages is not None \
                else config.undeclared_languages
        if stage1 is not None and self.stage1_languages is not None and language is None:
            from .language import LanguageDetector
            language = LanguageDetector(seed=config.lang_seed, min_letters=config.lang_min_letters,
                                        min_prob=config.lang_min_prob, max_chars=config.lang_max_chars)
        self.language = language
        self.s1_timeout_s = (config.stage1_timeout_ms or (stage1.timeout_ms if stage1 else 1)) / 1000
        self.s2_timeout_s = (config.stage2_timeout_ms or stage2.timeout_ms) / 1000
        self.batcher = MicroBatcher(det.infer, max_batch=config.max_batch, max_wait_ms=config.max_wait_ms,
                                    workers=config.workers, max_queue=config.max_queue,
                                    on_batch=self.metrics.observe_batch)
        self.shadow: ShadowMirror | None = None
        if shadow is not None:
            self.shadow = ShadowMirror(shadow, queue_size=config.shadow_queue,
                                       timeout_ms=config.shadow_timeout_ms or shadow.timeout_ms,
                                       max_chunks=config.shadow_max_chunks, stride=config.stride,
                                       structure_aware=config.structure_aware, metrics=self.metrics,
                                       log_path=config.shadow_log_path or None)
        self.inflight = 0
        self.inflight_chars = 0
        self._tokenizer_pool: ThreadPoolExecutor | None = None
        self._stage1_pool: ThreadPoolExecutor | None = None
        self.started = False

    @property
    def request_budget_ms(self) -> int:
        """Upper bound on gateway-side time per request (what clients should wait for)."""
        extra = self.config.deep_scan_timeout_ms if self.config.coverage_policy == "deep_scan" else 0
        return int(self.s2_timeout_s * 1000) + extra

    async def start(self) -> None:
        self._tokenizer_pool = ThreadPoolExecutor(max_workers=self.config.tokenize_workers,
                                                  thread_name_prefix="sentinel-tokenize")
        self._stage1_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-stage1")
        await self.batcher.start()
        if self.shadow is not None:
            await self.shadow.start()
        self.started = True

    async def stop(self) -> None:
        self.started = False
        if self.shadow is not None:
            await self.shadow.stop()
        await self.batcher.stop()
        for pool in (self._tokenizer_pool, self._stage1_pool):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ admission

    def _admit(self, n_chars: int) -> bool:
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

    # ------------------------------------------------------------------ steps (run in pools)

    def _prepare(self, text: str) -> ChunkSet:
        det = self.stage2.detector
        if self.config.structure_aware:
            segs = segment(text)
        else:
            segs = [Segment("body", text)] if text.strip() else []
        return build_chunks(segs, lambda t: tokenize_with_offsets(det, t), window=self.window, stride=self.stride,
                            structure_aware=self.config.structure_aware)

    def _run_stage1(self, cs: ChunkSet) -> _Stage1Out:
        s1 = self.stage1
        n_tokens = sum(len(TFIDF_V1_TOKEN_RE.findall(s.text)) for s in cs.segments)
        scores: dict[int, float] = {}
        langs: dict[int, str] = {}
        for i, ch in enumerate(cs.chunks[: self.config.stage1_max_chunks]):
            if ch.text is None:  # tokenizer gave no offsets: Stage 1 cannot see this chunk's text
                continue
            if self.stage1_languages is not None:
                langs[i] = self.language.detect(ch.text)[0]
            windows, _ = s1.detector.windows(ch.text)
            scores[i] = max(s1.detector.infer(windows))
        return _Stage1Out(scores=scores, langs=langs, n_tokens=n_tokens)

    async def _stage2_scores(self, cs: ChunkSet, idx: list[int]) -> dict[int, float]:
        det = self.stage2.detector
        rows = [[det.cls_id, *cs.chunks[i].ids, det.sep_id] for i in idx]
        futures = self.batcher.submit(rows)
        try:
            scores = await asyncio.gather(*futures)
        except BaseException:
            for f in futures:  # withdraw queued chunks; running ones cannot be stopped
                if not f.done():
                    f.cancel()
                elif not f.cancelled():
                    f.exception()
            raise
        return dict(zip(idx, scores))

    # ------------------------------------------------------------------ result builders

    def _result(self, spec: DetectorSpec, request_id: str, sha: str, *, status: str, reason: str | None,
                score: float | None, latency_ms: float, n_tokens: int, n_chunks: int, truncated: bool,
                stage: str) -> DetectorResult:
        return DetectorResult(
            request_id=request_id, detector_id=spec.detector_id, detector_version=spec.detector_version,
            status=status, score=score, threshold=spec.threshold,
            predicted_injection=None if score is None else bool(score >= spec.threshold),
            latency_ms=latency_ms, input_sha256=sha, input_tokens=n_tokens, n_chunks=max(1, n_chunks),
            truncated=truncated if status == "ok" else True, aggregation="max", stage=stage, role="active",
            failure_reason=None if status == "ok" else reason)

    @staticmethod
    def _policy_decision(policy: str) -> Decision:
        return "flag" if policy == "fail_closed" else "allow"

    # ------------------------------------------------------------------ main entry

    async def classify(self, text: str, request_id: str | None = None) -> ClassifyOutcome:
        request_id = request_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        if not self._admit(len(text)):
            out = self._overloaded(text, request_id, t0)
        else:
            try:
                out = await self._classify(text, request_id, t0)
            finally:
                self._release(len(text))
        self._observe(out)
        if self.shadow is not None:
            self.shadow.offer(ShadowItem(request_id=request_id, text=text, enqueued_at=time.perf_counter(), active={
                "decision": out.decision, "decided_by": out.decided_by, "policy_applied": out.policy_applied,
                "coverage_policy_applied": out.coverage_policy_applied, "stages_run": list(out.stages_run)}))
        return out

    def _overloaded(self, text: str, request_id: str, t0: float) -> ClassifyOutcome:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lat = (time.perf_counter() - t0) * 1000
        s2 = self.stage2
        results, stages = [], []
        if self.stage1 is not None:
            results.append(self._result(self.stage1, request_id, sha, status="error", reason="overloaded",
                                        score=None, latency_ms=lat, n_tokens=0, n_chunks=1, truncated=True,
                                        stage="stage1"))
            stages.append("stage1")
        results.append(self._result(s2, request_id, sha, status="error", reason="overloaded", score=None,
                                    latency_ms=lat, n_tokens=0, n_chunks=1, truncated=True, stage=s2.role))
        stages.append(s2.role)
        return ClassifyOutcome(request_id, self._policy_decision(s2.failure_policy), s2.detector_id,
                               s2.failure_policy, results, stages, None, {"path": "overloaded"})

    async def _classify(self, text: str, request_id: str, t0: float) -> ClassifyOutcome:
        c, s1, s2 = self.config, self.stage1, self.stage2
        loop = asyncio.get_running_loop()
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        deadline = t0 + self.s2_timeout_s
        info: dict[str, Any] = {}
        results: list[DetectorResult] = []
        stages: list[str] = []

        def remaining() -> float:
            return max(0.0, deadline - time.perf_counter())

        def s2_fail(status: str, reason: str, n_tokens: int = 0, n_chunks: int = 1) -> ClassifyOutcome:
            results.append(self._result(s2, request_id, sha, status=status, reason=reason, score=None,
                                        latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=n_tokens,
                                        n_chunks=n_chunks, truncated=True, stage=s2.role))
            stages.append(s2.role)
            info["path"] = f"stage2_{reason}"
            return ClassifyOutcome(request_id, self._policy_decision(s2.failure_policy), s2.detector_id,
                                   s2.failure_policy, results, stages, None, info)

        # -- prepare
        try:
            cs = await asyncio.wait_for(loop.run_in_executor(self._tokenizer_pool, self._prepare, text), remaining())
        except TimeoutError:
            if s1 is not None:
                results.append(self._result(s1, request_id, sha, status="timeout", reason="timeout", score=None,
                                            latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=0, n_chunks=1,
                                            truncated=True, stage="stage1"))
                stages.append("stage1")
            return s2_fail("timeout", "timeout")
        except Exception:
            log.exception("prepare failed on request %s", request_id)
            return s2_fail("error", "exception")
        n = len(cs.chunks)
        info["n_chunks"] = n
        info["chunk_kinds"] = [ch.kind for ch in cs.chunks]

        # -- stage 1
        cleared: set[int] = set()
        s1_scores: dict[int, float] = {}
        if s1 is not None:
            t1 = time.perf_counter()
            status, reason, s1out = "ok", None, None
            try:
                s1out = await asyncio.wait_for(loop.run_in_executor(self._stage1_pool, self._run_stage1, cs),
                                               min(self.s1_timeout_s, remaining()))
            except TimeoutError:
                status, reason = "timeout", "timeout"
            except Exception:
                log.exception("stage1 %s failed on request %s", s1.detector_id, request_id)
                status, reason = "error", "exception"
            lat1 = (time.perf_counter() - t1) * 1000
            if s1out is not None and s1out.scores:
                s1_scores = s1out.scores
                lo, hi = s1.gray_zone
                for i, sc in s1_scores.items():
                    lang_ok = self.stage1_languages is None or s1out.langs.get(i, UNKNOWN) in self.stage1_languages
                    if sc < lo and lang_ok:
                        cleared.add(i)
                info["stage1_langs"] = [s1out.langs.get(i) for i in range(n)]
                results.append(self._result(s1, request_id, sha, status="ok", reason=None,
                                            score=max(s1_scores.values()), latency_ms=lat1,
                                            n_tokens=s1out.n_tokens, n_chunks=len(s1_scores),
                                            truncated=len(s1_scores) < n, stage="stage1"))
            else:
                if status == "ok":  # ran but could score nothing (no chunk text)
                    status, reason = "error", "exception"
                results.append(self._result(s1, request_id, sha, status=status, reason=reason, score=None,
                                            latency_ms=lat1, n_tokens=0, n_chunks=n, truncated=True,
                                            stage="stage1"))
            stages.append("stage1")
            info["stage1_cleared"] = len(cleared)
            if len(cleared) == n:
                info["path"] = "stage1_allow"
                return ClassifyOutcome(request_id, "allow", s1.detector_id, None, results, stages, None, info)
            if c.stage1_finalize_flag and s1_scores and max(s1_scores.values()) >= s1.gray_zone[1]:
                info["path"] = "stage1_flag"
                return ClassifyOutcome(request_id, "flag", s1.detector_id, None, results, stages, None, info)

        # -- stage 2
        order = stage2_order(n, cleared, s1_scores)
        chosen = order[: c.max_chunks]
        info["stage2_candidates"] = len(order)
        try:
            s2_scores = await asyncio.wait_for(self._stage2_scores(cs, chosen), remaining())
        except TimeoutError:
            return s2_fail("timeout", "timeout", cs.n_tokens, len(chosen))
        except Overloaded:
            return s2_fail("error", "overloaded", cs.n_tokens, len(chosen))
        except Exception:
            log.exception("stage2 %s failed on request %s", s2.detector_id, request_id)
            return s2_fail("error", "exception", cs.n_tokens, len(chosen))

        coverage: Literal["flag", "allow", "quarantine", "deep_scan"] | None = None
        decided_by = s2.detector_id
        positive = any(v >= s2.threshold for v in s2_scores.values())
        # Content scored by neither stage: not cleared by Stage 1 and not scored by Stage 2.
        uncovered = [i for i in order if i not in s2_scores and i not in cleared]
        if not positive and uncovered and c.coverage_policy == "deep_scan":
            extra = uncovered[: c.deep_scan_max_chunks]
            try:
                more = await asyncio.wait_for(self._stage2_scores(cs, extra), c.deep_scan_timeout_ms / 1000)
                s2_scores.update(more)
                info["deep_scan"] = {"scored": len(extra)}
            except (TimeoutError, Overloaded) as exc:
                info["deep_scan"] = {"failed": "timeout" if isinstance(exc, TimeoutError) else "overloaded"}
            except Exception:
                log.exception("deep scan failed on request %s", request_id)
                info["deep_scan"] = {"failed": "exception"}
            positive = any(v >= s2.threshold for v in s2_scores.values())
            uncovered = [i for i in uncovered if i not in s2_scores]
            if positive or not uncovered:
                coverage = "deep_scan"
                decided_by = f"{s2.detector_id}+coverage_policy:deep_scan"
        s2_score = max(s2_scores.values())
        stage2_truncated = not cs.covered(s2_scores.keys())
        results.append(self._result(s2, request_id, sha, status="ok", reason=None, score=s2_score,
                                    latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=cs.n_tokens,
                                    n_chunks=len(s2_scores), truncated=stage2_truncated, stage=s2.role))
        stages.append(s2.role)
        info["stage2_scored"] = len(s2_scores)
        info["unscored_chunks"] = 0 if positive else len(uncovered)
        if positive:
            info["path"] = "stage2_flag"
            return ClassifyOutcome(request_id, "flag", decided_by, None, results, stages, coverage, info)
        if not uncovered:
            info["path"] = "stage2_allow"
            return ClassifyOutcome(request_id, "allow", decided_by, None, results, stages, coverage, info)
        policy = c.deep_scan_fallback if c.coverage_policy == "deep_scan" else c.coverage_policy
        info["path"] = f"coverage_{policy}"
        tag = f"coverage_policy:{c.coverage_policy}" + (f"->{policy}" if c.coverage_policy == "deep_scan" else "")
        decision: Decision = "allow" if policy == "allow" else "flag"
        return ClassifyOutcome(request_id, decision, tag, None, results, stages, policy, info)

    # ------------------------------------------------------------------ metrics

    def _observe(self, out: ClassifyOutcome) -> None:
        m = self.metrics
        m.requests.labels(out.decision, out.policy_applied or "none").inc()
        m.paths.labels(out.info.get("path", "unknown")).inc()
        if out.coverage_policy_applied is not None:
            m.coverage.labels(out.coverage_policy_applied).inc()
        if "n_chunks" in out.info:
            m.request_chunks.observe(out.info["n_chunks"])
        for r in out.results:
            m.detector_latency.labels(r.detector_id, r.status).observe(r.latency_ms / 1000)
            if r.status != "ok":
                m.detector_failures.labels(r.detector_id, r.status, r.failure_reason or "none").inc()
            else:
                m.chunks.labels(r.stage).observe(r.n_chunks)
            if r.truncated:
                m.truncated.labels(r.detector_id).inc()
