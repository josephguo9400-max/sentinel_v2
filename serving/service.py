"""Policy engine (engine 1.0.0): serves one ExecutionPolicy (contracts/policy.py) with its loaded capabilities.

This is the cascade service of milestone cascade-and-shadow refactored in place (D25, docs/design/capability-policy.md
§7 migration 2): the decision path keeps its code (`_classify_active`, `stage2_order`, `_stage1_pass`, coverage,
deep scan); its parameters now come from the policy instead of the registry pointers, manifests and GatewayConfig.
The pre-refactor file is kept verbatim as serving/legacy_service.py, the reference the equivalence tests (T2, T3)
compare against. GatewayConfig still supplies capacity settings only (batching, workers, admission limits,
tokenizer pool, shadow log path).

Request flow (all CPU work runs in bounded thread pools, never on the event loop):

1. Admission (event loop): at most `max_inflight` requests holding `max_inflight_chars` characters; otherwise every
   active step's result is status=error, failure_reason=overloaded, and the failure rules below decide (FALLBACK is
   never attempted on overloaded).
2. Prepare (tokenizer pool): segment the input (serving/structure.py, `chunking.structure_aware`) and build one
   chunk plan in the tokenizer units of `chunking.plan_tokenizer` (windows of max_len - 2 content ids,
   `chunking.stride`). Every step sees the same chunks.
3. Gate step (optional; text model; `gate.timeout_ms`, measured inside the decide budget): for each of the first
   `gate.max_chunks` chunks, in order, the gate score under the capability's `scoring` rule: max over the model's own
   windows of the chunk text, of each segment piece alone when the chunk packs several (`score_segment_pieces`), and
   of overlapping sub-windows (`subwindow_tokens` / `subwindow_stride`); then, only for a chunk scoring below
   `band.clear_below`, the chunk's language (policy `language_detector`). A chunk is CLEARED iff the gate scored it,
   score < clear_below, and its language is in `band.scope_languages` (when set). The gate is anytime: it starts no
   new chunk after (budget - deadline_margin_ms, the margin capped at half the budget) and returns the chunks it
   scored (status ok, truncated); the unscored rest is uncleared (not a failure).
4. The gate finalizes ALLOW iff every chunk is cleared and "allow" is in band.finalize; FLAG iff "flag" is in
   band.finalize and any chunk scored >= band.flag_at_or_above. Otherwise the request escalates (escalation_reason).
   Gate failure (timeout, exception, or a run that scored nothing) follows gate.on_failure: FAIL_OPEN -> allow,
   FAIL_CLOSED -> flag (decided by the gate, policy_applied), FALLBACK -> escalate with nothing cleared under the
   fallback's budget, and if the decide stage then fails, the fallback's `terminal` decides.
5. Decide stage scores at most `max_chunks` chunks per member: first and last chunk, then uncleared chunks by
   descending gate score (chunks the gate did not score first), ties by position.
   - single: any chunk score >= the member's calibrated threshold -> flag. A failed member follows its on_failure:
     FAIL_OPEN / FAIL_CLOSED, or FALLBACK: the fallback detector scores the same chunks under its own budget
     (not on overloaded); if it fails too, its `terminal` decides.
   - parallel (shape B; tested with fakes only, D25.5): both members score the same chunks concurrently; each votes
     flag/allow from its own threshold, a failed member votes by its on_failure; `aggregate` any_flag / all_flag.
     policy_applied is set when no ok member's vote alone produced the decision.
6. Coverage (content neither cleared by the gate nor scored by a decide member, outcome otherwise allow):
   flag | quarantine (decision "flag"; there is no quarantine decision, D25.1) | allow (explicit) |
   deep_scan: the decide member scores the remaining uncovered chunks under `deep_scan.max_chunks` /
   `deep_scan.timeout_ms`; a positive chunk flags, full coverage allows, anything still unscored falls back to
   `deep_scan.fallback`. A flag backed by a positive score is never overridden.
7. Observe steps (text models), after the decision is fixed and the admission slot is released, each on the same
   chunk plan under its own budget and at most `concurrency` at a time (busy -> dropped, error/overloaded, never
   queued). Result role="observe" (stage="stage1"), not listed in stages_run, never read by the decision. Its
   optional band is a what-if for metrics (would_finalize_allow / would_escalate).
8. Shadow step: the asynchronous mirror (serving/shadow.py), offered after the response is built.

Every response carries policy_id, policy_version, bundle_sha256, unscored_by_any_step and escalation_reason (v1.3).

Construction: `ClassifierService(bundle=LoadedBundle, config=...)` (registry v1 converted or v2), or the legacy
signature `ClassifierService(stage2, config, stage1=..., shadow=..., stage1_mode=...)` used by tests and scripts:
the specs and config are converted with serving.policy.legacy_to_v2 and served by the same engine.
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

from contracts import ArtifactManifest, ClassifyResponse, DetectorResult
from contracts.manifest import Runtime
from contracts.policy import Band, ExecutionPolicy

from .batcher import MicroBatcher, Overloaded
from .config import GatewayConfig
from .detector import TFIDF_V1_TOKEN_RE, tfidf_subwindows, tokenize_with_offsets
from .engine_version import ENGINE_VERSION
from .language import UNKNOWN
from .metrics import GatewayMetrics
from .registry import DetectorSpec, LoadedBundle
from .shadow import ShadowItem, ShadowMirror
from .structure import ChunkSet, Segment, build_chunks, segment

log = logging.getLogger("sentinel.serving")

Decision = Literal["allow", "flag"]
_MODE_POLICY = {"FAIL_OPEN": "fail_open", "FAIL_CLOSED": "fail_closed"}


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
    # v1.3 wire fields
    unscored_by_any_step: bool | None = None
    escalation_reason: Literal["band", "language", "coverage", "gate_failed"] | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    bundle_sha256: str | None = None

    def to_response(self) -> ClassifyResponse:
        """The validated wire object (contracts.ClassifyResponse v1.3). Raises if inconsistent."""
        return ClassifyResponse(request_id=self.request_id, decision=self.decision, decided_by=self.decided_by,
                                policy_applied=self.policy_applied,
                                coverage_policy_applied=self.coverage_policy_applied,
                                stages_run=list(self.stages_run), results=list(self.results),
                                policy_id=self.policy_id, policy_version=self.policy_version,
                                bundle_sha256=self.bundle_sha256, unscored_by_any_step=self.unscored_by_any_step,
                                escalation_reason=self.escalation_reason)

    def as_response(self) -> dict[str, Any]:
        return self.to_response().model_dump(mode="json")


@dataclass
class _Stage1Out:
    scores: dict[int, float]  # chunk index -> text-step score (only scored chunks)
    langs: dict[int, str]  # only chunks scoring below the band's clear_below (the only ones whose language matters)
    n_tokens: int  # text-model tokenizer units (regex tokens over all segments)
    deadline_hit: bool = False  # stopped at the soft deadline with chunks left unscored


@dataclass
class _Stage1Pass:
    result: DetectorResult
    cleared: set[int]
    scores: dict[int, float]
    out: _Stage1Out | None


@dataclass
class _TextStep:
    """A resolved gate or observe step (text model)."""
    kind: Literal["gate", "observe"]
    spec: DetectorSpec
    band: Band | None
    timeout_s: float
    margin_s: float  # capped at half the budget so a short budget still lets the step start
    max_chunks: int
    subwindow_tokens: int
    subwindow_stride: int
    score_segment_pieces: bool
    languages: list[str] | None
    failure: Any = None  # gate: contracts.policy.Failure
    concurrency: int = 1  # observe
    slots: int = 0  # observe runs in progress

    @property
    def soft_budget_s(self) -> float:
        return self.timeout_s - self.margin_s


@dataclass
class _Member:
    """A resolved decide member, or a fallback detector."""
    spec: DetectorSpec
    timeout_s: float
    max_chunks: int
    failure: Any  # contracts.policy.Failure (a fallback has none: its terminal is on the parent's Failure)
    stage: str  # DetectorResult.stage / stages_run label
    batcher: MicroBatcher
    fallback: "_Member | None" = None
    fallback_terminal: str | None = None


def stage2_order(n: int, cleared: set[int], s1: dict[int, float]) -> list[int]:
    """Decide-stage priority: first, last, then uncleared chunks (gate-unscored first, then by descending score)."""
    head = [0] if n == 1 else [0, n - 1]
    rest = [i for i in range(1, n - 1) if i not in cleared]
    rest.sort(key=lambda i: (-(s1.get(i, 2.0)), i))  # unscored by the gate -> 2.0 -> first
    return head + rest


class _TagMapped:
    """Applies a LanguageDetector.tag_map after the built-in region-subtag reduction."""

    def __init__(self, inner, tag_map: dict[str, str]):
        self.inner, self.tag_map = inner, dict(tag_map)

    def describe(self) -> dict:
        return {**self.inner.describe(), "tag_map": self.tag_map}

    def detect(self, text: str) -> tuple[str, float]:
        lang, p = self.inner.detect(text)
        return self.tag_map.get(lang, lang), p


# ---------------------------------------------------------------------- legacy construction (tests, scripts)

def _injected_manifest(spec: DetectorSpec, role: str) -> ArtifactManifest:
    """An unvalidated ArtifactManifest stand-in for a detector object injected without a manifest (tests)."""
    token = spec.kind == "token"
    sha = spec.sha256 if isinstance(spec.sha256, str) and len(spec.sha256) == 64 else hashlib.sha256(
        f"injected:{spec.detector_id}@{spec.detector_version}".encode()).hexdigest()
    return ArtifactManifest.model_construct(
        schema_version="1.2", detector_id=spec.detector_id, detector_version=spec.detector_version,
        artifact_format="onnx", artifact_path="onnx/model.onnx" if token else "model.onnx", sha256=sha,
        runtime=Runtime(python="0.0", onnxruntime="0.0", tokenizers="0.0" if token else None),
        preprocessing_version="deberta-v3-window-v1" if token else "tfidf-v1",
        label_mapping={"0": "benign", "1": "injection"}, threshold=spec.threshold,
        max_input_tokens=spec.max_input_tokens, failure_policy=spec.failure_policy, timeout_ms=spec.timeout_ms,
        cascade_role=role, gray_zone=spec.gray_zone,
        tokenizer=(spec.tokenizer if token and spec.tokenizer else f"hf:injected-{spec.detector_id}") if token
        else (spec.tokenizer or "regex:[\\p{L}\\p{N}_]{2,}"),
        input_languages=list(spec.input_languages) if spec.input_languages else None, training_data_version=None,
        evaluation_dataset_version=None, created_at="2026-01-01T00:00:00Z", metrics_path=None)


def bundle_from_specs(stage2: DetectorSpec, config: GatewayConfig, *, stage1: DetectorSpec | None = None,
                      shadow: DetectorSpec | None = None, stage1_mode: str = "route") -> LoadedBundle:
    """The legacy constructor's arguments as a bundle (legacy_to_v2 on the specs' manifests)."""
    from .convert_registry import CALIBRATION_RECORDS
    from .policy import bundle_sha256, legacy_to_v2

    specs = {"stage2": stage2, "stage1": stage1, "shadow": shadow}
    pointers: dict[str, Any] = {"stage1_mode": stage1_mode}
    manifests: dict[str, Any] = {}
    known = dict(config.known_tokenizers)
    for name, spec in specs.items():
        if spec is None:
            pointers[name] = None
            continue
        pointers[name] = spec.artifact_dir or f"injected/{name}"  # distinct per pointer
        m = spec.manifest if spec.manifest is not None else _injected_manifest(
            spec, "stage1" if name == "stage1" else spec.role if spec.role in ("stage2", "single") else "stage2")
        manifests[name] = m
        if spec.kind == "token" and m.tokenizer not in known:
            known[m.tokenizer] = "0" * 64  # an injected tokenizer: no file to pin
    policy, caps = legacy_to_v2(pointers, manifests, config, known_tokenizers=known,
                                calibrations=CALIBRATION_RECORDS, policy_id="injected")
    detectors = {}
    for name in ("stage2", "stage1"):
        spec = specs[name]
        if spec is not None:
            detectors[(spec.detector_id, spec.detector_version)] = spec
    return LoadedBundle(policy=policy, capabilities=caps, detectors=detectors, shadow=shadow,
                        bundle_sha256=bundle_sha256(policy), policy_source="injected")


class ClassifierService:
    """Serves one bundle: an ExecutionPolicy with loaded detectors. Built by serving/app.py from the registry (v1
    converted in memory, or v2), or with the legacy signature from injected detectors (tests)."""

    def __init__(self, stage2: DetectorSpec | Any = None, config: GatewayConfig | None = None,
                 metrics: GatewayMetrics | None = None, *, stage1: DetectorSpec | None = None,
                 shadow: DetectorSpec | None = None, language=None, stage1_mode: Literal["route", "observe"] = "route",
                 bundle: LoadedBundle | None = None):
        config = config or GatewayConfig()
        if bundle is None:
            if stage2 is None:
                raise ValueError("pass a bundle, or a Stage 2 detector (legacy signature)")
            if not isinstance(stage2, DetectorSpec):  # a bare detector object (tests)
                stage2 = DetectorSpec(detector=stage2, detector_id=stage2.detector_id,
                                      detector_version=stage2.detector_version, threshold=stage2.threshold,
                                      kind="token", role="single", failure_policy=config.failure_policy,
                                      timeout_ms=config.timeout_ms, max_input_tokens=stage2.max_len)
            if stage2.detector.max_len != config.max_len and stage2.artifact_dir is None:
                raise ValueError(f"detector max_len {stage2.detector.max_len} != config max_len {config.max_len}")
            if stage1 is not None and stage1.gray_zone is None:
                raise ValueError("stage1 needs a gray_zone")
            if stage1_mode not in ("route", "observe"):
                raise ValueError(f"stage1_mode must be route or observe, got {stage1_mode!r}")
            bundle = bundle_from_specs(stage2, config, stage1=stage1, shadow=shadow, stage1_mode=stage1_mode)
        self.bundle = bundle
        self.policy: ExecutionPolicy = bundle.policy
        p = self.policy
        if p.canary is not None:
            raise ValueError("canary is not implemented in this engine (D25.4)")
        self.config = config
        self.metrics = metrics or GatewayMetrics()
        self.engine_version = ENGINE_VERSION
        self._specs = bundle.detectors
        self._batchers: dict[tuple[str, str], MicroBatcher] = {}

        # -- chunk plan
        plan = self._specs[p.chunking.plan_tokenizer.key]
        self.window = plan.detector.max_len - 2
        self.stride = min(p.chunking.stride, self.window)
        self.structure_aware = p.chunking.structure_aware

        # -- language detector (only when a band declares scope_languages)
        if p.language_detector is not None and language is None:
            from .language import LanguageDetector
            ld = p.language_detector
            language = LanguageDetector(seed=ld.seed, min_letters=ld.min_letters, min_prob=ld.min_prob,
                                        max_chars=ld.max_chars)
            if ld.tag_map:
                language = _TagMapped(language, ld.tag_map)
        self.language = language

        # -- steps
        self.gate = self._text_step("gate", p.gate) if p.gate is not None else None
        self.observe_steps = [self._text_step("observe", o) for o in p.observe]
        self.members = [self._member(m) for m in p.decide.members]
        self.decide_mode = p.decide.mode
        self.aggregate = p.decide.aggregate
        self.coverage = p.coverage
        if self.gate is not None and self.gate.failure.mode == "FALLBACK":
            self.gate_fallback_timeout_s = self.gate.failure.fallback.timeout_ms / 1000
            self.gate_fallback_terminal = self.gate.failure.fallback.terminal
        self.shadow_spec = bundle.shadow
        self.shadow: ShadowMirror | None = None
        if p.shadow is not None and bundle.shadow is not None:
            s = p.shadow
            self.shadow = ShadowMirror(bundle.shadow, queue_size=s.queue, timeout_ms=s.timeout_ms,
                                       max_chunks=s.max_chunks, stride=p.chunking.stride,
                                       structure_aware=p.chunking.structure_aware, metrics=self.metrics,
                                       log_path=config.shadow_log_path or None)

        # -- names kept for /health, scripts and older callers
        self._s1_step = self.gate or (self.observe_steps[0] if self.observe_steps else None)
        self.stage1 = self._s1_step.spec if self._s1_step else None
        self.stage1_mode = "route" if self.gate else "observe" if self.observe_steps else stage1_mode
        self.stage1_languages = self._s1_step.languages if self._s1_step else None
        self.stage2 = self.members[0].spec
        self.detector = self.stage2.detector
        self.batcher = self.members[0].batcher
        self.s1_timeout_s = self._s1_step.timeout_s if self._s1_step else 0.001
        self.s1_margin_s = self._s1_step.margin_s if self._s1_step else 0.0
        self.s1_soft_budget_s = self.s1_timeout_s - self.s1_margin_s
        self.s2_timeout_s = self.members[0].timeout_s

        self.inflight = 0
        self.inflight_chars = 0
        self._tokenizer_pool: ThreadPoolExecutor | None = None
        self._stage1_pool: ThreadPoolExecutor | None = None
        self.started = False

    # ------------------------------------------------------------------ resolution

    def _batcher(self, spec: DetectorSpec) -> MicroBatcher:
        key = (spec.detector_id, spec.detector_version)
        if key not in self._batchers:
            c = self.config
            self._batchers[key] = MicroBatcher(spec.detector.infer, max_batch=c.max_batch, max_wait_ms=c.max_wait_ms,
                                               workers=c.workers, max_queue=c.max_queue,
                                               on_batch=self.metrics.observe_batch)
        return self._batchers[key]

    def _text_step(self, kind: str, step) -> _TextStep:
        spec = self._specs[step.detector.key]
        sc = self.bundle.capabilities[step.detector.key].scoring
        timeout_s = step.timeout_ms / 1000
        band = getattr(step, "band", None)
        return _TextStep(kind=kind, spec=spec, band=band, timeout_s=timeout_s,
                         margin_s=min(step.deadline_margin_ms / 1000, timeout_s / 2), max_chunks=step.max_chunks,
                         subwindow_tokens=sc.subwindow_tokens, subwindow_stride=sc.subwindow_stride or 1,
                         score_segment_pieces=sc.score_segment_pieces,
                         languages=list(band.scope_languages) if band is not None and band.scope_languages else None,
                         failure=getattr(step, "on_failure", None), concurrency=getattr(step, "concurrency", 1))

    def _label(self, spec: DetectorSpec) -> str:
        return spec.role if spec.role in ("stage2", "single") else "stage2"

    def _member(self, m) -> _Member:
        spec = self._specs[m.detector.key]
        out = _Member(spec=spec, timeout_s=m.timeout_ms / 1000, max_chunks=m.max_chunks, failure=m.on_failure,
                      stage=self._label(spec), batcher=self._batcher(spec))
        fb = m.on_failure.fallback
        if fb is not None:
            fspec = self._specs[fb.detector.key]
            out.fallback = _Member(spec=fspec, timeout_s=fb.timeout_ms / 1000, max_chunks=m.max_chunks,
                                   failure=None, stage=self._label(fspec), batcher=self._batcher(fspec))
            out.fallback_terminal = fb.terminal
        return out

    # ------------------------------------------------------------------ lifecycle

    @property
    def busy(self) -> bool:
        """Admitted requests or observe-only work still running (the app waits for this before retiring)."""
        return bool(self.inflight or any(s.slots for s in self.observe_steps))

    @property
    def _observe_slots(self) -> int:
        return sum(s.slots for s in self.observe_steps)

    @property
    def request_budget_ms(self) -> int:
        """Upper bound on gateway-side time per request (what clients should wait for)."""
        decide = max(m.timeout_s for m in self.members)
        if self.gate is not None and self.gate.failure.mode == "FALLBACK":
            decide = max(decide, self.gate_fallback_timeout_s)
        fallback = max([m.fallback.timeout_s for m in self.members if m.fallback is not None], default=0.0)
        extra = self.coverage.deep_scan.timeout_ms if self.coverage.deep_scan is not None else 0
        extra += sum(int(s.timeout_s * 1000) for s in self.observe_steps)  # observe steps run after the decision
        return int(decide * 1000) + int(fallback * 1000) + extra

    def describe_policy(self) -> dict:
        from .policy import describe_bundle
        return {**describe_bundle(self.policy, self.engine_version), "policy_source": self.bundle.policy_source,
                "policy_path": self.bundle.policy_path, "status": self.bundle.status, "previous": self.bundle.previous}

    async def start(self) -> None:
        self._tokenizer_pool = ThreadPoolExecutor(max_workers=self.config.tokenize_workers,
                                                  thread_name_prefix="sentinel-tokenize")
        self._stage1_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-stage1")
        for b in self._batchers.values():
            await b.start()
        if self.shadow is not None:
            await self.shadow.start()
        self.started = True

    async def stop(self) -> None:
        self.started = False
        if self.shadow is not None:
            await self.shadow.stop()
        for b in self._batchers.values():
            await b.stop()
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
        det = self._specs[self.policy.chunking.plan_tokenizer.key].detector
        if self.structure_aware:
            segs = segment(text)
        else:
            segs = [Segment("body", text)] if text.strip() else []
        return build_chunks(segs, lambda t: tokenize_with_offsets(det, t), window=self.window, stride=self.stride,
                            structure_aware=self.structure_aware)

    def _run_stage1(self, cs: ChunkSet, deadline: float | None = None) -> _Stage1Out:
        """The first text step (the gate, else the first observe step); kept under its legacy name for scripts."""
        return self._run_text_step(self._s1_step, cs, deadline)

    def _run_text_step(self, step: _TextStep, cs: ChunkSet, deadline: float | None = None) -> _Stage1Out:
        """Score chunks in order; with `deadline` (time.perf_counter() value), start no chunk after it."""
        det = step.spec.detector
        lo = step.band.clear_below if step.band is not None else None
        n_tokens = sum(len(TFIDF_V1_TOKEN_RE.findall(s.text)) for s in cs.segments)
        scores: dict[int, float] = {}
        langs: dict[int, str] = {}
        deadline_hit = False
        for i, ch in enumerate(cs.chunks[: step.max_chunks]):
            if ch.text is None:  # tokenizer gave no offsets: a text model cannot see this chunk's text
                continue
            if deadline is not None and time.perf_counter() >= deadline:
                deadline_hit = True
                break
            # Score the packed chunk and, when it packs several segment pieces, each piece on its own: a short
            # injected paragraph packed with long benign text is diluted in TF-IDF features (measured: 0.999 alone,
            # 0.067 packed), and the text model is cheap enough to score both. Chunk score = max.
            texts = [ch.text] + (ch.parts if step.score_segment_pieces and len(ch.parts) > 1 else [])
            windows = [w for t in texts for w in det.windows(t)[0]]
            if step.subwindow_tokens:
                windows += [w for t in texts for w in
                            tfidf_subwindows(t, step.subwindow_tokens, step.subwindow_stride)]
            scores[i] = max(det.infer(windows))
            # The language only matters for a chunk the band would clear; langdetect was 34-59% of Stage 1's
            # cost (p50, serving/bench_results/stage1_latency_before.json), so chunks at or above it skip it.
            if step.languages is not None and scores[i] < lo:
                langs[i] = self.language.detect(ch.text)[0]
        return _Stage1Out(scores=scores, langs=langs, n_tokens=n_tokens, deadline_hit=deadline_hit)

    async def _stage1_pass(self, step: _TextStep, cs: ChunkSet, request_id: str, sha: str, wait_s: float,
                           role: Literal["active", "observe"]) -> _Stage1Pass:
        """Run a text step under a soft deadline (budget - margin) and a hard wait `wait_s`; never raises."""
        s1 = step.spec
        loop = asyncio.get_running_loop()
        n = len(cs.chunks)
        t1 = time.perf_counter()
        deadline = t1 + max(0.0, wait_s - step.margin_s)
        run = self._run_stage1 if step is self._s1_step else (lambda c, d: self._run_text_step(step, c, d))
        status, reason, s1out = "ok", None, None
        try:
            s1out = await asyncio.wait_for(loop.run_in_executor(self._stage1_pool, run, cs, deadline), wait_s)
        except TimeoutError:
            status, reason = "timeout", "timeout"
        except Exception:
            log.exception("%s step %s failed on request %s", step.kind, s1.detector_id, request_id)
            status, reason = "error", "exception"
        lat1 = (time.perf_counter() - t1) * 1000
        cleared: set[int] = set()
        if s1out is not None and s1out.scores:
            if step.band is not None:
                lo = step.band.clear_below
                for i, sc in s1out.scores.items():
                    lang_ok = step.languages is None or s1out.langs.get(i, UNKNOWN) in step.languages
                    if sc < lo and lang_ok:
                        cleared.add(i)
            res = self._result(s1, request_id, sha, status="ok", reason=None, score=max(s1out.scores.values()),
                               latency_ms=lat1, n_tokens=s1out.n_tokens, n_chunks=len(s1out.scores),
                               truncated=len(s1out.scores) < n, stage="stage1", role=role)
            return _Stage1Pass(res, cleared, s1out.scores, s1out)
        if status == "ok":
            # Ran but scored nothing: the soft deadline passed before the first chunk (queued behind an abandoned
            # job), or no chunk had text.
            deadline_hit = s1out is not None and s1out.deadline_hit
            status, reason = ("timeout", "timeout") if deadline_hit else ("error", "exception")
        res = self._result(s1, request_id, sha, status=status, reason=reason, score=None, latency_ms=lat1,
                           n_tokens=0, n_chunks=n, truncated=True, stage="stage1", role=role)
        return _Stage1Pass(res, cleared, {}, s1out)

    async def _member_scores(self, m: _Member, cs: ChunkSet, idx: list[int]) -> dict[int, float]:
        det = m.spec.detector
        rows = [[det.cls_id, *cs.chunks[i].ids, det.sep_id] for i in idx]
        futures = m.batcher.submit(rows)
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

    async def _stage2_scores(self, cs: ChunkSet, idx: list[int]) -> dict[int, float]:
        """The first decide member's scores (kept under its legacy name: scripts instrument it)."""
        return await self._member_scores(self.members[0], cs, idx)

    # ------------------------------------------------------------------ result builders

    def _result(self, spec: DetectorSpec, request_id: str, sha: str, *, status: str, reason: str | None,
                score: float | None, latency_ms: float, n_tokens: int, n_chunks: int, truncated: bool,
                stage: str, role: Literal["active", "observe", "shadow"] = "active") -> DetectorResult:
        return DetectorResult(
            request_id=request_id, detector_id=spec.detector_id, detector_version=spec.detector_version,
            status=status, score=score, threshold=spec.threshold,
            predicted_injection=None if score is None else bool(score >= spec.threshold),
            latency_ms=latency_ms, input_sha256=sha, input_tokens=n_tokens, n_chunks=max(1, n_chunks),
            truncated=truncated if status == "ok" else True, aggregation="max", stage=stage, role=role,
            failure_reason=None if status == "ok" else reason)

    @staticmethod
    def _mode_decision(mode: str) -> Decision:
        return "flag" if mode == "FAIL_CLOSED" else "allow"

    def _outcome(self, request_id, decision, decided_by, policy, results, stages, coverage, info, *,
                 unscored: bool, escalation: str | None = None) -> ClassifyOutcome:
        b = self.bundle
        return ClassifyOutcome(request_id, decision, decided_by, policy, results, stages, coverage, info,
                               unscored_by_any_step=unscored, escalation_reason=escalation,
                               policy_id=self.policy.policy_id, policy_version=self.policy.policy_version,
                               bundle_sha256=b.bundle_sha256)

    # ------------------------------------------------------------------ main entry

    async def classify(self, text: str, request_id: str | None = None) -> ClassifyOutcome:
        request_id = request_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        if not self._admit(len(text)):
            out = self._overloaded(text, request_id, t0)
        else:
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            try:
                out, cs = await self._classify_active(text, request_id, t0, sha)
            finally:
                # The decision is final: give the admission slot back BEFORE any observe-only work, so observe
                # steps never hold max_inflight / max_inflight_chars against other requests.
                self._release(len(text))
            for step in self.observe_steps:
                await self._observe_guarded(step, out, cs, request_id, sha)
        self._observe(out)
        if self.shadow is not None:
            self.shadow.offer(ShadowItem(request_id=request_id, text=text, enqueued_at=time.perf_counter(), active={
                "decision": out.decision, "decided_by": out.decided_by, "policy_applied": out.policy_applied,
                "coverage_policy_applied": out.coverage_policy_applied, "stages_run": list(out.stages_run)}))
        return out

    def _overloaded(self, text: str, request_id: str, t0: float) -> ClassifyOutcome:
        """Load shedding before any work: every active step is overloaded and FALLBACK is never attempted."""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        lat = (time.perf_counter() - t0) * 1000
        results, stages = [], []
        info: dict[str, Any] = {"path": "overloaded", "stage1_mode": self.stage1_mode}

        def failed(spec, stage, role="active"):
            return self._result(spec, request_id, sha, status="error", reason="overloaded", score=None,
                                latency_ms=lat, n_tokens=0, n_chunks=1, truncated=True, stage=stage, role=role)

        if self.gate is not None:
            results.append(failed(self.gate.spec, "stage1"))
            stages.append("stage1")
        for m in self.members:
            results.append(failed(m.spec, m.stage))
            stages.append(m.stage)
        for i, step in enumerate(self.observe_steps):
            results.append(failed(step.spec, "stage1", role="observe"))
            self._record_observe(info, i, {"status": "error", "outcome": "not_scored"})
        if self.gate is not None and self.gate.failure.mode in _MODE_POLICY:
            mode, by = self.gate.failure.mode, self.gate.spec.detector_id
        elif self.gate is not None:  # FALLBACK: not attempted on overloaded, the terminal decides
            mode = self.gate_fallback_terminal
            by = self.gate.failure.fallback.detector.detector_id
        else:
            votes = [(m.fallback_terminal if m.failure.mode == "FALLBACK" else m.failure.mode, m) for m in self.members]
            decision, policy, by = self._aggregate_failures(votes)
            return self._outcome(request_id, decision, by, policy, results, stages, None, info, unscored=True)
        return self._outcome(request_id, self._mode_decision(mode), by, _MODE_POLICY[mode], results, stages, None,
                             info, unscored=True)

    def _aggregate_failures(self, votes: list[tuple[str, _Member]]) -> tuple[Decision, str, str]:
        """Every member failed: (decision, policy_applied, decided_by) from their failure votes."""
        flags = [m for mode, m in votes if mode == "FAIL_CLOSED"]
        if self.decide_mode == "single" or self.aggregate == "any_flag":
            flag = bool(flags)
        else:
            flag = len(flags) == len(votes)
        decisive = next(m for mode, m in votes if (mode == "FAIL_CLOSED") == flag)
        return ("flag" if flag else "allow"), ("fail_closed" if flag else "fail_open"), decisive.spec.detector_id

    @staticmethod
    def _record_observe(info: dict, i: int, obs: dict) -> None:
        info.setdefault("observe", []).append(obs)
        if i == 0:
            info["stage1_observe"] = obs  # legacy key (metrics, tests): the first observe step

    async def _observe_guarded(self, step: _TextStep, out: ClassifyOutcome, cs: ChunkSet | None, request_id: str,
                               sha: str) -> None:
        """An observe step after the decision, outside admission control, under its own bounded slot count. When
        every slot is busy the observation is dropped and counted: observe work never queues, so it cannot pile up
        behind a slow detector or add up across concurrent requests."""
        i = self.observe_steps.index(step)
        if step.slots >= step.concurrency:
            out.results.append(self._result(step.spec, request_id, sha, status="error", reason="overloaded",
                                            score=None, latency_ms=0.0, n_tokens=0, n_chunks=1, truncated=True,
                                            stage="stage1", role="observe"))
            self._record_observe(out.info, i, {"status": "error", "outcome": "dropped"})
            return
        step.slots += 1
        try:
            await self._observe_step(step, i, out, cs, request_id, sha)
        except Exception:  # defensive: nothing observe-only may fail the request
            log.exception("observe step %s failed on request %s", step.spec.detector_id, request_id)
            self._record_observe(out.info, i, {"status": "error", "outcome": "not_scored"})
        finally:
            step.slots -= 1

    async def _observe_step(self, step: _TextStep, i: int, out: ClassifyOutcome, cs: ChunkSet | None,
                            request_id: str, sha: str) -> None:
        path = out.info.get("path")
        obs: dict[str, Any]
        if cs is None or out.info.get("decide_overloaded"):
            # No chunk plan (prepare timed out or failed), or the decide stage shed load: not started.
            reason = "overloaded" if out.info.get("decide_overloaded") else "timeout" if path == "stage2_timeout" \
                else "exception"
            res = self._result(step.spec, request_id, sha, status="timeout" if reason == "timeout" else "error",
                               reason=reason, score=None, latency_ms=0.0, n_tokens=0, n_chunks=1, truncated=True,
                               stage="stage1", role="observe")
            obs = {"status": res.status, "outcome": "not_scored"}
        else:
            p = await self._stage1_pass(step, cs, request_id, sha, step.timeout_s, role="observe")
            res = p.result
            n = len(cs.chunks)
            would_allow = step.band is not None and "allow" in step.band.finalize and len(p.cleared) == n
            obs = {"status": res.status, "outcome": "not_scored" if not p.scores else
                   "would_finalize_allow" if would_allow else "would_escalate",
                   "chunks_scored": len(p.scores), "would_clear": len(p.cleared),
                   "deadline_hit": bool(p.out is not None and p.out.deadline_hit)}
        out.results.append(res)
        self._record_observe(out.info, i, obs)

    @staticmethod
    def _escalation(n: int, cleared: set[int], scores: dict[int, float], band: Band) -> str | None:
        """Why the gate did not clear every chunk (precedence band, language, coverage)."""
        unclear = [i for i in range(n) if i not in cleared]
        if not unclear:
            return None
        if any(i in scores and scores[i] >= band.clear_below for i in unclear):
            return "band"
        if any(i in scores for i in unclear):
            return "language"
        return "coverage"

    async def _classify_active(self, text: str, request_id: str, t0: float, sha: str
                               ) -> tuple[ClassifyOutcome, ChunkSet | None]:
        """The decision path: gate, decide stage, coverage."""
        gate, m0 = self.gate, self.members[0]
        loop = asyncio.get_running_loop()
        deadline = t0 + m0.timeout_s
        info: dict[str, Any] = {"stage1_mode": self.stage1_mode}
        results: list[DetectorResult] = []
        stages: list[str] = []

        def remaining() -> float:
            return max(0.0, deadline - time.perf_counter())

        def fail_outcome(status: str, reason: str) -> ClassifyOutcome:
            """No chunk plan: every decide member is reported failed; no fallback can run without chunks."""
            for m in self.members:
                results.append(self._result(m.spec, request_id, sha, status=status, reason=reason, score=None,
                                            latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=0, n_chunks=1,
                                            truncated=True, stage=m.stage))
                stages.append(m.stage)
            info["path"] = f"stage2_{reason}"
            votes = [(self.gate_fallback_terminal if escalated_by_failure else
                      m.fallback_terminal if m.failure.mode == "FALLBACK" else m.failure.mode, m)
                     for m in self.members]
            decision, policy, by = self._aggregate_failures(votes)
            return self._outcome(request_id, decision, by, policy, results, stages, None, info, unscored=True,
                                 escalation=escalation)

        escalated_by_failure = False
        escalation: str | None = None
        # -- prepare
        try:
            cs = await asyncio.wait_for(loop.run_in_executor(self._tokenizer_pool, self._prepare, text), remaining())
        except TimeoutError:
            if gate is not None:
                results.append(self._result(gate.spec, request_id, sha, status="timeout", reason="timeout",
                                            score=None, latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=0,
                                            n_chunks=1, truncated=True, stage="stage1"))
                stages.append("stage1")
            return fail_outcome("timeout", "timeout"), None
        except Exception:
            log.exception("prepare failed on request %s", request_id)
            return fail_outcome("error", "exception"), None
        n = len(cs.chunks)
        info["n_chunks"] = n
        info["chunk_kinds"] = [ch.kind for ch in cs.chunks]

        # -- gate
        cleared: set[int] = set()
        s1_scores: dict[int, float] = {}
        if gate is not None:
            p = await self._stage1_pass(gate, cs, request_id, sha, min(gate.timeout_s, remaining()), role="active")
            cleared, s1_scores = p.cleared, p.scores
            if p.out is not None and p.scores:
                info["stage1_langs"] = [p.out.langs.get(i) for i in range(n)]
            if p.out is not None and p.out.deadline_hit:
                info["stage1_deadline_hit"] = True
            results.append(p.result)
            stages.append("stage1")
            info["stage1_cleared"] = len(cleared)
            band = gate.band
            if p.result.status != "ok":
                mode = gate.failure.mode
                if mode in _MODE_POLICY:
                    info["path"] = f"stage1_{p.result.failure_reason}_{_MODE_POLICY[mode]}"
                    return self._outcome(request_id, self._mode_decision(mode), gate.spec.detector_id,
                                         _MODE_POLICY[mode], results, stages, None, info, unscored=True), cs
                if p.result.failure_reason == "overloaded":  # FALLBACK is never attempted on overloaded
                    info["path"] = "stage1_overloaded"
                    term = self.gate_fallback_terminal
                    return self._outcome(request_id, self._mode_decision(term),
                                         gate.failure.fallback.detector.detector_id, _MODE_POLICY[term], results,
                                         stages, None, info, unscored=True), cs
                escalated_by_failure, escalation = True, "gate_failed"
                deadline = t0 + self.gate_fallback_timeout_s
            else:
                if "allow" in band.finalize and len(cleared) == n:
                    info["path"] = "stage1_allow"
                    return self._outcome(request_id, "allow", gate.spec.detector_id, None, results, stages, None,
                                         info, unscored=False), cs
                if "flag" in band.finalize and s1_scores and max(s1_scores.values()) >= band.flag_at_or_above:
                    info["path"] = "stage1_flag"
                    unscored = any(i not in s1_scores for i in range(n))
                    return self._outcome(request_id, "flag", gate.spec.detector_id, None, results, stages, None,
                                         info, unscored=unscored), cs
                escalation = self._escalation(n, cleared, s1_scores, band)

        # -- decide
        order = stage2_order(n, cleared, s1_scores)
        chosen = order[: m0.max_chunks]
        info["stage2_candidates"] = len(order)
        if self.decide_mode == "parallel":
            return await self._decide_parallel(cs, order, chosen, cleared, request_id, sha, t0, results, stages,
                                               info, escalated_by_failure, escalation), cs
        try:
            s2_scores = await asyncio.wait_for(self._stage2_scores(cs, chosen), remaining())
            spec = m0.spec
        except (TimeoutError, Overloaded, Exception) as exc:
            status, reason = (("timeout", "timeout") if isinstance(exc, TimeoutError) else
                              ("error", "overloaded") if isinstance(exc, Overloaded) else ("error", "exception"))
            if reason == "exception":
                log.exception("decide %s failed on request %s", m0.spec.detector_id, request_id)
            results.append(self._result(m0.spec, request_id, sha, status=status, reason=reason, score=None,
                                        latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=cs.n_tokens,
                                        n_chunks=len(chosen), truncated=True, stage=m0.stage))
            stages.append(m0.stage)
            info["path"] = f"stage2_{reason}"
            info["decide_overloaded"] = reason == "overloaded"
            if escalated_by_failure:
                mode = self.gate_fallback_terminal
            elif m0.failure.mode == "FALLBACK" and reason != "overloaded":
                fb_scores = await self._run_fallback(m0, cs, chosen, request_id, sha, results, stages, info)
                if fb_scores is None:
                    mode = m0.fallback_terminal
                else:
                    return self._finish(cs, order, cleared, fb_scores, m0.fallback, request_id, sha, t0, results,
                                        stages, info, escalation, allow_deep_scan=False), cs
            else:
                mode = m0.fallback_terminal if m0.failure.mode == "FALLBACK" else m0.failure.mode
            return self._outcome(request_id, self._mode_decision(mode), m0.spec.detector_id, _MODE_POLICY[mode],
                                 results, stages, None, info, unscored=True, escalation=escalation), cs
        return await self._finish_async(cs, order, cleared, s2_scores, m0, request_id, sha, t0, results, stages,
                                        info, escalation), cs

    async def _run_fallback(self, m: _Member, cs: ChunkSet, chosen: list[int], request_id: str, sha: str,
                            results: list, stages: list, info: dict) -> dict[int, float] | None:
        """Depth-1 fallback on the same chunks under its own budget; None if it failed (its result is recorded)."""
        fb = m.fallback
        t = time.perf_counter()
        try:
            scores = await asyncio.wait_for(self._member_scores(fb, cs, chosen), fb.timeout_s)
            info["fallback"] = {"detector": fb.spec.detector_id, "status": "ok"}
            return scores
        except (TimeoutError, Overloaded, Exception) as exc:
            status, reason = (("timeout", "timeout") if isinstance(exc, TimeoutError) else
                              ("error", "overloaded") if isinstance(exc, Overloaded) else ("error", "exception"))
            results.append(self._result(fb.spec, request_id, sha, status=status, reason=reason, score=None,
                                        latency_ms=(time.perf_counter() - t) * 1000, n_tokens=cs.n_tokens,
                                        n_chunks=len(chosen), truncated=True, stage=fb.stage))
            stages.append(fb.stage)
            info["fallback"] = {"detector": fb.spec.detector_id, "status": status, "reason": reason}
            info["path"] = f"fallback_{reason}"
            return None

    def _finish(self, cs, order, cleared, scores, m: _Member, request_id, sha, t0, results, stages, info,
                escalation, *, allow_deep_scan: bool) -> ClassifyOutcome:
        """Decision from one detector's chunk scores, without deep scan (used after a fallback)."""
        positive = any(v >= m.spec.threshold for v in scores.values())
        uncovered = [i for i in order if i not in scores and i not in cleared]
        coverage = None
        results.append(self._result(m.spec, request_id, sha, status="ok", reason=None, score=max(scores.values()),
                                    latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=cs.n_tokens,
                                    n_chunks=len(scores), truncated=not cs.covered(scores.keys()), stage=m.stage))
        stages.append(m.stage)
        info["stage2_scored"] = len(scores)
        info["unscored_chunks"] = 0 if positive else len(uncovered)
        unscored = bool(uncovered)
        if positive:
            info["path"] = "fallback_flag"
            return self._outcome(request_id, "flag", m.spec.detector_id, None, results, stages, None, info,
                                 unscored=unscored, escalation=escalation)
        if not uncovered:
            info["path"] = "fallback_allow"
            return self._outcome(request_id, "allow", m.spec.detector_id, None, results, stages, None, info,
                                 unscored=unscored, escalation=escalation)
        cov = self.coverage
        policy = cov.deep_scan.fallback if cov.mode == "deep_scan" else cov.mode  # no deep scan with a fallback
        info["path"] = f"coverage_{policy}"
        tag = f"coverage_policy:{cov.mode}" + (f"->{policy}" if cov.mode == "deep_scan" else "")
        coverage = policy
        return self._outcome(request_id, "allow" if policy == "allow" else "flag", tag, None, results, stages,
                             coverage, info, unscored=unscored, escalation=escalation)

    async def _finish_async(self, cs, order, cleared, s2_scores, m0: _Member, request_id, sha, t0, results, stages,
                            info, escalation) -> ClassifyOutcome:
        """Single decide member succeeded: positive chunk, coverage policy and deep scan (legacy code path)."""
        cov = self.coverage
        s2 = m0.spec
        coverage: Literal["flag", "allow", "quarantine", "deep_scan"] | None = None
        decided_by = s2.detector_id
        positive = any(v >= s2.threshold for v in s2_scores.values())
        # Content neither cleared by the gate nor scored by the decide member.
        uncovered = [i for i in order if i not in s2_scores and i not in cleared]
        if not positive and uncovered and cov.mode == "deep_scan":
            ds = cov.deep_scan
            extra = uncovered[: ds.max_chunks]
            try:
                more = await asyncio.wait_for(self._stage2_scores(cs, extra), ds.timeout_ms / 1000)
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
                                    n_chunks=len(s2_scores), truncated=stage2_truncated, stage=m0.stage))
        stages.append(m0.stage)
        info["stage2_scored"] = len(s2_scores)
        info["unscored_chunks"] = 0 if positive else len(uncovered)
        unscored = bool(uncovered)
        if positive:
            info["path"] = "stage2_flag"
            return self._outcome(request_id, "flag", decided_by, None, results, stages, coverage, info,
                                 unscored=unscored, escalation=escalation)
        if not uncovered:
            info["path"] = "stage2_allow"
            return self._outcome(request_id, "allow", decided_by, None, results, stages, coverage, info,
                                 unscored=unscored, escalation=escalation)
        policy = cov.deep_scan.fallback if cov.mode == "deep_scan" else cov.mode
        info["path"] = f"coverage_{policy}"
        tag = f"coverage_policy:{cov.mode}" + (f"->{policy}" if cov.mode == "deep_scan" else "")
        decision: Decision = "allow" if policy == "allow" else "flag"
        return self._outcome(request_id, decision, tag, None, results, stages, policy, info, unscored=unscored,
                             escalation=escalation)

    async def _decide_parallel(self, cs, order, chosen, cleared, request_id, sha, t0, results, stages, info,
                               escalated_by_failure, escalation) -> ClassifyOutcome:
        """Shape B: both members score the same chunks concurrently; decision-level aggregation of their votes."""

        async def run(m: _Member):
            t = time.perf_counter()
            try:
                return m, await asyncio.wait_for(self._member_scores(m, cs, chosen),
                                                 max(0.0, t0 + m.timeout_s - time.perf_counter())), None, t
            except (TimeoutError, Overloaded, Exception) as exc:
                reason = ("timeout" if isinstance(exc, TimeoutError) else
                          "overloaded" if isinstance(exc, Overloaded) else "exception")
                if reason == "exception":
                    log.exception("decide %s failed on request %s", m.spec.detector_id, request_id)
                return m, None, reason, t

        outs = await asyncio.gather(*(run(m) for m in self.members))
        votes: list[tuple[bool, bool, _Member, str | None]] = []  # (flag, model_vote, decided-by member, policy)
        scored: set[int] = set()  # chunk indices scored by any member (or its fallback): coverage reads this
        info["members"] = []
        for m, scores, reason, t in outs:
            if scores is not None:
                flag = any(v >= m.spec.threshold for v in scores.values())
                results.append(self._result(m.spec, request_id, sha, status="ok", reason=None,
                                            score=max(scores.values()), latency_ms=(time.perf_counter() - t0) * 1000,
                                            n_tokens=cs.n_tokens, n_chunks=len(scores),
                                            truncated=not cs.covered(scores.keys()), stage=m.stage))
                stages.append(m.stage)
                votes.append((flag, True, m, None))
                scored |= set(scores)
                info["members"].append({"detector": m.spec.detector_id, "status": "ok", "flag": flag})
                continue
            status = "timeout" if reason == "timeout" else "error"
            results.append(self._result(m.spec, request_id, sha, status=status, reason=reason, score=None,
                                        latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=cs.n_tokens,
                                        n_chunks=len(chosen), truncated=True, stage=m.stage))
            stages.append(m.stage)
            info["members"].append({"detector": m.spec.detector_id, "status": status, "reason": reason})
            if reason == "overloaded":
                info["decide_overloaded"] = True
            if escalated_by_failure:
                mode = self.gate_fallback_terminal
            elif m.failure.mode == "FALLBACK" and reason != "overloaded":
                fb_scores = await self._run_fallback(m, cs, chosen, request_id, sha, results, stages, info)
                if fb_scores is not None:
                    fb = m.fallback
                    flag = any(v >= fb.spec.threshold for v in fb_scores.values())
                    results.append(self._result(fb.spec, request_id, sha, status="ok", reason=None,
                                                score=max(fb_scores.values()),
                                                latency_ms=(time.perf_counter() - t0) * 1000, n_tokens=cs.n_tokens,
                                                n_chunks=len(fb_scores), truncated=not cs.covered(fb_scores.keys()),
                                                stage=fb.stage))
                    stages.append(fb.stage)
                    votes.append((flag, True, fb, None))
                    scored |= set(fb_scores)
                    continue
                mode = m.fallback_terminal
            else:
                mode = m.fallback_terminal if m.failure.mode == "FALLBACK" else m.failure.mode
            votes.append((mode == "FAIL_CLOSED", False, m, _MODE_POLICY[mode]))
        flag = any(v[0] for v in votes) if self.aggregate == "any_flag" else all(v[0] for v in votes)
        decision: Decision = "flag" if flag else "allow"
        # Model-backed iff an ok vote alone yields the decision (any_flag: an ok flag; all_flag: an ok allow), or
        # every vote is a model vote.
        decisive_model = [v for v in votes if v[1] and v[0] == flag and (
            (self.aggregate == "any_flag") == flag or all(x[1] for x in votes))]
        model_backed = bool(decisive_model) or all(v[1] for v in votes)
        # Coverage is about content no step scored: a chunk one member scored is covered even if the other failed.
        uncovered = [i for i in order if i not in scored and i not in cleared]
        unscored = bool(uncovered)
        # The coverage rule applies to an aggregated allow whenever some member (or its fallback) scored chunks and
        # content remains that no step scored, whichever member's vote was a failure-policy vote (review r1 M2): a
        # failed member scores no chunk the other member did not, so its failure cannot shrink the unscored content
        # and must not turn a coverage flag into an allow. Only when no member scored anything does the failure
        # policy alone decide, as for a single decide member (FAIL_OPEN = the owner's choice when no detector
        # answered).
        coverage_applies = not flag and bool(uncovered) and bool(scored)
        if not model_backed and not coverage_applies:
            policy_vote = next(v for v in votes if not v[1] and v[0] == flag)
            info["path"] = f"parallel_{policy_vote[3]}"
            return self._outcome(request_id, decision, policy_vote[2].spec.detector_id, policy_vote[3], results,
                                 stages, None, info, unscored=unscored, escalation=escalation)
        by = "+".join(m.spec.detector_id for m in self.members) + f":{self.aggregate}"
        if flag:
            info["path"] = "parallel_flag"
            return self._outcome(request_id, "flag", by, None, results, stages, None, info, unscored=unscored,
                                 escalation=escalation)
        if not uncovered:
            info["path"] = "parallel_allow"
            return self._outcome(request_id, "allow", by, None, results, stages, None, info, unscored=unscored,
                                 escalation=escalation)
        # Coverage decides; a failed member stays visible in its result's status and info["members"].
        policy = self.coverage.mode  # deep_scan is refused for parallel stages by the policy validator
        info["path"] = f"coverage_{policy}"
        return self._outcome(request_id, "allow" if policy == "allow" else "flag", f"coverage_policy:{policy}", None,
                             results, stages, policy, info, unscored=unscored, escalation=escalation)

    # ------------------------------------------------------------------ metrics

    def _observe(self, out: ClassifyOutcome) -> None:
        m = self.metrics
        m.requests.labels(out.decision, out.policy_applied or "none").inc()
        m.paths.labels(out.info.get("path", "unknown")).inc()
        if out.coverage_policy_applied is not None:
            m.coverage.labels(out.coverage_policy_applied).inc()
        if out.unscored_by_any_step:
            m.unscored.labels(out.decision).inc()
        if "n_chunks" in out.info:
            m.request_chunks.observe(out.info["n_chunks"])
        for obs in out.info.get("observe", []):
            m.stage1_observed.labels(obs["outcome"]).inc()
        if out.info.get("stage1_deadline_hit"):
            m.stage1_deadline_hits.labels(self.gate.spec.detector_id, "route").inc()
        for step, obs in zip(self.observe_steps, out.info.get("observe", [])):
            if obs.get("deadline_hit"):
                m.stage1_deadline_hits.labels(step.spec.detector_id, "observe").inc()
        for r in out.results:
            m.detector_latency.labels(r.detector_id, r.status, r.role).observe(r.latency_ms / 1000)
            if r.status != "ok":
                m.detector_failures.labels(r.detector_id, r.status, r.failure_reason or "none", r.role).inc()
            else:
                m.chunks.labels(r.stage, r.role).observe(r.n_chunks)
            if r.truncated:
                m.truncated.labels(r.detector_id, r.role).inc()
