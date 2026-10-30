"""Execution Policy v1.0 — when and how detectors run and combine (docs/design/capability-policy.md §4, D25).

A fixed structure, not a graph: an optional gate step, one decide stage (one detector, or two in parallel), any number
of observe steps, an optional shadow, a coverage rule and an optional canary (designed, not implemented: the engine
rejects a policy that sets it, D25.4).

Two layers of validation:
- structural rules, checked when the model is built (pydantic); each message starts with a bracketed code;
- `validate_policy(policy, capabilities)`: rules that need the pinned CapabilityManifests (kinds, languages,
  tokenizers, sha256 pins). Raises PolicyError(code, detail).
`lint_policy` returns warnings that are allowed but suspicious (a FAIL_OPEN gate).

Differences from the design text (all recorded in the design doc §12):
- ObserveStep also carries `max_chunks`, `deadline_margin_ms` and an optional what-if `band`: the observe-only Stage 1
  of D23 scores at most STAGE1_MAX_CHUNKS chunks under an anytime soft deadline and reports would-allow /
  would-escalate outcomes from the gray-zone low; without these fields the conversion would not be faithful. The
  band of an observe step is never read by the decision.
- Band: a band that does not finalize "allow" must have clear_below 0 (it clears nothing), and `finalize` is not
  empty.
- Coverage deep_scan requires a single decide stage (engine 1.0.0).
- A decide member's FALLBACK must not target a decide member.
- Parallel members have equal max_chunks (they score the same chunk selection).

Owned by the project owner. Agents must not edit this file; file a request in docs/contract_requests.md.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from ._canonical import canonical_sha256
from ._validators import relative_artifact_path
from .capability import iso639_list

if TYPE_CHECKING:  # pragma: no cover
    from .capability import CapabilityManifest

POLICY_SCHEMA_VERSION = "1.0"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FailureMode = Literal["FAIL_OPEN", "FAIL_CLOSED", "FALLBACK", "ASYNC_ONLY"]
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class PolicyError(ValueError):
    """A policy that cannot be served with the given capabilities / engine. `code` names the violated rule."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail


def _component(v: str) -> str:
    relative_artifact_path(v)
    if "/" in v:
        raise ValueError(f"must be a single path component: {v!r}")
    return v


class DetectorRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector_id: str
    detector_version: str
    capability_sha256: Sha256  # pins the manifest, so an edited manifest cannot slip in

    @property
    def key(self) -> tuple[str, str]:
        return self.detector_id, self.detector_version

    def label(self) -> str:
        return f"{self.detector_id}@{self.detector_version}"


class FallbackStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    terminal: Literal["FAIL_OPEN", "FAIL_CLOSED"]  # depth 1: a fallback cannot fall back again


class Failure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: FailureMode
    fallback: FallbackStep | None = None  # required iff mode == FALLBACK

    @model_validator(mode="after")
    def _fallback(self) -> "Failure":
        if (self.mode == "FALLBACK") != (self.fallback is not None):
            raise ValueError("[failure_fallback] fallback is set iff mode is FALLBACK")
        return self


class Band(BaseModel):
    """Per-chunk routing on the gate detector's score. Independent of the gate detector's threshold."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    clear_below: float = Field(ge=0.0, le=1.0)  # chunk cleared iff score < clear_below (strict)
    flag_at_or_above: float | None = Field(default=None, ge=0.0, le=1.0)  # finalize flag iff any chunk score >= it
    finalize: set[Literal["allow", "flag"]]
    # Chunks whose language is outside it are never cleared; a subset of the capability's validated_languages.
    scope_languages: list[str] | None = None

    @field_validator("scope_languages")
    @classmethod
    def _langs(cls, v):
        return iso639_list(v, "scope_languages")

    @field_serializer("finalize")
    def _sorted(self, v: set[str]) -> list[str]:
        return sorted(v)

    @model_validator(mode="after")
    def _rules(self) -> "Band":
        if not self.finalize:
            raise ValueError("[band_finalize] finalize names at least one of allow, flag")
        if ("flag" in self.finalize) != (self.flag_at_or_above is not None):
            raise ValueError("[band_flag] flag_at_or_above is set iff finalize includes flag")
        if self.flag_at_or_above is not None and not self.clear_below < self.flag_at_or_above:
            raise ValueError("[band_flag] need clear_below < flag_at_or_above")
        if "allow" not in self.finalize and self.clear_below != 0.0:
            raise ValueError("[band_allow] a band that does not finalize allow clears nothing: clear_below must be 0")
        return self


class GateStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef
    band: Band
    timeout_ms: int = Field(gt=0)
    deadline_margin_ms: int = Field(ge=0)  # anytime soft deadline: no new chunk after (timeout - margin)
    max_chunks: int = Field(ge=1)
    on_failure: Failure  # FALLBACK -> must target the decide detector (= "escalate")


class DecideMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef
    timeout_ms: int = Field(gt=0)  # the request budget from admission, as today's Stage 2 budget
    max_chunks: int = Field(ge=1)  # first, last, then uncleared by gate score
    on_failure: Failure  # FAIL_OPEN | FAIL_CLOSED | FALLBACK


class DecideStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["single", "parallel"]
    members: list[DecideMember]  # 1 for single, exactly 2 for parallel
    aggregate: Literal["any_flag", "all_flag"] | None = None  # parallel only; decision-level, never score averaging

    @model_validator(mode="after")
    def _shape(self) -> "DecideStage":
        if self.mode == "single" and (len(self.members) != 1 or self.aggregate is not None):
            raise ValueError("[decide_shape] single needs exactly 1 member and no aggregate")
        if self.mode == "parallel" and (len(self.members) != 2 or self.aggregate is None):
            raise ValueError("[decide_shape] parallel needs exactly 2 members and an aggregate")
        if len({m.detector.key for m in self.members}) != len(self.members):
            raise ValueError("[decide_distinct] decide members must be distinct detectors")
        if len({m.max_chunks for m in self.members}) != 1:
            raise ValueError("[parallel_selection] parallel members score the same chunk selection: equal max_chunks")
        return self


class ObserveStep(BaseModel):  # today's stage1_mode=observe, generalized
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    deadline_margin_ms: int = Field(ge=0)
    max_chunks: int = Field(ge=1)
    concurrency: int = Field(ge=1)  # busy -> dropped (error/overloaded), never queued
    on_failure: Failure  # must be ASYNC_ONLY
    # What-if band for metrics only (would_finalize_allow / would_escalate). Never read by the decision.
    band: Band | None = None


class ShadowStep(BaseModel):  # the async mirror (serving/shadow.py)
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    queue: int = Field(ge=1)
    max_chunks: int = Field(ge=1)
    intra_op_threads: int = Field(ge=0)
    on_failure: Failure  # must be ASYNC_ONLY


class DeepScan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorRef  # must be the decide member
    max_chunks: int = Field(ge=1)
    timeout_ms: int = Field(gt=0)
    fallback: Literal["flag", "quarantine", "allow"]


class Coverage(BaseModel):  # content scored by no step
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["flag", "quarantine", "allow", "deep_scan"]
    deep_scan: DeepScan | None = None  # required iff mode == deep_scan

    @model_validator(mode="after")
    def _deep(self) -> "Coverage":
        if (self.mode == "deep_scan") != (self.deep_scan is not None):
            raise ValueError("[coverage_deep_scan] deep_scan is set iff mode is deep_scan")
        return self


class Chunking(BaseModel):  # one chunk plan shared by every step
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_tokenizer: DetectorRef  # the decide detector whose tokenizer defines chunk units
    stride: int = Field(ge=1)
    structure_aware: bool
    # Recorded for completeness; engine 1.0.0 does not read it (GatewayConfig.pack_overlap_tokens was read nowhere).
    pack_overlap_tokens: int = Field(ge=0)


class LanguageDetector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    impl: Literal["langdetect==1.0.9"]
    seed: int
    min_letters: int = Field(ge=0)
    min_prob: float = Field(ge=0.0, le=1.0)
    max_chars: int = Field(ge=1)
    # Applied after the built-in rule that reduces a region-tagged code to its language ("zh-cn" -> "zh").
    tag_map: dict[str, str] = Field(default_factory=dict)


class PolicyRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str
    policy_version: str
    policy_sha256: Sha256


class Canary(BaseModel):  # designed now, implemented later (design §6); the engine rejects it (D25.4)
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: PolicyRef
    share_bp: int = Field(ge=0, le=10_000)
    salt: str


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = POLICY_SCHEMA_VERSION
    policy_id: str
    policy_version: str
    entrypoint: Literal["classify"]
    engine_version: str  # semantic version of the policy interpreter it was validated on
    chunking: Chunking
    language_detector: LanguageDetector | None = None  # required iff a band declares scope_languages
    gate: GateStep | None = None
    decide: DecideStage
    observe: list[ObserveStep] = Field(default_factory=list)
    shadow: ShadowStep | None = None
    coverage: Coverage
    canary: Canary | None = None
    rationale: str

    @field_validator("policy_id", "policy_version")
    @classmethod
    def _ids(cls, v: str) -> str:
        return _component(v)

    @field_validator("engine_version")
    @classmethod
    def _engine(cls, v: str) -> str:
        if not _SEMVER.match(v):
            raise ValueError(f"[engine_version] must be MAJOR.MINOR.PATCH: {v!r}")
        return v

    @model_validator(mode="after")
    def _rules(self) -> "ExecutionPolicy":
        members = {m.detector.key for m in self.decide.members}
        for m in self.decide.members:
            if m.on_failure.mode == "ASYNC_ONLY":
                raise ValueError("[async_only] ASYNC_ONLY is not allowed on a decide member")
            if m.on_failure.fallback is not None and m.on_failure.fallback.detector.key in members:
                raise ValueError("[decide_fallback_target] a decide member's FALLBACK must name a detector outside "
                                 "the decide stage")
        if self.gate is not None:
            f = self.gate.on_failure
            if f.mode == "ASYNC_ONLY":
                raise ValueError("[async_only] ASYNC_ONLY is not allowed on the gate step")
            if f.fallback is not None and f.fallback.detector.key not in members:
                raise ValueError("[gate_fallback_target] the gate's FALLBACK must target a decide member")
            if self.gate.detector.key in members:
                raise ValueError("[gate_is_decider] the gate detector cannot also be a decide member")
        for step in [*self.observe, *([self.shadow] if self.shadow else [])]:
            if step.on_failure.mode != "ASYNC_ONLY":
                raise ValueError("[async_only] observe and shadow steps must use ASYNC_ONLY")
        bands = [b for b in [self.gate.band if self.gate else None, *(o.band for o in self.observe)] if b]
        scoped = any(b.scope_languages for b in bands)
        if scoped and self.language_detector is None:
            raise ValueError("[language_detector_required] a band declares scope_languages: set language_detector")
        if not scoped and self.language_detector is not None:
            raise ValueError("[language_detector_unused] language_detector is set but no band declares "
                             "scope_languages")
        if self.chunking.plan_tokenizer.key not in members:
            raise ValueError("[plan_tokenizer] chunking.plan_tokenizer must be a decide member")
        ds = self.coverage.deep_scan
        if ds is not None:
            if self.decide.mode != "single":
                raise ValueError("[deep_scan_parallel] coverage deep_scan needs a single decide stage (engine 1.0.0)")
            if ds.detector.key not in members:
                raise ValueError("[deep_scan_target] coverage deep_scan must use the decide member")
        refs = list(iter_refs(self))
        pins: dict[tuple[str, str], str] = {}
        for r in refs:
            if pins.setdefault(r.key, r.capability_sha256) != r.capability_sha256:
                raise ValueError(f"[ref_inconsistent] {r.label()} is pinned to two different capability_sha256")
        return self


def iter_refs(p: ExecutionPolicy):
    """Every DetectorRef in the policy, in a fixed order."""
    yield p.chunking.plan_tokenizer
    if p.gate is not None:
        yield p.gate.detector
        if p.gate.on_failure.fallback is not None:
            yield p.gate.on_failure.fallback.detector
    for m in p.decide.members:
        yield m.detector
        if m.on_failure.fallback is not None:
            yield m.on_failure.fallback.detector
    for o in p.observe:
        yield o.detector
    if p.shadow is not None:
        yield p.shadow.detector
    if p.coverage.deep_scan is not None:
        yield p.coverage.deep_scan.detector


def policy_sha256(p: ExecutionPolicy) -> str:
    return canonical_sha256(p.model_dump(mode="json"))


def validate_policy(p: ExecutionPolicy, capabilities: dict[tuple[str, str], "CapabilityManifest"], *,
                    allow_canary: bool = False) -> None:
    """Rules that need the pinned capabilities. Raises PolicyError with a named code."""
    from .capability import capability_sha256

    for r in iter_refs(p):
        cap = capabilities.get(r.key)
        if cap is None:
            raise PolicyError("ref_unresolved", f"{r.label()} is not a loaded capability")
        got = capability_sha256(cap)
        if got != r.capability_sha256:
            raise PolicyError("capability_sha256_mismatch", f"{r.label()}: policy pins {r.capability_sha256}, "
                                                             f"manifest is {got}")

    def kind(ref: DetectorRef) -> str:
        return capabilities[ref.key].input.kind

    if p.canary is not None and not allow_canary:
        raise PolicyError("canary_unsupported", "canary is designed but not implemented in this engine (D25.4)")
    for m in p.decide.members:
        if kind(m.detector) != "token":
            raise PolicyError("kind_mismatch", f"decide member {m.detector.label()} must be a token model")
        fb = m.on_failure.fallback
        if fb is not None and kind(fb.detector) != "token":
            raise PolicyError("kind_mismatch", f"fallback {fb.detector.label()} must be a token model")
    plan = capabilities[p.chunking.plan_tokenizer.key]
    token_refs = [m.detector for m in p.decide.members] + [m.on_failure.fallback.detector for m in p.decide.members
                                                           if m.on_failure.fallback is not None]
    for ref in token_refs:
        c = capabilities[ref.key]
        if (c.input.tokenizer, c.files[c.tokenizer_file], c.input.max_input_tokens) != (
                plan.input.tokenizer, plan.files[plan.tokenizer_file], plan.input.max_input_tokens):
            raise PolicyError("tokenizer_mismatch", f"{ref.label()} must share the chunk plan's tokenizer (same name, "
                                                    f"tokenizer file sha256 and max_input_tokens) to score its rows")
    if p.gate is not None:
        g = p.gate
        cap = capabilities[g.detector.key]
        if cap.input.kind != "text":
            raise PolicyError("kind_mismatch", f"gate {g.detector.label()} must be a text model (engine 1.0.0)")
        _check_band(g.band, cap, "gate")
    for o in p.observe:
        cap = capabilities[o.detector.key]
        if cap.input.kind != "text":
            raise PolicyError("kind_mismatch", f"observe step {o.detector.label()} must be a text model (engine "
                                               f"1.0.0; mirror a token model with the shadow step)")
        if o.band is not None:
            _check_band(o.band, cap, "observe")


def _check_band(band: Band, cap: "CapabilityManifest", where: str) -> None:
    langs = cap.input.validated_languages
    if band.scope_languages is not None and langs is not None and not set(band.scope_languages) <= set(langs):
        raise PolicyError("scope_languages", f"{where} band scope {band.scope_languages} is not within the "
                                             f"capability's validated_languages {langs}")
    if langs is None and band.scope_languages is None and band.clear_below > 0.0:
        raise PolicyError("undeclared_languages", f"{where} detector {cap.detector_id}@{cap.detector_version} "
                                                  f"declares no validated languages; a band that clears chunks must "
                                                  f"state scope_languages")


def lint_policy(p: ExecutionPolicy) -> list[str]:
    """Allowed but suspicious settings (design §4, 'Failure semantics')."""
    out = []
    if p.gate is not None and p.gate.on_failure.mode == "FAIL_OPEN":
        out.append("gate on_failure FAIL_OPEN turns a gate hiccup into an unscored allow")
    for m in p.decide.members:
        if m.on_failure.mode == "FAIL_OPEN":
            out.append(f"decide member {m.detector.label()} fails open")
    if p.coverage.mode == "allow":
        out.append("coverage allow: unscored content is allowed")
    return out
