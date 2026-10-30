"""/classify wire contract v1.3 — request and response shapes shared by the gateway and its callers.

v1.3 (owner decision D25, docs/design/capability-policy.md §7) adds the optional fields `policy_id`, `policy_version`,
`bundle_sha256`, `unscored_by_any_step` and `escalation_reason`. Every v1.2 response is a valid v1.3 response (all
new fields default to None). `decision` stays allow | flag: there is no "quarantine" decision in v1.3 (D25.1); a
quarantine is `decision="flag"` with `coverage_policy_applied="quarantine"`.

Owned by the project owner. Agents must not edit this file; file a request in docs/contract_requests.md.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .detector import DetectorResult


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    request_id: str | None = None


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    decision: Literal["allow", "flag"]
    decided_by: str  # detector_id that decided, or the policy name when a policy decided
    # A detector failure policy decided (timeout, error or overload) instead of a model score.
    policy_applied: Literal["fail_open", "fail_closed"] | None = None
    # v1.2: some input content was not scored and the coverage policy decided the outcome.
    coverage_policy_applied: Literal["flag", "allow", "quarantine", "deep_scan"] | None = None
    # v1.2: which detectors ran, in order (e.g. ["stage1"] or ["stage1", "stage2"]). Observe and shadow steps are
    # never listed.
    stages_run: list[Literal["stage1", "stage2", "single"]] = Field(default_factory=list)
    results: list[DetectorResult]
    # v1.3: the bundle that decided (ExecutionPolicy id and version, and sha256 over policy + pinned capabilities +
    # engine version). None from a pre-1.3 gateway.
    policy_id: str | None = None
    policy_version: str | None = None
    bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # v1.3: True iff some input content was neither cleared by the gate step nor scored by a decide detector, also
    # when the decision is a flag backed by a positive score (coverage_policy_applied is None then). None = not
    # reported (pre-1.3 gateway).
    unscored_by_any_step: bool | None = None
    # v1.3: why a request with a gate step reached the decide stage. None when there is no gate step or the gate
    # finalized. band: a chunk scored at or above the band's clear_below; language: a chunk below it was outside the
    # band's scope_languages; coverage: a chunk the gate did not score (chunk budget, soft deadline, no text);
    # gate_failed: the gate step failed and its FALLBACK escalated. Precedence when several apply: gate_failed, band,
    # language, coverage.
    escalation_reason: Literal["band", "language", "coverage", "gate_failed"] | None = None

    @model_validator(mode="after")
    def _consistency(self) -> "ClassifyResponse":
        if not self.results:
            raise ValueError("a response carries at least one DetectorResult")
        if any(r.request_id != self.request_id for r in self.results):
            raise ValueError("every result must carry the response's request_id")
        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("policy_id and policy_version are set together")
        return self
