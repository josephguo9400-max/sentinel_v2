"""/classify wire contract v1.2 — frozen request and response shapes shared by the gateway and its callers.

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
    # v1.2: which detectors ran, in order (e.g. ["stage1"] or ["stage1", "stage2"]).
    stages_run: list[Literal["stage1", "stage2", "single"]] = Field(default_factory=list)
    results: list[DetectorResult]

    @model_validator(mode="after")
    def _consistency(self) -> "ClassifyResponse":
        if not self.results:
            raise ValueError("a response carries at least one DetectorResult")
        if any(r.request_id != self.request_id for r in self.results):
            raise ValueError("every result must carry the response's request_id")
        return self
