"""DetectorResult v1.1 — the only shape any detector (stage1, stage2, shadow) may return.

Owned by the project owner. Agents must not edit this file; file a request in docs/contract_requests.md.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.1"


class DetectorResult(BaseModel):
    schema_version: Literal["1.1"] = SCHEMA_VERSION
    request_id: str
    detector_id: str
    detector_version: str
    status: Literal["ok", "timeout", "error", "skipped"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)  # higher = more likely injection; None unless ok
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_injection: bool | None = None
    latency_ms: float = Field(ge=0.0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # never store raw text here
    input_tokens: int = Field(ge=0)
    n_chunks: int = Field(default=1, ge=1)
    truncated: bool = False  # MUST be True whenever any input content was not scored
    aggregation: Literal["max", "mean", "none"] = "max"
    stage: Literal["stage1", "stage2", "single"] = "single"
    role: Literal["active", "shadow"] = "active"

    @model_validator(mode="after")
    def _status_consistency(self) -> "DetectorResult":
        if self.status == "ok":
            if self.score is None or self.predicted_injection is None:
                raise ValueError("status=ok requires score and predicted_injection")
            if self.predicted_injection != (self.score >= self.threshold):
                raise ValueError("predicted_injection must equal score >= threshold")
        else:
            if self.score is not None or self.predicted_injection is not None:
                raise ValueError("score and predicted_injection must be None unless status=ok")
        return self
