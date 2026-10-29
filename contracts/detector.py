"""DetectorResult v1.1 — the only shape any detector (stage1, stage2, shadow) may return.

Owned by the project owner. Agents must not edit this file; file a request in docs/contract_requests.md.
"""
import math
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, model_validator

from ._validators import coerce_bool

SCHEMA_VERSION = "1.1"
# Producers compute scores in float32 (onnxruntime). Within this distance of the threshold either
# decision is accepted, so float32-vs-float64 rounding never makes a valid result fail validation.
THRESHOLD_TOLERANCE = 1e-6


# Strict bool that also accepts numpy.bool_ (onnxruntime producers); strings and ints are still rejected.
Bool = Annotated[StrictBool, BeforeValidator(coerce_bool)]


class DetectorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1"] = SCHEMA_VERSION
    request_id: str
    detector_id: str
    detector_version: str
    status: Literal["ok", "timeout", "error", "skipped"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)  # higher = more likely injection; None unless ok
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_injection: Bool | None = None
    latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # never store raw text here
    input_tokens: int = Field(ge=0)
    n_chunks: int = Field(default=1, ge=1)
    truncated: Bool = False  # MUST be True whenever any input content was not scored
    aggregation: Literal["max", "mean", "none"] = "max"
    stage: Literal["stage1", "stage2", "single"] = "single"
    role: Literal["active", "shadow"] = "active"

    @model_validator(mode="after")
    def _consistency(self) -> "DetectorResult":
        if self.status == "ok":
            if self.score is None or self.predicted_injection is None:
                raise ValueError("status=ok requires score and predicted_injection")
            margin = self.score - self.threshold
            if abs(margin) > THRESHOLD_TOLERANCE and self.predicted_injection != (margin > 0):
                raise ValueError("predicted_injection must equal score >= threshold")
        else:
            if self.score is not None or self.predicted_injection is not None:
                raise ValueError("score and predicted_injection must be None unless status=ok")
        if self.n_chunks > 1 and self.aggregation == "none":
            raise ValueError("n_chunks > 1 requires an aggregation of max or mean")
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite")
        return self
