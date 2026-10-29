"""Artifact manifest v1.1 — what a researcher publishes to get a detector loaded (into shadow, never straight to active)."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Runtime(BaseModel):
    python: str
    onnxruntime: str
    sklearn: str | None = None


class ArtifactManifest(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    detector_id: str
    detector_version: str
    artifact_format: Literal["onnx"]  # pickle/joblib are not accepted
    artifact_path: str  # relative to the manifest's directory; must stay inside artifacts/
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: Runtime
    preprocessing_version: str
    label_mapping: dict[str, Literal["benign", "injection"]]
    threshold: float = Field(ge=0.0, le=1.0)
    max_input_tokens: int = Field(gt=0)
    failure_policy: Literal["fail_closed", "fail_open"]
    timeout_ms: int = Field(gt=0)
    cascade_role: Literal["stage1", "stage2", "single"]
    gray_zone: tuple[float, float] | None = None  # stage1 only: scores inside escalate to stage2
    training_data_version: str | None = None
    evaluation_dataset_version: str | None = None
    created_at: str  # ISO-8601 UTC
    metrics_path: str | None = None

    @model_validator(mode="after")
    def _checks(self) -> "ArtifactManifest":
        if ".." in self.artifact_path or self.artifact_path.startswith("/"):
            raise ValueError("artifact_path must be relative and stay inside the artifact directory")
        if self.cascade_role == "stage1":
            if self.gray_zone is None:
                raise ValueError("stage1 detectors must declare gray_zone")
            lo, hi = self.gray_zone
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError("gray_zone must satisfy 0 <= low < high <= 1")
        elif self.gray_zone is not None:
            raise ValueError("gray_zone is only valid for stage1")
        if set(self.label_mapping.values()) != {"benign", "injection"}:
            raise ValueError("label_mapping must map to exactly benign and injection")
        return self
