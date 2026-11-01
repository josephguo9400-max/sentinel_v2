"""Artifact manifest v1.2 — what a researcher publishes to get a detector loaded (into shadow, never straight to active)."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validators import utc_timestamp, relative_artifact_path, version_string


class Runtime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str
    onnxruntime: str
    tokenizers: str | None = None  # required by transformer detectors
    sklearn: str | None = None  # training-time version for sklearn-exported detectors
    skl2onnx: str | None = None

    @field_validator("python", "onnxruntime", "tokenizers", "sklearn", "skl2onnx")
    @classmethod
    def _versions(cls, v: str | None) -> str | None:
        return None if v is None else version_string(v)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Readers accept 1.1 manifests (no tokenizer or input_languages); 1.2 requires `tokenizer`.
    schema_version: Literal["1.1", "1.2"] = "1.2"
    detector_id: str
    detector_version: str
    artifact_format: Literal["onnx"]  # pickle/joblib are not accepted
    artifact_path: str  # relative to the manifest's directory
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: Runtime
    preprocessing_version: str
    # Keys are the model's output indices "0" and "1"; values say which index means what.
    label_mapping: dict[Literal["0", "1"], Literal["benign", "injection"]]
    threshold: float = Field(ge=0.0, le=1.0)
    max_input_tokens: int = Field(gt=0)
    failure_policy: Literal["fail_closed", "fail_open"]
    timeout_ms: int = Field(gt=0)
    cascade_role: Literal["stage1", "stage2", "single"]
    # stage1 only: below low -> final benign, above high -> final injection, inside -> escalate to stage2.
    gray_zone: tuple[float, float] | None = None
    # v1.2: the tokenizer whose units max_input_tokens and DetectorResult.input_tokens use,
    # e.g. "hf:deberta-v3" or "regex:[\\p{L}\\p{N}_]{2,}".
    tokenizer: str | None = Field(default=None, pattern=r"^(hf|regex):.+$")
    # v1.2: lowercase ISO 639 codes this detector is validated for; inputs outside them must not be
    # finalized by this detector (the router escalates them). None = no restriction declared.
    input_languages: list[str] | None = None
    training_data_version: str | None = None
    evaluation_dataset_version: str | None = None
    created_at: str  # ISO-8601 UTC
    metrics_path: str | None = None

    @field_validator("artifact_path")
    @classmethod
    def _artifact_path(cls, v: str) -> str:
        return relative_artifact_path(v, suffix=".onnx")

    @field_validator("metrics_path")
    @classmethod
    def _metrics_path(cls, v: str | None) -> str | None:
        return None if v is None else relative_artifact_path(v, suffix=".json")

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, v: str) -> str:
        return utc_timestamp(v)

    @model_validator(mode="after")
    def _checks(self) -> "ArtifactManifest":
        if self.schema_version == "1.2" and self.tokenizer is None:
            raise ValueError("schema_version 1.2 manifests must declare tokenizer")
        if self.schema_version == "1.1" and (self.tokenizer is not None or self.input_languages is not None):
            raise ValueError("tokenizer and input_languages require schema_version 1.2")
        if self.input_languages is not None:
            if not self.input_languages or len(set(self.input_languages)) != len(self.input_languages) or not all(
                    re.fullmatch(r"[a-z]{2,3}", code) for code in self.input_languages):
                raise ValueError("input_languages must be a non-empty list of distinct lowercase ISO 639 codes")
        if len(self.label_mapping) != 2 or set(self.label_mapping.values()) != {"benign", "injection"}:
            raise ValueError("label_mapping must map indices 0 and 1 to benign and injection")
        if self.cascade_role == "stage1":
            if self.gray_zone is None:
                raise ValueError("stage1 detectors must declare gray_zone")
            lo, hi = self.gray_zone
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError("gray_zone must satisfy 0 <= low < high <= 1")
            if not lo <= self.threshold <= hi:
                raise ValueError("stage1 threshold must lie inside gray_zone")
        elif self.gray_zone is not None:
            raise ValueError("gray_zone is only valid for stage1")
        return self
