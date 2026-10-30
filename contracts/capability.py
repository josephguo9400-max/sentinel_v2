"""Capability Manifest v2.0 — what a detector can do (docs/design/capability-policy.md §3, owner decision D25).

Immutable, keyed by (detector_id, detector_version): a new threshold or scoring rule is a new detector_version.
Nothing in it names another detector or a pipeline position; how detectors are combined is an ExecutionPolicy
(contracts/policy.py). `capability_sha256(m)` is the sha256 of the canonical JSON of the manifest; policies pin it.

Differences from the design text, all additive (recorded in the design doc §12):
- `ScoringRule.subwindow_stride` is None iff `subwindow_tokens` is 0 (no meaningless stride on a model without
  sub-windows, so one behaviour has one hash).
- `Calibration.calibrated_with` and `Calibration.notes`: the scoring rule the threshold was chosen with, when it
  differs from the served `scoring` (D25.6: recorded, not corrected).
- `tokenizer_file`: which key of `files` is the tokenizer (required for token models), so the loader does not guess.

Owned by the project owner. Agents must not edit this file; file a request in docs/contract_requests.md.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._canonical import canonical_sha256
from ._validators import relative_artifact_path, utc_timestamp
from .manifest import Runtime

CAPABILITY_SCHEMA_VERSION = "2.0"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
# The closed set of preprocessing the gateway implements, and the model kind each one is.
PREPROCESSING_KIND = {"tfidf-v1": "text", "deberta-v3-window-v1": "token"}


def iso639_list(v: list[str] | None, what: str) -> list[str] | None:
    if v is None:
        return None
    if not v or len(set(v)) != len(v) or not all(re.fullmatch(r"[a-z]{2,3}", c) for c in v):
        raise ValueError(f"{what} must be a non-empty list of distinct lowercase ISO 639 codes")
    return v


class InputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text", "token"]  # text: the ONNX graph takes strings; token: token ids
    preprocessing: Literal["tfidf-v1", "deberta-v3-window-v1"]
    tokenizer: str = Field(pattern=r"^(hf|regex):.+$")  # units of max_input_tokens / DetectorResult.input_tokens
    max_input_tokens: int = Field(gt=0)
    # ISO 639 codes the detector was validated on: a statement of evidence, not a routing rule (policies route).
    validated_languages: list[str] | None = None

    @field_validator("validated_languages")
    @classmethod
    def _langs(cls, v):
        return iso639_list(v, "validated_languages")

    @model_validator(mode="after")
    def _kind(self) -> "InputContract":
        if PREPROCESSING_KIND[self.preprocessing] != self.kind:
            raise ValueError(f"[input_kind] preprocessing {self.preprocessing} is a {PREPROCESSING_KIND[self.preprocessing]}"
                             f" model, not {self.kind}")
        return self


class ScoringRule(BaseModel):
    """How the model is applied to one chunk. Part of the calibration: the threshold is only valid with it."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregation: Literal["max"] = "max"  # over windows of a chunk and over chunks
    subwindow_tokens: int = Field(ge=0)  # 0 = off. TF-IDF anti-dilution sub-windows, in regex tokens
    subwindow_stride: int | None = Field(default=None, ge=1)  # None iff subwindow_tokens == 0
    score_segment_pieces: bool  # a text model also scores each packed segment piece of a chunk on its own

    @model_validator(mode="after")
    def _stride(self) -> "ScoringRule":
        if (self.subwindow_tokens == 0) != (self.subwindow_stride is None):
            raise ValueError("[scoring_stride] subwindow_stride is set iff subwindow_tokens > 0")
        return self


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)  # flag iff score >= threshold (DetectorResult rule)
    chosen_on: str  # dataset version, e.g. "splits-v2:val:d84853ecc84cc87e"
    rule: str  # e.g. "maximize F1 on val (pooled)"
    evidence_path: str | None = None  # relative to artifacts/, e.g. candidates/stage1-tfidf-v2.1/evaluation.json
    # The scoring rule the threshold was chosen with, when it is not `scoring` (D25.6). None = chosen with `scoring`.
    calibrated_with: ScoringRule | None = None
    notes: str | None = None

    @field_validator("evidence_path")
    @classmethod
    def _evidence(cls, v):
        return None if v is None else relative_artifact_path(v)


class OperationalLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recommended_timeout_ms: int = Field(gt=0)  # the researcher's budget; the policy sets the real deadline
    max_batch: int = Field(ge=1)
    latency_evidence_path: str | None = None  # repository-relative, e.g. serving/bench_results/stage1_latency.json


class CapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = CAPABILITY_SCHEMA_VERSION
    detector_id: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    artifact_format: Literal["onnx"]
    # Every file the detector loads, tokenizer included: path relative to artifacts/ -> sha256 of its bytes.
    files: dict[str, Sha256]
    model_file: str  # a key of `files`
    tokenizer_file: str | None = None  # a key of `files`; required iff input.kind == "token"
    runtime: Runtime
    label_mapping: dict[Literal["0", "1"], Literal["benign", "injection"]]
    input: InputContract
    scoring: ScoringRule
    calibration: Calibration
    limits: OperationalLimits
    training_data_version: str | None = None
    created_at: str  # RFC 3339 UTC

    @field_validator("files")
    @classmethod
    def _files(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("[files] at least the model file is required")
        for path in v:
            relative_artifact_path(path)
        return v

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, v: str) -> str:
        return utc_timestamp(v)

    @field_validator("detector_id", "detector_version")
    @classmethod
    def _id_component(cls, v: str) -> str:
        # Both name a directory under artifacts/capabilities/.
        relative_artifact_path(v)
        if "/" in v:
            raise ValueError(f"must be a single path component: {v!r}")
        return v

    @model_validator(mode="after")
    def _checks(self) -> "CapabilityManifest":
        if self.model_file not in self.files or not self.model_file.endswith(".onnx"):
            raise ValueError("[model_file] model_file must be a .onnx key of files")
        if len(self.label_mapping) != 2 or set(self.label_mapping.values()) != {"benign", "injection"}:
            raise ValueError("[label_mapping] label_mapping must map indices 0 and 1 to benign and injection")
        if self.input.kind == "token":
            if self.tokenizer_file is None or self.tokenizer_file not in self.files:
                raise ValueError("[tokenizer_file] token models name their tokenizer file, a key of files")
            if self.runtime.tokenizers is None:
                raise ValueError("[runtime] token models must declare runtime.tokenizers")
            if self.scoring.subwindow_tokens or self.scoring.score_segment_pieces:
                raise ValueError("[scoring_kind] sub-windows and segment-piece scoring are text-model rules")
        elif self.tokenizer_file is not None:
            raise ValueError("[tokenizer_file] text models carry their tokenizer inside the ONNX graph")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.detector_id, self.detector_version

    @property
    def injection_index(self) -> int:
        return int(next(k for k, v in self.label_mapping.items() if v == "injection"))


def capability_sha256(m: CapabilityManifest) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace) of the manifest. Policies pin it."""
    return canonical_sha256(m.model_dump(mode="json"))
