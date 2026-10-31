"""Load and score a Stage 1 ONNX artifact (verifies sha256 and the manifest contract first)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from contracts import ArtifactManifest
from training.common import sha256_file
from training.stage1_text import MAX_TOKENS, PREPROCESSING_VERSION, STRIDE, chunk_text


class Stage1Onnx:
    def __init__(self, artifact_dir: Path, verify: bool = True):
        artifact_dir = Path(artifact_dir)
        self.manifest = ArtifactManifest.model_validate_json((artifact_dir / "manifest.json").read_text())
        if self.manifest.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError(f"preprocessing {self.manifest.preprocessing_version} != {PREPROCESSING_VERSION}")
        path = (artifact_dir / self.manifest.artifact_path).resolve()
        if artifact_dir.resolve() not in path.parents:
            raise ValueError("artifact_path escapes the artifact directory")
        if verify and sha256_file(path) != self.manifest.sha256:
            raise ValueError(f"sha256 mismatch for {path}")
        self.injection_index = int(next(k for k, v in self.manifest.label_mapping.items() if v == "injection"))
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.max_tokens = self.manifest.max_input_tokens

    @classmethod
    def from_bytes(cls, model_bytes: bytes, injection_index: int = 1, max_tokens: int = MAX_TOKENS) -> "Stage1Onnx":
        self = cls.__new__(cls)
        self.manifest = None
        self.injection_index = injection_index
        self.session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.max_tokens = max_tokens
        return self

    def proba_raw(self, texts: list[str]) -> np.ndarray:
        """P(injection) for each text as a whole, no chunking."""
        if not texts:
            return np.zeros(0, dtype=np.float64)
        out = self.session.run(["probabilities"], {self.input_name: np.array(texts, dtype=object).reshape(-1, 1)})[0]
        return np.asarray(out[:, self.injection_index], dtype=np.float64)

    def score(self, text: str) -> dict:
        """Chunked, max-aggregated score with timing (preprocessing + inference)."""
        t0 = time.perf_counter()
        chunks, n_tokens = chunk_text(text, self.max_tokens, STRIDE)
        s = float(self.proba_raw(chunks).max())
        return {"score": s, "n_chunks": len(chunks), "input_tokens": n_tokens,
                "latency_ms": (time.perf_counter() - t0) * 1e3}


def load_evaluation(artifact_dir: Path) -> dict:
    return json.loads((Path(artifact_dir) / "evaluation.json").read_text())
