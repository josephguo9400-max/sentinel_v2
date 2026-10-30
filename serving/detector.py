"""Detector interface and the ONNX DeBERTa implementation.

A detector exposes tokenization and a synchronous, thread-safe batch inference call. The gateway
runs `infer` in a bounded worker pool; it is never called on the event loop.
"""
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

# Pinned by scripts/fetch_models.py (revision 90c9989b). Checked at load unless disabled.
DEBERTA_SHA256 = "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c"


class Detector(Protocol):
    detector_id: str
    detector_version: str
    threshold: float
    max_len: int  # model input length including [CLS] and [SEP]
    cls_id: int
    sep_id: int

    def tokenize(self, text: str) -> list[int]:
        """Content token ids, without special tokens."""

    def infer(self, batch: list[list[int]]) -> list[float]:
        """Injection probability for each input row (each row already includes [CLS]/[SEP])."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class OnnxDebertaDetector:
    """protectai/deberta-v3-base-prompt-injection-v2, ONNX fp32, CPUExecutionProvider."""

    def __init__(self, model_dir: str | Path, *, detector_id: str, detector_version: str, threshold: float,
                 max_len: int = 512, intra_op_threads: int = 0, expected_sha256: str | None = DEBERTA_SHA256):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        self.model_sha256 = sha256_file(model_path) if expected_sha256 else None
        if expected_sha256 and self.model_sha256 != expected_sha256:
            raise RuntimeError(f"model.onnx sha256 mismatch: got {self.model_sha256}, expected {expected_sha256}")

        id2label = json.loads((model_dir / "config.json").read_text())["id2label"]
        self._inj = next(int(k) for k, v in id2label.items() if v.upper() == "INJECTION")
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.cls_id = self._tok.token_to_id("[CLS]")
        self.sep_id = self._tok.token_to_id("[SEP]")
        opts = ort.SessionOptions()
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.detector_id = detector_id
        self.detector_version = detector_version
        self.threshold = threshold
        self.max_len = max_len

    def tokenize(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def infer(self, batch: list[list[int]]) -> list[float]:
        width = max(len(x) for x in batch)
        ids = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for r, x in enumerate(batch):
            ids[r, : len(x)], mask[r, : len(x)] = x, 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        logits = self._sess.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0].astype(np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return [float(p) for p in (e / e.sum(axis=1, keepdims=True))[:, self._inj]]
