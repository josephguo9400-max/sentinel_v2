"""Detector implementations loaded from manifests, and the protocols the gateway drives them through.

Two model families exist:

- Token models (`Detector` protocol): BERT-style classifiers over token ids. The gateway tokenizes,
  builds windows of `max_len - 2` content ids, wraps each as [CLS] ... [SEP] and calls `infer`.
  `OnnxDebertaDetector` (tokenizer "hf:deberta-v3") is the only real one; tests inject fakes.
- Text models (`OnnxTfidfDetector`, preprocessing "tfidf-v1"): the ONNX graph takes raw strings and does
  its own normalization and tokenization; the gateway only splits long texts into windows of
  `max_input_tokens` regex tokens.

Both expose a synchronous, thread-safe `infer`; the gateway runs it in a bounded worker pool, never on the
event loop.
"""
import hashlib
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
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
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokenize_with_offsets(det, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    """(content ids, character offsets of each id or None if the detector cannot report them)."""
    fn = getattr(det, "tokenize_with_offsets", None)
    if fn is not None:
        return fn(text)
    return det.tokenize(text), None


class OnnxDebertaDetector:
    """protectai/deberta-v3-base-prompt-injection-v2 (fp32 or int8 ONNX), CPUExecutionProvider.

    `model` / `tokenizer` are the verified bytes (or paths, for scripts). The label index comes from the
    manifest; this class does no hash checking of its own (serving/registry.py does, before it builds one).
    """

    tokenizer_name = "hf:deberta-v3"

    def __init__(self, model: bytes | str | Path, tokenizer: bytes | str | Path, *, detector_id: str,
                 detector_version: str, threshold: float, injection_index: int, max_len: int = 512,
                 intra_op_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._inj = injection_index
        self._tok = (Tokenizer.from_str(tokenizer.decode("utf-8")) if isinstance(tokenizer, bytes)
                     else Tokenizer.from_file(str(tokenizer)))
        self.cls_id = self._tok.token_to_id("[CLS]")
        self.sep_id = self._tok.token_to_id("[SEP]")
        opts = ort.SessionOptions()
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
        self._sess = ort.InferenceSession(model if isinstance(model, bytes) else str(model), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self.detector_id = detector_id
        self.detector_version = detector_version
        self.threshold = threshold
        self.max_len = max_len

    def tokenize(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def tokenize_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids, [tuple(o) for o in enc.offsets]

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


# Preprocessing "tfidf-v1", re-implemented here because serving must not import training/ (Track A never
# depends on Track B). tests/serving/test_registry.py checks parity with training/stage1_text.py.
TFIDF_V1_TOKEN_RE = re.compile(r"[^\W]{2,}")
TFIDF_V1_STRIDE = 384


def tfidf_windows(text: str, max_tokens: int, stride: int = TFIDF_V1_STRIDE) -> tuple[list[str], int]:
    """(windows, n_tokens): one window if n_tokens <= max_tokens, else overlapping windows covering the tail."""
    spans = [m.span() for m in TFIDF_V1_TOKEN_RE.finditer(text)]
    n = len(spans)
    if n <= max_tokens:
        return [text], n
    starts = list(range(0, n - max_tokens, stride)) + [n - max_tokens]
    return [text[spans[s][0]: spans[s + max_tokens - 1][1]] for s in starts], n


class OnnxTfidfDetector:
    """Stage 1 TF-IDF + logistic regression exported to ONNX; input is a [N, 1] string tensor."""

    tokenizer_name = "regex:[\\p{L}\\p{N}_]{2,}"

    def __init__(self, model: bytes | str | Path, *, detector_id: str, detector_version: str, threshold: float,
                 injection_index: int, max_input_tokens: int):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(model if isinstance(model, bytes) else str(model),
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        outputs = [o.name for o in self._sess.get_outputs()]
        if "probabilities" not in outputs:
            raise ValueError(f"tfidf model has no 'probabilities' output: {outputs}")
        self._inj = injection_index
        self.detector_id = detector_id
        self.detector_version = detector_version
        self.threshold = threshold
        self.max_input_tokens = max_input_tokens

    def windows(self, text: str) -> tuple[list[str], int]:
        return tfidf_windows(text, self.max_input_tokens)

    def infer(self, batch: list[str]) -> list[float]:
        x = np.array(batch, dtype=object).reshape(-1, 1)
        probs = self._sess.run(["probabilities"], {self._input: x})[0]
        return [float(p) for p in np.asarray(probs[:, self._inj], dtype=np.float64)]
