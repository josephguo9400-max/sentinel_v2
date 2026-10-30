"""Stage 1 preprocessing "tfidf-v1": the single definition shared by training, ONNX export and scoring.

Why it looks like this (measured, see training/DATA.md "ONNX parity"):
- onnxruntime's StringNormalizer needs an OS locale; this machine has only C / C.UTF-8 / POSIX, and
  C.UTF-8 lowercases some characters differently from Python (e.g. U+0130). With locale "C" it
  lowercases ASCII A-Z only. So the model lowercases ASCII only, on both sides:
  sklearn sees ascii_lower(text); the ONNX graph contains StringNormalizer(LOWER, locale="C").
- Python re and RE2 (onnxruntime Tokenizer) disagree on \\w, so each side gets an explicit pattern for
  the same character class: letters, numbers and underscore, runs of 2 or more.

Long inputs: max_input_tokens counts TOKEN_RE matches. Longer texts are scored as overlapping windows of
MAX_TOKENS tokens (stride STRIDE) cut at token boundaries on the raw text, aggregated by max.
"""
from __future__ import annotations

import re

PREPROCESSING_VERSION = "tfidf-v1"
TOKEN_PATTERN_PY = r"[^\W]{2,}"  # Python re: \w is Unicode letters/digits/underscore
TOKEN_PATTERN_RE2 = r"[\p{L}\p{N}_]{2,}"  # same class for RE2 inside onnxruntime
ONNX_LOCALE = "C"
MAX_TOKENS = 512
STRIDE = 384

TOKEN_RE = re.compile(TOKEN_PATTERN_PY)
_ASCII_LOWER = str.maketrans({c: c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})


def ascii_lower(text: str) -> str:
    return text.translate(_ASCII_LOWER)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in TOKEN_RE.finditer(text)]


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, stride: int = STRIDE) -> tuple[list[str], int]:
    """Return (chunks, n_tokens). One chunk if n_tokens <= max_tokens; else windows covering the tail.

    Each window spans from its first token's start to its last token's end, so no token is lost.
    """
    spans = token_spans(text)
    n = len(spans)
    if n <= max_tokens:
        return [text], n
    if not 0 < stride <= max_tokens:
        raise ValueError("stride must be in (0, max_tokens]")
    starts = list(range(0, n - max_tokens, stride)) + [n - max_tokens]
    return [text[spans[s][0] : spans[s + max_tokens - 1][1]] for s in starts], n
