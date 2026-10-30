"""Deterministic stand-in detector for fast gateway tests."""
import threading
import time

CLS, SEP = 1, 2
INJECTION_WORD = "INJECT"
INJECTION_ID = 3


class FakeDetector:
    """Whitespace tokenizer; a chunk scores 0.99 if it contains the word INJECT, otherwise a small
    value derived from its content (so different chunks get different, reproducible scores)."""

    detector_id = "fake"
    detector_version = "0.0.1"

    def __init__(self, *, max_len: int = 512, threshold: float = 0.5, delay_s: float = 0.0,
                 fail: Exception | None = None):
        self.max_len = max_len
        self.threshold = threshold
        self.cls_id, self.sep_id = CLS, SEP
        self.delay_s = delay_s
        self.fail = fail
        self.batches: list[int] = []
        self.calls_started = 0
        self.release = threading.Event()  # used by hang tests
        self.hang = False
        self._lock = threading.Lock()

    def tokenize(self, text: str) -> list[int]:
        return [INJECTION_ID if w == INJECTION_WORD else 10 + (sum(map(ord, w)) % 1000) for w in text.split()]

    def infer(self, batch: list[list[int]]) -> list[float]:
        with self._lock:
            self.calls_started += 1
            self.batches.append(len(batch))
        if self.hang:
            self.release.wait(10)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        out = []
        for row in batch:
            assert row[0] == CLS and row[-1] == SEP and len(row) <= self.max_len
            content = row[1:-1]
            out.append(0.99 if INJECTION_ID in content else (sum(content) % 97) / 1000)
        return out


def words(n: int, inject_at: int | None = None) -> str:
    ws = [f"w{i % 50}" for i in range(n)]
    if inject_at is not None:
        ws[inject_at] = INJECTION_WORD
    return " ".join(ws)
