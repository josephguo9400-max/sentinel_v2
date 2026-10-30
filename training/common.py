"""Shared helpers: paths, text cleaning, dedup keys, MinHash, JSONL I/O, hashing."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"
FROZEN_TEST_SHA = ROOT / "data" / "frozen_test.sha256"
SPLIT_NAMES = ("train", "val", "frozen_test")

SEED = 20261029

# Near-duplicate definition (shared by build_splits and check_leakage):
# character 5-gram shingles of the dedup key, MinHash with 128 permutations, LSH at Jaccard 0.8,
# candidates come from LSH tuned for Jaccard 0.6, and every candidate pair is confirmed with the exact
# Jaccard of the shingle sets (>= 0.8 is a near duplicate).
SHINGLE_K = 5
NUM_PERM = 128
NEAR_DUP_JACCARD = 0.8
LSH_CANDIDATE_JACCARD = 0.6

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def clean_text(text: object) -> str | None:
    """Minimal cleaning applied to stored text: CRLF -> LF, strip. Returns None for empty input.

    Stored text is kept close to raw because the gateway feeds raw text to the model.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return s or None


def dedup_key(text: str) -> str:
    """Aggressive normalization used ONLY for duplicate detection, never as model input."""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def key_hash(text: str) -> str:
    return hashlib.sha256(dedup_key(text).encode("utf-8")).hexdigest()


def shingles(key: str, k: int = SHINGLE_K) -> set[str]:
    if len(key) <= k:
        return {key}
    return {key[i : i + k] for i in range(len(key) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def minhashes(sets: list[set[str]]) -> list[MinHash]:
    """MinHash for each shingle set (fixed datasketch seed=1, so results are deterministic)."""
    encoded = [[s.encode("utf-8") for s in sh] for sh in sets]
    return MinHash.bulk(encoded, num_perm=NUM_PERM, seed=1)


def near_duplicate_pairs(keys: list[str], threshold: float = NEAR_DUP_JACCARD,
                         lsh_threshold: float = LSH_CANDIDATE_JACCARD) -> list[tuple[int, int, float]]:
    """All index pairs (i < j) whose shingle-set Jaccard is >= threshold, found via MinHash LSH.

    LSH candidates are generated at a lower threshold than the decision threshold to reduce false
    negatives near the boundary; every reported pair is verified with the exact Jaccard.
    """
    sets = [shingles(k) for k in keys]
    mhs = minhashes(sets)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=NUM_PERM)
    for i, m in enumerate(mhs):
        lsh.insert(str(i), m)
    pairs = []
    for i, m in enumerate(mhs):
        for c in lsh.query(m):
            j = int(c)
            if j <= i:
                continue
            jac = jaccard(sets[i], sets[j])
            if jac >= threshold:
                pairs.append((i, j, jac))
    pairs.sort()
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_unit(value: str, salt: str = str(SEED)) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from a string (no global RNG state)."""
    d = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
