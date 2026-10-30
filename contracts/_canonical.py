"""Canonical JSON for content hashes (capability_sha256, policy and bundle hashes).

Sorted keys, no whitespace, UTF-8, no NaN/Infinity. Sets must already be sorted lists (the models serialize their
set fields sorted), so the same content always gives the same bytes in every process.
"""
import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()
