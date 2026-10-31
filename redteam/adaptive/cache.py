"""On-disk cache of Anthropic responses, keyed by a hash of the request *content*.

The cache key is a SHA-256 over the canonical JSON of (model, system, messages, max_tokens,
temperature). Request headers are NEVER part of the key or the stored value, so no API key or auth
material is ever written. A hit returns the stored text + token usage and costs nothing, which makes
reruns of the whole experiment cheap and deterministic.

The cache directory is gitignored (redteam/adaptive/cache/). It is local-only: it is not shared
between viewers and is not read back by anything but this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def request_hash(*, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int,
                 temperature: float) -> str:
    """Stable content hash of a request. Excludes every header and any auth material by construction."""
    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ResponseCache:
    directory: Path = DEFAULT_CACHE_DIR
    _mem: dict[str, dict] | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # A .gitignore with "*" belt-and-braces guarantees nothing here is committed even if the
        # top-level ignore entry is edited away.
        gi = self.directory / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n", encoding="utf-8")
        self._mem = {}

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if key in self._mem:  # type: ignore[operator]
            return self._mem[key]  # type: ignore[index]
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        self._mem[key] = value  # type: ignore[index]
        return value

    def put(self, key: str, value: dict) -> None:
        # Defensive: never persist anything that looks like a header/auth field.
        for banned in ("headers", "authorization", "x-api-key", "api_key", "apikey"):
            if banned in value:
                raise ValueError(f"refusing to cache a value containing {banned!r}")
        self._mem[key] = value  # type: ignore[index]
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(key))
