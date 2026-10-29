"""Shared validators for contracts. Owned by the project owner."""
import re
from datetime import datetime
from pathlib import PurePosixPath

_VERSION = re.compile(r"^\d+(\.\d+){0,3}$")
_DRIVE = re.compile(r"^[A-Za-z]:")


def iso8601_utc(value: str) -> str:
    """Accept ISO-8601 timestamps with an explicit UTC offset (Z or +00:00)."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp must be UTC: {value!r}")
    return value


def version_string(value: str) -> str:
    if not _VERSION.match(value):
        raise ValueError(f"not a numeric version like 1.29.0: {value!r}")
    return value


def relative_artifact_path(value: str, suffix: str | None = None) -> str:
    """A POSIX-style path relative to the manifest directory that cannot escape it.

    The loader must additionally resolve() the joined path and confirm it stays inside the
    artifact directory, because symlinks cannot be checked here.
    """
    if not value or value.strip() != value:
        raise ValueError("path must be non-empty with no surrounding whitespace")
    if "\\" in value or _DRIVE.match(value) or value.startswith(("/", "~")):
        raise ValueError(f"path must be relative POSIX style: {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or any(p in ("..", ".") for p in value.split("/")) or "" in value.split("/"):
        raise ValueError(f"path must not contain empty, '.' or '..' components: {value!r}")
    if suffix and not value.endswith(suffix):
        raise ValueError(f"path must end with {suffix}: {value!r}")
    return value
