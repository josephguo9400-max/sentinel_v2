"""Shared validators for contracts. Owned by the project owner."""
import re
from datetime import datetime, timezone
from typing import Any

# RFC 3339 in UTC only: 2026-10-29T00:00:00Z or with fractional seconds, or the +00:00 offset.
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|\+00:00)$")
# PEP 440-style release with optional pre/post/dev/local segments, e.g. 1.29.0, 3.11.0rc1, 2.1.0+cpu.
_VERSION = re.compile(r"^\d+(\.\d+){0,3}((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[A-Za-z0-9.]+)?$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f:\u200b-\u200f\u2028-\u202e\ufeff]")


def utc_timestamp(value: str) -> str:
    """Accept RFC 3339 UTC timestamps and normalize them to YYYY-MM-DDTHH:MM:SS[.ffffff]Z.

    Normalized strings sort chronologically as plain text.
    """
    if not isinstance(value, str) or not _RFC3339_UTC.match(value):
        raise ValueError(f"timestamp must be RFC 3339 UTC like 2026-10-29T00:00:00Z: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%f" if parsed.microsecond else "%Y-%m-%dT%H:%M:%S") + "Z"


def version_string(value: str) -> str:
    if not _VERSION.match(value):
        raise ValueError(f"not a version like 1.29.0 or 3.11.0rc1: {value!r}")
    return value


def coerce_bool(value: Any) -> Any:
    """Accept Python and numpy booleans; leave everything else for strict validation to reject."""
    if type(value).__name__ in ("bool_", "bool") and type(value).__module__ == "numpy":
        return bool(value)
    return value


def relative_artifact_path(value: str, suffix: str | None = None) -> str:
    """A POSIX-style path relative to the manifest directory that cannot escape it.

    The loader must additionally resolve() the joined path and confirm it stays inside the
    artifact directory, because symlinks cannot be checked here.
    """
    if not value or value.strip() != value:
        raise ValueError("path must be non-empty with no surrounding whitespace")
    if _UNSAFE_CHARS.search(value):
        raise ValueError(f"path contains control, colon or invisible characters: {value!r}")
    if "\\" in value or _DRIVE.match(value) or value.startswith(("/", "~")):
        raise ValueError(f"path must be relative POSIX style: {value!r}")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"path must not contain empty, '.' or '..' components: {value!r}")
    if suffix:
        if not value.endswith(suffix):
            raise ValueError(f"path must end with {suffix}: {value!r}")
        if parts[-1][: -len(suffix)].strip(".") == "":
            raise ValueError(f"file name must have a stem before {suffix}: {value!r}")
    return value
