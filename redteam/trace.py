"""Trace writer: every event is validated as contracts.TraceEvent and appended to one JSONL file per run."""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from contracts import TraceEvent

DEFAULT_RUNS_DIR = Path("runs")
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def to_jsonable(value: object) -> object:
    """Round-trip through JSON so traces never hold objects that cannot be persisted."""
    return json.loads(json.dumps(value, default=repr, ensure_ascii=False))


class RunTrace:
    """Collects the events of one run (many attempts) and persists them as runs/<run_id>.jsonl.

    out_dir=None keeps events in memory only (used by unit tests that do not need files).
    """

    def __init__(self, run_id: str | None = None, out_dir: str | Path | None = DEFAULT_RUNS_DIR,
                 clock: Callable[[], str] = utc_now):
        self.run_id = run_id or new_run_id()
        if not _RUN_ID.match(self.run_id):
            raise ValueError(f"run_id must match {_RUN_ID.pattern}: {self.run_id!r}")
        self.clock = clock
        self.events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self.path: Path | None = None
        if out_dir is not None:
            d = Path(out_dir)
            created = not d.exists()
            d.mkdir(parents=True, exist_ok=True)
            # Traces are run output, never source: a directory this writer creates is kept out of git
            # without editing the root .gitignore (owned by the project owner). Existing directories
            # (e.g. a future fixtures/recorded_runs/) are left alone.
            ignore = d / ".gitignore"
            if created:
                ignore.write_text("# created by redteam.trace: run output is not committed\n*\n", encoding="utf-8")
            self.path = d / f"{self.run_id}.jsonl"
            if self.path.exists():
                raise FileExistsError(f"trace file already exists: {self.path}")
            self.path.touch()

    def emit(self, **fields) -> TraceEvent:
        with self._lock:
            if fields.get("tool_arguments") is not None:
                fields["tool_arguments"] = to_jsonable(fields["tool_arguments"])
            if fields.get("detail") is not None:
                fields["detail"] = to_jsonable(fields["detail"])
            event = TraceEvent(run_id=self.run_id, timestamp=self.clock(), **fields)
            self.events.append(event)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(event.model_dump_json() + "\n")
            return event

    def attempt_events(self, attempt_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.attempt_id == attempt_id]


def read_jsonl(path: str | Path) -> list[TraceEvent]:
    """Load and validate every event of a trace file. Raises on the first invalid line."""
    events = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(TraceEvent.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - re-raise with the line number
            raise ValueError(f"{path}:{i}: invalid TraceEvent: {exc}") from exc
    return events


def group_by_attempt(events: Iterable[TraceEvent]) -> dict[str, list[TraceEvent]]:
    out: dict[str, list[TraceEvent]] = {}
    for e in events:
        out.setdefault(e.attempt_id, []).append(e)
    return out
