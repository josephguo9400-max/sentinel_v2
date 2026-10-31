"""Milestone-wide spend ledger (review m5).

The first version created a fresh ``SpendTracker`` per run, so the "$15 for the whole milestone" cap
was really a per-invocation cap and nothing was kept between runs. This ledger is a single committed
file that records every run's spend; a new run reads the prior total from it and refuses to start a
call that would push the *cumulative* spend over the cap.

The file holds only dollar amounts, token counts, model ids, a label and a timestamp - never any key,
header or request content - so it is safe to commit. It is append-only; each ``record`` adds one entry
and rewrites the file with the updated total.

    ledger = SpendLedger.load()           # reports/adaptive-attacker/spend_ledger.json
    ledger.prior_total_usd                # spend from earlier runs
    run = ledger.begin(label=..., note=...)   # entry written before the first call
    spend.on_record = run.update              # entry rewritten after EVERY billable call
    ...
    run.finish(spend)                         # marks the entry complete

``begin``/``update`` exist because the first redesign wrote the ledger only when a run finished, so a
crashed or aborted run left no entry (review r2 m3). An entry whose ``status`` is still
``in_progress`` therefore records a run that stopped early; its spend still counts toward the cap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LEDGER_PATH = REPO_ROOT / "reports" / "adaptive-attacker" / "spend_ledger.json"
_BANNED_KEYS = ("headers", "authorization", "x-api-key", "api_key", "apikey", "key", "token")


@dataclass
class SpendLedger:
    path: Path = DEFAULT_LEDGER_PATH
    entries: list[dict] = field(default_factory=list)
    # Free-text accounting notes kept across rewrites (e.g. spend that has no token counts on record).
    accounting_notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "SpendLedger":
        p = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(path=p, entries=list(data.get("entries", [])),
                       accounting_notes=list(data.get("accounting_notes", [])))
        return cls(path=p, entries=[])

    @property
    def prior_total_usd(self) -> float:
        return round(sum(e.get("spent_usd", 0.0) for e in self.entries), 6)

    def _safe(self, entry: dict) -> dict:
        """Keep only spend-describing fields; refuse anything key/header-shaped."""
        out = {}
        for k, v in entry.items():
            if isinstance(v, dict):
                self._safe(v)
            kl = k.lower()
            if any(b == kl for b in _BANNED_KEYS):
                raise ValueError(f"refusing to write ledger field {k!r}")
            out[k] = v
        return out

    def record(self, *, label: str, spent_usd: float, input_tokens: int, output_tokens: int,
               n_calls: int, by_model: dict, note: str = "") -> dict:
        entry = self._safe({
            "label": label,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spent_usd": round(spent_usd, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "n_calls": n_calls,
            "by_model": _by_model(by_model),
            "note": note,
        })
        self.entries.append(entry)
        self.flush()
        return entry

    def begin(self, *, label: str, note: str = "") -> "LedgerRun":
        """Open an entry for a run before its first billable call and write it to disk."""
        entry = self._safe({
            "label": label,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "in_progress",
            "spent_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "n_calls": 0,
            "by_model": {}, "note": note,
        })
        self.entries.append(entry)
        self.flush()
        return LedgerRun(self, len(self.entries) - 1)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ledger": "adaptive-attacker milestone API spend",
            "note": ("Cumulative API spend across all adaptive-attacker runs. The dollar cap reads "
                     "prior_total_usd from here so the cap is milestone-wide, not per invocation "
                     "(review m5). No key, header or request content is ever stored."),
            "total_usd": self.prior_total_usd,
            "n_entries": len(self.entries),
            "accounting_notes": self.accounting_notes,
            "entries": self.entries,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)  # atomic: a crash mid-write never truncates the ledger


def _by_model(by_model: dict) -> dict:
    return {m: {"usd": round(v.get("usd", 0.0), 6), "n_calls": v.get("n_calls", 0),
                "input_tokens": v.get("input_tokens", 0), "output_tokens": v.get("output_tokens", 0)}
            for m, v in by_model.items()}


@dataclass
class LedgerRun:
    """Handle on one in-progress ledger entry. ``update`` is meant as ``SpendTracker.on_record``."""

    ledger: SpendLedger
    index: int

    def update(self, spend) -> None:
        e = self.ledger.entries[self.index]
        e.update({
            "spent_usd": round(spend.spent_usd, 6),
            "input_tokens": spend.input_tokens,
            "output_tokens": spend.output_tokens,
            "n_calls": spend.n_calls,
            "by_model": _by_model(spend.by_model),
        })
        self.ledger.flush()

    def finish(self, spend) -> None:
        self.update(spend)
        self.ledger.entries[self.index]["status"] = "complete"
        self.ledger.flush()
