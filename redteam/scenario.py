"""Scenario fixtures: an invented inbox, the user's request, what the user authorized, and the facts
a correct answer must mention. Loaded from fixtures/scenarios/<split>/<scenario_id>.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .addresses import RecipientError, parse_one

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_ROOT = REPO_ROOT / "fixtures" / "scenarios"
RESERVED_DOMAINS = ("example.com", "example.org", "example.net")


def _reserved(address: str) -> bool:
    domain = address.rsplit("@", 1)[-1]
    if domain.endswith(".invalid") or domain == "invalid":
        return True
    return any(domain == d or domain.endswith("." + d) for d in RESERVED_DOMAINS)


def _strict_reserved(value: str) -> str:
    try:
        addr = parse_one(value).address
    except RecipientError as exc:
        raise ValueError(str(exc)) from exc
    if not _reserved(addr):
        raise ValueError(f"fixture addresses must use .invalid or example.com/org/net: {value!r}")
    return value


class Email(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    sender: str = Field(alias="from")
    to: tuple[str, ...]
    subject: str
    date: str
    body: str

    @field_validator("sender")
    @classmethod
    def _sender(cls, v: str) -> str:
        return _strict_reserved(v)

    @field_validator("to")
    @classmethod
    def _to(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strict_reserved(x) for x in v)


class Doc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    content: str


class SendGrant(BaseModel):
    """One send the user explicitly authorized: the exact recipient set and how many times."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    to: tuple[str, ...] = ()
    cc: tuple[str, ...] = ()
    bcc: tuple[str, ...] = ()
    max_uses: int = Field(default=1, ge=1)
    required: bool = True  # legitimate_task_success requires this send to be executed

    @field_validator("to", "cc", "bcc")
    @classmethod
    def _addrs(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strict_reserved(x) for x in v)

    @model_validator(mode="after")
    def _nonempty(self) -> "SendGrant":
        if not (self.to or self.cc or self.bcc):
            raise ValueError("a send grant needs at least one recipient")
        return self

    def recipient_set(self) -> frozenset[str]:
        return frozenset(parse_one(a).address for a in (*self.to, *self.cc, *self.bcc))


class ExpectedFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    any_of: tuple[str, ...] = Field(min_length=1)  # phrase alternatives; see judge.normalize_text


class ScriptedHints(BaseModel):
    """Inputs for scripted test defenders only. Never shown to a live (LLM) defender."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    search_queries: tuple[str, ...] = ()


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^[a-z0-9_]+$")
    split: Literal["dev", "heldout"]
    description: str
    user_address: str
    user_request: str
    authorized_sends: tuple[SendGrant, ...] = ()
    emails: tuple[Email, ...] = Field(min_length=1)
    attacker_email_id: str
    docs: tuple[Doc, ...] = ()
    expected_facts: tuple[ExpectedFact, ...] = Field(min_length=1)
    scripted_hints: ScriptedHints = ScriptedHints()

    @field_validator("user_address")
    @classmethod
    def _user(cls, v: str) -> str:
        return _strict_reserved(v)

    @model_validator(mode="after")
    def _checks(self) -> "Scenario":
        ids = [e.id for e in self.emails]
        if len(ids) != len(set(ids)):
            raise ValueError("email ids must be unique")
        if self.attacker_email_id not in ids:
            raise ValueError("attacker_email_id must name one of the emails")
        sets = [g.recipient_set() for g in self.authorized_sends]
        if len(sets) != len(set(sets)):
            raise ValueError("authorized_sends must have distinct recipient sets")
        return self

    def email(self, email_id: str) -> Email:
        for e in self.emails:
            if e.id == email_id:
                return e
        raise KeyError(email_id)


def load_scenario(path: str | Path) -> Scenario:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    sc = Scenario.model_validate(data)
    if p.parent.name in ("dev", "heldout") and p.parent.name != sc.split:
        raise ValueError(f"{p}: split field {sc.split!r} does not match directory {p.parent.name!r}")
    if p.stem != sc.scenario_id:
        raise ValueError(f"{p}: file name must equal scenario_id {sc.scenario_id!r}")
    return sc


def scenario_path(split: str, scenario_id: str, root: Path = SCENARIO_ROOT) -> Path:
    return root / split / f"{scenario_id}.json"


def load_split(split: str, root: Path = SCENARIO_ROOT) -> list[Scenario]:
    return [load_scenario(p) for p in sorted((root / split).glob("*.json"))]


def find_scenario(scenario_id: str, root: Path = SCENARIO_ROOT) -> Scenario:
    for split in ("dev", "heldout"):
        p = scenario_path(split, scenario_id, root)
        if p.exists():
            return load_scenario(p)
    raise FileNotFoundError(f"no scenario {scenario_id!r} under {root}")
