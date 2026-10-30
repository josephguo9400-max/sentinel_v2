"""Trace event v1.2 — every observable step of a sandbox run. proposed / blocked / executed are never conflated."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._validators import utc_timestamp

EventType = Literal[
    "tool_proposed",
    "tool_blocked",
    "tool_executed",
    "tool_failed",
    "model_response",
    "detector_prediction",
    "detector_timeout",
    "shadow_failed",
    "content_quarantined",
    "attempt_completed",
]
# Detector events written when no detector answered (gateway unreachable) use this detector_version,
# with detail.fallback = true, so the event stays attributable.
UNAVAILABLE_DETECTOR_VERSION = "unavailable"
TOOL_EVENTS = {"tool_proposed", "tool_blocked", "tool_executed", "tool_failed"}


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1", "1.2"] = "1.2"
    event_id: str
    run_id: str
    attempt_id: str
    timestamp: str  # ISO-8601 UTC
    event_type: EventType
    mode: Literal["A", "B", "C"]  # A: no defenses, B: classifier only, C: classifier + authorization firewall
    source: Literal["live", "recorded", "scripted"]
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    policy_reason: str | None = None
    detector_version: str | None = None
    detail: dict[str, Any] | None = None
    seq: int | None = Field(default=None, ge=0)  # v1.2: monotonic order within an attempt
    scenario_id: str | None = None  # v1.2

    @field_validator("timestamp")
    @classmethod
    def _timestamp(cls, v: str) -> str:
        return utc_timestamp(v)

    @model_validator(mode="after")
    def _required_by_type(self) -> "TraceEvent":
        if self.schema_version == "1.1" and (self.seq is not None or self.scenario_id is not None):
            raise ValueError("seq and scenario_id require schema_version 1.2")
        if self.event_type in TOOL_EVENTS and not self.tool_name:
            raise ValueError(f"{self.event_type} requires tool_name")
        if self.event_type in ("tool_blocked", "content_quarantined") and not self.policy_reason:
            raise ValueError(f"{self.event_type} requires policy_reason")
        if self.event_type in ("detector_prediction", "detector_timeout") and not self.detector_version:
            raise ValueError(f"{self.event_type} requires detector_version")
        return self
