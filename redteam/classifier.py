"""Classifier interface used by mode B and C to decide which email bodies are quarantined.

A classifier's output can only ever *remove* content from the defender's context (quarantine).
It is never passed to the authorization firewall and cannot authorize any tool call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from contracts import DetectorResult

GatewayStatus = Literal["ok", "timeout", "unreachable", "bad_response"]


@dataclass(frozen=True)
class ClassifierVerdict:
    decision: Literal["allow", "flag"]
    decided_by: str | None = None
    policy_applied: str | None = None
    results: tuple[DetectorResult, ...] = ()
    gateway_status: GatewayStatus = "ok"
    fallback: bool = False  # True when the decision came from on_unavailable, not from the gateway
    error: str | None = None
    latency_ms: float | None = None

    @property
    def detector_version(self) -> str:
        """Value for TraceEvent.detector_version: which detectors produced this verdict."""
        if self.results:
            return ",".join(f"{r.detector_id}@{r.detector_version}" for r in self.results)
        if self.decided_by:
            return self.decided_by
        return "unavailable"

    @property
    def flag_cause(self) -> str | None:
        """Why decision == "flag", or None for allow.

        "classifier_flag"         a detector scored the text and the gateway flagged it;
        "gateway_failure_policy"  the gateway answered, but no detector produced a positive score and the
                                  gateway applied its failure policy (fail_closed after a timeout/error);
        "fallback"                the client could not use the gateway and applied on_unavailable="flag".
        Only "classifier_flag" may be reported as the classifier stopping an attack.
        """
        if self.decision != "flag":
            return None
        if self.fallback:
            return "fallback"
        positive = any(r.status == "ok" and r.predicted_injection for r in self.results)
        if self.policy_applied == "fail_closed" and not positive:
            return "gateway_failure_policy"
        return "classifier_flag"

    @property
    def truncated(self) -> bool:
        """True when any detector result says part of the input was not scored."""
        return any(r.truncated for r in self.results)

    def as_detail(self) -> dict:
        return {
            "decision": self.decision,
            "decided_by": self.decided_by,
            "policy_applied": self.policy_applied,
            "gateway_status": self.gateway_status,
            "fallback": self.fallback,
            "flag_cause": self.flag_cause,
            "truncated": self.truncated,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "results": [r.model_dump(mode="json") for r in self.results],
        }


class ClassifierUnavailable(RuntimeError):
    """Raised only when a client is configured with on_unavailable='error'."""


@runtime_checkable
class Classifier(Protocol):
    def classify(self, text: str, request_id: str) -> ClassifierVerdict: ...


# ---------------------------------------------------------------- fakes for scripted tests

@dataclass
class KeywordClassifier:
    """Deterministic fake: flags text matching any pattern. For tests only; scores are not measurements."""

    patterns: tuple[str, ...] = (r"send_email", r"ignore (all |any )?(previous|prior) instructions",
                                 r"\bforward\b.*\bto\b", r"SYSTEM NOTICE")
    detector_id: str = "fake-keyword"
    detector_version: str = "scripted-0"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def classify(self, text: str, request_id: str) -> ClassifierVerdict:
        import hashlib

        self.calls.append((request_id, text))
        hit = any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in self.patterns)
        score = 1.0 if hit else 0.0
        result = DetectorResult(
            request_id=request_id, detector_id=self.detector_id, detector_version=self.detector_version,
            status="ok", score=score, threshold=0.5, predicted_injection=hit, latency_ms=0.0,
            input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), input_tokens=len(text.split()),
        )
        return ClassifierVerdict(decision="flag" if hit else "allow", decided_by=self.detector_id,
                                 policy_applied="scripted_keyword", results=(result,))


@dataclass
class FixedClassifier:
    """Returns the same decision for every input (e.g. allow-all to model a classifier miss)."""

    decision: Literal["allow", "flag"] = "allow"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def classify(self, text: str, request_id: str) -> ClassifierVerdict:
        self.calls.append((request_id, text))
        return ClassifierVerdict(decision=self.decision, decided_by="fixed", policy_applied="scripted_fixed")
