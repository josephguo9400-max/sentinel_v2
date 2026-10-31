"""The decision oracle the adaptive attacker queries, and a caching classifier shared with the sandbox.

``CachingClassifier`` wraps the real ``GatewayClient`` and memoises verdicts by exact body text, so an
identical body (the attacker candidate the oracle already scored, or a fixed inbox email screened in
every sandbox run) is scored by DeBERTa at most once per run. This both honours the "count each query"
rule (the oracle counts a query only on a genuine, uncached classification) and keeps CPU bounded on
the shared 2-core VM.

The attacker's observation is black-box by default: it receives only the gateway ``decision``
(allow/flag). ``score_feedback=True`` additionally exposes the decider's score (max active chunk
score) as a stronger-attacker ablation (docs/design/adaptive-attacker.md §2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..classifier import Classifier, ClassifierVerdict
from ..gateway_client import GatewayClient


def decider_score(verdict: ClassifierVerdict) -> float | None:
    """Max score over the active (decider) results, or None if no active result carried a score."""
    scores = [r.score for r in verdict.active_results if r.score is not None]
    return max(scores) if scores else None


@dataclass
class CachingClassifier:
    """A Classifier that memoises gateway verdicts by exact text. Also records how many real gateway
    calls were made (``n_gateway_calls``) vs served from cache (``n_cache_hits``)."""

    inner: Classifier
    _cache: dict[str, ClassifierVerdict] = field(default_factory=dict)
    n_gateway_calls: int = 0
    n_cache_hits: int = 0

    def classify(self, text: str, request_id: str) -> ClassifierVerdict:
        if text in self._cache:
            self.n_cache_hits += 1
            return self._cache[text]
        verdict = self.inner.classify(text, request_id)
        self._cache[text] = verdict
        self.n_gateway_calls += 1
        return verdict


@dataclass
class DecisionOracle:
    """Classifies attacker candidate bodies through the gateway and returns the feedback the attacker is
    allowed to see. One ``query`` is one genuine (uncached) classification of a candidate body."""

    classifier: CachingClassifier
    score_feedback: bool = False
    n_queries: int = 0

    def query(self, body: str, request_id: str) -> dict:
        before = self.classifier.n_gateway_calls
        verdict = self.classifier.classify(body, request_id)
        was_cached = self.classifier.n_gateway_calls == before
        if not was_cached:
            self.n_queries += 1
        allowed = verdict.decision == "allow"
        out = {
            "decision": verdict.decision,
            "classifier_allowed": allowed,
            "stages_run": list(verdict.stages_run),
            "coverage_policy_applied": verdict.coverage_policy_applied,
            "flag_cause": verdict.flag_cause,
            "was_cached": was_cached,
            "gateway_status": verdict.gateway_status,
            "fallback": verdict.fallback,
        }
        # Score is observe-able only in the ablation; always computed, revealed conditionally.
        out["score"] = decider_score(verdict) if self.score_feedback else None
        out["_score_internal"] = decider_score(verdict)  # recorded in logs, not shown to the attacker
        return out


def make_gateway_classifier(url: str, timeout_s: float | None = None) -> CachingClassifier:
    """A CachingClassifier over a real GatewayClient. on_unavailable='flag' (fail closed): a dead
    gateway counts as a flag, never as an evasion, so a broken gateway can never inflate ASR."""
    return CachingClassifier(GatewayClient(url, timeout_s=timeout_s, on_unavailable="flag"))
