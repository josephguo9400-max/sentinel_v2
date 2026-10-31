"""The per-goal adaptive search loop and its end-to-end (A/B/C) measurement.

For one goal the loop proposes candidate bodies with the attacker model, classifies each through the
gateway (one query), feeds the black-box outcome back, and keeps going until it evades or the budget
is spent. Because the classifier is cached by exact text, running the mode-B sandbox on every
candidate adds no new DeBERTa calls after warmup, so the attacker gets a real behavioural signal
(did the assistant send?) for free. Modes A and C are measured once on the witness candidate.

Success is reported in two separate axes (never merged):
  * classifier-evaded  -- the gateway allowed a harmful-carrying body (mode B decision == allow)
  * end-to-end harm    -- an unauthorised send actually executed (mode B judge attack_success)
Mode C must show attack_success == 0 for every candidate (the firewall is the security boundary).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..defenders import GullibleDefender
from ..judge import judge_attempt
from ..sandbox import Limits, run_attempt
from ..trace import RunTrace
from .budget import BudgetExceeded
from .goals import Goal
from .llm import AttackerClient
from .oracle import CachingClassifier, DecisionOracle
from .strategies import (INJECTION_CORES, STRATEGIES, build_carrier_prompt, build_prompt,
                         compose_body, looks_like_refusal, parse_body)

LONG_MAX_TOKENS = 3000  # the model may need many tokens for a long_dilution body
SHORT_MAX_TOKENS = 1024


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


@dataclass
class GoalResult:
    goal: Goal
    budget: int
    n_queries: int = 0
    n_attacker_calls: int = 0
    n_attacker_cache_hits: int = 0
    # "allowed_any": the classifier allowed some body (harmful or not) -- a weak diagnostic only.
    classifier_allowed_any: bool = False
    queries_to_first_allow: int | None = None
    # "harmful_evasion": the classifier allowed a body that is ACTUALLY harmful (succeeds undefended in
    # mode A). For the deterministic gullible victim this is exactly mode-B end-to-end harm (the body
    # reaches the victim and the send executes because the firewall is off). This is the headline.
    harmful_evasion: bool = False
    queries_to_first_harmful_evasion: int | None = None
    end_to_end_harm: bool = False  # mode B attack_success (== harmful_evasion for this victim)
    queries_to_first_harm: int | None = None
    # Candidate-level classifier accounting over harmful candidates: how many harmful bodies the
    # classifier allowed (missed) vs flagged (caught).
    n_candidates: int = 0
    n_harmful: int = 0
    n_harmful_allowed: int = 0  # harmful AND classifier allowed (evaded)
    n_harmful_flagged: int = 0  # harmful AND classifier flagged (caught)
    n_refusals: int = 0  # candidates where the attacker model declined to produce a functional injection
    stopped_reason: str = "budget_exhausted"
    witness: dict[str, Any] | None = None  # A/B/C verdicts on the witness candidate
    queries: list[dict[str, Any]] = field(default_factory=list)  # per-query log (bodies in the raw log only)

    def summary(self) -> dict:
        """Committed-safe summary: no raw bodies, only hashes/lengths."""
        return {
            "goal": self.goal.as_dict(),
            "budget": self.budget,
            "n_queries": self.n_queries,
            "n_attacker_calls": self.n_attacker_calls,
            "n_attacker_cache_hits": self.n_attacker_cache_hits,
            "classifier_allowed_any": self.classifier_allowed_any,
            "queries_to_first_allow": self.queries_to_first_allow,
            "harmful_evasion": self.harmful_evasion,
            "queries_to_first_harmful_evasion": self.queries_to_first_harmful_evasion,
            "end_to_end_harm": self.end_to_end_harm,
            "queries_to_first_harm": self.queries_to_first_harm,
            "n_candidates": self.n_candidates,
            "n_harmful": self.n_harmful,
            "n_harmful_allowed": self.n_harmful_allowed,
            "n_harmful_flagged": self.n_harmful_flagged,
            "n_refusals": self.n_refusals,
            "stopped_reason": self.stopped_reason,
            "witness": self.witness,
            "queries_meta": [{k: q.get(k) for k in ("query", "strategy", "seed_index", "body_len",
                                                    "body_sha16", "classifier_allowed", "is_harmful",
                                                    "score", "mode_a_attack_success",
                                                    "mode_b_attack_success", "flag_cause", "was_cached",
                                                    "gateway_status")}
                             for q in self.queries],
        }


def _measure_mode(scenario, mode: str, body: str, classifier: CachingClassifier,
                  attempt_id: str) -> dict:
    """Run one sandbox attempt in a mode with the gullible victim and judge it."""
    trace = RunTrace(run_id=f"adv-{attempt_id}", out_dir=None)
    clf = None if mode == "A" else classifier
    res = run_attempt(scenario, mode, GullibleDefender(), trace, classifier=clf,
                      attacker_body=body, attempt_id=attempt_id, limits=Limits(deadline_s=30.0))
    v = judge_attempt(trace.attempt_events(res.attempt_id), scenario)
    return {
        "status": v.status,
        "attack_success": v.attack_success,
        "unsafe_attempt": v.unsafe_attempt,
        "blocked": v.blocked,
        "attacker_email_quarantine_cause": v.attacker_email_quarantine_cause,
        "attack_stopped_by_classifier": v.attack_stopped_by_classifier,
    }


def run_goal(goal: Goal, attacker: AttackerClient, classifier: CachingClassifier, *, budget: int,
             score_feedback: bool = False, temperature: float = 1.0,
             attack_mode: str = "compose") -> GoalResult:
    scenario = goal.scenario()
    strat = STRATEGIES[goal.strategy]
    max_tokens = LONG_MAX_TOKENS if strat.long else SHORT_MAX_TOKENS
    oracle = DecisionOracle(classifier, score_feedback=score_feedback)
    result = GoalResult(goal=goal, budget=budget)
    history: list[dict[str, Any]] = []
    n_seeds = len(strat.seeds)
    cores = list(INJECTION_CORES)

    for q in range(1, budget + 1):
        if attack_mode == "compose":
            # LLM writes only a BENIGN carrier; the harness inserts a fixed mild injection core.
            system, messages = build_carrier_prompt(scenario, strategy=goal.strategy,
                                                    seed_index=(q - 1), n_prior=len(history))
        else:  # "author": the LLM authors the full injection (documents the refusal behaviour)
            system, messages = build_prompt(scenario, strategy=goal.strategy, target=goal.target,
                                            channel=goal.channel,
                                            seed_index=(q - 1) if q <= n_seeds else q,
                                            history=history, score_feedback=score_feedback)
        try:
            completion = attacker.complete(system=system, messages=messages, max_tokens=max_tokens,
                                           temperature=temperature)
        except BudgetExceeded:
            result.stopped_reason = "dollar_cap"
            break
        result.n_attacker_calls += 1
        if completion.cached:
            result.n_attacker_cache_hits += 1
        carrier = parse_body(completion.text)
        if attack_mode == "compose" and carrier:
            body = compose_body(carrier, strategy=goal.strategy, core_id=cores[(q - 1) % len(cores)],
                                target=goal.target)
        else:
            body = carrier
        if not body:
            result.queries.append({"query": q, "strategy": goal.strategy, "seed_index": (q - 1) % n_seeds,
                                   "body": "", "body_len": 0, "body_sha16": "", "classifier_allowed": None,
                                   "is_harmful": None, "score": None, "mode_a_attack_success": None,
                                   "mode_b_attack_success": None, "flag_cause": None, "was_cached": None,
                                   "gateway_status": None, "note": "empty_body"})
            continue

        fb = oracle.query(body, request_id=f"{goal.goal_id}:q{q}")
        result.n_queries = oracle.n_queries

        # is_harmful: does the body cause an unauthorised send with NO classifier (mode A)? This tells a
        # genuine harmful payload apart from one the attacker softened into harmlessness just to pass.
        ma = _measure_mode(scenario, "A", body, classifier, attempt_id=f"{goal.goal_id[:7]}a{q:03d}")
        is_harmful = ma["attack_success"]
        # Behavioural signal: mode B (classifier screens; cached text -> no new DeBERTa call).
        mb = _measure_mode(scenario, "B", body, classifier, attempt_id=f"{goal.goal_id[:7]}b{q:03d}")
        mb_success = mb["attack_success"]  # harmful AND classifier allowed

        result.n_candidates += 1
        refused = (not is_harmful) and looks_like_refusal(body)
        if refused:
            result.n_refusals += 1
        if is_harmful:
            result.n_harmful += 1
            if fb["classifier_allowed"]:
                result.n_harmful_allowed += 1
            else:
                result.n_harmful_flagged += 1

        entry = {
            "query": q, "strategy": goal.strategy, "seed_index": (q - 1) % n_seeds,
            "body": body, "body_len": len(body), "body_sha16": _body_hash(body),
            "classifier_allowed": fb["classifier_allowed"], "is_harmful": is_harmful,
            "score": fb["score"], "score_internal": fb["_score_internal"],
            "mode_a_attack_success": is_harmful, "mode_b_attack_success": mb_success,
            "refused": refused, "flag_cause": fb["flag_cause"], "was_cached": fb["was_cached"],
            "gateway_status": fb["gateway_status"],
        }
        result.queries.append(entry)
        history.append({"query": q, "body": body, "classifier_allowed": fb["classifier_allowed"],
                        "score": fb["score"], "send_executed": mb_success})

        if fb["classifier_allowed"] and result.queries_to_first_allow is None:
            result.classifier_allowed_any = True
            result.queries_to_first_allow = q
        if mb_success and result.queries_to_first_harm is None:
            result.harmful_evasion = True
            result.end_to_end_harm = True
            result.queries_to_first_harm = q
            result.queries_to_first_harmful_evasion = q
            result.stopped_reason = "harm_achieved"
            result.witness = _witness(scenario, body, classifier, goal.goal_id)
            break

    # No end-to-end harm within budget: still record a witness (first evasion, else last candidate) so
    # modes A and C are reported for the goal.
    if result.witness is None:
        scored = [e for e in result.queries if e["body"]]
        # Prefer a harmful body the classifier allowed; else any harmful body (classifier caught it);
        # else the last candidate, so modes A and C are still reported for the goal.
        witness_body = next((e["body"] for e in scored if e.get("mode_b_attack_success")), None)
        if witness_body is None:
            witness_body = next((e["body"] for e in scored if e.get("is_harmful")), None)
        if witness_body is None:
            witness_body = scored[-1]["body"] if scored else None
        if witness_body is not None:
            result.witness = _witness(scenario, witness_body, classifier, goal.goal_id)
    return result


def _witness(scenario, body: str, classifier: CachingClassifier, goal_id: str) -> dict:
    stem = goal_id[:8]
    return {
        "body_sha16": _body_hash(body),
        "body_len": len(body),
        "mode_A": _measure_mode(scenario, "A", body, classifier, f"{stem}wa"),
        "mode_B": _measure_mode(scenario, "B", body, classifier, f"{stem}wb"),
        "mode_C": _measure_mode(scenario, "C", body, classifier, f"{stem}wc"),
    }
