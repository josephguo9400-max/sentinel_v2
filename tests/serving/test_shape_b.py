"""T7 (docs/design/capability-policy.md §8): shape B, a parallel decide stage, with fake detectors only.

D25.5: shape B is implemented and tested with fakes; it is not served. On a 2-core host two DeBERTa-class members
compete for CPU and turn into timeouts (design §10), so no committed policy uses it and the gate never runs it.

Checked here: the any_flag / all_flag truth tables over both members' votes, that a failed member votes by its own
on_failure, that both members score identical rows, that aggregation is decision-level (never score averaging),
and that coverage still applies to content no member scored.
"""
import asyncio

import pytest

from fakes import FakeDetector, FakeStage1, RecordingDetector, words
from policy_helpers import default_caps, policy, ref, service, spec
from serving.config import GatewayConfig

CAPS = default_caps()
A_KEY, B_KEY, GATE_KEY, FB_KEY = ("fake", "0.0.1"), ("fake-b", "0.0.1"), ("fake-s1", "0.0.1"), ("fake-fb", "0.0.1")


def cfg(**kw) -> GatewayConfig:
    kw.setdefault("shadow_log_path", "")
    return GatewayConfig(**kw)


class Fixed(FakeDetector):
    """Scores every row the same, so a member's vote is set by the test."""

    def __init__(self, score: float, name: str, **kw):
        super().__init__(**kw)
        self.detector_id = name
        self.score = score
        self.rows: list[list[int]] = []

    def infer(self, batch):
        with self._lock:
            self.rows.extend(r[1:-1] for r in batch)
        super().infer(batch)  # keeps the counters, delays and faults of FakeDetector
        return [self.score] * len(batch)


def build(aggregate: str, a: float, b: float, *, gate: bool = False, coverage: str = "flag", config=None,
          a_kwargs=None, b_kwargs=None, a_failure=None, b_failure=None):
    da, db = Fixed(a, "fake", **(a_kwargs or {})), Fixed(b, "fake-b", **(b_kwargs or {}))
    specs = {A_KEY: spec(da), B_KEY: spec(db), FB_KEY: spec(Fixed(0.0, "fake-fb"))}
    doc = policy(CAPS, parallel=aggregate, gate=gate, coverage=coverage)
    for i, failure in ((0, a_failure), (1, b_failure)):
        doc["decide"]["members"][i]["timeout_ms"] = 200
        if failure is not None:
            doc["decide"]["members"][i]["on_failure"] = failure
    if gate:
        specs[GATE_KEY] = spec(FakeStage1(), kind="text", timeout_ms=2000)
        doc["gate"]["timeout_ms"] = 2000
    return service(doc, specs, config=config or cfg()), da, db


def run(svc, texts):
    async def scenario():
        await svc.start()
        try:
            return [await svc.classify(t, f"r{i}") for i, t in enumerate(texts)]
        finally:
            await svc.stop()

    return asyncio.run(scenario())


# ---------------------------------------------------------------- truth tables

TRUTH = {("any_flag", True, True): "flag", ("any_flag", True, False): "flag",
         ("any_flag", False, True): "flag", ("any_flag", False, False): "allow",
         ("all_flag", True, True): "flag", ("all_flag", True, False): "allow",
         ("all_flag", False, True): "allow", ("all_flag", False, False): "allow"}


@pytest.mark.parametrize("b_flag", [True, False])
@pytest.mark.parametrize("a_flag", [True, False])
@pytest.mark.parametrize("aggregate", ["any_flag", "all_flag"])
def test_t7_truth_table(aggregate, a_flag, b_flag):
    svc, da, db = build(aggregate, 0.9 if a_flag else 0.1, 0.9 if b_flag else 0.1)
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.decision == TRUTH[(aggregate, a_flag, b_flag)]
    assert o.policy_applied is None and o.decided_by == f"fake+fake-b:{aggregate}"
    assert [m["flag"] for m in o.info["members"]] == [a_flag, b_flag]
    # Both members ran and reported their own score and threshold.
    assert sorted((r.detector_id, r.score) for r in o.results) == [("fake", 0.9 if a_flag else 0.1),
                                                                   ("fake-b", 0.9 if b_flag else 0.1)]
    assert all(r.role == "active" and r.stage == "stage2" for r in o.results)


def test_t7_aggregation_is_decision_level_not_score_averaging():
    """0.9 and 0.1 average below the threshold, yet any_flag flags: the votes are aggregated, not the scores."""
    svc, _, _ = build("any_flag", 0.9, 0.1)
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.decision == "flag" and max(r.score for r in o.results) == 0.9
    svc, _, _ = build("all_flag", 0.9, 0.1)
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.decision == "allow"


# ---------------------------------------------------------------- identical chunk selection

@pytest.mark.parametrize("text", ["w1 w2 w3", words(3000), words(3000, inject_at=1500)])
@pytest.mark.parametrize("gate", [False, True])
def test_t7_both_members_score_identical_rows(text, gate):
    svc, da, db = build("any_flag", 0.1, 0.1, gate=gate)
    run(svc, [text])
    assert da.rows == db.rows
    assert da.batches == db.batches  # same number of rows per call, too


def test_t7_members_share_the_chunk_budget():
    svc, da, db = build("any_flag", 0.1, 0.1, config=cfg())
    run(svc, [words(3000)])
    assert len(da.rows) == len(db.rows) == 4  # max_chunks of both members


# ---------------------------------------------------------------- failure votes

@pytest.mark.parametrize("fault", ["timeout", "exception"])
@pytest.mark.parametrize("mode,other,expect", [
    ("FAIL_CLOSED", 0.1, ("flag", "fail_closed")),   # failed member votes flag, any_flag flags
    ("FAIL_OPEN", 0.1, ("allow", "fail_open")),      # both allow, but one vote is a policy vote
    ("FAIL_OPEN", 0.9, ("flag", None)),              # the healthy member's flag alone decides: model-backed
])
def test_t7_failed_member_votes_by_its_on_failure(fault, mode, other, expect):
    kwargs = {"delay_s": 0.5} if fault == "timeout" else {"fail": RuntimeError("boom")}
    svc, _, _ = build("any_flag", 0.1, other, a_kwargs=kwargs, a_failure={"mode": mode, "fallback": None})
    (o,) = run(svc, ["w1 w2 w3"])
    assert (o.decision, o.policy_applied) == expect
    a = next(m for m in o.info["members"] if m["detector"] == "fake")
    assert a["status"] in ("timeout", "error")


def test_t7_all_flag_with_a_failed_member():
    """all_flag: a FAIL_CLOSED vote is not enough on its own, and an allow by the healthy member decides."""
    svc, _, _ = build("all_flag", 0.1, 0.1, a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_CLOSED", "fallback": None})
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.decision == "allow" and o.policy_applied is None
    svc, _, _ = build("all_flag", 0.1, 0.9, a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_CLOSED", "fallback": None})
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.decision == "flag" and o.policy_applied == "fail_closed"


def test_t7_member_fallback_scores_the_same_rows():
    fb = {"mode": "FALLBACK", "fallback": {"detector": ref(CAPS["fallback"]), "timeout_ms": 500,
                                           "terminal": "FAIL_CLOSED"}}
    svc, da, db = build("any_flag", 0.1, 0.1, a_kwargs={"fail": RuntimeError("boom")}, a_failure=fb,
                        coverage="allow")  # words(3000) has more chunks than the budget: coverage would decide
    (o,) = run(svc, [words(3000)])
    fallback = svc._specs[FB_KEY].detector
    assert o.decision == "allow" and o.policy_applied is None and o.coverage_policy_applied == "allow"
    assert fallback.rows == db.rows  # the fallback took over the failed member's chunk selection
    assert sorted(r.detector_id for r in o.results if r.status == "ok") == ["fake-b", "fake-fb"]


def test_t7_coverage_counts_a_chunk_scored_by_one_member_as_covered():
    """One member failed, the other scored every chunk: no content is unscored, so coverage does not apply."""
    svc, _, _ = build("any_flag", 0.1, 0.1, a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_CLOSED", "fallback": None}, coverage="flag")
    (o,) = run(svc, ["w1 w2 w3"])
    assert o.coverage_policy_applied is None and o.unscored_by_any_step is False
    assert (o.decision, o.policy_applied) == ("flag", "fail_closed")  # the failed member's vote, not coverage


def test_t7_both_members_fail():
    svc, _, _ = build("any_flag", 0.1, 0.1, a_kwargs={"fail": RuntimeError("a")},
                      b_kwargs={"fail": RuntimeError("b")},
                      a_failure={"mode": "FAIL_OPEN", "fallback": None},
                      b_failure={"mode": "FAIL_CLOSED", "fallback": None})
    (o,) = run(svc, ["w1 w2 w3"])
    assert (o.decision, o.policy_applied) == ("flag", "fail_closed")


# ---------------------------------------------------------------- coverage and the gate

@pytest.mark.parametrize("coverage,expect", [("flag", "flag"), ("quarantine", "flag"), ("allow", "allow")])
def test_t7_coverage_applies_to_content_no_member_scored(coverage, expect):
    svc, _, _ = build("any_flag", 0.1, 0.1, coverage=coverage)
    (o,) = run(svc, [words(3000)])  # more chunks than the members' budget
    assert o.decision == expect and o.coverage_policy_applied == coverage
    assert o.unscored_by_any_step is True


def test_t7_gate_can_finalize_before_a_parallel_stage():
    svc, da, db = build("any_flag", 0.9, 0.9, gate=True)
    (o,) = run(svc, ["w1 w2 w3"])  # the gate clears the only chunk
    assert o.decision == "allow" and o.stages_run == ["stage1"] and o.decided_by == "fake-s1"
    assert da.rows == db.rows == []
    (o,) = run(svc, ["w1 GRAY w3"])  # escalates to both members
    assert o.decision == "flag" and o.stages_run == ["stage1", "stage2", "stage2"]
    assert o.escalation_reason == "band"


def test_t7_parallel_is_not_served_by_any_committed_policy():
    """D25.5: shape B stays unserved. Every committed policy has a single decide stage."""
    import json
    from pathlib import Path

    from contracts import ExecutionPolicy
    root = Path(__file__).resolve().parents[2] / "artifacts"
    found = sorted(root.glob("policies/*/*/policy.json"))
    assert found, "no committed policies"
    for path in found:
        p = ExecutionPolicy.model_validate(json.loads(path.read_text()))
        assert p.decide.mode == "single", f"{path} serves shape B"
