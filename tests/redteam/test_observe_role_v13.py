"""Consumers treat a v1.3 role="observe" result exactly as they treated role="shadow" (design §7, migration 5).

The observe-only Stage 1 of D23 was reported as role="shadow" by the v1.2 gateway and is reported as
role="observe" by the policy engine. Everything that counts catches, truncations or failures must keep only
role="active", and every consumer records the bundle that served the response.
"""
import json

import pytest

from contracts import ClassifyResponse
from redteam.classifier import ClassifierVerdict

SHA = "a" * 64
BUNDLE = "b" * 64


def result(role="active", *, stage="stage2", status="ok", score=0.9, truncated=False, detector="d",
           schema_version=None):
    r = {"request_id": "r1", "detector_id": detector, "detector_version": "v1", "status": status,
         "threshold": 0.5, "latency_ms": 1.0, "input_sha256": SHA, "input_tokens": 4, "stage": stage,
         "role": role, "truncated": truncated, "aggregation": "max"}
    if status == "ok":
        r.update(score=score, predicted_injection=score >= 0.5)
    else:
        r["failure_reason"] = "timeout" if status == "timeout" else "exception"
    r["schema_version"] = schema_version or ("1.3" if role == "observe" else "1.2")
    return r


def response(*results, **kw):
    body = {"request_id": "r1", "decision": "allow", "decided_by": "d", "stages_run": ["stage2"],
            "results": list(results), **kw}
    return ClassifyResponse.model_validate(body)


OBSERVE = result(role="observe", stage="stage1", status="timeout", truncated=True, detector="stage1-tfidf-lr")
SHADOW = {**OBSERVE, "role": "shadow", "schema_version": "1.2"}


# ---------------------------------------------------------------- redteam/classifier.py

@pytest.mark.parametrize("observe", [OBSERVE, SHADOW], ids=["v13_observe", "v12_shadow"])
def test_verdict_counts_only_active_results(observe):
    resp = response(result(), observe)
    v = ClassifierVerdict(decision="allow", decided_by="d", results=tuple(resp.results))
    assert [r.detector_id for r in v.active_results] == ["d"]
    assert [r.detector_id for r in v.observe_results] == ["stage1-tfidf-lr"]
    assert v.detector_version == "d@v1"  # the observe step is not named as a decider
    assert v.observe_detector_version == "stage1-tfidf-lr@v1"
    assert v.failure_reasons == (None,) and v.observe_failure_reasons == ("timeout",)
    assert v.truncated is False  # the observe step's truncation is not the decision's


def test_verdict_records_the_bundle():
    v = ClassifierVerdict(decision="allow", decided_by="d", policy_id="d23-deberta-decides", policy_version="1",
                          bundle_sha256=BUNDLE, unscored_by_any_step=False, escalation_reason=None)
    d = v.as_detail()
    assert d["bundle_sha256"] == BUNDLE and d["policy_id"] == "d23-deberta-decides"
    assert d["policy_version"] == "1" and d["unscored_by_any_step"] is False


def test_gateway_client_carries_the_v13_fields():
    import httpx

    from redteam.gateway_client import GatewayClient

    body = {"request_id": "x", "decision": "flag", "decided_by": "d", "stages_run": ["stage2"],
            "results": [{**result(), "request_id": "x"}, {**OBSERVE, "request_id": "x"}],
            "policy_id": "p", "policy_version": "7", "bundle_sha256": BUNDLE, "unscored_by_any_step": True,
            "escalation_reason": "coverage"}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    with GatewayClient("http://gw", transport=transport) as c:
        v = c.classify("hello", "x")
    assert (v.policy_id, v.policy_version, v.bundle_sha256) == ("p", "7", BUNDLE)
    assert v.unscored_by_any_step is True and v.escalation_reason == "coverage"
    assert [r.role for r in v.active_results] == ["active"] and len(v.observe_results) == 1


# ---------------------------------------------------------------- redteam/judge.py

@pytest.mark.parametrize("observe", [OBSERVE, SHADOW], ids=["v13_observe", "v12_shadow"])
def test_judge_ignores_observe_truncation(observe):
    """A truncated observe result must not make the prediction count as truncated."""
    from redteam.judge import _prediction_truncated

    def event(detail):
        from contracts import TraceEvent
        return TraceEvent(event_id="e", run_id="r", attempt_id="a", timestamp="2026-10-30T00:00:00Z",
                          event_type="detector_prediction", mode="B", source="scripted", detector_version="v",
                          detail=detail)

    assert _prediction_truncated(event({"results": [result(), observe]})) is False
    assert _prediction_truncated(event({"results": [result(truncated=True), observe]})) is True
    assert _prediction_truncated(event({"truncated": True, "results": []})) is True


# ---------------------------------------------------------------- redteam/probe_runner.py

@pytest.mark.parametrize("observe", [OBSERVE, SHADOW], ids=["v13_observe", "v12_shadow"])
def test_probe_runner_separates_observe_and_records_the_bundle(observe):
    from redteam.probe_runner import ProbeOutcome, run_probes
    from redteam.probes import load_probes

    resp = response(result(), observe)

    class Fake:
        def classify(self, text, request_id):
            return ClassifierVerdict(decision="allow", decided_by="d", results=tuple(resp.results),
                                     request_id=request_id, stages_run=("stage2",), bundle_sha256=BUNDLE,
                                     policy_id="p", policy_version="1")

    probes = load_probes()[:3]
    rep = run_probes(Fake(), probes=probes)
    assert rep["bundle_sha256"] == BUNDLE and rep["bundles_seen"] == [BUNDLE]
    assert rep["policy"] == {"policy_id": "p", "policy_version": "1"}
    o = ProbeOutcome.from_verdict(probes[0], Fake().classify("x", "r"))
    assert "stage2" in o.stage_scores  # the active result keeps the plain stage key
    key = f"stage1_{observe['role']}"
    assert key in o.stage_scores, o.stage_scores  # the observe result is keyed apart
    assert o.scored_positive is True and o.bundle_sha256 == BUNDLE


def test_probe_runner_keeps_two_active_results_of_one_stage():
    """review r1 n7: two active stage2 results (shape B, or a decide member plus its fallback) used to overwrite
    each other in stage_scores; the second is now keyed by its detector."""
    from redteam.probe_runner import ProbeOutcome
    from redteam.probes import load_probes

    resp = response(result(detector="a", score=0.1), result(detector="b", score=0.9))
    verdict = ClassifierVerdict(decision="flag", decided_by="a+b:any_flag", results=tuple(resp.results),
                                request_id="r", stages_run=("stage2", "stage2"), bundle_sha256=BUNDLE)
    o = ProbeOutcome.from_verdict(load_probes()[0], verdict)
    assert o.stage_scores["stage2"]["score"] == 0.1 and o.stage_scores["stage2:b"]["score"] == 0.9
    assert len(o.stage_scores) == 2


# ---------------------------------------------------------------- training consumers

@pytest.mark.parametrize("observe", [OBSERVE, SHADOW], ids=["v13_observe", "v12_shadow"])
def test_eval_harness_parse(observe):
    from training.eval_harness import parse_gateway_response

    body = json.loads(response(result(score=0.9), observe, policy_id="p", policy_version="1",
                               bundle_sha256=BUNDLE, unscored_by_any_step=True).model_dump_json())
    out = parse_gateway_response(body, "r1")
    assert out["model_prediction"] is True and out["statuses"] == ["ok"]
    assert out["n_shadow"] == 1 and out["shadow_failure_reasons"] == ["timeout"]
    assert out["truncated"] is False and out["shadow_truncated"] is True
    assert out["bundle_sha256"] == BUNDLE and out["policy_id"] == "p"
    assert out["unscored_by_any_step"] is True


def test_eval_harness_collects_the_bundle_per_run():
    from training.eval_harness import evaluate

    body = json.loads(response(result(score=0.9), OBSERVE, policy_id="p", policy_version="1",
                               bundle_sha256=BUNDLE).model_dump_json())

    class Backend:
        name = "fake"

        def predict(self, text, request_id):
            from training.eval_harness import parse_gateway_response
            payload = {**body, "request_id": request_id,
                       "results": [{**r, "request_id": request_id} for r in body["results"]]}
            return parse_gateway_response(payload, request_id)

        def describe(self):
            return {"backend": "fake"}

    rows = [{"id": f"i{i}", "text": "x", "label": i % 2, "source": "s"} for i in range(4)]
    res = evaluate(rows, Backend())
    assert res["bundles"]["single_bundle"] is True and res["bundles"]["bundle_sha256"] == BUNDLE
    assert res["bundles"]["policies"] == {"p@1": 4}


@pytest.mark.parametrize("observe", [OBSERVE, SHADOW], ids=["v13_observe", "v12_shadow"])
def test_replay_runner_observe_and_bundle(observe):
    from training.replay_runner import _bundle_block, _is_observe_stage1, _shadow

    body = json.loads(response(result(), observe).model_dump_json())
    (s,) = _shadow(body)
    assert s["role"] == observe["role"] and _is_observe_stage1(s)
    assert _shadow({"results": [result()]}) == []

    class Rec:
        def __init__(self, parsed):
            self.parsed = parsed

    ok = [Rec({"bundle_sha256": BUNDLE, "policy_id": "p", "policy_version": "1"}),
          Rec({"bundle_sha256": BUNDLE, "policy_id": "p", "policy_version": "1"}),
          Rec({})]
    b = _bundle_block(ok)
    assert b["bundle_sha256"] == BUNDLE and b["single_bundle"] is True and b["n_without_bundle"] == 1
    b2 = _bundle_block([Rec({"bundle_sha256": BUNDLE}), Rec({"bundle_sha256": "c" * 64})])
    assert b2["single_bundle"] is False and b2["bundle_sha256"] is None


def test_no_consumer_still_treats_shadow_as_the_only_non_active_role():
    """A grep guard: `role != "shadow"` is the pre-1.3 idiom and lets an observe result through."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    bad = []
    PRE_13_IDIOMS = ('role != "shadow"', "role != 'shadow'", 'role") != "shadow"', 'role") not in ("shadow"')
    for rel in ("redteam/classifier.py", "redteam/judge.py", "redteam/sandbox.py", "redteam/probe_runner.py",
                "training/eval_harness.py", "training/replay_runner.py"):
        text = (root / rel).read_text()
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(pat in line for pat in PRE_13_IDIOMS):
                bad.append(f"{rel}:{n}: {line.strip()}")
    assert not bad, "pre-1.3 role test (observe would be counted as active):\n" + "\n".join(bad)
