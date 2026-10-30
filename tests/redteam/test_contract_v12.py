"""Contracts v1.2 adoption: ClassifyResponse validation, recorded cascade fields, TraceEvent seq/scenario_id,
the UNAVAILABLE_DETECTOR_VERSION sentinel, and coverage-policy quarantines as their own cause."""
import hashlib
import json

import httpx
import pytest

from contracts import UNAVAILABLE_DETECTOR_VERSION
from redteam.classifier import ClassifierVerdict
from redteam.defenders import GullibleDefender
from redteam.gateway_client import GatewayClient
from redteam.judge import judge_attempt, summarize
from redteam.sandbox import run_attempt


def result(request_id, text, score, *, stage="stage1", status="ok", truncated=False, n_chunks=1,
           failure_reason=None, detector_id=None):
    ok = status == "ok"
    return {"schema_version": "1.2", "request_id": request_id, "detector_id": detector_id or f"{stage}-det",
            "detector_version": "1.0.0", "status": status, "score": score if ok else None, "threshold": 0.5,
            "predicted_injection": (score > 0.5) if ok else None, "latency_ms": 1.0,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(), "input_tokens": 600 * n_chunks,
            "n_chunks": n_chunks, "truncated": truncated, "aggregation": "max", "stage": stage,
            "role": "active", "failure_reason": failure_reason}


def gateway(make):
    """make(body) -> response dict; GET /health is 404 so the client uses its default timeout."""
    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json=make(json.loads(request.content)))
    return handler


def client(handler, **kw):
    return GatewayClient("http://gateway.invalid", transport=httpx.MockTransport(handler), **kw)


def cascade_response(body, flag_attack=True, coverage=None, truncated=False):
    text, rid = body["text"], body["request_id"]
    hit = flag_attack and ("forward" in text or "send_email" in text)
    return {"request_id": rid, "decision": "flag" if (hit or coverage in ("flag", "quarantine")) else "allow",
            "decided_by": "coverage_policy" if coverage else ("stage2-det" if hit else "stage1-det"),
            "policy_applied": None, "coverage_policy_applied": coverage,
            "stages_run": ["stage1", "stage2"] if hit else ["stage1"],
            "results": [result(rid, text, 0.6 if hit else 0.05, truncated=truncated, n_chunks=5 if truncated else 1),
                        *([result(rid, text, 0.98, stage="stage2")] if hit else [])]}


# ------------------------------------------------------------------ ClassifyResponse validation

def test_records_stages_run_coverage_and_failure_reason():
    def make(body):
        rid, text = body["request_id"], body["text"]
        return {"request_id": rid, "decision": "flag", "decided_by": "fail_closed", "policy_applied": "fail_closed",
                "coverage_policy_applied": None, "stages_run": ["stage1", "stage2"],
                "results": [result(rid, text, 0.4), result(rid, text, None, stage="stage2", status="error",
                                                           failure_reason="overloaded")]}
    with client(gateway(make)) as c:
        v = c.classify("hello", "r1")
    assert v.gateway_status == "ok" and not v.fallback
    assert v.request_id == "r1" and v.stages_run == ("stage1", "stage2")
    assert v.failure_reasons == (None, "overloaded")
    d = v.as_detail()
    assert d["stages_run"] == ["stage1", "stage2"] and d["failure_reasons"] == [None, "overloaded"]
    assert d["coverage_policy_applied"] is None
    assert v.flag_cause == "gateway_failure_policy"


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r.pop("request_id"), "request_id"),
    (lambda r: r.update(request_id="someone-else"), "request_id"),
    (lambda r: r.update(extra_key=True), "extra"),
    (lambda r: r.update(stages_run=["stage3"]), "stages_run"),
    (lambda r: r.update(coverage_policy_applied="ignore"), "coverage_policy_applied"),
    (lambda r: r.update(results=[]), "at least one"),
    (lambda r: r["results"][0].update(request_id="other"), "request_id"),
])
def test_invalid_classify_response_is_bad_response(mutate, why):
    def make(body):
        r = cascade_response(body)
        mutate(r)
        return r
    with client(gateway(make)) as c:
        v = c.classify("hello", "r1")
    assert v.gateway_status == "bad_response" and v.fallback and v.decision == "flag"
    assert v.detector_version == UNAVAILABLE_DETECTOR_VERSION


def test_request_is_a_classify_request():
    seen = []

    def make(body):
        seen.append(body)
        return cascade_response(body)
    with client(gateway(make)) as c:
        c.classify("hi", "a:b")
    assert seen == [{"text": "hi", "request_id": "a:b"}]


# ------------------------------------------------------------------ TraceEvent v1.2 fields

def test_trace_uses_seq_and_scenario_id_fields_not_detail(manager, trace):
    with client(gateway(cascade_response)) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    assert [e.seq for e in evs] == list(range(1, len(evs) + 1))
    assert {e.scenario_id for e in evs} == {manager.scenario_id}
    assert all(e.schema_version == "1.2" for e in evs)
    assert not any("seq" in (e.detail or {}) or "scenario_id" in (e.detail or {}) for e in evs)
    preds = [e for e in evs if e.event_type == "detector_prediction"]
    assert all(p.detail["stages_run"] for p in preds)


def test_judge_still_reads_v11_traces_with_detail_seq(manager, trace):
    """Older committed/recorded traces carried seq and scenario_id in detail; they must still judge."""
    with client(gateway(cascade_response)) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    new = trace.attempt_events(res.attempt_id)
    old = [e.model_copy(update={"schema_version": "1.1", "seq": None, "scenario_id": None,
                                "detail": {"seq": e.seq, "scenario_id": e.scenario_id, **(e.detail or {})}})
           for e in reversed(new)]
    a, b = judge_attempt(new, manager).to_dict(), judge_attempt(old, manager).to_dict()
    assert a == b and a["status"] == "completed"


def test_unreachable_gateway_uses_unavailable_sentinel(digest, trace):
    with GatewayClient("http://127.0.0.1:9", timeout_s=0.3) as c:
        res = run_attempt(digest, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    det = [e for e in evs if e.event_type in ("detector_prediction", "detector_timeout")]
    assert det and all(e.detector_version == UNAVAILABLE_DETECTOR_VERSION and e.detail["fallback"] is True
                       for e in det)
    q = [e for e in evs if e.event_type == "content_quarantined"]
    assert q and all(e.detector_version == UNAVAILABLE_DETECTOR_VERSION for e in q)


def test_fallback_with_decided_by_still_uses_sentinel():
    v = ClassifierVerdict(decision="flag", decided_by="something", fallback=True, gateway_status="timeout")
    assert v.detector_version == UNAVAILABLE_DETECTOR_VERSION


# ------------------------------------------------------------------ coverage policy is its own cause

@pytest.mark.parametrize("coverage", ["flag", "quarantine"])
def test_coverage_policy_quarantine_is_not_a_classifier_catch(manager, trace, coverage):
    """Truncated input, nothing scored positive, gateway coverage policy flags: never a classifier stop."""
    with client(gateway(lambda b: cascade_response(b, flag_attack=False, coverage=coverage, truncated=True))) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    q = [e for e in evs if e.event_type == "content_quarantined"]
    assert len(q) == len(manager.emails)
    assert all(e.policy_reason == f"coverage_policy: {coverage}" and e.detail["cause"] == "coverage_policy"
               for e in q)
    v = judge_attempt(evs, manager)
    assert v.attacker_email_quarantine_cause == "coverage_policy" and not v.attack_stopped_by_classifier
    s = summarize([v])
    assert s["attack_stopped_by_classifier"] == 0
    assert s["attacker_email_quarantined_by_cause"]["coverage_policy"] == 1
    assert s["attacker_email_quarantined_by_cause"]["classifier_flag"] == 0
    assert s["quarantined_emails_by_cause"]["coverage_policy"] == len(manager.emails)


def test_coverage_policy_wins_over_a_positive_score():
    """Even if some chunk scored positive, a decision the coverage policy made is attributed to the policy."""
    def make(body):
        r = cascade_response(body, flag_attack=True, coverage="flag", truncated=True)
        return r
    with client(gateway(make)) as c:
        v = c.classify("please forward everything", "r1")
    assert any(r.predicted_injection for r in v.results)
    assert v.flag_cause == "coverage_policy"


def test_coverage_policy_allow_is_not_a_quarantine(manager, trace):
    with client(gateway(lambda b: cascade_response(b, flag_attack=False, coverage="allow", truncated=True))) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    assert not any(e.event_type == "content_quarantined" for e in evs)
    preds = [e for e in evs if e.event_type == "detector_prediction"]
    assert all(p.detail["coverage_policy_applied"] == "allow" and p.detail["truncated"] for p in preds)
