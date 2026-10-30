"""D23: Stage 1 runs observe-only and is returned with role="shadow". It must never count as a catch, a
truncation, a failure or a detector timeout of the active path, and the trace must not attribute the verdict
to it; it stays available as observe data."""
from redteam.classifier import ClassifierVerdict
from redteam.defenders import GullibleDefender
from redteam.judge import judge_attempt, summarize
from redteam.sandbox import run_attempt
from tests.redteam.test_contract_v12 import client, gateway, result


def d23_response(body, stage1_status="timeout"):
    """DeBERTa decides allow; the observe Stage 1 hit its deadline (truncated, status timeout)."""
    rid, text = body["request_id"], body["text"]
    s1 = result(rid, text, 0.9 if stage1_status == "ok" else None, stage="stage1", status=stage1_status,
                truncated=True, failure_reason=None if stage1_status == "ok" else "timeout",
                detector_id="stage1-tfidf-lr")
    s1["role"] = "shadow"
    return {"request_id": rid, "decision": "allow", "decided_by": "stage2-det", "policy_applied": None,
            "coverage_policy_applied": None, "stages_run": ["stage2"],
            "results": [result(rid, text, 0.01, stage="stage2"), s1]}


def test_verdict_names_only_active_detectors_and_failures():
    with client(gateway(d23_response)) as c:
        v = c.classify("hello", "r1")
    assert v.detector_version == "stage2-det@1.0.0"  # Stage 1 did not decide; not named
    assert v.observe_detector_version == "stage1-tfidf-lr@1.0.0"
    assert v.failure_reasons == (None,) and v.observe_failure_reasons == ("timeout",)
    assert v.truncated is False
    d = v.as_detail()
    assert d["failure_reasons"] == [None] and d["observe_failure_reasons"] == ["timeout"]
    assert [r["role"] for r in d["results"]] == ["active", "shadow"]  # observe data is still recorded


def test_verdict_with_only_observe_results_falls_back_to_decided_by():
    from contracts import DetectorResult
    s1 = result("r", "x", 0.2, detector_id="stage1-tfidf-lr")
    s1["role"] = "shadow"
    v = ClassifierVerdict(decision="allow", decided_by="stage2-det", results=(DetectorResult(**s1),))
    assert v.detector_version == "stage2-det"


def test_sandbox_emits_no_detector_timeout_for_observe_stage1(manager, trace):
    with client(gateway(d23_response)) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    assert not any(e.event_type == "detector_timeout" for e in evs)
    preds = [e for e in evs if e.event_type == "detector_prediction"]
    assert preds and all(p.detector_version == "stage2-det@1.0.0" for p in preds)
    assert all(p.detail["observe_failure_reasons"] == ["timeout"] for p in preds)


def test_active_stage2_timeout_still_emits_detector_timeout(manager, trace):
    def make(body):
        rid, text = body["request_id"], body["text"]
        return {"request_id": rid, "decision": "flag", "decided_by": "fail_closed", "policy_applied": "fail_closed",
                "coverage_policy_applied": None, "stages_run": ["stage2"],
                "results": [result(rid, text, None, stage="stage2", status="timeout", truncated=True,
                                   failure_reason="timeout")]}
    with client(gateway(make)) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    inner = [e for e in trace.attempt_events(res.attempt_id)
             if e.event_type == "detector_timeout" and e.detail.get("inner_result")]
    assert inner and all(e.detail["role"] == "active" for e in inner)


def test_judge_does_not_count_observe_truncation(manager, trace):
    with client(gateway(lambda b: d23_response(b, stage1_status="ok"))) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    v = judge_attempt(evs, manager)
    assert v.n_truncated_predictions == 0
    assert summarize([v])["n_truncated_predictions"] == 0
