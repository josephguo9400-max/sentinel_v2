"""Eval harness against the v1.2 /classify contract: ClassifyResponse validation, stages_run,
coverage_policy_applied and failure_reason counts per slice; overloaded separate from timeouts/exceptions."""
import json

import httpx
import pytest

from training.eval_harness import GatewayResponseError, HttpGatewayBackend, evaluate, parse_gateway_response

SHA = "b" * 64


def dr(rid, stage="stage1", status="ok", score=0.1, reason=None, role="active", truncated=False, threshold=0.5):
    ok = status == "ok"
    return {"schema_version": "1.2", "request_id": rid, "detector_id": f"d-{stage}", "detector_version": "v",
            "status": status, "score": score if ok else None, "threshold": threshold,
            "predicted_injection": (score >= threshold) if ok else None, "latency_ms": 1.0, "input_sha256": SHA,
            "input_tokens": 5, "n_chunks": 1, "truncated": truncated if ok else True, "aggregation": "max",
            "stage": stage, "role": role, "failure_reason": reason}


def resp(rid, decision, results, stages, policy=None, coverage=None, decided_by="d-stage1"):
    return {"request_id": rid, "decision": decision, "decided_by": decided_by, "policy_applied": policy,
            "coverage_policy_applied": coverage, "stages_run": stages, "results": results}


def handler(request: httpx.Request) -> httpx.Response:
    """Prefix of the text picks the scenario."""
    b = json.loads(request.content)
    rid, t = b["request_id"], b["text"]
    if t.startswith("s1allow"):
        return httpx.Response(200, json=resp(rid, "allow", [dr(rid, score=0.05)], ["stage1"]))
    if t.startswith("esc-flag"):
        return httpx.Response(200, json=resp(rid, "flag", [dr(rid, score=0.6, threshold=0.5),
                                                           dr(rid, "stage2", score=0.99)],
                                             ["stage1", "stage2"], decided_by="d-stage2"))
    if t.startswith("overload"):
        return httpx.Response(200, json=resp(rid, "flag", [dr(rid, score=0.6), dr(rid, "stage2", "error",
                                                                                  reason="overloaded")],
                                             ["stage1", "stage2"], policy="fail_closed", decided_by="fail_closed"))
    if t.startswith("timeout"):
        return httpx.Response(200, json=resp(rid, "flag", [dr(rid, score=0.6), dr(rid, "stage2", "timeout",
                                                                                  reason="timeout")],
                                             ["stage1", "stage2"], policy="fail_closed", decided_by="fail_closed"))
    if t.startswith("exception"):
        return httpx.Response(200, json=resp(rid, "allow", [dr(rid, score=0.6), dr(rid, "stage2", "error",
                                                                                   reason="exception")],
                                             ["stage1", "stage2"], policy="fail_open", decided_by="fail_open"))
    if t.startswith("coverage"):
        return httpx.Response(200, json=resp(rid, "flag", [dr(rid, score=0.1, truncated=True)], ["stage1"],
                                             coverage="quarantine", decided_by="coverage"))
    if t.startswith("deepscan"):
        return httpx.Response(200, json=resp(rid, "allow", [dr(rid, score=0.1), dr(rid, "stage2", score=0.2)],
                                             ["stage1", "stage2"], coverage="deep_scan", decided_by="d-stage2"))
    if t.startswith("shadowfail"):
        return httpx.Response(200, json=resp(rid, "allow", [dr(rid, score=0.05),
                                                            dr(rid, "stage2", "error", reason="exception",
                                                               role="shadow")], ["stage1"]))
    if t.startswith("extra"):
        r = resp(rid, "allow", [dr(rid)], ["stage1"])
        r["debug"] = 1
        return httpx.Response(200, json=r)
    raise AssertionError(t)


ROWS = [
    {"id": "a", "text": "s1allow", "label": 0, "source": "x", "lang": "en"},
    {"id": "b", "text": "esc-flag", "label": 1, "source": "x", "lang": "en"},
    {"id": "c", "text": "overload", "label": 0, "source": "y", "lang": "de"},
    {"id": "d", "text": "timeout", "label": 1, "source": "y", "lang": "en"},
    {"id": "e", "text": "exception", "label": 1, "source": "y", "lang": "en"},
    {"id": "f", "text": "coverage", "label": 0, "source": "x", "lang": "en"},
    {"id": "g", "text": "deepscan", "label": 0, "source": "x", "lang": "en"},
    {"id": "h", "text": "shadowfail", "label": 0, "source": "x", "lang": "en"},
    {"id": "i", "text": "extra", "label": 0, "source": "x", "lang": "en"},
]


@pytest.fixture
def result():
    be = HttpGatewayBackend("http://gw.test", transport=httpx.MockTransport(handler), retries=0)
    return evaluate(ROWS, be)


def test_response_with_unknown_key_fails_classify_response_validation(result):
    assert result["n_errors"] == 1
    (err,) = result["errors"]
    assert err.startswith("bad response: not a valid ClassifyResponse") and "debug" in err
    with pytest.raises(GatewayResponseError, match="ClassifyResponse"):
        parse_gateway_response({**resp("r", "allow", [dr("r")], ["stage1"]), "debug": 1})
    with pytest.raises(GatewayResponseError, match="ClassifyResponse"):  # results must carry the request_id
        parse_gateway_response(resp("r", "allow", [dr("other")], ["stage1"]))


def test_stages_run_counts_and_escalation_per_slice(result):
    s = result["slices"]["overall"]["served"]
    assert s["stages_run"] == {"stage1": 3, "stage1+stage2": 5}
    assert s["escalated_to_stage2"] == 5 and s["n_responses"] == 8
    assert result["slices"]["source=y"]["served"]["stages_run"] == {"stage1+stage2": 3}


def test_failure_reasons_overloaded_counted_apart_from_timeout_and_exception(result):
    o = result["slices"]["overall"]["served"]
    assert o["failure_reason"] == {"exception": 1, "overloaded": 1, "timeout": 1}
    assert o["failures"] == {"overloaded": 1, "timeout_or_exception": 2}
    assert o["shadow_failure_reason"] == {"exception": 1}  # a failed shadow is not an active failure
    assert result["slices"]["lang=non_en"]["served"]["failures"] == {"overloaded": 1, "timeout_or_exception": 0}


def test_coverage_policy_decision_is_not_a_model_prediction_but_is_served(result):
    o = result["slices"]["overall"]
    cov = o["served"]["coverage_policy_applied"]
    assert cov["by_policy"] == {"None": 6, "deep_scan": 1, "quarantine": 1}
    assert cov["by_policy_decision_and_label"]["quarantine:flag:label=0"] == 1
    # model view: a (tn), b (tp), g deep_scan (tn), h (tn); c/d/e failure policy, f coverage -> not predictions
    assert o["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 3}
    assert o["policy_decisions"]["by_status"] == {"coverage_quarantine": 1, "error": 2, "timeout": 1}
    # as served (gate view): c flag -> fp, d flag -> tp, e allow -> fn, f flag -> fp
    assert o["as_served"]["confusion"] == {"tp": 2, "fp": 2, "fn": 1, "tn": 3}
    assert result["gate_view"] == "as_served"


def test_v11_style_response_without_stages_run_falls_back_to_result_stages():
    r = {"request_id": "r", "decision": "allow", "decided_by": "d", "policy_applied": None,
         "results": [{k: v for k, v in dr("r").items() if k != "failure_reason"} | {"schema_version": "1.1"}]}
    out = parse_gateway_response(r)
    assert out["stages"] == ["stage1"] and not out["stages_run_reported"] and out["model_prediction"]


def test_truncated_observe_stage1_does_not_mark_item_truncated():
    """D23: DeBERTa decides; a role="shadow" Stage 1 that hit its deadline (truncated, status timeout) is observe
    data only. It must not count as a truncation or failure of the decision in the release-gate tool."""
    def h(request: httpx.Request) -> httpx.Response:
        b = json.loads(request.content)
        rid = b["request_id"]
        return httpx.Response(200, json=resp(rid, "allow", [dr(rid, "stage2", score=0.01),
                                                            dr(rid, "stage1", "timeout", reason="timeout",
                                                               role="shadow")],
                                             ["stage2"], decided_by="d-stage2"))
    rows = [{"id": "a", "text": "x", "label": 0, "source": "x", "lang": "en"}]
    one = parse_gateway_response(h(httpx.Request("POST", "http://gw/classify",
                                                 json={"text": "x", "request_id": "r"})).json())
    assert one["truncated"] is False and one["shadow_truncated"] is True
    assert one["model_prediction"] and one["failure_reasons"] == [] and one["shadow_failure_reasons"] == ["timeout"]
    out = evaluate(rows, HttpGatewayBackend("http://gw.test", transport=httpx.MockTransport(h), retries=0))
    t = out["slices"]["overall"]["truncated"]
    assert t["n_truncated"] == 0 and t["n_truncated_model_predictions"] == 0 and t["n_shadow_truncated"] == 1
    assert out["slices"]["overall"]["served"]["failures"] == {"overloaded": 0, "timeout_or_exception": 0}
