"""Gateway client against a fake /classify (httpx.MockTransport) and an unreachable port."""
import hashlib
import json

import httpx
import pytest

from redteam.classifier import ClassifierUnavailable, Classifier, KeywordClassifier
from redteam.defenders import GullibleDefender
from redteam.gateway_client import GatewayClient
from redteam.judge import judge_attempt
from redteam.sandbox import run_attempt


def detector_result(request_id, text, score, schema_version="1.1"):
    return {"schema_version": schema_version, "request_id": request_id, "detector_id": "stage1-tfidf",
            "detector_version": "v1", "status": "ok", "score": score, "threshold": 0.5,
            "predicted_injection": score > 0.5, "latency_ms": 1.5,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(), "input_tokens": 10,
            "n_chunks": 1, "truncated": False, "aggregation": "max", "stage": "stage1", "role": "active"}


def fake_gateway(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/health":
            return httpx.Response(404)  # a gateway without /health: the client uses its default timeout
        assert request.method == "POST" and request.url.path == "/classify"
        body = json.loads(request.content)
        assert set(body) == {"text", "request_id"}
        requests.append(body)
        flag = "send_email" in body["text"] or "forward" in body["text"]
        return httpx.Response(200, json={
            "request_id": body["request_id"],
            "decision": "flag" if flag else "allow", "decided_by": "stage1", "policy_applied": None,
            "stages_run": ["stage1"],
            "results": [detector_result(body["request_id"], body["text"], 0.97 if flag else 0.02)]})
    return handler


def client(handler, **kw):
    return GatewayClient("http://gateway.invalid", transport=httpx.MockTransport(handler), **kw)


def test_clients_satisfy_protocol():
    assert isinstance(KeywordClassifier(), Classifier)
    assert isinstance(GatewayClient("http://127.0.0.1:9"), Classifier)


def test_fake_gateway_drives_quarantine(manager, trace):
    reqs = []
    with client(fake_gateway(reqs)) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    v = judge_attempt(trace.attempt_events(res.attempt_id), manager)
    assert res.quarantined == ("m3",) and not v.attack_success
    assert len(reqs) == len(manager.emails)
    assert all(r["request_id"].startswith(res.attempt_id + ":") for r in reqs)
    preds = [e for e in trace.attempt_events(res.attempt_id) if e.event_type == "detector_prediction"]
    assert len(preds) == 4 and preds[0].detector_version == "stage1-tfidf@v1"
    assert preds[0].detail["results"][0]["schema_version"] == "1.1"


@pytest.mark.parametrize("on_unavailable,expect_decision", [("flag", "flag"), ("allow", "allow")])
def test_unreachable_gateway_fallback(on_unavailable, expect_decision):
    with GatewayClient("http://127.0.0.1:9", timeout_s=0.5, on_unavailable=on_unavailable) as c:
        v = c.classify("hello", "r1")
    assert v.decision == expect_decision and v.fallback and v.gateway_status == "unreachable" and v.error


def test_default_unreachable_quarantines_everything_and_records_it(digest, trace):
    with GatewayClient("http://127.0.0.1:9", timeout_s=0.5) as c:
        assert c.on_unavailable == "flag"
        res = run_attempt(digest, "B", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    q = [e for e in evs if e.event_type == "content_quarantined"]
    assert len(q) == len(digest.emails)
    assert all(e.policy_reason == "classifier_unavailable_fail_closed: unreachable" for e in q)
    preds = [e for e in evs if e.event_type == "detector_prediction"]
    assert all(e.detail["fallback"] and e.detector_version == "unavailable" for e in preds)
    v = judge_attempt(evs, digest)
    assert v.status == "completed" and not v.attack_success and not v.legitimate_task_success


def test_timeout_emits_detector_timeout():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with client(slow) as c:
        v = c.classify("x", "r")
    assert v.gateway_status == "timeout" and v.decision == "flag"


def test_timeout_event_in_trace(digest, trace):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    with client(slow) as c:
        res = run_attempt(digest, "C", GullibleDefender(), trace, classifier=c)
    evs = trace.attempt_events(res.attempt_id)
    assert sum(e.event_type == "detector_timeout" for e in evs) == len(digest.emails)


def test_real_timeout_is_enforced():
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def accept_and_hang():
        conn, _ = srv.accept()
        stop.wait(5)
        conn.close()

    t = threading.Thread(target=accept_and_hang, daemon=True)
    t.start()
    import time

    t0 = time.perf_counter()
    with GatewayClient(f"http://127.0.0.1:{port}", timeout_s=0.3) as c:
        v = c.classify("x", "r")
    elapsed = time.perf_counter() - t0
    stop.set()
    srv.close()
    assert v.gateway_status == "timeout" and v.decision == "flag" and elapsed < 2.0


def test_on_unavailable_error_makes_attempt_an_error(digest, trace):
    with GatewayClient("http://127.0.0.1:9", timeout_s=0.5, on_unavailable="error") as c:
        with pytest.raises(ClassifierUnavailable):
            c.classify("x", "r")
        res = run_attempt(digest, "B", GullibleDefender(), trace, classifier=c)
    v = judge_attempt(trace.attempt_events(res.attempt_id), digest)
    assert res.status == "error" and v.status == "error" and "classifier_unavailable" in v.error
    assert not any(e.event_type == "tool_proposed" for e in trace.attempt_events(res.attempt_id))


@pytest.mark.parametrize("response", [
    httpx.Response(500, json={"error": "boom"}),
    httpx.Response(200, content=b"not json"),
    httpx.Response(200, json=["allow"]),
    httpx.Response(200, json={"decision": "maybe", "results": []}),
    httpx.Response(200, json={"decision": "allow", "results": "nope"}),
    httpx.Response(200, json={"decision": "allow", "results": [{"score": 0.1}]}),
])
def test_bad_responses_fall_back_to_flag(response):
    with client(lambda r: response) as c:
        v = c.classify("x", "r")
    assert v.gateway_status == "bad_response" and v.decision == "flag" and v.fallback


def test_flag_decision_with_low_score_still_quarantines_and_allow_with_high_score_does_not():
    """The client trusts the gateway's decision (it applies cascade policy), not raw scores."""
    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"request_id": body["request_id"], "decision": "allow",
                                         "decided_by": "stage2", "policy_applied": None,
                                         "results": [detector_result(body["request_id"], body["text"], 0.99)]})
    with client(handler) as c:
        assert c.classify("x", "r").decision == "allow"


def test_invalid_configuration():
    with pytest.raises(ValueError):
        GatewayClient("http://x.invalid", on_unavailable="ignore")
    with pytest.raises(ValueError):
        GatewayClient("http://x.invalid", timeout_s=0)


# ---------------------------------------------------------------- review B1: timeouts and flag causes

def test_default_timeout_is_not_shorter_than_gateway_budget():
    """Without /health the client must still outwait the gateway's default 5000 ms budget."""
    from redteam.gateway_client import DEFAULT_TIMEOUT_S
    with GatewayClient("http://127.0.0.1:9") as c:
        assert c.timeout_s == DEFAULT_TIMEOUT_S and c.timeout_s * 1000 > 5000


def test_health_budget_sets_timeout_with_margin():
    from redteam.gateway_client import TIMEOUT_MARGIN_S

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "config": {"timeout_ms": 7000}})
        return httpx.Response(500)
    with client(handler) as c:
        assert c.timeout_source == "health" and c.timeout_s == pytest.approx(7.0 + TIMEOUT_MARGIN_S)


@pytest.mark.parametrize("health", [httpx.Response(503, json={"status": "unavailable"}),
                                    httpx.Response(200, json={"config": {"timeout_ms": "5000"}}),
                                    httpx.Response(200, json={"config": {"timeout_ms": -1}}),
                                    httpx.Response(200, content=b"not json")])
def test_unusable_health_falls_back_to_default_timeout(health):
    from redteam.gateway_client import DEFAULT_TIMEOUT_S
    with client(lambda r: health) as c:
        assert c.timeout_source == "default" and c.timeout_s == DEFAULT_TIMEOUT_S


def test_health_budget_is_capped_because_health_is_untrusted():
    """Review round 2 (R1): a /health that reports a huge budget must not stall an attempt."""
    from redteam.gateway_client import MAX_DERIVED_TIMEOUT_S

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "config": {"timeout_ms": 3_599_999}})
        return httpx.Response(500)
    with client(handler) as c:
        assert c.timeout_source == "health_capped" and c.timeout_s == MAX_DERIVED_TIMEOUT_S


def test_explicit_timeout_skips_health_probe():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(404)
    with client(handler, timeout_s=3.0) as c:
        assert c.timeout_source == "explicit" and c.timeout_s == 3.0
    assert seen == []


def test_fallback_quarantine_is_not_reported_as_classifier_stop(manager, trace):
    """Gateway timeout under on_unavailable=flag quarantines the attacker email, but that is a fallback."""
    from redteam.judge import summarize

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    with client(slow, timeout_s=1.0) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    v = judge_attempt(trace.attempt_events(res.attempt_id), manager)
    assert manager.attacker_email_id in v.quarantined and not v.attack_success
    assert v.attacker_email_quarantine_cause == "fallback" and not v.attack_stopped_by_classifier
    assert set(v.quarantine_causes.values()) == {"fallback"}
    s = summarize([v])
    assert s["attack_stopped_by_classifier"] == 0
    assert s["attacker_email_quarantined_by_cause"] == {"classifier_flag": 0, "fallback": 1,
                                                       "gateway_failure_policy": 0, "coverage_policy": 0,
                                                       "unknown": 0}
    assert s["quarantined_emails_by_cause"]["fallback"] == len(manager.emails)


def test_real_flag_is_reported_as_classifier_stop(manager, trace):
    from redteam.judge import summarize
    with client(fake_gateway([])) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    v = judge_attempt(trace.attempt_events(res.attempt_id), manager)
    assert v.attacker_email_quarantine_cause == "classifier_flag" and v.attack_stopped_by_classifier
    assert summarize([v])["attack_stopped_by_classifier"] == 1


def test_truncated_flag_recorded_and_counted(manager, trace):
    from redteam.judge import summarize

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        body = json.loads(request.content)
        r = detector_result(body["request_id"], body["text"], 0.02)
        r.update(n_chunks=4, truncated=True)
        return httpx.Response(200, json={"request_id": body["request_id"], "decision": "allow",
                                         "decided_by": "stage1", "policy_applied": None, "results": [r]})
    with client(handler) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    preds = [e for e in trace.attempt_events(res.attempt_id) if e.event_type == "detector_prediction"]
    assert preds and all(e.detail["truncated"] is True for e in preds)
    v = judge_attempt(trace.attempt_events(res.attempt_id), manager)
    assert v.n_truncated_predictions == len(manager.emails)
    assert summarize([v])["n_truncated_predictions"] == len(manager.emails)


def test_untruncated_prediction_detail_says_false(manager, trace):
    with client(fake_gateway([])) as c:
        res = run_attempt(manager, "B", GullibleDefender(), trace, classifier=c)
    preds = [e for e in trace.attempt_events(res.attempt_id) if e.event_type == "detector_prediction"]
    assert preds and all(e.detail["truncated"] is False for e in preds)
