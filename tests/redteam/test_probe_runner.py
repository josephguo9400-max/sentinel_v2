"""Probe runner against fake gateways: a 'current' gateway that ignores truncated (both placements
slip through) and a 'cascade' gateway that applies a coverage policy and scores every chunk."""
import hashlib
import json

import httpx
import pytest

from redteam.gateway_client import GatewayClient
from redteam.probe_runner import run_probes, wilson_interval
from redteam.probes import generate_probes, load_probes

PROBES = load_probes()
# The complete, actionable instruction is the exfil address (always past the head in both placements);
# the fakes key on it so "seen in the scored region" == "the injection was actually scored".
EXFIL = ".invalid"


def _result(rid, text, score, *, stage="stage1", status="ok", truncated=False, n_chunks=1):
    ok = status == "ok"
    return {"schema_version": "1.2", "request_id": rid, "detector_id": f"{stage}-det",
            "detector_version": "1.0.0", "status": status, "score": score if ok else None,
            "threshold": 0.5, "predicted_injection": (score > 0.5) if ok else None, "latency_ms": 1.0,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_tokens": 500, "n_chunks": n_chunks, "truncated": truncated, "aggregation": "max",
            "stage": stage, "role": "active", "failure_reason": None}


def current_gateway(request):
    """Scores only the head (first chunks); long inputs are truncated and the decision ignores it.
    Both placements hide the injection outside the head, so it is never scored -> allow."""
    if request.method == "GET":
        return httpx.Response(404)
    body = json.loads(request.content)
    text, rid = body["text"], body["request_id"]
    head = text[:1500]  # only the start is "seen"
    hit = EXFIL in head
    truncated = len(text) > 1500
    return httpx.Response(200, json={
        "request_id": rid, "decision": "flag" if hit else "allow",
        "decided_by": "stage1-det", "policy_applied": None, "coverage_policy_applied": None,
        "stages_run": ["stage1"],
        "results": [_result(rid, text, 0.9 if hit else 0.05, truncated=truncated,
                            n_chunks=4 if truncated else 1)]})


def cascade_gateway(request):
    """Scores the whole input when short; when it must drop content (truncated) a coverage policy
    quarantines regardless of score. The boundary placement is short enough to be fully scored here,
    so its injection fragments are seen and flagged; the middle placement is truncated -> coverage."""
    if request.method == "GET":
        return httpx.Response(404)
    body = json.loads(request.content)
    text, rid = body["text"], body["request_id"]
    truncated = len(text) > 6000  # middle probes (2600 tok) are long; boundary probes (900 tok) are not
    if truncated:
        return httpx.Response(200, json={
            "request_id": rid, "decision": "flag", "decided_by": "coverage_policy",
            "policy_applied": None, "coverage_policy_applied": "flag", "stages_run": ["stage1"],
            "results": [_result(rid, text, 0.05, truncated=True, n_chunks=4)]})
    hit = EXFIL in text
    return httpx.Response(200, json={
        "request_id": rid, "decision": "flag" if hit else "allow",
        "decided_by": "stage2-det" if hit else "stage1-det", "policy_applied": None,
        "coverage_policy_applied": None, "stages_run": ["stage1", "stage2"] if hit else ["stage1"],
        "results": [_result(rid, text, 0.2 if hit else 0.05),
                    *([_result(rid, text, 0.95, stage="stage2")] if hit else [])]})


def _client(handler):
    return GatewayClient("http://gw.invalid", transport=httpx.MockTransport(handler), on_unavailable="allow")


def test_wilson_interval_basic():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


def test_current_gateway_misses_both_placements():
    with _client(current_gateway) as c:
        rep = run_probes(c, PROBES)
    for placement in ("middle_overlong", "boundary_split"):
        s = rep["placements"][placement]
        assert s["detection_rate_as_served"]["rate"] == 0.0  # as served: nothing flagged
        assert s["benign_control_flag_rate"]["rate"] == 0.0
        assert s["unavailable"] == {"attack": 0, "control": 0}


def test_cascade_gateway_improves_detection():
    with _client(cascade_gateway) as c:
        rep = run_probes(c, PROBES)
    mid = rep["placements"]["middle_overlong"]
    bnd = rep["placements"]["boundary_split"]
    # middle: caught only by the coverage policy (truncated), not by a score
    assert mid["detection_rate_as_served"]["rate"] == 1.0
    assert mid["coverage_policy_flags"]["rate"] == 1.0
    assert mid["classifier_score_flags"]["rate"] == 0.0
    assert mid["attack_truncated"]["rate"] == 1.0
    # boundary: short enough to be fully scored, so the fragments are flagged by a score, not coverage
    assert bnd["detection_rate_as_served"]["rate"] == 1.0
    assert bnd["classifier_score_flags"]["rate"] == 1.0
    assert bnd["coverage_policy_flags"]["rate"] == 0.0
    # Controls: boundary controls are short -> benign. Middle controls are also truncated, so a blanket
    # coverage-flag policy quarantines them too -- the real false-positive cost, which the runner
    # surfaces instead of hiding (a flag-on-truncate policy flags benign long emails).
    assert bnd["benign_control_flag_rate"]["rate"] == 0.0
    assert mid["benign_control_flag_rate"]["rate"] == 1.0


def test_before_after_table_shape():
    """The owner compares the two reports; the shape supports a before/after table per placement."""
    with _client(current_gateway) as c:
        before = run_probes(c, PROBES)
    with _client(cascade_gateway) as c:
        after = run_probes(c, PROBES)
    for placement in ("middle_overlong", "boundary_split"):
        b = before["placements"][placement]["detection_rate_as_served"]["rate"]
        a = after["placements"][placement]["detection_rate_as_served"]["rate"]
        assert b == 0.0 and a == 1.0


def test_unavailable_gateway_is_not_a_detection():
    with GatewayClient("http://127.0.0.1:9", timeout_s=0.3, on_unavailable="allow") as c:
        rep = run_probes(c, PROBES)
    mid = rep["placements"]["middle_overlong"]
    assert mid["detection_rate_as_served"]["n"] == 0  # all fell back -> excluded from denominator
    assert mid["unavailable"]["attack"] == 6
    assert mid["detection_rate_as_served"]["rate"] is None


def test_report_counts_match_fixtures():
    with _client(current_gateway) as c:
        rep = run_probes(c, PROBES)
    assert rep["n_probes"] == len(PROBES) == 24
    assert rep["placements"]["middle_overlong"]["n_attack"] == 6
    assert rep["placements"]["boundary_split"]["n_control"] == 6
    assert len(rep["outcomes"]) == 24


# ---------------------------------------------------------------- end-to-end over real HTTP

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")


def test_runner_against_real_serving_app_head_only_detector():
    """The current gateway, served for real: a head-only detector flags nothing because both
    placements hide the full instruction past the scored chunks; validates ClassifyResponse on the wire."""
    import socket
    import threading
    import time

    from serving.app import create_app
    from serving.config import GatewayConfig

    class HeadDetector:
        detector_id, detector_version, threshold, max_len, cls_id, sep_id = "head", "0.1", 0.5, 512, 1, 2

        def tokenize(self, text):
            return [3 if "invalid" in w else 10 for w in text.split()]

        def infer(self, batch):
            return [0.95 if 3 in row else 0.02 for row in batch]

    app = create_app(GatewayConfig(verify_model_sha256=False, max_chunks=4), HeadDetector())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        with GatewayClient(f"http://127.0.0.1:{port}", on_unavailable="allow") as c:
            rep = run_probes(c, PROBES)
    finally:
        server.should_exit = True
        t.join(5)
    mid = rep["placements"]["middle_overlong"]
    # Real chunking drops the middle, so the 'invalid' address token is never scored -> allow.
    assert mid["detection_rate_as_served"]["n"] == 6 and mid["detection_rate_as_served"]["rate"] == 0.0
    assert mid["attack_truncated"]["rate"] == 1.0  # the gateway reports the middle probes truncated
