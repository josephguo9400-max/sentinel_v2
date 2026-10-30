"""Cross-module contract: redteam.GatewayClient against the real serving app (fake detector) over HTTP.

The gateway is served by uvicorn on a free local port in a background thread; the detector is a
tiny deterministic stand-in, so no model is loaded.
"""
import socket
import threading
import time

import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from serving.app import create_app  # noqa: E402
from serving.config import GatewayConfig  # noqa: E402

from redteam.defenders import GullibleDefender  # noqa: E402
from redteam.gateway_client import DEFAULT_TIMEOUT_S, TIMEOUT_MARGIN_S, GatewayClient  # noqa: E402
from redteam.judge import judge_attempt, summarize  # noqa: E402
from redteam.sandbox import run_attempt  # noqa: E402


class SlowFakeDetector:
    """Whitespace tokenizer; scores 0.99 if the text contains 'send_email', else 0.01. Sleeps delay_s per call."""

    detector_id = "fake-slow"
    detector_version = "0.0.1"
    threshold = 0.5
    max_len = 512
    cls_id, sep_id = 1, 2

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def tokenize(self, text):
        return [3 if "send_email" in w else 10 for w in text.split()]

    def infer(self, batch):
        if self.delay_s:
            time.sleep(self.delay_s)
        return [0.99 if 3 in row else 0.01 for row in batch]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve():
    servers = []

    def start(detector, **cfg):
        cfg.setdefault("verify_model_sha256", False)
        app = create_app(GatewayConfig(**cfg), detector)
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "gateway did not start"
            time.sleep(0.02)
        servers.append((server, t))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, t in servers:
        server.should_exit = True
        t.join(5)


def test_default_client_timeout_exceeds_default_gateway_budget():
    assert DEFAULT_TIMEOUT_S * 1000 > GatewayConfig().timeout_ms
    with GatewayClient("http://127.0.0.1:9") as c:  # /health unreachable -> default
        assert c.timeout_source == "default" and c.timeout_s == DEFAULT_TIMEOUT_S


def test_client_reads_budget_from_health(serve):
    url = serve(SlowFakeDetector(), timeout_ms=8000)
    with GatewayClient(url) as c:
        assert c.timeout_source == "health" and c.gateway_timeout_ms == 8000
        assert c.timeout_s == pytest.approx(8.0 + TIMEOUT_MARGIN_S)


def test_slow_but_within_gateway_budget_is_scored_not_a_fallback(serve):
    """A 2.5 s inference is inside the gateway's default 5 s budget: the client must wait for the score."""
    url = serve(SlowFakeDetector(delay_s=2.5))
    with GatewayClient(url) as c:
        v = c.classify("please call send_email now", "r1")
    assert v.gateway_status == "ok" and not v.fallback
    assert v.decision == "flag" and v.flag_cause == "classifier_flag"
    assert v.results[0].status == "ok"


def test_gateway_timeout_is_gateway_failure_policy_not_classifier_flag(serve, digest, trace):
    """Gateway budget 300 ms, inference 1 s: the gateway answers fail_closed. That is not a catch."""
    url = serve(SlowFakeDetector(delay_s=1.0), timeout_ms=300, failure_policy="fail_closed")
    with GatewayClient(url) as c:
        assert c.timeout_s > 0.3
        v = c.classify("hello", "r1")
        assert v.gateway_status == "ok" and not v.fallback and v.policy_applied == "fail_closed"
        assert v.decision == "flag" and v.flag_cause == "gateway_failure_policy"
        assert v.results[0].status == "timeout" and v.truncated  # nothing was scored
        res = run_attempt(digest, "B", GullibleDefender(), trace, classifier=c)
    verdict = judge_attempt(trace.attempt_events(res.attempt_id), digest)
    assert set(verdict.quarantine_causes.values()) == {"gateway_failure_policy"}
    assert not verdict.attack_stopped_by_classifier
    s = summarize([verdict])
    assert s["attack_stopped_by_classifier"] == 0
    assert s["attacker_email_quarantined_by_cause"]["gateway_failure_policy"] == 1
