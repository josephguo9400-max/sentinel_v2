"""Async shadow mirror: the shadow can crash, time out, hang or be slow without changing the active response or
holding it up; results go to a JSONL log and metrics."""
import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from contracts import ClassifyResponse, DetectorResult
from fakes import FakeDetector, make_specs, words
from serving.app import create_app
from serving.config import GatewayConfig
from serving.language import FixedLanguage
from serving.registry import DetectorSpec
from serving.service import ClassifierService

TEXTS = ["hello there", "please INJECT now", words(1500, inject_at=1400), "", words(40)]


def shadow_spec(det, timeout_ms=200) -> DetectorSpec:
    return DetectorSpec(detector=det, detector_id=det.detector_id, detector_version="9.9.9-shadow",
                        threshold=det.threshold, kind="token", role="stage2", failure_policy="fail_closed",
                        timeout_ms=timeout_ms, max_input_tokens=det.max_len)


class Named(FakeDetector):
    def __init__(self, name="shadow", **kw):
        super().__init__(**kw)
        self.detector_id = name


class Inverted(Named):
    """Disagrees with the active detector on every chunk."""

    def infer(self, batch):
        return [1.0 - s for s in super().infer(batch)]


def comparable_view(out) -> dict:
    """The active response minus wall-clock latency and the bundle identity (a policy with a shadow step is a
    different bundle from the same policy without one)."""
    body = out.as_response()
    for k in ("bundle_sha256", "policy_version"):
        body.pop(k)
    for r in body["results"]:
        r.pop("latency_ms")
    return body


async def run_service(texts, shadow=None, *, config=None, settle_s=0.0):
    s2, s1 = make_specs(FakeDetector(), None)
    config = config or GatewayConfig(shadow_log_path="")
    svc = ClassifierService(s2, config, stage1=s1, shadow=shadow, language=FixedLanguage())
    await svc.start()
    try:
        outs, lats = [], []
        for i, t in enumerate(texts):
            t0 = time.perf_counter()
            outs.append(await svc.classify(t, f"r{i}"))
            lats.append(time.perf_counter() - t0)
        if settle_s:
            await asyncio.sleep(settle_s)
        return outs, lats, (svc.shadow.describe() if svc.shadow else None)
    finally:
        await svc.stop()


def baseline():
    outs, _, _ = asyncio.run(run_service(TEXTS))
    return [comparable_view(o) for o in outs]


@pytest.mark.parametrize("make", [
    lambda: Named(fail=RuntimeError("shadow crash")),
    lambda: Named(delay_s=0.5),
    lambda: Inverted(),
], ids=["crash", "timeout", "disagree"])
def test_shadow_faults_leave_active_response_identical(make, tmp_path):
    base = baseline()
    log = tmp_path / "shadow.jsonl"
    cfgx = GatewayConfig(shadow_log_path=str(log), shadow_timeout_ms=100)
    outs, _, info = asyncio.run(run_service(TEXTS, shadow_spec(make()), config=cfgx, settle_s=1.0))
    assert [comparable_view(o) for o in outs] == base
    for o in outs:
        assert all(r.role == "active" for r in o.results)  # shadow never on the wire
        ClassifyResponse.model_validate(o.as_response())
    assert info["counts"]["offered"] == len(TEXTS)


def test_shadow_crash_and_timeout_are_logged_with_failure_reason(tmp_path):
    log = tmp_path / "s.jsonl"
    cfgx = GatewayConfig(shadow_log_path=str(log), shadow_timeout_ms=100)
    asyncio.run(run_service(TEXTS[:2], shadow_spec(Named(fail=RuntimeError("boom"))), config=cfgx, settle_s=0.3))
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(recs) == 2
    for rec in recs:
        r = DetectorResult.model_validate(rec["shadow"])
        assert r.role == "shadow" and r.status == "error" and r.failure_reason == "exception" and r.truncated
        assert rec["agreement"] is None and "text" not in json.dumps(rec)
    log.unlink()
    asyncio.run(run_service(TEXTS[:1], shadow_spec(Named(delay_s=0.4)), config=cfgx, settle_s=0.6))
    (rec,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert rec["shadow"]["status"] == "timeout" and rec["shadow"]["failure_reason"] == "timeout"


def test_hung_shadow_sheds_with_overloaded_and_does_not_block_active(tmp_path):
    det = Named()
    det.hang = True
    log = tmp_path / "s.jsonl"
    cfgx = GatewayConfig(shadow_log_path=str(log), shadow_timeout_ms=50, shadow_queue=64)
    try:
        outs, lats, info = asyncio.run(run_service(TEXTS * 4, shadow_spec(det), config=cfgx, settle_s=0.3))
    finally:
        det.release.set()
    assert max(lats) < 0.5  # the shadow hangs for 10 s; active requests never wait for it
    c = info["counts"]
    assert c["timeout"] == 1 and c["overloaded"] >= 1 and det.calls_started == 1  # one stuck thread, no pile-up
    reasons = {json.loads(line)["shadow"]["failure_reason"] for line in log.read_text().splitlines()}
    assert reasons == {"timeout", "overloaded"}


def test_full_shadow_queue_drops_without_blocking():
    det = Named(delay_s=0.2)
    cfgx = GatewayConfig(shadow_log_path="", shadow_queue=1, shadow_timeout_ms=1000)
    outs, lats, info = asyncio.run(run_service(["a b"] * 10, shadow_spec(det), config=cfgx))
    assert info["counts"]["dropped"] >= 7 and max(lats) < 0.15


def test_agreement_metric_and_log(tmp_path):
    log = tmp_path / "s.jsonl"
    cfgx = GatewayConfig(shadow_log_path=str(log))
    _, _, same = asyncio.run(run_service(TEXTS, shadow_spec(Named()), config=cfgx, settle_s=0.3))
    assert same["agreement_rate"] == 1.0 and same["n_compared"] == len(TEXTS)
    _, _, inv = asyncio.run(run_service(TEXTS, shadow_spec(Inverted()), config=cfgx, settle_s=0.3))
    # Inverted chunk scores disagree on 4 of 5 texts; on the long injected text its benign chunks score 0.99
    # after inversion, so under max aggregation it also flags and agrees with the active flag.
    assert inv["agreement_rate"] == pytest.approx(0.2) and inv["counts"]["disagree"] == 4
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(recs) == 2 * len(TEXTS)
    assert {r["shadow"]["detector_version"] for r in recs} == {"9.9.9-shadow"}
    assert all(r["shadow"]["input_sha256"] and r["active"]["decision"] in ("allow", "flag") for r in recs)


def test_policy_decided_active_is_not_compared():
    s2, _ = make_specs(FakeDetector(fail=RuntimeError("active down")), stage1=False)

    async def go():
        svc = ClassifierService(s2, GatewayConfig(shadow_log_path=""), shadow=shadow_spec(Named()))
        await svc.start()
        try:
            o = await svc.classify("hello", "x")
            await asyncio.sleep(0.2)
            return o, svc.shadow.describe()
        finally:
            await svc.stop()

    o, info = asyncio.run(go())
    assert o.policy_applied == "fail_closed" and info["counts"]["not_comparable"] == 1 and info["n_compared"] == 0


def test_shadow_over_http_metrics_and_health():
    shadow = Named(fail=RuntimeError("x"))
    app = create_app(GatewayConfig(shadow_log_path=""), FakeDetector(), shadow=shadow_spec(shadow))
    with TestClient(app) as c:
        for t in TEXTS:
            assert c.post("/classify", json={"text": t}).status_code == 200
        time.sleep(0.3)
        h = c.get("/health").json()
        m = c.get("/metrics").text
    assert h["shadow"]["counts"]["exception"] == len(TEXTS)
    assert f'sentinel_shadow_results_total{{detector_id="shadow",reason="exception",status="error"}} {len(TEXTS)}.0' in m
    assert "sentinel_shadow_latency_seconds_count" in m and "sentinel_shadow_dropped_total" in m


def test_shadow_offer_cost_is_small():
    """Active-path overhead of the mirror: building the item and one put_nowait. Bounded by a loose ceiling here;
    the measured number is in serving/README.md (benchmarks under the lock)."""
    import statistics

    from serving.metrics import GatewayMetrics
    from serving.shadow import ShadowItem, ShadowMirror

    async def go():
        m = ShadowMirror(shadow_spec(Named(delay_s=0.0)), queue_size=100000, timeout_ms=100, max_chunks=1,
                         stride=384, structure_aware=True, metrics=GatewayMetrics(), log_path=None)
        await m.start()
        try:
            ts = []
            for i in range(2000):
                t0 = time.perf_counter()
                m.offer(ShadowItem(str(i), "x", {}, t0))
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts)
        finally:
            await m.stop()

    assert asyncio.run(go()) < 0.001
