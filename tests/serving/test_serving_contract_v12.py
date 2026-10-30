"""Contract v1.2 adoption: every /classify response is a contracts.ClassifyResponse, and every non-ok
DetectorResult carries failure_reason (timeout / exception / overloaded)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from contracts import ClassifyResponse
from fakes import FakeDetector
from serving.app import create_app
from serving.config import GatewayConfig
from serving.service import ClassifierService


def cfg(**kw) -> GatewayConfig:
    kw.setdefault("verify_model_sha256", False)
    return GatewayConfig(**kw)


def test_ok_response_is_v12_classify_response():
    with TestClient(create_app(cfg(), FakeDetector())) as c:
        body = c.post("/classify", json={"text": "hello", "request_id": "a"}).json()
    resp = ClassifyResponse.model_validate(body)
    assert resp.stages_run and resp.coverage_policy_applied is None
    assert all(r.schema_version == "1.2" and r.failure_reason is None for r in resp.results)


def test_request_contract_rejects_extra_and_long_request_id():
    with TestClient(create_app(cfg(), FakeDetector())) as c:
        assert c.post("/classify", json={"text": "x", "bogus": 1}).status_code == 422
        assert c.post("/classify", json={"text": "x", "request_id": "r" * 201}).status_code == 422


@pytest.mark.parametrize("kind,status,reason", [("timeout", "timeout", "timeout"),
                                                ("exception", "error", "exception")])
def test_failure_reason_on_timeout_and_exception(kind, status, reason):
    det = FakeDetector(delay_s=0.5) if kind == "timeout" else FakeDetector(fail=RuntimeError("boom"))
    with TestClient(create_app(cfg(timeout_ms=50), det)) as c:
        body = c.post("/classify", json={"text": "hello"}).json()
    resp = ClassifyResponse.model_validate(body)
    failed = [r for r in resp.results if r.status != "ok"]
    assert failed and all(r.status == status and r.failure_reason == reason for r in failed)


def test_failure_reason_overloaded():
    det = FakeDetector(tokenize_delay_s=0.05)

    async def scenario():
        service = ClassifierService(det, cfg(max_inflight=1))
        await service.start()
        try:
            return await asyncio.gather(*(service.classify("x y", str(i)) for i in range(3)))
        finally:
            await service.stop()

    outs = asyncio.run(scenario())
    reasons = sorted(str(r.failure_reason) for o in outs for r in o.results)
    assert reasons == ["None", "overloaded", "overloaded"]
    for o in outs:
        ClassifyResponse.model_validate(o.as_response())
