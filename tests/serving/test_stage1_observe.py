"""Observe-only Stage 1 (registry stage1_mode="observe", owner decision D23) and the anytime Stage 1 deadline.

Observe mode contract under test:
- Stage 1 never finalizes, clears or reorders anything: the response minus its role="shadow" result is the
  response of a gateway built without Stage 1 (every field except latency_ms), and Stage 2 scores the same rows.
- Stage 1 still runs and is reported as one DetectorResult(stage="stage1", role="shadow") after the active results;
  stages_run does not list it.
- A Stage 1 timeout, exception or hang in observe mode changes neither the decision nor the coverage outcome.

Fakes (tests/serving/fakes.py): Stage 1 scores a text 0.95 if it contains INJECT, 0.5 if GRAY, else 0.01, gray
zone [0.2, 0.8]; Stage 2 scores a chunk 0.99 if it contains the INJECT token.
"""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from artifact_factory import (ROOT, extra_tokenizers_cfg, write_registry, write_stage1_artifact,
                              write_token_artifact)
from contracts import ClassifyResponse
from fakes import FakeDetector, FakeStage1, RecordingDetector, make_specs
from serving.app import create_app
from serving.config import GatewayConfig
from serving.language import FixedLanguage
from artifact_factory import known_tokenizers
from serving.registry import RegistryError, load_registry, read_registry
from serving.service import ClassifierService


def cfg(**kw) -> GatewayConfig:
    kw.setdefault("shadow_log_path", "")
    return GatewayConfig(**kw)


class MapLanguage:
    def __init__(self, mapping: dict[str, str], default: str = "en"):
        self.mapping, self.default = mapping, default
        self.calls = 0

    def describe(self):
        return {"method": "map"}

    def detect(self, text):
        self.calls += 1
        for k, v in self.mapping.items():
            if k in text:
                return v, 1.0
        return self.default, 1.0


class SlowStage1(FakeStage1):
    """Each infer call (one per chunk) sleeps `delay_s`; optionally hangs on every call."""

    def __init__(self, delay_s: float, **kw):
        super().__init__(**kw)
        self.delay_s = delay_s
        self.calls = 0

    def infer(self, batch):
        self.calls += 1
        return super().infer(batch)


def paras(n_paras: int, words_per: int = 60, inject_in=(), gray_in=(), foreign_in=()) -> str:
    out = []
    for p in range(n_paras):
        ws = [f"w{(p * 7 + i) % 50}" for i in range(words_per)]
        if p in inject_in:
            ws[words_per // 2] = "INJECT"
        if p in gray_in:
            ws[1] = "GRAY"
        if p in foreign_in:
            ws[2] = "DEUTSCH"
        out.append(" ".join(ws))
    return "\n\n".join(out)


# Inputs: short, gray, positive, long (truncated by the Stage 2 budget), long with the injection in the middle
# (beyond the first/last chunks), a foreign-language chunk, quoted/signature/hidden structure.
INPUTS = [
    "w1 w2 w3",
    "w1 GRAY w3",
    "w1 INJECT w3",
    paras(12, words_per=300),                          # 7+ chunks, all benign -> Stage 2 budget truncates
    paras(12, words_per=300, inject_in=(6,)),          # injection in a middle chunk
    paras(12, words_per=300, gray_in=(3, 9)),
    paras(3, words_per=300, foreign_in=(1,)),
    "w1 w2\n\n> w3 INJECT w4\n> w5\n\n-- \nw6 w7",
    "<p>w1 w2</p><div style=\"display:none\">w3 INJECT</div><p>w4</p>",
    "",
]


def run(texts, *, stage1, mode, config=None, language=None, stage2=None, **spec_kw):
    s2, s1 = make_specs(stage2 or RecordingDetector(), stage1, **spec_kw)
    config = config or cfg()

    async def scenario():
        svc = ClassifierService(s2, config, stage1=s1, language=language or MapLanguage({"DEUTSCH": "de"}),
                                stage1_mode=mode)
        await svc.start()
        try:
            return [await svc.classify(t, f"r{i}") for i, t in enumerate(texts)], svc
        finally:
            await svc.stop()

    outs, svc = asyncio.run(scenario())
    for o in outs:
        ClassifyResponse.model_validate(o.as_response())
    return outs, s2.detector, svc


def active_view(out) -> dict:
    """The response without role="shadow" results and without latency_ms."""
    d = out.as_response()
    d["results"] = [{k: v for k, v in r.items() if k != "latency_ms"} for r in d["results"] if r["role"] != "shadow"]
    return d


def no_stage1(texts, **kw):
    return run(texts, stage1=False, mode="route", **kw)


# ----------------------------------------------------------------------------------------- observe mode


def test_observe_stage1_never_finalizes_and_is_reported_as_shadow():
    (o,), det, _ = run(["w1 w2 w3"], stage1=FakeStage1(), mode="observe")
    assert o.decision == "allow" and o.decided_by == "fake" and o.stages_run == ["stage2"]
    assert det.calls_started == 1  # Stage 2 scored it although Stage 1 would have finalized allow
    assert [(r.stage, r.role) for r in o.results] == [("stage2", "active"), ("stage1", "shadow")]
    s1 = o.results[1]
    assert s1.status == "ok" and s1.score == pytest.approx(0.01) and s1.detector_id == "fake-s1"
    assert o.info["stage1_observe"]["outcome"] == "would_finalize_allow"
    # The same input in route mode is finalized by Stage 1: the observe test above is not vacuous.
    (r,), det_r, _ = run(["w1 w2 w3"], stage1=FakeStage1(), mode="route")
    assert r.decided_by == "fake-s1" and r.stages_run == ["stage1"] and det_r.calls_started == 0


@pytest.mark.parametrize("coverage", ["flag", "allow", "quarantine", "deep_scan"])
def test_observe_decisions_equal_gateway_without_stage1(coverage):
    config = cfg(coverage_policy=coverage, max_chunks=3)
    obs, det_obs, _ = run(INPUTS, stage1=FakeStage1(), mode="observe", config=config)
    ref, det_ref, _ = no_stage1(INPUTS, config=config)
    assert [active_view(o) for o in obs] == [active_view(o) for o in ref]
    assert det_obs.rows == det_ref.rows  # same chunks, same order, same Stage 2 calls
    for o in obs:
        assert o.stages_run == [r.stage for r in o.results if r.role == "active"]
        assert [r.role for r in o.results].count("shadow") == 1 and o.results[-1].stage == "stage1"
    # Long inputs really are truncated by the Stage 2 budget (so coverage policy is exercised), and route mode
    # makes different routing choices on this set.
    assert any(o.coverage_policy_applied is not None for o in obs)
    routed, det_routed, _ = run(INPUTS, stage1=FakeStage1(), mode="route", config=config)
    assert det_routed.rows != det_ref.rows


@pytest.mark.parametrize("fault", ["timeout_hard", "timeout_partial", "exception"])
def test_observe_stage1_failure_never_changes_decision_or_coverage(fault):
    if fault == "timeout_hard":  # first chunk alone outlives the 40 ms budget
        stage1, kw = SlowStage1(0.3), {"s1_timeout_ms": 40}
    elif fault == "timeout_partial":  # anytime: some chunks scored before the soft deadline
        stage1, kw = SlowStage1(0.01), {"s1_timeout_ms": 40}
    else:
        stage1, kw = FakeStage1(fail=RuntimeError("boom")), {}
    config = cfg(stage1_deadline_margin_ms=15, max_chunks=3)
    obs, _, _ = run(INPUTS, stage1=stage1, mode="observe", config=config, **kw)
    ref, _, _ = no_stage1(INPUTS, config=config)
    assert [active_view(o) for o in obs] == [active_view(o) for o in ref]
    statuses = {o.results[-1].status for o in obs}
    if fault == "exception":
        assert statuses == {"error"} and all(o.results[-1].failure_reason == "exception" for o in obs)
    elif fault == "timeout_hard":
        assert "timeout" in statuses
    else:
        long_obs = obs[3]
        assert long_obs.results[-1].status == "ok" and long_obs.results[-1].truncated
        assert long_obs.info["stage1_observe"]["deadline_hit"] is True
        assert long_obs.info["stage1_observe"]["outcome"] == "would_escalate"


def test_observe_stage1_adds_at_most_its_budget_after_the_decision():
    stage1 = SlowStage1(2.0)  # hangs far past its 50 ms budget
    t = time.perf_counter()
    (o,), _, svc = run(["w1 w2 w3"], stage1=stage1, mode="observe", s1_timeout_ms=50)
    elapsed = time.perf_counter() - t
    assert o.decision == "allow" and o.results[-1].status == "timeout" and o.results[-1].role == "shadow"
    assert o.results[-1].latency_ms < 500
    assert svc.request_budget_ms == 5000 + 50
    assert elapsed < 2.0 + 1.0  # stop() does not wait for the abandoned thread; generous bound for CI


def test_observe_overloaded_request_reports_stage1_after_stage2():
    s2, s1 = make_specs(FakeDetector(), FakeStage1())

    async def scenario():
        svc = ClassifierService(s2, cfg(max_inflight=1), stage1=s1, language=FixedLanguage("en"),
                                stage1_mode="observe")
        await svc.start()
        try:
            svc.inflight = 1  # the only slot is taken
            return await svc.classify("w1", "r")
        finally:
            svc.inflight = 0
            await svc.stop()

    o = asyncio.run(scenario())
    ClassifyResponse.model_validate(o.as_response())
    assert o.stages_run == ["stage2"] and o.policy_applied == "fail_closed"
    assert [(r.stage, r.role, r.failure_reason) for r in o.results] == [
        ("stage2", "active", "overloaded"), ("stage1", "shadow", "overloaded")]


def test_observe_mode_metrics_and_health_via_injected_app():
    det, s1 = FakeDetector(), FakeStage1()
    app = create_app(cfg(), det, stage1=s1, language=FixedLanguage("en"), stage1_mode="observe")
    with TestClient(app) as c:
        body = c.post("/classify", json={"text": "w1 w2", "request_id": "x"}).json()
        h = c.get("/health").json()
        m = c.get("/metrics").text
    resp = ClassifyResponse.model_validate(body)
    assert resp.stages_run == ["single"] and [r.role for r in resp.results] == ["active", "shadow"]
    assert h["stage1_mode"] == "observe" and h["config"]["timeout_ms"] == 5000 + h["config"]["stage1_timeout_ms"]
    assert 'sentinel_stage1_observed_total{outcome="would_finalize_allow"} 1.0' in m


# ----------------------------------------------------------------------------------------- anytime deadline


def test_route_partial_stage1_clears_only_scored_chunks_and_never_finalizes():
    text = paras(12, words_per=300)
    stage1 = SlowStage1(0.01)
    (o,), _, svc = run([text], stage1=stage1, mode="route", s1_timeout_ms=40,
                       config=cfg(stage1_deadline_margin_ms=15))
    s1 = o.results[0]
    n = o.info["n_chunks"]
    assert s1.status == "ok" and s1.truncated and 1 <= s1.n_chunks < n
    assert o.info["stage1_deadline_hit"] is True and o.info["stage1_cleared"] == s1.n_chunks
    assert o.stages_run == ["stage1", "stage2"] and o.info["path"] != "stage1_allow"
    assert stage1.calls == s1.n_chunks  # the thread stopped at the soft deadline instead of running on


def test_deadline_frees_the_stage1_thread_for_the_next_request():
    """Pre-D23, a timed-out Stage 1 job kept the single Stage 1 thread, so the next request's Stage 1 queued behind
    it and timed out too. With the soft deadline the job returns within one chunk of it."""
    long = paras(12, words_per=300)
    (a, b), _, _ = run([long, "w1 w2 w3"], stage1=SlowStage1(0.01), mode="route", s1_timeout_ms=40,
                       config=cfg(stage1_deadline_margin_ms=15))
    assert a.results[0].status == "ok" and a.results[0].truncated
    assert b.results[0].status == "ok" and b.decided_by == "fake-s1"


def test_margin_is_capped_at_half_the_budget():
    s2, s1 = make_specs(FakeDetector(), FakeStage1(), s1_timeout_ms=20)
    svc = ClassifierService(s2, cfg(stage1_deadline_margin_ms=500), stage1=s1, language=FixedLanguage("en"))
    assert svc.s1_soft_budget_s == pytest.approx(0.010)


def test_language_is_identified_only_for_chunks_below_gray_low():
    lang = MapLanguage({})
    text = paras(6, words_per=300, inject_in=(0, 2))
    (o,), _, _ = run([text], stage1=FakeStage1(), mode="route", language=lang)
    n = o.info["n_chunks"]
    above = sum(v is None for v in o.info["stage1_langs"])
    assert above >= 1 and lang.calls == n - above


# ----------------------------------------------------------------------------------------- registry


def test_registry_stage1_mode_is_validated(tmp_path):
    p = write_registry(tmp_path, stage2="baseline/x", stage1_mode="observe")
    assert read_registry(p)["stage1_mode"] == "observe"
    p = write_registry(tmp_path, stage2="baseline/x")  # no stage1: the key may be omitted
    assert read_registry(p)["stage1_mode"] is None
    p = write_registry(tmp_path, stage2="baseline/x", stage1_mode="shadow")
    with pytest.raises(RegistryError) as exc:
        read_registry(p)
    assert exc.value.code == "registry_invalid"


def test_stage1_without_stage1_mode_is_rejected_not_routed(tmp_path):
    """M5: no silent default. Stage 1 is rejected (stage1_mode_missing), Stage 2 serves alone, health is degraded,
    and a reload to such a registry is refused."""
    write_token_artifact(tmp_path, "baseline/tiny")
    write_stage1_artifact(tmp_path, "candidates/s1")
    reg = write_registry(tmp_path, stage1="candidates/s1", stage2="baseline/tiny")
    loaded = load_registry(reg, tmp_path, known_tokenizers=known_tokenizers())
    assert "stage1" not in loaded.specs and "stage2" in loaded.specs
    assert [(r.pointer, r.code) for r in loaded.rejections] == [("stage1", "stage1_mode_missing")]
    config = GatewayConfig(registry_path=str(reg), artifacts_root=str(tmp_path),
                           extra_tokenizers=extra_tokenizers_cfg(), shadow_log_path="")
    with TestClient(create_app(config, language=FixedLanguage("en"))) as c:
        h = c.get("/health").json()
        body = ClassifyResponse.model_validate(c.post("/classify", json={"text": "w1 w2", "request_id": "r"}).json())
        write_registry(tmp_path, stage1="candidates/s1", stage1_mode="observe", stage2="baseline/tiny")
        assert c.post("/admin/reload").status_code == 200
        write_registry(tmp_path, stage1="candidates/s1", stage2="baseline/tiny")
        r = c.post("/admin/reload")
        h2 = c.get("/health").json()
    assert h["status"] == "degraded" and h["stage1"] is None
    assert h["registry"]["rejections"][0]["code"] == "stage1_mode_missing"
    assert body.stages_run == ["stage2"] and len(body.results) == 1
    assert r.status_code == 409 and r.json()["rejections"][0]["code"] == "stage1_mode_missing"
    assert h2["stage1_mode"] == "observe"  # last good config keeps serving


def test_historical_integration_registry_is_rejected_and_its_route_copy_loads():
    old = ROOT / "reports" / "cascade-and-shadow" / "registry_integration.json"
    new = ROOT / "reports" / "cascade-and-shadow" / "registry_integration_route.json"
    assert "stage1_mode" not in json.loads(old.read_text())
    assert read_registry(old)["stage1_mode"] is None
    a, b = json.loads(old.read_text()), json.loads(new.read_text())
    assert b.pop("stage1_mode") == "route" and {k: v for k, v in a.items() if k != "comment"} == {
        k: v for k, v in b.items() if k != "comment"}


def test_committed_registry_is_deberta_decides_with_stage1_observe():
    reg = json.loads((ROOT / "artifacts" / "registry.json").read_text())
    assert reg["stage1_mode"] == "observe" and reg["stage1"] == "candidates/stage1-tfidf-v2.1"
    assert reg["stage2"] == "baseline/deberta-pi-v2" and reg["shadow"] is None


def test_gateway_from_registry_in_observe_mode(tmp_path):
    write_token_artifact(tmp_path, "baseline/tiny")
    write_stage1_artifact(tmp_path, "candidates/s1")
    write_registry(tmp_path, stage1="candidates/s1", stage1_mode="observe", stage2="baseline/tiny")
    config = GatewayConfig(registry_path=str(tmp_path / "registry.json"), artifacts_root=str(tmp_path),
                           extra_tokenizers=extra_tokenizers_cfg(), shadow_log_path="")
    with TestClient(create_app(config, language=FixedLanguage("en"))) as c:
        h = c.get("/health").json()
        body = c.post("/classify", json={"text": "w1 w2 w3", "request_id": "r"}).json()
        write_registry(tmp_path, stage1="candidates/s1", stage1_mode="route", stage2="baseline/tiny")
        assert c.post("/admin/reload").status_code == 200
        routed = c.post("/classify", json={"text": "w1 w2 w3", "request_id": "r2"}).json()
    assert h["stage1_mode"] == "observe" and h["registry"]["pointers"]["stage1_mode"] == "observe"
    resp = ClassifyResponse.model_validate(body)
    assert resp.stages_run == ["stage2"] and resp.decided_by == "tiny-token"
    assert [(r.stage, r.role) for r in resp.results] == [("stage2", "active"), ("stage1", "shadow")]
    assert ClassifyResponse.model_validate(routed).stages_run[0] == "stage1"


# ----------------------------------------------------------------------------------------- concurrency (M1)


def run_concurrent(texts_and_delays, *, stage1, mode, config):
    """Send each text after its delay (seconds from start), concurrently; return outcomes in input order."""
    s2, s1 = make_specs(FakeDetector(), stage1)

    async def scenario():
        svc = ClassifierService(s2, config, stage1=s1, language=FixedLanguage("en"), stage1_mode=mode)
        await svc.start()

        async def one(i, text, delay):
            await asyncio.sleep(delay)
            return await svc.classify(text, f"r{i}")

        try:
            return await asyncio.gather(*(one(i, t, d) for i, (t, d) in enumerate(texts_and_delays)))
        finally:
            await svc.stop()

    outs = asyncio.run(scenario())
    for o in outs:
        ClassifyResponse.model_validate(o.as_response())
    return outs


def test_observe_stage1_does_not_hold_admission_against_other_requests():
    """The verifier's M1 case: max_inflight=1, request B arrives 150 ms after A, while A's observe-only Stage 1 (300 ms)
    is still running after A's decision. Before the fix A still held the admission slot, so B was shed as overloaded
    and Stage 2's fail_closed flagged it. B must get the decision a gateway without Stage 1 gives it."""
    reqs = [("w1 w2 w3", 0.0), ("w4 w5 w6", 0.15)]
    config = cfg(max_inflight=1)
    ref = run_concurrent(reqs, stage1=False, mode="route", config=config)
    obs = run_concurrent(reqs, stage1=FakeStage1(delay_s=0.3), mode="observe", config=config)
    assert [(o.decision, o.policy_applied, o.info["path"]) for o in ref] == [("allow", None, "stage2_allow")] * 2
    assert [active_view(o) for o in obs] == [active_view(o) for o in ref]
    assert obs[0].results[-1].status == "ok" and obs[0].info["stage1_observe"]["outcome"] == "would_finalize_allow"
    # A's observation still holds the single observe slot when B is decided: B's is dropped, not queued.
    assert obs[1].info["stage1_observe"]["outcome"] == "dropped"
    assert obs[1].results[-1].failure_reason == "overloaded" and obs[1].results[-1].role == "shadow"


def test_observe_work_is_bounded_and_drops_instead_of_queueing():
    reqs = [(f"w{i} w2 w3", 0.01 * i) for i in range(4)]
    config = cfg(stage1_observe_concurrency=1)
    ref = run_concurrent(reqs, stage1=False, mode="route", config=config)
    obs = run_concurrent(reqs, stage1=FakeStage1(delay_s=0.2), mode="observe", config=config)
    assert [active_view(o) for o in obs] == [active_view(o) for o in ref]
    outcomes = [o.info["stage1_observe"]["outcome"] for o in obs]
    assert outcomes.count("dropped") == 3 and outcomes[0] == "would_finalize_allow"


def test_observe_failures_are_labelled_shadow_in_metrics():
    det = FakeDetector()
    app = create_app(cfg(), det, stage1=FakeStage1(fail=RuntimeError("boom")), language=FixedLanguage("en"),
                     stage1_mode="observe")
    with TestClient(app) as c:
        c.post("/classify", json={"text": "w1 w2", "request_id": "x"})
        m = c.get("/metrics").text
    assert ('sentinel_detector_failures_total{detector_id="fake-s1",reason="exception",role="shadow",'
            'status="error"} 1.0') in m
    assert 'role="active",status="error"' not in m
