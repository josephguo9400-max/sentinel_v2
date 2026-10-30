"""T2 (docs/design/capability-policy.md §8): the policy engine decides exactly what the legacy cascade service
decided.

Every input of tests/serving/test_cascade.py, test_stage1_observe.py, test_gateway.py and test_shadow.py is run
through serving/legacy_service.py (the frozen pre-refactor service) and through serving/service.py (the policy
engine built from the same specs by serving.policy.legacy_to_v2), in route and observe mode, in all four coverage
modes, with fakes that answer normally, hang, raise or hit the soft deadline.

Compared: every response field except `latency_ms` and the v1.3 additions (policy_id, policy_version,
bundle_sha256, unscored_by_any_step, escalation_reason, and schema_version, which the legacy copy writes as 1.2
through the shared contract). Also compared: the rows Stage 2 scored, in order, and the `path` diagnostic.

Not compared, and why: legacy `info["stage1_observe"]`'s key set (the engine adds `info["observe"]` per step) and
the observe result's `role` ("shadow" in legacy, "observe" in v1.3, which is the point of the contract change).
"""
import asyncio

import pytest

from fakes import FakeDetector, FakeStage1, RecordingDetector, make_specs, words
from serving.config import GatewayConfig
from serving.language import FixedLanguage
from serving.legacy_service import ClassifierService as LegacyService
from serving.service import ClassifierService

V13_ONLY = ("policy_id", "policy_version", "bundle_sha256", "unscored_by_any_step", "escalation_reason")


def cfg(**kw) -> GatewayConfig:
    kw.setdefault("shadow_log_path", "")
    return GatewayConfig(**kw)


def paras(n_paras: int, words_per: int = 60, inject_in=(), gray_in=(), foreign_in=()) -> str:
    """The generator of test_cascade.py / test_stage1_observe.py (inject_in accepts an int or a set)."""
    inject_in = {inject_in} if isinstance(inject_in, int) else set(inject_in)
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


# Every distinct input used by the four test files, plus the shadow suite's TEXTS.
TEXTS = [
    "", "hello", "hello there", "x y", "a b", "c", "d", "e f", "warm", "w1 w2", "w1 w2 w3", "INJECT",
    "please INJECT now", "hello INJECT", "Can we move our meeting to Thursday at 3pm?",
    words(40), words(1200), words(1500, inject_at=1400), words(3000, inject_at=1500), words(3000, inject_at=2999),
    " ".join([f"w{i % 50}" for i in range(400)] + ["INJECT"]),
    paras(3, words_per=60) + "\n\nplease INJECT now",
    paras(3, words_per=300), paras(8, words_per=300), paras(12, words_per=300), paras(40, words_per=300),
    paras(30, words_per=300, inject_in=17), paras(10, words_per=300, inject_in=5, gray_in=(2, 3, 4, 6, 7)),
    paras(10, words_per=300, inject_in=0, gray_in=(2, 3, 4, 5, 6, 7)),
    paras(10, words_per=300, gray_in=(2, 3, 4, 5, 6, 7)), paras(12, words_per=300, gray_in=tuple(range(1, 11))),
    paras(12, words_per=300, gray_in=(5,)), paras(6, words_per=300, inject_in=(0, 2)),
    paras(3, words_per=300) + "\n\nDEUTSCH w1 w2", paras(12, words_per=300, foreign_in=(4,)),
]


def view(out) -> dict:
    """The comparable part of a response: no latency, no v1.3 additions, no schema_version."""
    d = out.as_response()
    for k in V13_ONLY:
        d.pop(k, None)
    d["results"] = [{k: v for k, v in r.items() if k not in ("latency_ms", "schema_version")} for r in d["results"]]
    return d


def active_only(d: dict) -> dict:
    """The response with observe/shadow results dropped (their role differs between the engines by design)."""
    out = dict(d)
    out["results"] = [r for r in d["results"] if r["role"] == "active"]
    return out


def run_both(texts, *, config=None, stage1=True, stage1_mode="route", s1_kwargs=None, s2_kwargs=None,
             language="en", **spec_kw):
    """Both engines on the same texts with freshly built, identical fakes. Returns (legacy, policy) results."""
    config = config or cfg()
    outs = []
    rows = []
    for Service in (LegacyService, ClassifierService):
        s2det = RecordingDetector(**(s2_kwargs or {}))
        s1det = FakeStage1(**(s1_kwargs or {})) if stage1 else False
        s2, s1 = make_specs(s2det, s1det, **spec_kw)

        async def scenario():
            svc = Service(s2, config, stage1=s1, language=FixedLanguage(language), stage1_mode=stage1_mode)
            await svc.start()
            try:
                return [await svc.classify(t, f"r{i}") for i, t in enumerate(texts)]
            finally:
                await svc.stop()

        res = asyncio.run(scenario())
        outs.append(res)
        rows.append(list(s2det.rows))
    return outs, rows


def compare(texts, **kw):
    (legacy, policy), (legacy_rows, policy_rows) = run_both(texts, **kw)
    assert [active_only(view(o)) for o in policy] == [active_only(view(o)) for o in legacy]
    assert [o.info.get("path") for o in policy] == [o.info.get("path") for o in legacy]
    assert policy_rows == legacy_rows, "Stage 2 must score the same rows in the same order"
    return legacy, policy


# ---------------------------------------------------------------- the two served shapes, every coverage mode

@pytest.mark.parametrize("coverage", ["flag", "quarantine", "allow", "deep_scan"])
@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_every_input_both_modes_every_coverage(mode, coverage):
    compare(TEXTS, config=cfg(coverage_policy=coverage), stage1_mode=mode)


@pytest.mark.parametrize("coverage", ["flag", "quarantine", "allow", "deep_scan"])
def test_t2_without_stage1(coverage):
    compare(TEXTS, config=cfg(coverage_policy=coverage), stage1=False)


# ---------------------------------------------------------------- non-default decision settings

@pytest.mark.parametrize("config", [
    cfg(max_chunks=1), cfg(max_chunks=2), cfg(stage1_max_chunks=1), cfg(stage1_max_chunks=2),
    cfg(stage1_finalize_flag=True), cfg(stage1_subwindow_tokens=0), cfg(stage1_subwindow_tokens=16,
                                                                        stage1_subwindow_stride=8),
    cfg(structure_aware=False), cfg(stride=128), cfg(coverage_policy="deep_scan", deep_scan_max_chunks=1),
    cfg(coverage_policy="deep_scan", deep_scan_fallback="allow"),
    cfg(coverage_policy="deep_scan", deep_scan_fallback="quarantine"),
], ids=["max_chunks1", "max_chunks2", "s1_chunks1", "s1_chunks2", "finalize_flag", "no_subwindows", "subwindows16",
        "flat", "stride128", "deep_scan1", "deep_scan_allow", "deep_scan_quarantine"])
@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_decision_settings(config, mode):
    compare(TEXTS, config=config, stage1_mode=mode)


@pytest.mark.parametrize("gray_zone", [(0.2, 0.8), (0.02, 0.98), (0.6, 0.9), (0.96, 0.99)])
def test_t2_bands(gray_zone):
    compare(TEXTS, gray_zone=gray_zone, config=cfg(stage1_finalize_flag=True))


@pytest.mark.parametrize("language", ["en", "de"])
def test_t2_language_gate(language):
    compare(TEXTS, language=language)
    compare(TEXTS, language=language, languages=("en", "fr"))


@pytest.mark.parametrize("policy", ["fail_closed", "fail_open"])
def test_t2_decide_failure_policy(policy):
    compare(TEXTS, s2_policy=policy)


# ---------------------------------------------------------------- faults

FAULTY = [paras(6, words_per=300, inject_in=(0, 2)), "hello", words(1500, inject_at=1400), "", "w1 w2"]


@pytest.mark.parametrize("mode", ["route", "observe"])
@pytest.mark.parametrize("coverage", ["flag", "allow", "deep_scan"])
def test_t2_stage1_raises(mode, coverage):
    compare(FAULTY, s1_kwargs={"fail": RuntimeError("s1 boom")}, stage1_mode=mode,
            config=cfg(coverage_policy=coverage))


@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_stage1_times_out(mode):
    compare(FAULTY, s1_kwargs={"delay_s": 0.3}, s1_timeout_ms=50, stage1_mode=mode)


@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_stage1_hits_the_soft_deadline(mode):
    # A per-chunk delay with a short budget: some chunks are scored, the rest stay uncleared (not a failure).
    compare([paras(12, words_per=300), paras(40, words_per=300)], s1_kwargs={"delay_s": 0.02}, s1_timeout_ms=60,
            config=cfg(stage1_deadline_margin_ms=20), stage1_mode=mode)


@pytest.mark.parametrize("mode", ["route", "observe"])
@pytest.mark.parametrize("policy", ["fail_closed", "fail_open"])
def test_t2_stage2_raises(mode, policy):
    compare(FAULTY, s2_kwargs={"fail": RuntimeError("s2 boom")}, s2_policy=policy, stage1_mode=mode)


@pytest.mark.parametrize("mode", ["route", "observe"])
@pytest.mark.parametrize("policy", ["fail_closed", "fail_open"])
def test_t2_stage2_times_out(mode, policy):
    compare(FAULTY, s2_kwargs={"delay_s": 0.5}, s2_timeout_ms=60, s2_policy=policy, stage1_mode=mode)


@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_prepare_times_out(mode):
    compare(["hello", paras(3, words_per=300)], s2_kwargs={"tokenize_delay_s": 0.4}, s2_timeout_ms=60,
            stage1_mode=mode)


@pytest.mark.parametrize("mode", ["route", "observe"])
def test_t2_stage2_hangs_until_the_budget_ends(mode):
    """A hung ONNX call cannot be cancelled: the request times out and the worker stays busy (both engines)."""
    for Service, expect in ((LegacyService, None), (ClassifierService, None)):
        det = FakeDetector()
        det.hang = True
        s2, s1 = make_specs(det, FakeStage1(), s2_timeout_ms=80)

        async def scenario():
            svc = Service(s2, cfg(), stage1=s1, language=FixedLanguage("en"), stage1_mode=mode)
            await svc.start()
            try:
                return await svc.classify("hello INJECT", "hang")
            finally:
                det.release.set()
                await svc.stop()

        out = asyncio.run(scenario())
        got = (out.decision, out.policy_applied, out.results[-1].status if mode == "route" else out.results[0].status)
        expect = expect or got
        assert got == expect == ("flag", "fail_closed", "timeout")


# ---------------------------------------------------------------- overload (admission control)

@pytest.mark.parametrize("mode", ["route", "observe"])
@pytest.mark.parametrize("policy", ["fail_closed", "fail_open"])
def test_t2_overloaded_requests(mode, policy):
    views = []
    for Service in (LegacyService, ClassifierService):
        s2, s1 = make_specs(RecordingDetector(delay_s=0.2), FakeStage1(), s2_policy=policy)

        async def scenario():
            svc = Service(s2, cfg(max_inflight=1), stage1=s1, language=FixedLanguage("en"), stage1_mode=mode)
            await svc.start()
            try:
                first = asyncio.create_task(svc.classify("w1 w2 INJECT", "first"))
                await asyncio.sleep(0.02)
                shed = await svc.classify("w1 w2", "shed")
                return shed, await first
            finally:
                await svc.stop()

        shed, _ = asyncio.run(scenario())
        assert shed.info["path"] == "overloaded"
        views.append(active_only(view(shed)))
    assert views[1] == views[0]


# ---------------------------------------------------------------- the engine's own additions

GRAY_MANY = paras(12, words_per=300, gray_in=tuple(range(1, 11)))  # escalates, more chunks than the decide budget


def test_policy_engine_reports_bundle_identity_and_unscored_flag():
    _, policy = compare(["w1 w2", GRAY_MANY, "please INJECT now"])
    for o in policy:
        assert o.policy_id == "injected" and o.bundle_sha256 and len(o.bundle_sha256) == 64
        assert o.unscored_by_any_step is not None
    assert policy[1].unscored_by_any_step is True and policy[1].coverage_policy_applied == "flag"
    assert policy[0].unscored_by_any_step is False and policy[2].unscored_by_any_step is False


def test_unscored_by_any_step_is_set_for_a_flag_backed_by_a_score():
    """coverage_policy_applied is None for a positive flag, so only unscored_by_any_step shows the gap
    (docs/contract_requests.md, `unscored_by_any_stage`)."""
    _, policy = compare([paras(6, words_per=300, inject_in=(0,))], stage1=False, config=cfg(max_chunks=2))
    assert policy[0].decision == "flag" and policy[0].coverage_policy_applied is None
    assert policy[0].unscored_by_any_step is True


@pytest.mark.parametrize("text,config,reason", [
    (GRAY_MANY, cfg(), "band"),
    (paras(12, words_per=300), cfg(stage1_max_chunks=1), "coverage"),
])
def test_escalation_reason_in_route_mode(text, config, reason):
    (_, policy), _ = run_both([text], config=config)
    assert policy[0].escalation_reason == reason


def test_escalation_reason_language_and_gate_failure():
    (_, policy), _ = run_both([paras(3, words_per=300)], language="de")
    assert policy[0].escalation_reason == "language"
    (_, policy), _ = run_both(["hello"], s1_kwargs={"fail": RuntimeError("boom")})
    assert policy[0].escalation_reason == "gate_failed"


def test_observe_mode_reports_role_observe_and_no_escalation_reason():
    (legacy, policy), _ = run_both(["w1 w2"], stage1_mode="observe")
    assert [r.role for r in legacy[0].results] == ["active", "shadow"]
    assert [r.role for r in policy[0].results] == ["active", "observe"]
    assert policy[0].escalation_reason is None and policy[0].results[-1].stage == "stage1"
