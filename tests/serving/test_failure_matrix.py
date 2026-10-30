"""T6 (docs/design/capability-policy.md §8): the failure matrix, with fakes.

{gate, decide single, decide parallel member} x {timeout, exception, overloaded} x {FAIL_OPEN, FAIL_CLOSED,
FALLBACK whose target succeeds, FALLBACK whose target fails too}. MATRIX below is the documented outcome per cell;
the test asserts it and prints the table with `-s`.

How each fault is produced (no model, no patching of the engine):
- timeout      the detector sleeps past the step's budget
- exception    the detector raises
- overloaded   gate: admission control sheds the request (its characters exceed max_inflight_chars), which is
               the only way a gate step reports overloaded, since it does not go through the micro-batcher;
               decide / parallel: the batcher's queue is full (max_queue smaller than the request's chunks).

The two invariants the design names:
- FALLBACK is never started on `overloaded` (a second model under load shedding adds load): the fallback's
  `terminal` applies directly and the fallback detector is never called.
- an ASYNC_ONLY step (observe, shadow) never changes any decision field.

Gate semantics worth stating: a gate's FAIL_OPEN means the gate step itself votes allow, which in a sequential
shape finalizes the request (the decider never runs); its FALLBACK targets a decide member, which is today's
"Stage 1 failure clears nothing, escalate". Both are reached by the cells below.
"""
import asyncio
import copy

import pytest

from fakes import FakeDetector, FakeStage1, words
from policy_helpers import default_caps, member, policy, ref, service, spec
from serving.config import GatewayConfig

BENIGN = "w1 w2 w3"
MANY_CHUNKS = words(3000)  # several decide chunks: enough to overflow a max_queue of 1

CAPS = default_caps()
GATE_KEY, DECIDE_KEY, SECOND_KEY, FB_KEY = ("fake-s1", "0.0.1"), ("fake", "0.0.1"), ("fake-b", "0.0.1"), \
    ("fake-fb", "0.0.1")

# (step, fault, mode) -> (decision, policy_applied, fallback_detector_ran)
MATRIX: dict[tuple[str, str, str], tuple[str, str | None, bool]] = {}
for _f in ("timeout", "exception"):
    MATRIX[("gate", _f, "FAIL_OPEN")] = ("allow", "fail_open", False)
    MATRIX[("gate", _f, "FAIL_CLOSED")] = ("flag", "fail_closed", False)
    MATRIX[("gate", _f, "FALLBACK_ok")] = ("allow", None, True)  # escalates; the decider allows this text
    MATRIX[("gate", _f, "FALLBACK_fails")] = ("flag", "fail_closed", True)  # terminal FAIL_CLOSED
    MATRIX[("decide", _f, "FAIL_OPEN")] = ("allow", "fail_open", False)
    MATRIX[("decide", _f, "FAIL_CLOSED")] = ("flag", "fail_closed", False)
    MATRIX[("decide", _f, "FALLBACK_ok")] = ("allow", None, True)
    MATRIX[("decide", _f, "FALLBACK_fails")] = ("flag", "fail_closed", True)
    # any_flag: an allow needs both members to allow, so the failed member's fail-open vote is load-bearing
    # and the decision is not model-backed (policy_applied is set).
    MATRIX[("parallel", _f, "FAIL_OPEN")] = ("allow", "fail_open", False)
    MATRIX[("parallel", _f, "FAIL_CLOSED")] = ("flag", "fail_closed", False)
    MATRIX[("parallel", _f, "FALLBACK_ok")] = ("allow", None, True)
    MATRIX[("parallel", _f, "FALLBACK_fails")] = ("flag", "fail_closed", True)
for _s in ("gate", "decide", "parallel"):
    MATRIX[(_s, "overloaded", "FAIL_OPEN")] = ("allow", "fail_open", False)
    MATRIX[(_s, "overloaded", "FAIL_CLOSED")] = ("flag", "fail_closed", False)
    # No fallback is started on overloaded: the terminal decides.
    MATRIX[(_s, "overloaded", "FALLBACK_ok")] = ("flag", "fail_closed", False)
    MATRIX[(_s, "overloaded", "FALLBACK_fails")] = ("flag", "fail_closed", False)
# A full queue overloads both parallel members at once; member 1 is FAIL_CLOSED in this rig, so any_flag flags
# whatever member 0's mode is.
MATRIX[("parallel", "overloaded", "FAIL_OPEN")] = ("flag", "fail_closed", False)


def cfg(**kw) -> GatewayConfig:
    kw.setdefault("shadow_log_path", "")
    return GatewayConfig(**kw)


def failure_doc(mode: str, caps=CAPS) -> dict:
    if mode in ("FAIL_OPEN", "FAIL_CLOSED"):
        return {"mode": mode, "fallback": None}
    # A failing fallback gets a budget shorter than the fault's delay, so the "timeout" fault really times it out.
    return {"mode": "FALLBACK", "fallback": {"detector": ref(caps["fallback"]),
                                             "timeout_ms": 100 if mode == "FALLBACK_fails" else 500,
                                             "terminal": "FAIL_CLOSED"}}


def build(step: str, fault: str, mode: str):
    """(service, detectors) for one cell. The gate's FALLBACK must target a decide member, so for the gate the
    fallback detector *is* the decide member (design §4: gate FALLBACK = escalate)."""
    gate_det = FakeStage1(**({"delay_s": 0.4} if (step == "gate" and fault == "timeout") else
                             {"fail": RuntimeError("gate boom")} if (step == "gate" and fault == "exception") else {}))
    broken = ({"delay_s": 0.4} if fault == "timeout" else {"fail": RuntimeError("decide boom")}
              if fault == "exception" else {})
    faulty_member = step in ("decide", "parallel") and fault in ("timeout", "exception")
    decide_det = FakeDetector(**(broken if faulty_member else {}))
    second_det = FakeDetector()
    fb_det = FakeDetector(**(broken if mode == "FALLBACK_fails" else {}))
    fb_det.detector_id, second_det.detector_id = "fake-fb", "fake-b"

    specs = {DECIDE_KEY: spec(decide_det), SECOND_KEY: spec(second_det), FB_KEY: spec(fb_det)}
    if step == "gate":
        specs[GATE_KEY] = spec(gate_det, kind="text", timeout_ms=80)

    doc = policy(CAPS, gate=step == "gate", parallel="any_flag" if step == "parallel" else None)
    doc["decide"]["members"][0]["timeout_ms"] = 120
    if step == "parallel":
        doc["decide"]["members"][1]["timeout_ms"] = 120
    if step == "gate":
        doc["gate"]["timeout_ms"] = 80
        doc["gate"]["deadline_margin_ms"] = 0
        if mode in ("FAIL_OPEN", "FAIL_CLOSED"):
            doc["gate"]["on_failure"] = {"mode": mode, "fallback": None}
        else:
            # The gate escalates to the decide member; "FALLBACK_fails" breaks that member.
            doc["gate"]["on_failure"] = {"mode": "FALLBACK", "fallback": {
                "detector": ref(CAPS["decide"]), "timeout_ms": 100 if mode == "FALLBACK_fails" else 500,
                "terminal": "FAIL_CLOSED"}}
            if mode == "FALLBACK_fails":
                specs[DECIDE_KEY] = spec(FakeDetector(**({"fail": RuntimeError("decide boom")})))
    else:
        doc["decide"]["members"][0]["on_failure"] = failure_doc(mode)
        if step == "parallel":
            doc["decide"]["members"][1]["on_failure"] = {"mode": "FAIL_CLOSED", "fallback": None}
    over = fault == "overloaded"
    svc = service(doc, specs, config=cfg(
        # gate: admission control sheds on the character budget (deterministic, no race with another request);
        # decide / parallel: the batcher queue is too small for the request's chunks.
        max_text_chars=4 if over and step == "gate" else 200_000,
        max_inflight_chars=4 if over and step == "gate" else 1_000_000,
        max_queue=1 if over and step != "gate" else 64))
    ran = {"fallback": False}
    for key, det in (((DECIDE_KEY, specs[DECIDE_KEY].detector) if step == "gate" else (FB_KEY, fb_det)),):
        inner = det.infer

        def wrapped(batch, _inner=inner):
            ran["fallback"] = True
            return _inner(batch)

        det.infer = wrapped
        svc._batchers.clear()
        svc.members = [svc._member(m) for m in svc.policy.decide.members]
        svc.batcher = svc.members[0].batcher
    return svc, ran, (decide_det, fb_det)


async def _run(svc, step: str, fault: str, text: str):
    await svc.start()
    try:
        return await svc.classify(text, "r")
    finally:
        await svc.stop()


@pytest.mark.parametrize("mode", ["FAIL_OPEN", "FAIL_CLOSED", "FALLBACK_ok", "FALLBACK_fails"])
@pytest.mark.parametrize("fault", ["timeout", "exception", "overloaded"])
@pytest.mark.parametrize("step", ["gate", "decide", "parallel"])
def test_t6_failure_matrix(step, fault, mode):
    svc, ran, _ = build(step, fault, mode)
    text = MANY_CHUNKS if fault == "overloaded" and step != "gate" else BENIGN
    out = asyncio.run(_run(svc, step, fault, text))
    got = (out.decision, out.policy_applied, ran["fallback"])
    print(f"{step:9} {fault:11} {mode:15} -> decision={got[0]:5} policy_applied={got[1]} fallback_ran={got[2]}")
    assert got == MATRIX[(step, fault, mode)]
    if fault == "overloaded" and mode.startswith("FALLBACK"):
        assert ran["fallback"] is False, "FALLBACK must never be started on overloaded"


def test_t6_matrix_is_complete():
    assert set(MATRIX) == {(s, f, m) for s in ("gate", "decide", "parallel")
                           for f in ("timeout", "exception", "overloaded")
                           for m in ("FAIL_OPEN", "FAIL_CLOSED", "FALLBACK_ok", "FALLBACK_fails")}


# ---------------------------------------------------------------- ASYNC_ONLY steps never decide

@pytest.mark.parametrize("fault", ["ok", "timeout", "exception", "flagging"])
def test_t6_async_only_steps_never_change_a_decision_field(fault):
    """Every decision field of a response with observe + shadow equals the same policy without them."""
    fields = ("decision", "decided_by", "policy_applied", "coverage_policy_applied", "stages_run",
              "unscored_by_any_step", "escalation_reason")
    texts = [BENIGN, "w1 INJECT w3", words(3000, inject_at=1500)]
    obs_kwargs = {"ok": {}, "timeout": {"delay_s": 0.5}, "exception": {"fail": RuntimeError("obs boom")},
                  "flagging": {}}[fault]
    out = {}
    for with_observe in (False, True):
        doc = policy(CAPS, gate=True, observe=with_observe, shadow=with_observe)
        doc["gate"]["timeout_ms"] = 2000
        if with_observe:
            doc["observe"][0]["timeout_ms"] = 60
            doc["observe"][0]["band"] = copy.deepcopy(doc["gate"]["band"])
        specs = {DECIDE_KEY: spec(FakeDetector()), GATE_KEY: spec(FakeStage1(), kind="text", timeout_ms=2000)}
        shadow = None
        if with_observe:
            obs = FakeStage1(**obs_kwargs)
            # "flagging": the observe detector would flag everything; it still must not change a decision.
            if fault == "flagging":
                obs.infer = lambda batch: [0.99] * len(batch)
            obs.detector_id = "fake-obs"
            specs[("fake-obs", "0.0.1")] = spec(obs, kind="text", timeout_ms=60)
            sh = FakeDetector()
            sh.detector_id = "fake-shadow"
            shadow = spec(sh)
        svc = service(doc, specs, shadow=shadow)

        async def scenario():
            await svc.start()
            try:
                return [await svc.classify(t, f"r{i}") for i, t in enumerate(texts)]
            finally:
                await svc.stop()

        outs = asyncio.run(scenario())
        out[with_observe] = [{f: getattr(o, f) for f in fields} for o in outs]
        if with_observe:
            for o in outs:
                assert [r.role for r in o.results if r.role != "active"] == ["observe"]
    assert out[True] == out[False]
