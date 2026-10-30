"""Reviewer probes, round 2 (reviews/policy-runtime-r2.md). Not part of the suite.

Two kinds of tests:
- "fixed_*": the round-1 finding no longer reproduces (each asserts the corrected behaviour).
- "r2_*": a round-2 observation; the test asserts the behaviour the review reports, so a pass means it is
  reproduced.

Fakes only: no model, nothing under data/splits/ is read (the T3 probe reads data/replay/replay-v1.jsonl only to
reach the point where the row cache is built, and stops there).

    flock /home/claude/measure.lock python -m pytest -q -p no:cacheprovider reviews/policy-runtime-r2
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "serving"))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import FakeDetector, FakeStage1, words  # noqa: E402
from policy_helpers import caps_by_key, default_caps, member, policy, ref, service, spec  # noqa: E402
from test_shape_b import CAPS, GATE_KEY, build, cfg, run  # noqa: E402

from contracts import ExecutionPolicy, validate_policy  # noqa: E402
from contracts.policy import PolicyError  # noqa: E402

D23 = "policies/d23-deberta-decides/1/policy.json"
CASCADE = "policies/cascade-tfidf-allow-only/1/policy.json"


# ---------------------------------------------------------------- M2

def test_fixed_m2_parallel_fail_open_member_no_longer_skips_coverage():
    text = words(3000)
    svc, _, _ = build("any_flag", 0.1, 0.1, coverage="flag", a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_OPEN", "fallback": None})
    (o,) = run(svc, [text])
    assert (o.decision, o.coverage_policy_applied, o.policy_applied) == ("flag", "flag", None)
    assert o.unscored_by_any_step is True


def test_r2_carveout_matches_the_single_member_fail_open_semantics():
    """The carve-out (no member scored anything -> the failure policy alone decides) is the same outcome a single
    FAIL_OPEN decide member gives on the same multi-chunk text: allow, policy_applied fail_open, unscored True."""
    doc = policy(CAPS)
    doc["decide"]["members"][0]["on_failure"] = {"mode": "FAIL_OPEN", "fallback": None}
    svc = service(doc, {("fake", "0.0.1"): spec(FakeDetector(fail=RuntimeError("x")))}, config=cfg())
    (single,) = run(svc, [words(3000)])
    svc, _, _ = build("any_flag", 0.1, 0.1, coverage="flag", a_kwargs={"fail": RuntimeError("a")},
                      b_kwargs={"fail": RuntimeError("b")}, a_failure={"mode": "FAIL_OPEN", "fallback": None},
                      b_failure={"mode": "FAIL_OPEN", "fallback": None})
    (both,) = run(svc, [words(3000)])
    for o in (single, both):
        assert (o.decision, o.policy_applied, o.coverage_policy_applied) == ("allow", "fail_open", None)
        assert o.unscored_by_any_step is True


def test_r2_coverage_allow_after_a_failed_member_hides_the_failure_policy_field():
    """coverage: allow + a FAIL_OPEN member failure: the response says policy_applied null, decided_by
    coverage_policy:allow. The member failure is visible only in results[].status / info.members."""
    svc, _, _ = build("any_flag", 0.1, 0.1, coverage="allow", a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_OPEN", "fallback": None})
    (o,) = run(svc, [words(3000)])
    assert (o.decision, o.decided_by, o.policy_applied) == ("allow", "coverage_policy:allow", None)
    assert any(r.status != "ok" for r in o.results if r.role == "active")


# ---------------------------------------------------------------- m10

def test_r2_gate_fallback_budget_also_bounds_the_member_the_gate_fallback_does_not_name():
    """After a gate failure both parallel members run until t0 + gate fallback timeout_ms, including the member the
    gate's FALLBACK does not target (here B; the gate FALLBACK names A). Documented in design §11a (engine 1.1.0)."""
    svc, _, _ = build("any_flag", 0.1, 0.1, gate=True, b_kwargs={"delay_s": 0.5})
    doc = svc.policy.model_dump(mode="json")
    doc["gate"]["on_failure"] = {"mode": "FALLBACK", "fallback": {"detector": ref(CAPS["decide"]),
                                                                  "timeout_ms": 150, "terminal": "FAIL_OPEN"}}
    for m in doc["decide"]["members"]:
        m["timeout_ms"] = 2000
    specs = dict(svc._specs)
    specs[GATE_KEY] = spec(FakeStage1(fail=RuntimeError("gate boom")), kind="text", timeout_ms=2000)
    svc = service(doc, specs, config=cfg())
    (o,) = run(svc, ["w1 GRAY w3"])
    b = next(m for m in o.info["members"] if m["detector"] == "fake-b")
    assert b["status"] == "timeout"
    # B votes the gate terminal (FAIL_OPEN), A scored the only chunk: allow decided by the failure policy.
    assert (o.decision, o.escalation_reason) == ("allow", "gate_failed")


# ---------------------------------------------------------------- m1 / n9

def test_fixed_m1_band_without_scope_is_refused_for_gate_and_observe():
    caps = default_caps()
    doc = policy(caps, gate=True)
    doc["gate"]["band"]["scope_languages"] = None
    doc["language_detector"] = None  # required iff some band declares a scope
    with pytest.raises(PolicyError) as e:
        validate_policy(ExecutionPolicy.model_validate(doc), caps_by_key(caps))
    assert e.value.code == "scope_languages_required"
    # a flag-only band clears nothing and needs no scope
    doc["gate"]["band"].update({"clear_below": 0.0, "finalize": ["flag"], "flag_at_or_above": 0.9})
    validate_policy(ExecutionPolicy.model_validate(doc), caps_by_key(caps))


def test_fixed_n9_scope_order_does_not_change_the_hash():
    from serving.policy import bundle_sha256
    caps = default_caps()
    a, b = policy(caps, gate=True), policy(caps, gate=True)
    a["gate"]["band"]["scope_languages"] = ["en", "de"]
    b["gate"]["band"]["scope_languages"] = ["de", "en"]
    assert bundle_sha256(ExecutionPolicy.model_validate(a)) == bundle_sha256(ExecutionPolicy.model_validate(b))


def test_fixed_all_committed_policies_still_validate():
    from serving.registry import load_policy_file
    for rel in (D23, CASCADE, "policies/cascade-tfidf-provisional/1/policy.json"):
        assert load_policy_file(ROOT / "artifacts", rel).policy_id


# ---------------------------------------------------------------- m2 / m3 / m4 (gate.py)

def _registry(tmp_path, active, previous):
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": "sentinel-registry-v2", "active": active, "previous": previous,
                               "capabilities_root": "capabilities", "status": "pending_gate"}))
    return reg


def _report_for(tmp_path, rel, **run_over):
    from training import gate
    rep = json.loads((ROOT / "reports/policy-runtime/gate_val_d23.json").read_text())
    rep["bundle"]["bundle_sha256"] = gate._bundle_of_policy(tmp_path / "registry.json", rel).bundle_sha256
    rep["bundle"]["policy_path"] = rel
    rep["run"].update(run_over)
    p = tmp_path / "rep.json"
    p.write_text(json.dumps(rep))
    return p


def test_fixed_m2_promoting_the_active_bundle_is_refused_and_previous_kept(tmp_path):
    from training import gate
    reg = _registry(tmp_path, D23, CASCADE)
    rep = _report_for(tmp_path, D23)
    before = reg.read_text()
    with pytest.raises(SystemExit, match="already the active bundle"):
        gate.promote(reg, rep, "probe", log=tmp_path / "a.jsonl", allow_not_pass=True)
    assert reg.read_text() == before and not (tmp_path / "a.jsonl").exists()


def test_fixed_m3_incomplete_report_is_refused_even_with_pass(tmp_path):
    from training import gate
    reg = _registry(tmp_path, CASCADE, None)
    rep = _report_for(tmp_path, D23, complete=False, n_items_evaluated=10)
    data = json.loads(rep.read_text())
    data["verdict"] = gate.PASS
    rep.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="incomplete"):
        gate.promote(reg, rep, "probe", log=tmp_path / "a.jsonl", allow_not_pass=True)


def test_r2_m3_a_complete_val_report_is_still_promotable(tmp_path):
    """m3's second half (val accepted like frozen test) was kept by design: val is a gate split; the audit entry
    records gate_split=val."""
    from training import gate
    reg = _registry(tmp_path, CASCADE, None)
    rep = _report_for(tmp_path, D23)
    entry = gate.promote(reg, rep, "probe", log=tmp_path / "a.jsonl", allow_not_pass=True)
    assert entry["gate_split"] == "val" and json.loads(reg.read_text())["active"] == D23


def _frozen_args(tmp_path, **over):
    ns = dict(split="frozen_test", frozen_test_approved_by="owner", audit_log=tmp_path / "audit.jsonl",
              registry=tmp_path / "registry.json", out=tmp_path / "gate_frozen.json", max_seconds=None, limit=None,
              splits_dir=tmp_path, active_report=None)
    ns.update(over)
    return argparse.Namespace(**ns)


def test_fixed_m4_started_is_written_before_the_run(tmp_path, monkeypatch):
    from training import gate
    args = _frozen_args(tmp_path)
    seen = {}

    def stub(*a, **k):
        seen["log"] = args.audit_log.read_text()
        raise KeyboardInterrupt

    monkeypatch.setattr(gate, "run_bundle", stub)
    with pytest.raises(KeyboardInterrupt):
        gate.evaluate(args)
    assert '"started"' in seen["log"]
    assert [json.loads(x)["status"] for x in args.audit_log.read_text().splitlines()] == ["started", "failed"]


def test_r2_m4_an_error_after_the_split_was_read_leaves_only_started(tmp_path, monkeypatch):
    """The try/except covers run_bundle only. An exception after the whole frozen split was scored (here a missing
    --active-report; also build_report / write_json) leaves just "started", which the docstring says means a killed
    process."""
    from training import gate
    args = _frozen_args(tmp_path, active_report=tmp_path / "missing.json")
    monkeypatch.setattr(gate, "run_bundle", lambda *a, **k: ([], {}, None, 0.0, 0))
    with pytest.raises(FileNotFoundError):
        gate.evaluate(args)
    assert [json.loads(x)["status"] for x in args.audit_log.read_text().splitlines()] == ["started"]


# ---------------------------------------------------------------- M1 and the T3 --no-cache flag

def test_fixed_m1_t3_reads_the_preserved_v1_registry():
    from training import replay_policy_equivalence as eq
    state = eq.registry_state()
    assert state["schema"] == "sentinel-registry-v1" and state["stage1_mode"] == "observe"


def test_r2_t3_no_cache_still_shares_an_in_memory_cache_between_both_engines(tmp_path, monkeypatch):
    """--no-cache sets cache_path None, but t3() still builds RowCache(None) and hands it to build_engines, which
    gives the SAME cache to both Recorders. The legacy engine scores a row, the policy engine gets the memoized
    score: "no cache" runs are memoized across engines (in memory), so their scores are identical by construction."""
    from training import replay_policy_equivalence as eq
    captured = {}

    class Stop(Exception):
        pass

    def fake_build(pointers, config, cache, specs):
        captured["cache"] = cache
        raise Stop

    monkeypatch.setattr(eq, "load_specs", lambda *a, **k: {})
    monkeypatch.setattr(eq, "build_engines", fake_build)
    with pytest.raises(Stop):
        eq.t3(tmp_path / "t3.json", None, None, 1, 0)
    cache = captured["cache"]
    assert cache is not None and cache.path is None

    class Det:
        detector_id, detector_version, threshold, max_len, cls_id, sep_id = "d", "v", 0.5, 512, 1, 2
        calls = 0

        def infer(self, batch):
            Det.calls += len(batch)
            return [0.3] * len(batch)

    legacy, pol = eq.Recorder(Det(), cache), eq.Recorder(Det(), cache)
    legacy.infer([[5, 6, 7]])
    pol.infer([[5, 6, 7]])
    assert Det.calls == 1  # the second engine never ran the model
