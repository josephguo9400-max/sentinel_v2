"""Reviewer probes for milestone policy-runtime (reviews/policy-runtime-r1.md). Not part of the suite.

Each test asserts the behaviour the review reports (so a passing probe means the finding is reproduced). Fakes only:
no model, no data split, nothing under data/splits/ is read.

    flock /home/claude/measure.lock python -m pytest -q -p no:cacheprovider reviews/policy-runtime-r1
"""
import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "serving"))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import FakeStage1, words  # noqa: E402
from policy_helpers import caps_by_key, default_caps, policy, service, spec  # noqa: E402
from test_shape_b import build, run  # noqa: E402

from contracts import ExecutionPolicy, validate_policy  # noqa: E402
from serving.language import FixedLanguage  # noqa: E402


# ---------------------------------------------------------------- R1: parallel FAIL_OPEN member bypasses coverage

def test_r1_parallel_fail_open_member_turns_a_coverage_flag_into_an_allow():
    """any_flag, both members would vote allow, the text has more chunks than max_chunks, coverage=flag.

    Both healthy: content no member scored -> coverage flag. Member A fails with FAIL_OPEN: its vote is the same
    "allow", and member B still scores exactly the same chunks, so the unscored content is the same - yet the
    engine returns allow (policy_applied fail_open) and never applies coverage. A failing detector yields a more
    permissive decision than the same detector succeeding with an allow vote."""
    text = words(3000)
    svc, _, _ = build("any_flag", 0.1, 0.1, coverage="flag")
    (healthy,) = run(svc, [text])
    assert (healthy.decision, healthy.coverage_policy_applied) == ("flag", "flag")
    assert healthy.unscored_by_any_step is True

    svc, _, _ = build("any_flag", 0.1, 0.1, coverage="flag", a_kwargs={"fail": RuntimeError("boom")},
                      a_failure={"mode": "FAIL_OPEN", "fallback": None})
    (failed,) = run(svc, [text])
    assert failed.decision == "allow"
    assert failed.policy_applied == "fail_open" and failed.coverage_policy_applied is None
    assert failed.unscored_by_any_step is True  # the engine knows content was unscored, and allows anyway


# ---------------------------------------------------------------- R2: band without scope_languages drops the gate

def test_r2_band_without_scope_clears_any_language_although_the_capability_declares_en():
    """The gate capability declares validated_languages=["en"]; a hand-written band omits scope_languages.
    validate_policy accepts it (the subset rule only applies 'when both are set'), lint_policy is silent, and the
    engine clears a French chunk. The legacy gateway always gated on the manifest's input_languages."""
    from contracts.policy import lint_policy

    caps = default_caps()
    assert caps["gate"].input.validated_languages == ["en"]
    doc = policy(caps, gate=True)
    doc["gate"]["band"]["scope_languages"] = None
    doc["language_detector"] = None
    p = ExecutionPolicy.model_validate(doc)
    validate_policy(p, caps_by_key(caps))  # accepted
    assert not [w for w in lint_policy(p) if "language" in w]

    from fakes import FakeDetector
    specs = {("fake", "0.0.1"): spec(FakeDetector()), ("fake-s1", "0.0.1"): spec(FakeStage1(), kind="text")}
    svc = service(p, specs, language=FixedLanguage("fr"))

    async def go():
        await svc.start()
        try:
            return await svc.classify("bonjour tout le monde, ceci est un texte", "r")
        finally:
            await svc.stop()

    out = asyncio.run(go())
    assert out.decision == "allow" and out.stages_run == ["stage1"]  # finalized by the gate, language ignored

    # With the scope the converter writes, the same request escalates on language.
    doc2 = policy(caps, gate=True)
    svc2 = service(ExecutionPolicy.model_validate(doc2), specs, language=FixedLanguage("fr"))

    async def go2():
        await svc2.start()
        try:
            return await svc2.classify("bonjour tout le monde, ceci est un texte", "r")
        finally:
            await svc2.stop()

    out2 = asyncio.run(go2())
    assert out2.escalation_reason == "language" and "stage2" in out2.stages_run


# ---------------------------------------------------------------- R3: promote of the active bundle loses `previous`

def _registry(tmp_path: Path, active: str, previous: str | None) -> Path:
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema": "sentinel-registry-v2", "active": active, "previous": previous,
                               "capabilities_root": "capabilities", "status": "pending_gate"}))
    return reg


D23 = "policies/d23-deberta-decides/1/policy.json"
CASCADE = "policies/cascade-tfidf-allow-only/1/policy.json"


def test_r3_promoting_the_already_active_bundle_overwrites_previous_with_itself(tmp_path):
    from training import gate

    reg = _registry(tmp_path, active=D23, previous=CASCADE)
    report = ROOT / "reports" / "policy-runtime" / "gate_val_d23.json"  # bound to the D23 bundle
    entry = gate.promote(reg, report, "reviewer-probe", log=tmp_path / "audit.jsonl", allow_not_pass=True)
    state = json.loads(reg.read_text())
    assert state["active"] == D23 and state["previous"] == D23  # the real previous (CASCADE) is gone
    assert state["status"] == "approved"
    assert entry["bundle_sha256"] == entry["previous_bundle_sha256"]
    # A rollback is now a no-op flip.
    gate.rollback(reg, "reviewer-probe", log=tmp_path / "audit.jsonl")
    assert json.loads(reg.read_text())["active"] == D23


def test_r3b_promote_accepts_an_incomplete_report(tmp_path):
    """promote checks verdict and bundle hash but not run.complete (nor the split)."""
    from training import gate

    rep = json.loads((ROOT / "reports" / "policy-runtime" / "gate_val_d23.json").read_text())
    rep = copy.deepcopy(rep)
    rep["verdict"] = "PASS"
    rep["run"]["complete"] = False
    rep["run"]["n_items_evaluated"] = 10
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(rep))
    reg = _registry(tmp_path, active=CASCADE, previous=None)
    gate.promote(reg, path, "reviewer-probe", candidate_policy=D23, log=tmp_path / "audit.jsonl")
    assert json.loads(reg.read_text())["active"] == D23


# ---------------------------------------------------------------- R4: T3 cannot be re-run at HEAD

def test_r4_t3_reads_artifacts_registry_which_is_now_v2():
    """training/replay_policy_equivalence.t3 hard-codes artifacts/registry.json as a v1 state; at HEAD that file is
    sentinel-registry-v2, so load_specs() returns {} and build_engines raises KeyError('stage2') before any model
    is loaded."""
    from training import replay_policy_equivalence as eq

    state = eq.registry_state()
    assert state["schema"] == "sentinel-registry-v2"
    assert eq.load_specs(state) == {}
