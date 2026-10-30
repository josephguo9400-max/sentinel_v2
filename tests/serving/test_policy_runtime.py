"""Converter (T1), bundle hash and env-var guard (T8), registry v2 loader. No real models are loaded except the
tiny factory artifacts."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from artifact_factory import (ROOT, known_tokenizers, write_capability, write_policy, write_registry,
                              write_registry_v2, write_v2_tree)
from contracts import ArtifactManifest
from contracts._canonical import canonical_json
from contracts.capability import CapabilityManifest, capability_sha256
from contracts.policy import ExecutionPolicy, PolicyError
from serving.config import GatewayConfig
from serving.convert_registry import CALIBRATION_RECORDS, SOURCES, convert, read_manifests
from serving.policy import (DECISION_FIELDS, DecisionSettingError, bundle_sha256, check_decision_settings,
                            check_engine_compat, env_name, legacy_to_v2)
from serving.registry import (KNOWN_TOKENIZERS, RegistryError, load_bundle_v2, load_registry, bundle_from_v1,
                              registry_schema)

GOLDEN = Path(__file__).with_name("golden")
S2 = ("deberta-pi-v2", "2.0.0+90c9989b.onnx.fp32")
S1 = ("stage1-tfidf-lr", "stage1-tfidf-v2.1")


# ---------------------------------------------------------------- T1: converter golden files

@pytest.mark.parametrize("pid", sorted(SOURCES))
def test_t1_converter_golden(pid):
    p, _ = convert(pid)
    assert canonical_json(p.model_dump(mode="json")) + b"\n" == (GOLDEN / f"{pid}.policy.canonical.json").read_bytes()


def test_t1_d23_matches_design_section_5():
    p, caps = convert("d23-deberta-decides")
    assert p.policy_id == "d23-deberta-decides" and p.policy_version == "1" and p.engine_version == "1.0.0"
    assert p.gate is None and p.shadow is None and p.canary is None and p.coverage.mode == "flag"
    assert p.chunking.plan_tokenizer.key == S2 and p.chunking.stride == 384 and p.chunking.structure_aware
    assert p.chunking.pack_overlap_tokens == 126
    (m,) = p.decide.members
    assert p.decide.mode == "single" and m.detector.key == S2
    assert (m.timeout_ms, m.max_chunks, m.on_failure.mode) == (5000, 4, "FAIL_CLOSED")
    (o,) = p.observe
    assert o.detector.key == S1 and (o.timeout_ms, o.concurrency, o.on_failure.mode) == (50, 1, "ASYNC_ONLY")
    # Additions to the design's YAML (design §12 / §4 notes): the observe step's chunk budget, soft deadline and
    # what-if band, and therefore a language detector.
    assert (o.max_chunks, o.deadline_margin_ms) == (512, 15)
    assert o.band.clear_below == 0.505068 and o.band.finalize == {"allow"} and o.band.scope_languages == ["en"]
    assert p.language_detector.impl == "langdetect==1.0.9" and p.language_detector.min_prob == 0.8
    assert set(caps) == {S1, S2}


def test_t1_cascade_allow_only_matches_design_section_5():
    p, _ = convert("cascade-tfidf-allow-only")
    g = p.gate
    assert g.detector.key == S1
    assert (g.band.clear_below, g.band.flag_at_or_above, g.band.finalize, g.band.scope_languages) == (
        0.505068, None, {"allow"}, ["en"])
    assert (g.timeout_ms, g.deadline_margin_ms, g.max_chunks) == (50, 15, 512)
    fb = g.on_failure.fallback
    assert g.on_failure.mode == "FALLBACK" and fb.detector.key == S2
    assert (fb.timeout_ms, fb.terminal) == (5000, "FAIL_CLOSED")
    ld = p.language_detector
    assert (ld.seed, ld.min_letters, ld.min_prob, ld.max_chars, ld.tag_map) == (0, 20, 0.8, 2000, {})
    assert p.observe == [] and p.coverage.mode == "flag" and p.decide.members[0].max_chunks == 4


def test_t1_provisional_policy_and_both_stage1_capabilities():
    p, caps = convert("cascade-tfidf-provisional")
    assert p.gate.band.clear_below == 0.282715 and p.gate.band.finalize == {"allow"}
    # v2 is a v1.1 manifest without input_languages: the scope comes from STAGE1_UNDECLARED_LANGUAGES ("en").
    assert p.gate.band.scope_languages == ["en"]
    v2 = caps[("stage1-tfidf-lr", "stage1-tfidf-v2")]
    _, caps21 = convert("cascade-tfidf-allow-only")
    v21 = caps21[S1]
    # D25.7: two immutable capability versions of the same bytes and threshold.
    assert v2.files == {"candidates/stage1-tfidf-v2/model.onnx": v21.files["candidates/stage1-tfidf-v2.1/model.onnx"]}
    assert v2.calibration.threshold == v21.calibration.threshold == 0.505068
    assert capability_sha256(v2) != capability_sha256(v21)
    # D25.6: served scoring rule recorded; the calibration records the different rule it was chosen with.
    assert (v21.scoring.subwindow_tokens, v21.scoring.subwindow_stride, v21.scoring.score_segment_pieces) == (
        64, 32, True)
    assert v21.calibration.calibrated_with.subwindow_tokens == 0
    assert not v21.calibration.calibrated_with.score_segment_pieces and "D25.6" in v21.calibration.notes


V1_D23 = "reports/policy-runtime/registry_v1_d23.json"  # the v1 state D23 served (artifacts/registry.json is v2)


def test_converter_is_pure_and_follows_config():
    reg = json.loads((ROOT / V1_D23).read_text())
    ms = read_manifests(reg)
    a, _ = legacy_to_v2(reg, ms, GatewayConfig(), known_tokenizers=KNOWN_TOKENIZERS, calibrations=CALIBRATION_RECORDS)
    b, _ = legacy_to_v2(reg, ms, GatewayConfig(), known_tokenizers=KNOWN_TOKENIZERS, calibrations=CALIBRATION_RECORDS)
    assert a == b
    c, caps = legacy_to_v2(reg, ms, GatewayConfig(coverage_policy="deep_scan", max_chunks=2,
                                                  stage1_subwindow_tokens=0),
                           known_tokenizers=KNOWN_TOKENIZERS)
    assert c.coverage.mode == "deep_scan" and c.coverage.deep_scan.max_chunks == 16
    assert c.decide.members[0].max_chunks == 2
    assert caps[S1].scoring.subwindow_tokens == 0 and caps[S1].scoring.subwindow_stride is None
    route = dict(reg, stage1_mode="route")
    d, _ = legacy_to_v2(route, ms, GatewayConfig(stage1_finalize_flag=True), known_tokenizers=KNOWN_TOKENIZERS)
    assert d.gate.band.finalize == {"allow", "flag"} and d.gate.band.flag_at_or_above == 1.0


def test_converter_refuses_unscoped_undeclared_stage1():
    reg = {"stage1": "candidates/stage1-tfidf-v2", "stage1_mode": "route", "stage2": "baseline/deberta-pi-v2",
           "shadow": None}
    with pytest.raises(PolicyError) as exc:
        legacy_to_v2(reg, read_manifests(reg), GatewayConfig(stage1_undeclared_languages=""),
                     known_tokenizers=KNOWN_TOKENIZERS)
    assert exc.value.code == "undeclared_languages"


# ---------------------------------------------------------------- T8: bundle hash

def _with_policy(p: ExecutionPolicy, **path_values) -> ExecutionPolicy:
    d = p.model_dump(mode="json")
    for path, v in path_values.items():
        cur = d
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur[int(k)] if k.isdigit() else cur[k]
        cur[keys[-1]] = v
    return ExecutionPolicy.model_validate(d)


def test_t8_bundle_changes_with_band_capability_or_engine():
    p, caps = convert("cascade-tfidf-allow-only")
    base = bundle_sha256(p)
    assert bundle_sha256(p) == base  # deterministic
    assert bundle_sha256(_with_policy(p, **{"gate.band.clear_below": 0.4})) != base
    # Only a capability's threshold: a new capability hash, pinned by the policy.
    cap = caps[S1].model_dump(mode="json")
    cap["calibration"]["threshold"] = 0.6
    new_sha = capability_sha256(CapabilityManifest.model_validate(cap))
    assert new_sha != capability_sha256(caps[S1])
    assert bundle_sha256(_with_policy(p, **{"gate.detector.capability_sha256": new_sha})) != base
    assert bundle_sha256(p, engine_version="1.0.1") != base
    assert bundle_sha256(_with_policy(p, engine_version="1.0.0")) == base


def test_engine_compat():
    p, _ = convert("d23-deberta-decides")
    check_engine_compat(p, "1.0.0")
    check_engine_compat(p, "1.4.2")
    for bad in ("2.0.0", "0.9.0"):
        with pytest.raises(PolicyError) as exc:
            check_engine_compat(p, bad)
        assert exc.value.code == "engine_version_incompatible"


# ---------------------------------------------------------------- T8: decision-affecting settings

def test_t8_env_var_guard_names_the_policy_field():
    check_decision_settings(GatewayConfig(), environ={"SENTINEL_MAX_BATCH": "8", "SENTINEL_WORKERS": "2"})
    with pytest.raises(DecisionSettingError) as exc:
        check_decision_settings(GatewayConfig(), environ={"SENTINEL_COVERAGE_POLICY": "allow"})
    assert "SENTINEL_COVERAGE_POLICY" in str(exc.value) and "coverage.mode" in str(exc.value)
    with pytest.raises(DecisionSettingError) as exc:
        check_decision_settings(GatewayConfig(max_chunks=8), environ={})
    assert "decide.members[].max_chunks" in str(exc.value)
    for name in DECISION_FIELDS:
        with pytest.raises(DecisionSettingError):
            check_decision_settings(GatewayConfig(), environ={env_name(name): "1"})


def test_capacity_settings_are_not_decision_settings():
    cap = {"max_batch", "workers", "max_queue", "max_inflight", "max_inflight_chars", "tokenize_workers",
           "intra_op_threads", "max_text_chars", "max_wait_ms", "registry_path", "artifacts_root", "registry_poll_s",
           "shadow_log_path"}
    assert not cap & set(DECISION_FIELDS)


# ---------------------------------------------------------------- registry v2 loader

def test_v2_loads_and_hashes(tmp_path):
    policy, caps, reg = write_v2_tree(tmp_path, stage1_mode="observe")
    assert registry_schema(reg) == "sentinel-registry-v2"
    b = load_bundle_v2(reg, tmp_path)
    assert b.policy == policy and b.bundle_sha256 == bundle_sha256(policy) and b.policy_source == "registry_v2"
    assert set(b.detectors) == set(caps) and b.shadow is None and b.previous is None
    spec = b.detectors[("tiny-token", "0.0.1")]
    assert spec.kind == "token" and spec.capability == caps[("tiny-token", "0.0.1")]
    assert spec.detector.infer([[1, 3, 2]])[0] > 0.5


def test_v2_previous_kept_and_broken_previous_reported(tmp_path):
    policy, caps, reg = write_v2_tree(tmp_path)
    prev = policy.model_copy(update={"policy_version": "0"})
    rel = write_policy(tmp_path, prev)
    write_registry_v2(tmp_path, f"policies/{policy.policy_id}/1/policy.json", rel)
    b = load_bundle_v2(tmp_path / "registry.json", tmp_path)
    assert b.previous["bundle_sha256"] == bundle_sha256(prev)
    write_registry_v2(tmp_path, f"policies/{policy.policy_id}/1/policy.json", "policies/missing/1/policy.json")
    b = load_bundle_v2(tmp_path / "registry.json", tmp_path)
    assert b.previous["error"].startswith("missing_file")


def _reject(tmp_path, code):
    with pytest.raises(RegistryError) as exc:
        load_bundle_v2(tmp_path / "registry.json", tmp_path)
    assert exc.value.code == code, str(exc.value)


def test_v2_rejects_tampered_file(tmp_path):
    write_v2_tree(tmp_path)
    (tmp_path / "baseline/tiny/onnx/tokenizer.json").write_text("{}")
    _reject(tmp_path, "tokenizer_mismatch")


def test_v2_rejects_tampered_model(tmp_path):
    write_v2_tree(tmp_path)
    p = tmp_path / "candidates/s1/model.onnx"
    p.write_bytes(p.read_bytes() + b"x")
    _reject(tmp_path, "sha256_mismatch")


def test_v2_rejects_edited_capability(tmp_path):
    policy, caps, _ = write_v2_tree(tmp_path)
    cap = caps[("tiny-token", "0.0.1")].model_dump(mode="json")
    cap["calibration"]["threshold"] = 0.4
    write_capability(tmp_path, CapabilityManifest.model_validate(cap))
    _reject(tmp_path, "capability_sha256_mismatch")


def test_v2_rejects_path_and_policy_problems(tmp_path):
    policy, _, _ = write_v2_tree(tmp_path)
    write_registry_v2(tmp_path, "../outside/policy.json")
    _reject(tmp_path, "path_escape")
    other = tmp_path / "policies/other/9/policy.json"
    other.parent.mkdir(parents=True)
    other.write_text(policy.model_dump_json())
    write_registry_v2(tmp_path, "policies/other/9/policy.json")
    _reject(tmp_path, "policy_invalid")
    rel = write_policy(tmp_path, policy.model_copy(update={"policy_version": "2", "engine_version": "9.0.0"}))
    write_registry_v2(tmp_path, rel)
    _reject(tmp_path, "engine_version_incompatible")
    d = policy.model_dump(mode="json")
    d.update(policy_version="3", canary={"candidate": {"policy_id": "x", "policy_version": "1",
                                                       "policy_sha256": "a" * 64}, "share_bp": 1, "salt": "s"})
    rel = write_policy(tmp_path, ExecutionPolicy.model_validate(d))
    write_registry_v2(tmp_path, rel)
    _reject(tmp_path, "canary_unsupported")
    reg = json.loads((tmp_path / "registry.json").read_text())
    (tmp_path / "registry.json").write_text(json.dumps({**reg, "stage2": "x"}))
    _reject(tmp_path, "registry_invalid")


def test_v1_registry_converts_in_memory(tmp_path):
    from artifact_factory import write_stage1_artifact, write_token_artifact
    write_token_artifact(tmp_path, "baseline/tiny")
    write_stage1_artifact(tmp_path, "candidates/s1")
    reg = write_registry(tmp_path, stage1="candidates/s1", stage1_mode="observe", stage2="baseline/tiny", shadow=None)
    loaded = load_registry(reg, tmp_path, known_tokenizers=known_tokenizers())
    cfg = GatewayConfig(extra_tokenizers=",".join(f"{k}={v}" for k, v in known_tokenizers().items()))
    b = bundle_from_v1(loaded, cfg, registry_sha256="ab" * 32)
    assert b.policy_source == "legacy_registry" and b.policy.policy_id == "legacy-registry"
    assert b.policy.policy_version == "abababababab" and len(b.policy.observe) == 1 and b.policy.gate is None
    assert set(b.detectors) == {("tiny-token", "0.0.1"), ("stage1-tfidf-lr", "stage1-tfidf-v2")}
    # A rejected Stage 1 is absent from the converted policy, as the legacy gateway served it.
    reg = write_registry(tmp_path, stage1="candidates/missing", stage1_mode="route", stage2="baseline/tiny")
    loaded = load_registry(reg, tmp_path, known_tokenizers=known_tokenizers())
    b = bundle_from_v1(loaded, cfg)
    assert b.policy.gate is None and b.policy.observe == [] and b.rejections


def test_committed_v1_registry_converts_to_the_d23_capabilities():
    reg = json.loads((ROOT / V1_D23).read_text())
    p, caps = legacy_to_v2(reg, read_manifests(reg), GatewayConfig(), known_tokenizers=KNOWN_TOKENIZERS,
                           calibrations=CALIBRATION_RECORDS)
    d23, d23_caps = convert("d23-deberta-decides")
    assert caps == d23_caps
    strip = lambda x: {k: v for k, v in x.model_dump(mode="json").items()  # noqa: E731
                       if k not in ("policy_id", "policy_version", "rationale")}
    assert strip(p) == strip(d23)


# ---------------------------------------------------------------- the committed registry (step g)

def test_committed_registry_is_a_loadable_v2_bundle():
    """artifacts/registry.json serves the D23 bundle; its hash is the one T3 and the smoke run recorded."""
    from serving.registry import REGISTRY_SCHEMA_V2, load_policy_file, read_registry_v2

    reg = ROOT / "artifacts" / "registry.json"
    assert registry_schema(reg) == REGISTRY_SCHEMA_V2
    state = read_registry_v2(reg)
    assert state["active"] == "policies/d23-deberta-decides/1/policy.json"
    assert state["capabilities_root"] == "capabilities"
    # D23 was pending the gate before the conversion and stays pending (design §7, migration 4).
    assert state["status"] == "pending_gate"
    assert state["previous"] is None
    policy = load_policy_file(ROOT / "artifacts", state["active"])
    assert policy.decide.mode == "single" and policy.gate is None and len(policy.observe) == 1
    assert policy.canary is None and policy.engine_version == "1.0.0"
    converted, _ = convert("d23-deberta-decides")
    assert policy == converted  # the committed policy is exactly what the converter derives
    t3 = json.loads((ROOT / "reports/policy-runtime/t3_d23_equivalence.json").read_text())
    assert bundle_sha256(policy) == t3["bundle"]["committed_bundle_sha256"]


def test_committed_capabilities_are_pinned_by_the_committed_policies():
    """Every DetectorRef of every committed policy resolves to a committed capability with that sha256."""
    from contracts.policy import iter_refs
    from serving.registry import load_capability, load_policy_file

    for path in sorted((ROOT / "artifacts").glob("policies/*/*/policy.json")):
        rel = str(path.relative_to(ROOT / "artifacts"))
        policy = load_policy_file(ROOT / "artifacts", rel)
        for r in iter_refs(policy):
            cap = load_capability(ROOT / "artifacts", "capabilities", r)  # raises on a mismatch
            assert cap.key == r.key


def test_t3_and_t4_reports_record_a_pass():
    """The merge condition of the refactor (design §8): T3 identical on every item."""
    t3 = json.loads((ROOT / "reports/policy-runtime/t3_d23_equivalence.json").read_text())
    assert t3["complete"] and t3["identical_on_every_item"] and t3["n_items_with_differences"] == 0
    assert t3["n_items_compared"] == t3["n_items_total"] == 2000
    assert t3["bundle"]["only_timeout_fields_differ"] is True
    t4 = json.loads((ROOT / "reports/policy-runtime/t4_cascade_equivalence.json").read_text())
    assert t4["identical_on_every_item"] and t4["headline_201_306_reproduced"]
    assert t4["recall"]["as_fraction"] == "201/306" and t4["fpr"]["as_fraction"] == "191/1185"
    t5 = json.loads((ROOT / "reports/policy-runtime/t5_live_comparison.json").read_text())
    assert t5["identical_on_every_comparable_item"] and t5["n_different"] == 0
