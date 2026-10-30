"""T9 (docs/design/capability-policy.md §8): one negative test per validator rule of CapabilityManifest 2.0 and
ExecutionPolicy 1.0, each rejected with a named error; plus the 1.3 wire additions (DetectorResult role "observe",
ClassifyResponse bundle fields)."""
import copy

import pytest
from pydantic import ValidationError

from contracts import ClassifyResponse, DetectorResult
from contracts.capability import CapabilityManifest, capability_sha256
from contracts.policy import ExecutionPolicy, PolicyError, lint_policy, policy_sha256, validate_policy
from tests.policy_factory import capability, caps_by_key, default_caps, member, policy, ref

SHA = "a" * 64


def rejects(model, doc, code):
    with pytest.raises(ValidationError) as exc:
        model.model_validate(doc)
    assert f"[{code}]" in str(exc.value), str(exc.value)


# ---------------------------------------------------------------- CapabilityManifest 2.0

def test_valid_capabilities_and_stable_hash():
    for c in default_caps().values():
        again = CapabilityManifest.model_validate_json(c.model_dump_json())
        assert capability_sha256(again) == capability_sha256(c)


@pytest.mark.parametrize("code,over", [
    ("input_kind", {"input": {**capability("x")["input"], "kind": "text"}}),
    ("scoring_stride", {"scoring": {"aggregation": "max", "subwindow_tokens": 0, "subwindow_stride": 32,
                                    "score_segment_pieces": False}}),
    ("model_file", {"model_file": "x/other.onnx"}),
    ("label_mapping", {"label_mapping": {"0": "benign", "1": "benign"}}),
    ("tokenizer_file", {"tokenizer_file": None}),
    ("runtime", {"runtime": {"python": "3.13.0", "onnxruntime": "1.29.0"}}),
    ("scoring_kind", {"scoring": {"aggregation": "max", "subwindow_tokens": 64, "subwindow_stride": 32,
                                  "score_segment_pieces": False}}),
    ("files", {"files": {}}),
])
def test_capability_rules(code, over):
    rejects(CapabilityManifest, capability("x", **over), code)


def test_text_capability_has_no_tokenizer_file():
    rejects(CapabilityManifest, capability("t", kind="text", tokenizer_file="t/model.onnx"), "tokenizer_file")


@pytest.mark.parametrize("bad", [
    {"files": {"../escape/model.onnx": SHA}, "model_file": "../escape/model.onnx", "tokenizer_file": None},
    {"validated_languages": ["EN"]},
    {"created_at": "2026-10-30 00:00"},
    {"detector_version": "a/b"},
    {"extra_field": 1},
])
def test_capability_rejects(bad):
    doc = capability("t", kind="text")
    if "validated_languages" in bad:
        doc["input"]["validated_languages"] = bad["validated_languages"]
    else:
        doc.update(bad)
    with pytest.raises(ValidationError):
        CapabilityManifest.model_validate(doc)


def test_capability_is_frozen_and_threshold_change_changes_hash():
    c = default_caps()["gate"]
    with pytest.raises(ValidationError):
        c.detector_version = "x"
    doc = c.model_dump(mode="json")
    doc["calibration"]["threshold"] = 0.6
    assert capability_sha256(CapabilityManifest.model_validate(doc)) != capability_sha256(c)


# ---------------------------------------------------------------- ExecutionPolicy 1.0, structural rules

def test_valid_policies():
    caps = default_caps()
    for kw in ({}, {"gate": True}, {"parallel": "any_flag"}, {"gate": True, "parallel": "all_flag"},
               {"observe": True, "shadow": True}, {"coverage": "deep_scan"}, {"gate": True, "coverage": "allow"}):
        p = ExecutionPolicy.model_validate(policy(caps, **kw))
        validate_policy(p, caps_by_key(caps))
        assert ExecutionPolicy.model_validate_json(p.model_dump_json()) == p


def test_canonical_hash_independent_of_set_order():
    caps = default_caps()
    a = policy(caps, gate=True)
    a["gate"]["band"].update(finalize=["allow", "flag"], flag_at_or_above=0.9)
    b = copy.deepcopy(a)
    b["gate"]["band"]["finalize"] = ["flag", "allow"]
    assert policy_sha256(ExecutionPolicy.model_validate(a)) == policy_sha256(ExecutionPolicy.model_validate(b))


def _with(doc, path, value):
    d = doc
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    return doc


def structural_cases():
    caps = default_caps()
    g = lambda: policy(caps, gate=True)  # noqa: E731
    fb_decide = {"mode": "FALLBACK", "fallback": {"detector": ref(caps["fallback"]), "timeout_ms": 100,
                                                  "terminal": "FAIL_CLOSED"}}
    return [
        ("failure_fallback", _with(policy(caps), ["decide", "members", 0, "on_failure"],
                                   {"mode": "FALLBACK", "fallback": None})),
        ("failure_fallback", _with(policy(caps), ["decide", "members", 0, "on_failure"],
                                   {**fb_decide, "mode": "FAIL_CLOSED"})),
        ("band_finalize", _with(g(), ["gate", "band", "finalize"], [])),
        ("band_flag", _with(g(), ["gate", "band", "finalize"], ["allow", "flag"])),
        ("band_flag", _with(g(), ["gate", "band", "flag_at_or_above"], 0.9)),
        ("band_flag", _with(_with(g(), ["gate", "band", "finalize"], ["allow", "flag"]),
                            ["gate", "band", "flag_at_or_above"], 0.1)),
        ("band_allow", _with(_with(_with(g(), ["gate", "band", "finalize"], ["flag"]),
                                   ["gate", "band", "flag_at_or_above"], 0.9), ["gate", "band", "clear_below"], 0.2)),
        ("decide_shape", _with(policy(caps), ["decide", "members"],
                               [member(caps["decide"]), member(caps["decide2"])])),
        ("decide_shape", _with(policy(caps), ["decide", "aggregate"], "any_flag")),
        ("decide_shape", _with(policy(caps, parallel="any_flag"), ["decide", "aggregate"], None)),
        ("decide_shape", _with(policy(caps, parallel="any_flag"), ["decide", "members"], [member(caps["decide"])])),
        ("decide_distinct", _with(policy(caps, parallel="any_flag"), ["decide", "members"],
                                  [member(caps["decide"]), member(caps["decide"])])),
        ("async_only", _with(policy(caps), ["decide", "members", 0, "on_failure"], {"mode": "ASYNC_ONLY"})),
        ("async_only", _with(g(), ["gate", "on_failure"], {"mode": "ASYNC_ONLY"})),
        ("async_only", _with(policy(caps, observe=True), ["observe", 0, "on_failure"], {"mode": "FAIL_OPEN"})),
        ("async_only", _with(policy(caps, shadow=True), ["shadow", "on_failure"], {"mode": "FAIL_CLOSED"})),
        ("decide_fallback_target", _with(policy(caps), ["decide", "members", 0, "on_failure"],
                                         {**fb_decide, "fallback": {**fb_decide["fallback"],
                                                                    "detector": ref(caps["decide"])}})),
        ("gate_fallback_target", _with(g(), ["gate", "on_failure", "fallback", "detector"], ref(caps["fallback"]))),
        ("gate_is_decider", _with(g(), ["gate", "detector"], ref(caps["decide"]))),
        ("language_detector_required", _with(g(), ["language_detector"], None)),
        ("language_detector_unused", _with(policy(caps), ["language_detector"],
                                           {"impl": "langdetect==1.0.9", "seed": 0, "min_letters": 20,
                                            "min_prob": 0.8, "max_chars": 2000, "tag_map": {}})),
        ("plan_tokenizer", _with(policy(caps), ["chunking", "plan_tokenizer"], ref(caps["shadow"]))),
        ("coverage_deep_scan", _with(policy(caps), ["coverage"], {"mode": "deep_scan", "deep_scan": None})),
        ("coverage_deep_scan", _with(policy(caps, coverage="deep_scan"), ["coverage", "mode"], "flag")),
        ("deep_scan_parallel", policy(caps, coverage="deep_scan", parallel="any_flag")),
        ("deep_scan_target", _with(policy(caps, coverage="deep_scan"), ["coverage", "deep_scan", "detector"],
                                   ref(caps["fallback"]))),
        ("ref_inconsistent", _with(policy(caps, shadow=True), ["shadow", "detector"],
                                   {**ref(caps["decide"]), "capability_sha256": "c" * 64})),
        ("engine_version", _with(policy(caps), ["engine_version"], "1.0")),
    ]


@pytest.mark.parametrize("code,doc", structural_cases(), ids=[c for c, _ in structural_cases()])
def test_policy_structural_rules(code, doc):
    rejects(ExecutionPolicy, doc, code)


def test_fallback_depth_is_one():
    caps = default_caps()
    doc = _with(policy(caps), ["decide", "members", 0, "on_failure"],
                {"mode": "FALLBACK", "fallback": {"detector": ref(caps["fallback"]), "timeout_ms": 100,
                                                  "terminal": "FALLBACK"}})
    with pytest.raises(ValidationError):
        ExecutionPolicy.model_validate(doc)


@pytest.mark.parametrize("bad", [{"policy_id": "../x"}, {"policy_version": "a/b"}, {"entrypoint": "exchange"},
                                 {"unknown": 1}])
def test_policy_rejects_fields(bad):
    with pytest.raises(ValidationError):
        ExecutionPolicy.model_validate(policy(default_caps(), **bad))


# ---------------------------------------------------------------- validate_policy (needs the capabilities)

def cross(code, doc, caps):
    p = ExecutionPolicy.model_validate(doc)
    with pytest.raises(PolicyError) as exc:
        validate_policy(p, caps_by_key(caps))
    assert exc.value.code == code, str(exc.value)


def test_ref_unresolved_and_sha_mismatch():
    caps = default_caps()
    doc = policy(caps)
    other = dict(caps)
    other.pop("decide")
    cross("ref_unresolved", doc, other)
    doc = policy(caps)
    pinned = {**ref(caps["decide"]), "capability_sha256": "c" * 64}
    doc["decide"]["members"][0]["detector"] = pinned
    doc["chunking"]["plan_tokenizer"] = pinned
    cross("capability_sha256_mismatch", doc, caps)


def test_canary_rejected_by_engine():
    caps = default_caps()
    doc = policy(caps, canary={"candidate": {"policy_id": "c", "policy_version": "2", "policy_sha256": SHA},
                               "share_bp": 100, "salt": "s"})
    cross("canary_unsupported", doc, caps)


def test_kind_rules():
    caps = default_caps()
    swapped = dict(caps, gate=caps["decide2"])
    cross("kind_mismatch", policy(swapped, gate=True), swapped)  # token model as gate
    textual = dict(caps, decide2=caps["observe"])
    cross("kind_mismatch", policy(textual, parallel="any_flag"), textual)  # text model as decide member
    obs = dict(caps, observe=caps["shadow"])
    cross("kind_mismatch", policy(obs, observe=True), obs)  # token model as observe step
    doc = policy(caps)
    doc["decide"]["members"][0]["on_failure"] = {"mode": "FALLBACK", "fallback": {
        "detector": ref(caps["gate"]), "timeout_ms": 100, "terminal": "FAIL_CLOSED"}}
    cross("kind_mismatch", doc, caps)  # text model as fallback


def test_parallel_members_share_the_chunk_tokenizer():
    caps = default_caps()
    caps["decide2"] = CapabilityManifest.model_validate(capability("fake-b", tokenizer_sha="d" * 64))
    cross("tokenizer_mismatch", policy(caps, parallel="any_flag"), caps)


def test_band_languages():
    caps = default_caps()
    doc = policy(caps, gate=True)
    doc["gate"]["band"]["scope_languages"] = ["en", "de"]
    cross("scope_languages", doc, caps)
    caps["gate"] = CapabilityManifest.model_validate(capability("fake-s1", kind="text", languages=None))
    doc = policy(caps, gate=True)
    doc["gate"]["band"]["scope_languages"] = None
    doc["language_detector"] = None
    cross("undeclared_languages", doc, caps)
    doc["gate"]["band"]["scope_languages"] = ["en"]  # a stated scope makes an undeclared detector usable
    doc["language_detector"] = policy(caps, gate=True)["language_detector"]
    validate_policy(ExecutionPolicy.model_validate(doc), caps_by_key(caps))


def test_no_rule_ties_band_to_the_gate_threshold():
    """Design §1 item 2 / §4: clear_below may lie above the gate detector's calibrated threshold."""
    caps = default_caps()
    doc = policy(caps, gate=True)
    doc["gate"]["band"]["clear_below"] = 0.9  # gate threshold is 0.5
    validate_policy(ExecutionPolicy.model_validate(doc), caps_by_key(caps))


def test_lint_flags_fail_open_gate():
    caps = default_caps()
    doc = policy(caps, gate=True)
    doc["gate"]["on_failure"] = {"mode": "FAIL_OPEN", "fallback": None}
    assert any("gate" in w for w in lint_policy(ExecutionPolicy.model_validate(doc)))
    assert lint_policy(ExecutionPolicy.model_validate(policy(caps, gate=True))) == []


# ---------------------------------------------------------------- 1.3 wire additions

def _result(**kw):
    base = dict(request_id="r1", detector_id="pi", detector_version="v1", status="ok", score=0.9, threshold=0.5,
                predicted_injection=True, latency_ms=1.0, input_sha256=SHA, input_tokens=4)
    base.update(kw)
    return DetectorResult(**base)


def test_observe_role_needs_v13():
    assert _result(role="observe").schema_version == "1.3"
    for v in ("1.1", "1.2"):
        with pytest.raises(ValidationError):
            _result(role="observe", schema_version=v)
    assert _result(role="shadow", schema_version="1.2").role == "shadow"


def test_classify_response_v13_fields():
    r = _result()
    resp = ClassifyResponse(request_id="r1", decision="flag", decided_by="pi", results=[r], policy_id="p",
                            policy_version="1", bundle_sha256=SHA, unscored_by_any_step=False,
                            escalation_reason="band")
    assert resp.bundle_sha256 == SHA
    v12 = ClassifyResponse(request_id="r1", decision="allow", decided_by="pi", results=[r])
    assert v12.unscored_by_any_step is None and v12.escalation_reason is None and v12.policy_id is None
    for bad in ({"bundle_sha256": "xyz"}, {"policy_id": "p"}, {"escalation_reason": "gray_zone"},
                {"decision": "quarantine"}):
        with pytest.raises(ValidationError):
            ClassifyResponse(**{"request_id": "r1", "decision": "flag", "decided_by": "pi", "results": [r], **bad})
