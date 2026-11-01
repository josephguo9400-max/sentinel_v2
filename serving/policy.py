"""Policy plumbing that is not the decision path: the legacy converter, bundle hashes and the env-var guard.

legacy_to_v2(registry_v1, manifests, config) -> (ExecutionPolicy, {key: CapabilityManifest})
    Pure (no file or model access). Converts a sentinel-registry-v1 state, its ArtifactManifests (v1.1 / v1.2) and a
    GatewayConfig into the policy the legacy gateway actually served and the capabilities it pins. This is what the
    gateway does in memory for a v1 registry (/health: policy_source "legacy_registry") and what
    `python -m serving.convert_registry` writes to artifacts/ (docs/design/capability-policy.md §7, migration 1).

bundle_sha256(policy, engine_version)
    sha256 over canonical(policy) + sorted(pinned capability_sha256) + engine_version (design §7). The policy pins
    every capability by sha256, and capabilities pin every file by sha256, so model and tokenizer bytes are covered.

check_decision_settings(config, environ)
    Under registry v2 every decision-affecting setting lives in the policy; setting one through SENTINEL_* (or a
    non-default GatewayConfig value) is a startup error naming the policy field (D25.3).
"""
from __future__ import annotations

import os
from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any, Mapping

from contracts import ArtifactManifest
from contracts._canonical import canonical_sha256
from contracts.capability import CapabilityManifest, capability_sha256
from contracts.policy import ExecutionPolicy, PolicyError, iter_refs, policy_sha256, validate_policy

from .config import GatewayConfig
from .engine_version import ENGINE_VERSION

TFIDF_TOKENIZER = "regex:[\\p{L}\\p{N}_]{2,}"
TOKENIZER_FILE = "tokenizer.json"
LANGDETECT_IMPL = "langdetect==1.0.9"


class DecisionSettingError(RuntimeError):
    """A decision-affecting setting was given outside the policy while a v2 registry is in use."""


# ---------------------------------------------------------------- engine version compatibility

def _semver(v: str) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in v.split("."))
    return a, b, c


def check_engine_compat(policy: ExecutionPolicy, engine_version: str = ENGINE_VERSION) -> None:
    """A policy validated on engine X.Y.Z runs on engine X.y.z >= X.Y.Z (same major)."""
    want, have = _semver(policy.engine_version), _semver(engine_version)
    if want[0] != have[0] or want > have:
        raise PolicyError("engine_version_incompatible", f"policy {policy.policy_id}@{policy.policy_version} was "
                                                         f"validated on engine {policy.engine_version}, this engine "
                                                         f"is {engine_version}")


# ---------------------------------------------------------------- bundle

def bundle_sha256(policy: ExecutionPolicy, engine_version: str = ENGINE_VERSION) -> str:
    caps = sorted({r.capability_sha256 for r in iter_refs(policy)})
    return canonical_sha256({"policy": policy.model_dump(mode="json"), "capability_sha256": caps,
                             "engine_version": engine_version})


def describe_bundle(policy: ExecutionPolicy, engine_version: str = ENGINE_VERSION) -> dict:
    return {"policy_id": policy.policy_id, "policy_version": policy.policy_version,
            "policy_sha256": policy_sha256(policy), "engine_version": engine_version,
            "policy_engine_version": policy.engine_version, "bundle_sha256": bundle_sha256(policy, engine_version),
            "capabilities": sorted(f"{r.label()}:{r.capability_sha256[:12]}" for r in {r.key: r for r in
                                                                                       iter_refs(policy)}.values())}


# ---------------------------------------------------------------- legacy converter (pure)

def manifest_to_capability(m: ArtifactManifest, rel_dir: str, config: GatewayConfig, *,
                           known_tokenizers: Mapping[str, str],
                           calibration: Mapping[str, Any] | None = None) -> CapabilityManifest:
    """ArtifactManifest v1.x -> CapabilityManifest 2.0 with the scoring rule the legacy gateway applied.

    `calibration` overrides fields of the derived Calibration (v1 manifests do not record how the threshold was
    chosen); everything else comes from the manifest and the config.
    """
    kind = {"tfidf-v1": "text", "deberta-v3-window-v1": "token"}.get(m.preprocessing_version)
    if kind is None:
        raise PolicyError("unsupported_preprocessing", f"{m.preprocessing_version!r} is not a gateway preprocessing")
    model_rel = f"{rel_dir}/{m.artifact_path}"
    files = {model_rel: m.sha256}
    tokenizer_file = None
    if kind == "token":
        if m.tokenizer not in known_tokenizers:
            raise PolicyError("tokenizer_mismatch", f"token model tokenizer {m.tokenizer!r} has no pinned sha256")
        tokenizer_file = str(PurePosixPath(model_rel).parent / TOKENIZER_FILE)
        files[tokenizer_file] = known_tokenizers[m.tokenizer]
        scoring = {"aggregation": "max", "subwindow_tokens": 0, "subwindow_stride": None,
                   "score_segment_pieces": False}
        tokenizer = m.tokenizer
    else:
        sub = config.stage1_subwindow_tokens
        scoring = {"aggregation": "max", "subwindow_tokens": sub,
                   "subwindow_stride": config.stage1_subwindow_stride if sub else None, "score_segment_pieces": True}
        tokenizer = m.tokenizer or TFIDF_TOKENIZER
    cal = {"threshold": m.threshold,
           "chosen_on": m.evaluation_dataset_version or m.training_data_version or "unrecorded",
           "rule": "not recorded in ArtifactManifest v1",
           "evidence_path": f"{rel_dir}/{m.metrics_path}" if m.metrics_path else None,
           "calibrated_with": None, "notes": None}
    cal.update(calibration or {})
    return CapabilityManifest.model_validate({
        "detector_id": m.detector_id, "detector_version": m.detector_version, "artifact_format": m.artifact_format,
        "files": files, "model_file": model_rel, "tokenizer_file": tokenizer_file,
        "runtime": m.runtime.model_dump(), "label_mapping": dict(m.label_mapping),
        "input": {"kind": kind, "preprocessing": m.preprocessing_version, "tokenizer": tokenizer,
                  "max_input_tokens": m.max_input_tokens, "validated_languages": m.input_languages},
        "scoring": scoring, "calibration": cal,
        "limits": {"recommended_timeout_ms": m.timeout_ms, "max_batch": 1, "latency_evidence_path": None},
        "training_data_version": m.training_data_version, "created_at": m.created_at,
    })


def _ref(c: CapabilityManifest) -> dict:
    return {"detector_id": c.detector_id, "detector_version": c.detector_version,
            "capability_sha256": capability_sha256(c)}


def _failure(policy: str) -> dict:
    return {"mode": {"fail_closed": "FAIL_CLOSED", "fail_open": "FAIL_OPEN"}[policy], "fallback": None}


def legacy_to_v2(registry_v1: Mapping[str, str | None], manifests: Mapping[str, ArtifactManifest],
                 config: GatewayConfig, *, known_tokenizers: Mapping[str, str],
                 calibrations: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
                 policy_id: str = "legacy-registry", policy_version: str = "1",
                 rationale: str = "converted in memory from a sentinel-registry-v1 state by serving.policy.legacy_to_v2",
                 engine_version: str = ENGINE_VERSION) -> tuple[ExecutionPolicy, dict[tuple[str, str],
                                                                                         CapabilityManifest]]:
    """The ExecutionPolicy equivalent to what the legacy ClassifierService served for this registry state.

    `registry_v1`: the pointers ("stage1", "stage2", "shadow" -> directory under artifacts/ or None) and
    "stage1_mode". `manifests`: the ArtifactManifest of every non-null pointer, keyed by pointer name.
    """
    calibrations = calibrations or {}
    mode = registry_v1.get("stage1_mode")
    caps: dict[str, CapabilityManifest] = {}
    for name in ("stage1", "stage2", "shadow"):
        if registry_v1.get(name) is None:
            continue
        m = manifests[name]
        caps[name] = manifest_to_capability(m, registry_v1[name], config, known_tokenizers=known_tokenizers,
                                            calibration=calibrations.get((m.detector_id, m.detector_version)))
    s2m, s2 = manifests["stage2"], caps["stage2"]
    s2_timeout = config.stage2_timeout_ms or s2m.timeout_ms
    decide_failure = _failure(s2m.failure_policy)
    policy: dict[str, Any] = {
        "policy_id": policy_id, "policy_version": policy_version, "entrypoint": "classify",
        "engine_version": engine_version, "rationale": rationale,
        "chunking": {"plan_tokenizer": _ref(s2), "stride": config.stride, "structure_aware": config.structure_aware,
                     "pack_overlap_tokens": config.pack_overlap_tokens},
        "language_detector": None, "gate": None,
        "decide": {"mode": "single", "aggregate": None, "members": [
            {"detector": _ref(s2), "timeout_ms": s2_timeout, "max_chunks": config.max_chunks,
             "on_failure": decide_failure}]},
        "observe": [], "shadow": None, "canary": None,
        "coverage": {"mode": config.coverage_policy, "deep_scan": None},
    }
    if config.coverage_policy == "deep_scan":
        policy["coverage"]["deep_scan"] = {"detector": _ref(s2), "max_chunks": config.deep_scan_max_chunks,
                                           "timeout_ms": config.deep_scan_timeout_ms,
                                           "fallback": config.deep_scan_fallback}
    if "stage1" in caps:
        s1m, s1 = manifests["stage1"], caps["stage1"]
        if s1m.gray_zone is None:
            raise PolicyError("band", "a legacy Stage 1 manifest without gray_zone cannot be converted")
        lo, hi = s1m.gray_zone
        scope = list(s1m.input_languages) if s1m.input_languages is not None else config.undeclared_languages
        s1_timeout = config.stage1_timeout_ms or s1m.timeout_ms
        band = {"clear_below": lo, "flag_at_or_above": None, "finalize": ["allow"], "scope_languages": scope}
        if mode == "route":
            if config.stage1_finalize_flag:
                band.update(finalize=["allow", "flag"], flag_at_or_above=hi)
            policy["gate"] = {
                "detector": _ref(s1), "band": band, "timeout_ms": s1_timeout,
                "deadline_margin_ms": config.stage1_deadline_margin_ms, "max_chunks": config.stage1_max_chunks,
                # Legacy: "Stage 1 failure clears nothing ... every chunk escalates" = FALLBACK to the decider.
                "on_failure": {"mode": "FALLBACK", "fallback": {"detector": _ref(s2), "timeout_ms": s2_timeout,
                                                                "terminal": decide_failure["mode"]}}}
        elif mode == "observe":
            policy["observe"] = [{
                "detector": _ref(s1), "timeout_ms": s1_timeout,
                "deadline_margin_ms": config.stage1_deadline_margin_ms, "max_chunks": config.stage1_max_chunks,
                "concurrency": config.stage1_observe_concurrency,
                "on_failure": {"mode": "ASYNC_ONLY", "fallback": None}, "band": band}]
        else:
            raise PolicyError("stage1_mode_missing", "a registry with stage1 needs stage1_mode route or observe")
        if scope:
            policy["language_detector"] = {"impl": LANGDETECT_IMPL, "seed": config.lang_seed,
                                           "min_letters": config.lang_min_letters, "min_prob": config.lang_min_prob,
                                           "max_chars": config.lang_max_chars, "tag_map": {}}
    if "shadow" in caps:
        sm = manifests["shadow"]
        policy["shadow"] = {"detector": _ref(caps["shadow"]), "timeout_ms": config.shadow_timeout_ms or sm.timeout_ms,
                            "queue": config.shadow_queue, "max_chunks": config.shadow_max_chunks,
                            "intra_op_threads": config.shadow_intra_op_threads,
                            "on_failure": {"mode": "ASYNC_ONLY", "fallback": None}}
    p = ExecutionPolicy.model_validate(policy)
    by_key = {c.key: c for c in caps.values()}
    validate_policy(p, by_key)
    return p, by_key


# ---------------------------------------------------------------- decision-affecting settings (D25.3)

# GatewayConfig field -> the policy (or capability) field that owns it under registry v2.
DECISION_FIELDS: dict[str, str] = {
    "coverage_policy": "coverage.mode",
    "deep_scan_max_chunks": "coverage.deep_scan.max_chunks",
    "deep_scan_timeout_ms": "coverage.deep_scan.timeout_ms",
    "deep_scan_fallback": "coverage.deep_scan.fallback",
    "max_chunks": "decide.members[].max_chunks",
    "stage2_timeout_ms": "decide.members[].timeout_ms",
    "stage1_max_chunks": "gate.max_chunks / observe[].max_chunks",
    "stage1_finalize_flag": "gate.band.finalize / gate.band.flag_at_or_above",
    "stage1_subwindow_tokens": "the gate/observe capability's scoring.subwindow_tokens (a new detector_version)",
    "stage1_subwindow_stride": "the gate/observe capability's scoring.subwindow_stride (a new detector_version)",
    "stage1_undeclared_languages": "gate.band.scope_languages / observe[].band.scope_languages",
    "stage1_timeout_ms": "gate.timeout_ms / observe[].timeout_ms",
    "stage1_deadline_margin_ms": "gate.deadline_margin_ms / observe[].deadline_margin_ms",
    "stage1_observe_concurrency": "observe[].concurrency",
    "lang_seed": "language_detector.seed",
    "lang_min_letters": "language_detector.min_letters",
    "lang_min_prob": "language_detector.min_prob",
    "lang_max_chars": "language_detector.max_chars",
    "stride": "chunking.stride",
    "structure_aware": "chunking.structure_aware",
    "pack_overlap_tokens": "chunking.pack_overlap_tokens",
    "shadow_queue": "shadow.queue",
    "shadow_timeout_ms": "shadow.timeout_ms",
    "shadow_max_chunks": "shadow.max_chunks",
    "shadow_intra_op_threads": "shadow.intra_op_threads",
    "extra_tokenizers": "the capability's files (tokenizer sha256 pinned in the CapabilityManifest)",
    # Fields that only ever applied to injected test detectors are not decision settings of a registry gateway.
}


def env_name(field: str) -> str:
    return f"SENTINEL_{field.upper()}"


def check_decision_settings(config: GatewayConfig, environ: Mapping[str, str] | None = None) -> None:
    """Raise DecisionSettingError if any decision-affecting setting is given outside the policy."""
    environ = os.environ if environ is None else environ
    defaults = {f.name: f.default for f in fields(GatewayConfig)}
    problems = []
    for name, target in DECISION_FIELDS.items():
        if env_name(name) in environ:
            problems.append(f"{env_name(name)} is set: under registry v2 change {target} in the active policy")
        elif getattr(config, name) != defaults[name]:
            problems.append(f"GatewayConfig.{name}={getattr(config, name)!r}: under registry v2 change {target} "
                            f"in the active policy")
    if problems:
        raise DecisionSettingError("decision-affecting settings outside the policy (D25.3): " + "; ".join(problems))
