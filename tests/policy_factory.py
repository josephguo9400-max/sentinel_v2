"""Builders for CapabilityManifest / ExecutionPolicy documents in tests (no files, no models).

    caps = default_caps()                        # {"gate": text cap, "decide": token cap, "decide2": ..., ...}
    p = policy(caps, gate=True)                  # a valid shape-A policy dict
    ExecutionPolicy.model_validate(p)
"""
from __future__ import annotations

import copy

from contracts.capability import CapabilityManifest, capability_sha256

SHA = "a" * 64
TOK_SHA = "b" * 64
TS = "2026-10-30T00:00:00Z"


def capability(detector_id: str, version: str = "0.0.1", *, kind: str = "token", threshold: float = 0.5,
               languages=None, tokenizer: str | None = None, tokenizer_sha: str = TOK_SHA, **over) -> dict:
    token = kind == "token"
    d = {
        "schema_version": "2.0", "detector_id": detector_id, "detector_version": version, "artifact_format": "onnx",
        "files": ({f"{detector_id}/onnx/model.onnx": SHA, f"{detector_id}/onnx/tokenizer.json": tokenizer_sha}
                  if token else {f"{detector_id}/model.onnx": SHA}),
        "model_file": f"{detector_id}/onnx/model.onnx" if token else f"{detector_id}/model.onnx",
        "tokenizer_file": f"{detector_id}/onnx/tokenizer.json" if token else None,
        "runtime": {"python": "3.13.0", "onnxruntime": "1.29.0", "tokenizers": "0.23.3" if token else None,
                    "sklearn": None, "skl2onnx": None},
        "label_mapping": {"0": "benign", "1": "injection"},
        "input": {"kind": kind, "preprocessing": "deberta-v3-window-v1" if token else "tfidf-v1",
                  "tokenizer": tokenizer or ("hf:test" if token else "regex:[\\p{L}\\p{N}_]{2,}"),
                  "max_input_tokens": 512, "validated_languages": languages},
        "scoring": ({"aggregation": "max", "subwindow_tokens": 0, "subwindow_stride": None,
                     "score_segment_pieces": False} if token else
                    {"aggregation": "max", "subwindow_tokens": 64, "subwindow_stride": 32,
                     "score_segment_pieces": True}),
        "calibration": {"threshold": threshold, "chosen_on": "test", "rule": "test", "evidence_path": None,
                        "calibrated_with": None, "notes": None},
        "limits": {"recommended_timeout_ms": 1000, "max_batch": 1, "latency_evidence_path": None},
        "training_data_version": None, "created_at": TS,
    }
    for k, v in over.items():
        d[k] = v
    return d


def default_caps() -> dict[str, CapabilityManifest]:
    return {
        "gate": CapabilityManifest.model_validate(capability("fake-s1", kind="text", languages=["en"])),
        "decide": CapabilityManifest.model_validate(capability("fake")),
        "decide2": CapabilityManifest.model_validate(capability("fake-b")),
        "fallback": CapabilityManifest.model_validate(capability("fake-fb")),
        "observe": CapabilityManifest.model_validate(capability("fake-obs", kind="text", languages=["en"])),
        "shadow": CapabilityManifest.model_validate(capability("fake-shadow")),
    }


def ref(cap: CapabilityManifest) -> dict:
    return {"detector_id": cap.detector_id, "detector_version": cap.detector_version,
            "capability_sha256": capability_sha256(cap)}


def member(cap, *, timeout_ms=5000, max_chunks=4, failure=None) -> dict:
    return {"detector": ref(cap), "timeout_ms": timeout_ms, "max_chunks": max_chunks,
            "on_failure": failure or {"mode": "FAIL_CLOSED", "fallback": None}}


def policy(caps: dict, *, gate: bool = False, parallel: str | None = None, observe: bool = False,
           shadow: bool = False, coverage: str = "flag", **over) -> dict:
    d = {
        "schema_version": "1.0", "policy_id": "test-policy", "policy_version": "1", "entrypoint": "classify",
        "engine_version": "1.0.0",
        "chunking": {"plan_tokenizer": ref(caps["decide"]), "stride": 384, "structure_aware": True,
                     "pack_overlap_tokens": 126},
        "language_detector": None, "gate": None,
        "decide": {"mode": "single", "members": [member(caps["decide"])], "aggregate": None},
        "observe": [], "shadow": None, "coverage": {"mode": coverage, "deep_scan": None}, "canary": None,
        "rationale": "test",
    }
    if gate:
        d["gate"] = {"detector": ref(caps["gate"]),
                     "band": {"clear_below": 0.2, "flag_at_or_above": None, "finalize": ["allow"],
                              "scope_languages": ["en"]},
                     "timeout_ms": 1000, "deadline_margin_ms": 15, "max_chunks": 512,
                     "on_failure": {"mode": "FALLBACK", "fallback": {"detector": ref(caps["decide"]),
                                                                     "timeout_ms": 5000, "terminal": "FAIL_CLOSED"}}}
    if parallel:
        d["decide"] = {"mode": "parallel", "members": [member(caps["decide"]), member(caps["decide2"])],
                       "aggregate": parallel}
    if observe:
        d["observe"] = [{"detector": ref(caps["observe"]), "timeout_ms": 1000, "deadline_margin_ms": 15,
                         "max_chunks": 512, "concurrency": 1, "on_failure": {"mode": "ASYNC_ONLY", "fallback": None},
                         "band": None}]
    if shadow:
        d["shadow"] = {"detector": ref(caps["shadow"]), "timeout_ms": 1000, "queue": 4, "max_chunks": 8,
                       "intra_op_threads": 1, "on_failure": {"mode": "ASYNC_ONLY", "fallback": None}}
    if coverage == "deep_scan":
        d["coverage"]["deep_scan"] = {"detector": ref(caps["decide"]), "max_chunks": 16, "timeout_ms": 10000,
                                      "fallback": "flag"}
    if gate or any(o.get("band") and o["band"].get("scope_languages") for o in d["observe"]):
        d["language_detector"] = {"impl": "langdetect==1.0.9", "seed": 0, "min_letters": 20, "min_prob": 0.8,
                                  "max_chars": 2000, "tag_map": {}}
    for k, v in over.items():
        d[k] = v
    return copy.deepcopy(d)


def caps_by_key(caps: dict) -> dict:
    return {c.key: c for c in caps.values()}
