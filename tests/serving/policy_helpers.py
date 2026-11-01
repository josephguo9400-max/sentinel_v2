"""Build a ClassifierService from an arbitrary ExecutionPolicy and fake detectors (no files, no models).

The converter (serving.policy.legacy_to_v2) can only express the two legacy shapes; the failure matrix (T6) and
shape B (T7) need policies the legacy registry cannot describe, so these helpers pin fake detectors directly.

    specs = {("fake", "0.0.1"): spec(FakeDetector()), ("fake-s1", "0.0.1"): spec(FakeStage1(), kind="text")}
    svc = service(policy(caps, gate=True), specs)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ for policy_factory

from contracts.policy import ExecutionPolicy, iter_refs  # noqa: E402
from policy_factory import caps_by_key, capability, default_caps, member, policy, ref  # noqa: E402,F401
from serving.config import GatewayConfig  # noqa: E402
from serving.language import FixedLanguage  # noqa: E402
from serving.policy import bundle_sha256  # noqa: E402
from serving.registry import DetectorSpec, LoadedBundle  # noqa: E402
from serving.service import ClassifierService  # noqa: E402


def spec(det, *, kind: str = "token", timeout_ms: int = 5000, failure_policy: str = "fail_closed",
         version: str | None = None) -> DetectorSpec:
    return DetectorSpec(detector=det, detector_id=det.detector_id, detector_version=version or det.detector_version,
                        threshold=det.threshold, kind=kind, role="stage1" if kind == "text" else "stage2",
                        failure_policy=failure_policy, timeout_ms=timeout_ms,
                        max_input_tokens=getattr(det, "max_len", getattr(det, "max_input_tokens", 512)),
                        gray_zone=(0.2, 0.8) if kind == "text" else None)


def bundle(policy_doc: dict | ExecutionPolicy, specs: dict[tuple[str, str], DetectorSpec], *,
           shadow: DetectorSpec | None = None) -> LoadedBundle:
    p = policy_doc if isinstance(policy_doc, ExecutionPolicy) else ExecutionPolicy.model_validate(policy_doc)
    caps = caps_by_key(default_caps())
    missing = [r.key for r in iter_refs(p)
               if r.key not in specs and not (shadow is not None and r.key == p.shadow.detector.key)]
    assert not missing, f"no detector for {missing}"
    return LoadedBundle(policy=p, capabilities=caps, detectors=dict(specs), shadow=shadow,
                        bundle_sha256=bundle_sha256(p), policy_source="injected")


def service(policy_doc: dict | ExecutionPolicy, specs: dict[tuple[str, str], DetectorSpec], *,
            config: GatewayConfig | None = None, language: Any = None, shadow: DetectorSpec | None = None,
            metrics=None) -> ClassifierService:
    cfg = config or GatewayConfig(shadow_log_path="")
    return ClassifierService(config=cfg, metrics=metrics, language=language or FixedLanguage("en"),
                             bundle=bundle(policy_doc, specs, shadow=shadow))
