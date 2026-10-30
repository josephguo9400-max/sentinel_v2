"""Write the v2 files converted from the committed v1 registry and manifests (migration step 1 / 4).

    python -m serving.convert_registry            # writes artifacts/capabilities/... and artifacts/policies/...
    python -m serving.convert_registry --check    # exit 1 if a committed file differs from the conversion

Pure conversion (serving/policy.py legacy_to_v2) with the default GatewayConfig, plus CALIBRATION_RECORDS: what the
v1 manifests do not record about how each threshold was chosen, with its sources. The same records are used when the
gateway converts a v1 registry in memory, so the derived capability_sha256 values equal the committed ones.

Policies written (design §5 / §7):
- d23-deberta-decides/1        reports/policy-runtime/registry_v1_d23.json, the v1 state D23 served
                               (DeBERTa decides, Stage 1 observe-only); artifacts/registry.json now holds the
                               resulting v2 bundle
- cascade-tfidf-allow-only/1   reports/cascade-and-shadow/registry_integration_route.json (route mode, v2.1;
                               rejected by D23)
- cascade-tfidf-provisional/1  route mode with stage1-tfidf-v2's gray zone (historical; never served by default)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contracts import ArtifactManifest
from contracts._canonical import canonical_json
from contracts.capability import CapabilityManifest
from contracts.policy import ExecutionPolicy

from .config import GatewayConfig
from .policy import legacy_to_v2
from .registry import KNOWN_TOKENIZERS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

_TRAINING_RULE = {"aggregation": "max", "subwindow_tokens": 0, "subwindow_stride": None,
                  "score_segment_pieces": False}
_S1_NOTE = ("D25.6: the threshold was chosen on the training-side score (max over 512-regex-token windows of the whole "
            "text, stride 384: training/stage1_onnx.py score; rule in training/train_stage1.py choose_threshold). The "
            "gateway scores per chunk, per packed segment piece and over 64/32-token sub-windows (`scoring`). Not "
            "re-calibrated: Stage 1 is observe-only under D23, so the mismatch has no decision impact; a gate band "
            "on this capability is chosen on its served scores, not on this threshold.")
CALIBRATION_RECORDS: dict[tuple[str, str], dict] = {
    ("stage1-tfidf-lr", "stage1-tfidf-v2"): {
        "chosen_on": "splits-v2:val:d84853ecc84cc87e",
        "rule": "maximize F1 on val (pooled); training/train_stage1.py choose_threshold",
        "calibrated_with": _TRAINING_RULE, "notes": _S1_NOTE},
    ("stage1-tfidf-lr", "stage1-tfidf-v2.1"): {
        "chosen_on": "splits-v2:val:d84853ecc84cc87e",
        "rule": "maximize F1 on val (pooled); training/train_stage1.py choose_threshold; same model bytes and "
                "threshold as stage1-tfidf-v2 (candidates/stage1-tfidf-v2.1/evaluation.json derived_from)",
        "evidence_path": "candidates/stage1-tfidf-v2/evaluation.json",
        "calibrated_with": _TRAINING_RULE, "notes": _S1_NOTE},
    ("deberta-pi-v2", "2.0.0+90c9989b.onnx.fp32"): {
        "chosen_on": "upstream:protectai/deberta-v3-base-prompt-injection-v2@90c9989b1a342275dd0d1a95aad283c04e075671",
        "rule": "upstream default 0.5 (argmax of two classes); not re-calibrated in this project",
        "notes": "Scored per 510-token window of the chunk plan, max over the chunks the policy selects."},
    ("deberta-pi-v2-int8", "2.0.0+90c9989b.onnx.int8-dyn-qint8-matmul-perchannel.v2"): {
        "chosen_on": "upstream:protectai/deberta-v3-base-prompt-injection-v2@90c9989b1a342275dd0d1a95aad283c04e075671",
        "rule": "upstream default 0.5 kept after dynamic int8 quantization (scripts/quantize_deberta.py); not "
                "re-calibrated",
        "notes": "serving/bench_results/int8_compare.json measures its agreement with fp32."},
}

# Frozen: this text is part of the hashed content of the committed d23-deberta-decides/1 policy, so the converter must
# keep emitting it for policy_version "1" (T1 golden, `--check`). "artifacts/registry.json (sentinel-registry-v1)"
# was true when /1 was converted (before 764e160); the same v1 pointers are preserved in
# reports/policy-runtime/registry_v1_d23.json, which SOURCES reads. A new version of this policy needs a new
# policy_version with its own rationale (review r1 n4).
D23_RATIONALE = ("D23 (docs/RFC.md, milestone 3): the cascade's kill criterion fired on the replay set; DeBERTa decides "
                 "every request and Stage 1 runs observe-only. Registry change pending the release gate "
                 "(serving/README.md 'Registry'). Converted from artifacts/registry.json (sentinel-registry-v1) by "
                 "serving.convert_registry.")
SOURCES = {
    # The v1 state this was converted from; artifacts/registry.json now holds the resulting v2 bundle.
    "d23-deberta-decides": {"registry": "reports/policy-runtime/registry_v1_d23.json",
                            "rationale": D23_RATIONALE},
    "cascade-tfidf-allow-only": {
        "registry": "reports/cascade-and-shadow/registry_integration_route.json",
        "rationale": "Rejected by D23: replay recall -15.0 pt [-19.4, -10.7] "
                     "(reports/cascade-and-shadow/replay_counterfactual_served_faithful.json). Route mode with "
                     "stage1-tfidf-v2.1 (allow-only gray zone [0.505068, 1.0]). Converted by serving.convert_registry."},
    "cascade-tfidf-provisional": {
        "registry": {"schema": "sentinel-registry-v1", "stage1": "candidates/stage1-tfidf-v2", "stage1_mode": "route",
                     "stage2": "baseline/deberta-pi-v2", "shadow": None},
        "rationale": "Historical: route mode with stage1-tfidf-v2's provisional gray zone [0.282715, 0.764614] "
                     "(training/CASCADE.md). Never the default deployment. Converted by serving.convert_registry."},
}


def read_manifests(pointers: dict, artifacts: Path = ARTIFACTS) -> dict[str, ArtifactManifest]:
    return {name: ArtifactManifest.model_validate_json((artifacts / rel / "manifest.json").read_bytes())
            for name, rel in pointers.items() if name in ("stage1", "stage2", "shadow") and rel is not None}


def convert(policy_id: str, config: GatewayConfig | None = None, artifacts: Path = ARTIFACTS
            ) -> tuple[ExecutionPolicy, dict[tuple[str, str], CapabilityManifest]]:
    src = SOURCES[policy_id]
    reg = src["registry"]
    pointers = json.loads((ROOT / reg).read_text()) if isinstance(reg, str) else dict(reg)
    return legacy_to_v2(pointers, read_manifests(pointers, artifacts), config or GatewayConfig(),
                        known_tokenizers=KNOWN_TOKENIZERS, calibrations=CALIBRATION_RECORDS,
                        policy_id=policy_id, policy_version="1", rationale=src["rationale"])


def capability_path(c: CapabilityManifest) -> str:
    return f"capabilities/{c.detector_id}/{c.detector_version}/capability.json"


def policy_path(p: ExecutionPolicy) -> str:
    return f"policies/{p.policy_id}/{p.policy_version}/policy.json"


def render(model) -> str:
    """Committed form: indented, sorted keys; its canonical JSON is what the hashes cover."""
    return json.dumps(json.loads(canonical_json(model.model_dump(mode="json"))), indent=1, sort_keys=True) + "\n"


def outputs(artifacts: Path = ARTIFACTS) -> dict[str, str]:
    out: dict[str, str] = {}
    for pid in SOURCES:
        p, caps = convert(pid, artifacts=artifacts)
        out[policy_path(p)] = render(p)
        for c in caps.values():
            out[capability_path(c)] = render(c)
    # The int8 candidate is not in any policy yet; its capability is published so a future policy can pin it.
    m = ArtifactManifest.model_validate_json((artifacts / "candidates/deberta-pi-v2-int8/manifest.json").read_bytes())
    from .policy import manifest_to_capability
    c = manifest_to_capability(m, "candidates/deberta-pi-v2-int8", GatewayConfig(), known_tokenizers=KNOWN_TOKENIZERS,
                               calibration=CALIBRATION_RECORDS.get((m.detector_id, m.detector_version)))
    out[capability_path(c)] = render(c)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    bad = []
    for rel, text in sorted(outputs().items()):
        path = ARTIFACTS / rel
        if a.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                bad.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote artifacts/{rel}")
    if bad:
        print("differs from the conversion: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
