"""Manifest-driven detector registry.

`artifacts/registry.json` names the active Stage 1, the active Stage 2 and an optional shadow detector, each
as a directory under `artifacts/` holding a `manifest.json` (contracts.ArtifactManifest v1.1 or v1.2):

    {"schema": "sentinel-registry-v1",
     "stage1": "candidates/stage1-tfidf-v2",
     "stage1_mode": "route",
     "stage2": "baseline/deberta-pi-v2",
     "shadow": "candidates/deberta-pi-v2-int8"}

`stage1_mode` says what the Stage 1 pointer may do. It is REQUIRED whenever `stage1` is set: a registry with a Stage 1
pointer and no `stage1_mode` gets a per-pointer rejection `stage1_mode_missing` (Stage 1 is not loaded, Stage 2
serves alone, /health is degraded; a reload is refused). There is no default, so dropping the key can never hand
Stage 1 the power to finalize.
- "route":   Stage 1 clears chunks and may finalize "allow" (the cascade, serving/service.py steps 3-5).
- "observe": Stage 1 never routes or decides. Stage 2 scores the request exactly as if Stage 1 were absent; Stage 1
             runs after the decision is fixed and its result is reported with role="shadow" (owner decision D23).

Every detector is verified before it is loaded. A detector is rejected, with a reason code, when:

- `path_escape`        the pointer is not a plain relative path, or the artifact directory, manifest, model or
                       tokenizer resolves (symlinks followed) outside its directory / outside artifacts/
- `missing_file`       the manifest, model or tokenizer file does not exist
- `manifest_invalid`   manifest.json is not a valid ArtifactManifest (schema 1.1 or 1.2)
- `role_mismatch`      the manifest's cascade_role does not fit the pointer (stage1 needs stage1; stage2 needs
                       stage2 or single; shadow accepts any role)
- `unsupported_preprocessing` / `tokenizer_mismatch`   the gateway does not implement this preprocessing, or the
                       declared tokenizer does not belong to it, or the tokenizer file's sha256 is not the pinned one
- `runtime_mismatch`   see RUNTIME RULE below
- `sha256_mismatch`    sha256 of the model bytes differs from the manifest
- `load_failed`        onnxruntime or the tokenizer refused the verified bytes
- `stage1_mode_missing` (stage1 only) the registry sets `stage1` without `stage1_mode`

RUNTIME RULE: python and onnxruntime must match the installed versions in major.minor (3.13.x == 3.13.y,
1.29.x == 1.29.y); patch releases are treated as compatible. Token models must declare `tokenizers`, and it
must match in major.minor as well. `sklearn` / `skl2onnx` are export-time versions and are not needed at
serving time (the ONNX graph is self-contained), so they are recorded but not checked.

The model file is read into memory once; that same byte string is hashed and handed to onnxruntime, so the
file cannot change between verification and load.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from contracts import ArtifactManifest
from contracts._validators import relative_artifact_path

log = logging.getLogger("sentinel.registry")

REGISTRY_SCHEMA = "sentinel-registry-v1"
POINTERS = ("stage1", "stage2", "shadow")
STAGE1_MODES = ("route", "observe")
ROLES_FOR_POINTER = {"stage1": {"stage1"}, "stage2": {"stage2", "single"}, "shadow": {"stage1", "stage2", "single"}}

# preprocessing_version -> model kind the gateway implements for it.
TEXT_PREPROCESSING = "tfidf-v1"  # ONNX graph takes raw strings (serving/detector.py OnnxTfidfDetector)
TOKEN_PREPROCESSING = "deberta-v3-window-v1"  # [CLS] + <=510 content ids + [SEP] (serving/encoding.py)
TFIDF_TOKENIZER = "regex:[\\p{L}\\p{N}_]{2,}"
# Token-model tokenizers this gateway knows, with the sha256 of the tokenizer.json that the name stands for.
# The tokenizer file sits next to the model file. (Manifests have one sha256 field, for the model; see
# docs/contract_requests.md for the request to pin auxiliary files in the manifest itself.)
KNOWN_TOKENIZERS = {
    "hf:deberta-v3": "752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f",
}
TOKENIZER_FILE = "tokenizer.json"


class RegistryError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class InstalledRuntime:
    python: str
    onnxruntime: str
    tokenizers: str | None

    @classmethod
    def current(cls) -> "InstalledRuntime":
        import onnxruntime

        try:
            import tokenizers
            tok = tokenizers.__version__
        except ImportError:  # pragma: no cover
            tok = None
        return cls(python=platform.python_version(), onnxruntime=onnxruntime.__version__, tokenizers=tok)


def major_minor(version: str) -> tuple[int, int]:
    m = re.match(r"^(\d+)\.(\d+)", version)
    if not m:
        raise ValueError(f"cannot read major.minor from {version!r}")
    return int(m.group(1)), int(m.group(2))


def check_runtime(manifest: Any, installed: InstalledRuntime, kind: str) -> None:
    """`manifest`: anything with a contracts.Runtime `.runtime` (ArtifactManifest or CapabilityManifest)."""
    rt = manifest.runtime
    pairs = [("python", rt.python, installed.python), ("onnxruntime", rt.onnxruntime, installed.onnxruntime)]
    if kind == "token":
        if rt.tokenizers is None:
            raise RegistryError("runtime_mismatch", "token models must declare runtime.tokenizers")
        pairs.append(("tokenizers", rt.tokenizers, installed.tokenizers or "0.0"))
    for name, want, have in pairs:
        if major_minor(want) != major_minor(have):
            raise RegistryError("runtime_mismatch",
                                f"{name} {want} in manifest, {have} installed (rule: same major.minor)")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_inside(base: Path, rel: str, what: str) -> Path:
    """base/rel with symlinks resolved; it must stay inside base (also resolved)."""
    base_r = base.resolve()
    target = (base / rel).resolve()
    if not _inside(target, base_r) or target == base_r:
        raise RegistryError("path_escape", f"{what} {rel!r} resolves to {target}, outside {base_r}")
    return target


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class DetectorSpec:
    """A loaded detector plus the routing settings the gateway applies to it."""

    detector: Any
    detector_id: str
    detector_version: str
    threshold: float
    kind: Literal["token", "text"]
    role: Literal["stage1", "stage2", "single"]  # cascade_role from the manifest (or "single" when injected)
    failure_policy: Literal["fail_closed", "fail_open"]
    timeout_ms: int
    gray_zone: tuple[float, float] | None = None
    input_languages: list[str] | None = None
    max_input_tokens: int = 512
    tokenizer: str | None = None
    manifest_schema: str | None = None
    artifact_dir: str | None = None
    sha256: str | None = None
    manifest: Any = None  # the ArtifactManifest (v1 registry) it was loaded from
    capability: Any = None  # the CapabilityManifest (v2 registry, or converted from a v1 manifest)

    def describe(self) -> dict:
        return {"detector_id": self.detector_id, "detector_version": self.detector_version,
                "threshold": self.threshold, "kind": self.kind, "cascade_role": self.role,
                "failure_policy": self.failure_policy, "timeout_ms": self.timeout_ms,
                "gray_zone": list(self.gray_zone) if self.gray_zone else None,
                "input_languages": self.input_languages, "max_input_tokens": self.max_input_tokens,
                "tokenizer": self.tokenizer, "manifest_schema": self.manifest_schema,
                "artifact_dir": self.artifact_dir, "model_sha256": self.sha256}


@dataclass
class Rejection:
    pointer: str
    path: str | None
    code: str
    detail: str

    def as_dict(self) -> dict:
        return {"pointer": self.pointer, "path": self.path, "code": self.code, "detail": self.detail}


@dataclass
class LoadedRegistry:
    pointers: dict[str, str | None]  # the three POINTERS plus "stage1_mode"
    specs: dict[str, DetectorSpec] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def stage1_mode(self) -> str:
        """The mode to serve with. None in the file (only possible with no or a rejected Stage 1) -> observe."""
        return self.pointers.get("stage1_mode") or "observe"


def read_registry(path: Path) -> dict[str, str | None]:
    """The three pointers plus "stage1_mode" (None when the file omits it; load_registry then rejects stage1)."""
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise RegistryError("missing_file", f"registry {path} not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("registry_invalid", f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError("registry_invalid", f"registry must be an object with schema {REGISTRY_SCHEMA!r}")
    extra = set(raw) - {"schema", "comment", "stage1_mode", *POINTERS}
    if extra:
        raise RegistryError("registry_invalid", f"unknown registry keys: {sorted(extra)}")
    out: dict[str, str | None] = {}
    for p in POINTERS:
        v = raw.get(p)
        if v is not None and not isinstance(v, str):
            raise RegistryError("registry_invalid", f"{p} must be a string or null")
        out[p] = v
    if out["stage2"] is None:
        raise RegistryError("registry_invalid", "stage2 is required (it is the detector of last resort)")
    mode = raw.get("stage1_mode")
    if mode is not None and mode not in STAGE1_MODES:
        raise RegistryError("registry_invalid", f"stage1_mode must be one of {list(STAGE1_MODES)}, got {mode!r}")
    out["stage1_mode"] = mode
    return out


class ModelCache:
    """Reuses onnxruntime sessions across reloads when the verified model bytes are identical."""

    def __init__(self) -> None:
        self._items: dict[tuple, Any] = {}

    def get(self, key: tuple, build):
        if key not in self._items:
            self._items[key] = build()
        return self._items[key]


def load_detector(artifacts_root: Path, pointer_name: str, rel_dir: str, *, intra_op_threads: int = 0,
                  installed: InstalledRuntime | None = None, cache: ModelCache | None = None,
                  known_tokenizers: dict[str, str] | None = None) -> DetectorSpec:
    """Verify and load one detector. Raises RegistryError with a reason code."""
    from .detector import OnnxDebertaDetector, OnnxTfidfDetector

    installed = installed or InstalledRuntime.current()
    known = KNOWN_TOKENIZERS if known_tokenizers is None else known_tokenizers
    try:
        relative_artifact_path(rel_dir)
    except ValueError as exc:
        raise RegistryError("path_escape", f"pointer {rel_dir!r}: {exc}") from exc
    root = Path(artifacts_root)
    art_dir = resolve_inside(root, rel_dir, "artifact directory")
    manifest_path = resolve_inside(art_dir, "manifest.json", "manifest")
    if not manifest_path.is_file():
        raise RegistryError("missing_file", f"{manifest_path} not found")
    try:
        manifest = ArtifactManifest.model_validate_json(manifest_path.read_bytes())
    except Exception as exc:  # pydantic ValidationError or decode error
        raise RegistryError("manifest_invalid", str(exc).splitlines()[0] + f" ({manifest_path})") from exc
    if manifest.cascade_role not in ROLES_FOR_POINTER[pointer_name]:
        raise RegistryError("role_mismatch", f"{pointer_name} pointer cannot use cascade_role "
                                             f"{manifest.cascade_role!r}")
    if manifest.preprocessing_version == TEXT_PREPROCESSING:
        kind = "text"
        if manifest.tokenizer not in (None, TFIDF_TOKENIZER):
            raise RegistryError("tokenizer_mismatch", f"tfidf-v1 counts {TFIDF_TOKENIZER}, manifest declares "
                                                      f"{manifest.tokenizer!r}")
    elif manifest.preprocessing_version == TOKEN_PREPROCESSING:
        kind = "token"
        if manifest.tokenizer not in known:
            raise RegistryError("tokenizer_mismatch", f"token model needs a known tokenizer "
                                                      f"{sorted(known)}, got {manifest.tokenizer!r}")
    else:
        raise RegistryError("unsupported_preprocessing", f"{manifest.preprocessing_version!r} is not one of "
                                                         f"{[TEXT_PREPROCESSING, TOKEN_PREPROCESSING]}")
    check_runtime(manifest, installed, kind)

    model_path = resolve_inside(art_dir, manifest.artifact_path, "artifact_path")
    if not model_path.is_file():
        raise RegistryError("missing_file", f"{model_path} not found (weights are not committed; see the "
                                            f"script named in serving/README.md)")
    model_bytes = model_path.read_bytes()
    digest = sha256_bytes(model_bytes)
    if digest != manifest.sha256:
        raise RegistryError("sha256_mismatch", f"{manifest.artifact_path}: got {digest}, manifest {manifest.sha256}")
    injection_index = int(next(k for k, v in manifest.label_mapping.items() if v == "injection"))

    tok_bytes = None
    if kind == "token":
        tok_rel = str(Path(manifest.artifact_path).parent / TOKENIZER_FILE).removeprefix("./")
        tok_path = resolve_inside(art_dir, tok_rel, "tokenizer")
        if not tok_path.is_file():
            raise RegistryError("missing_file", f"{tok_path} not found")
        tok_bytes = tok_path.read_bytes()
        tok_digest = sha256_bytes(tok_bytes)
        if tok_digest != known[manifest.tokenizer]:
            raise RegistryError("tokenizer_mismatch", f"{tok_rel} sha256 {tok_digest} is not the pinned "
                                                      f"{manifest.tokenizer} tokenizer")

    def build():
        if kind == "text":
            return OnnxTfidfDetector(model_bytes, detector_id=manifest.detector_id,
                                     detector_version=manifest.detector_version, threshold=manifest.threshold,
                                     injection_index=injection_index, max_input_tokens=manifest.max_input_tokens)
        return OnnxDebertaDetector(model_bytes, tok_bytes, detector_id=manifest.detector_id,
                                   detector_version=manifest.detector_version, threshold=manifest.threshold,
                                   injection_index=injection_index, max_len=manifest.max_input_tokens,
                                   intra_op_threads=intra_op_threads)

    key = (str(art_dir), digest, manifest.model_dump_json(), intra_op_threads)
    try:
        det = (cache.get(key, build) if cache is not None else build())
    except Exception as exc:
        raise RegistryError("load_failed", f"{type(exc).__name__}: {exc}") from exc
    return DetectorSpec(detector=det, detector_id=manifest.detector_id, detector_version=manifest.detector_version,
                        threshold=manifest.threshold, kind=kind, role=manifest.cascade_role,
                        failure_policy=manifest.failure_policy, timeout_ms=manifest.timeout_ms,
                        gray_zone=tuple(manifest.gray_zone) if manifest.gray_zone else None,
                        input_languages=list(manifest.input_languages) if manifest.input_languages else None,
                        max_input_tokens=manifest.max_input_tokens,
                        tokenizer=manifest.tokenizer or (TFIDF_TOKENIZER if kind == "text" else None),
                        manifest_schema=manifest.schema_version, artifact_dir=rel_dir, sha256=digest,
                        manifest=manifest)


def load_registry(registry_path: Path, artifacts_root: Path, *, intra_op_threads: int = 0,
                  intra_op_by_pointer: dict[str, int] | None = None, installed: InstalledRuntime | None = None,
                  cache: ModelCache | None = None, known_tokenizers: dict[str, str] | None = None) -> LoadedRegistry:
    """Load every pointer. Per-pointer failures become Rejections (logged); a broken registry file raises."""
    pointers = read_registry(registry_path)
    out = LoadedRegistry(pointers=pointers)
    for name in POINTERS:
        rel = pointers[name]
        if rel is None:
            continue
        if name == "stage1" and pointers.get("stage1_mode") is None:
            detail = (f"registry sets stage1 -> {rel} without stage1_mode; add \"stage1_mode\": \"observe\" or "
                      f"\"route\" (there is no default)")
            log.error("registry: rejected stage1 -> %s: stage1_mode_missing", rel)
            out.rejections.append(Rejection(name, rel, "stage1_mode_missing", detail))
            continue
        try:
            threads = (intra_op_by_pointer or {}).get(name, intra_op_threads)
            spec = load_detector(artifacts_root, name, rel, intra_op_threads=threads, installed=installed,
                                 cache=cache, known_tokenizers=known_tokenizers)
        except RegistryError as exc:
            log.error("registry: rejected %s -> %s: %s", name, rel, exc)
            out.rejections.append(Rejection(name, rel, exc.code, exc.detail))
            continue
        if name == "stage1" and spec.kind != "text":
            out.rejections.append(Rejection(name, rel, "role_mismatch",
                                            "stage1 must be a text model (tfidf-v1) in this gateway"))
            continue
        if name == "stage2" and spec.kind != "token":
            out.rejections.append(Rejection(name, rel, "role_mismatch", "stage2 must be a token model"))
            continue
        log.info("registry: loaded %s -> %s (%s@%s, sha256 %s)", name, rel, spec.detector_id,
                 spec.detector_version, spec.sha256[:12])
        out.specs[name] = spec
    return out


# ====================================================================== registry v2 (D25)
#
#   {"schema": "sentinel-registry-v2",
#    "active": "policies/d23-deberta-decides/1/policy.json",
#    "previous": null,                      # or another policy path: kept loaded, rollback is a pointer flip
#    "capabilities_root": "capabilities",
#    "status": "pending_gate",              # pending_gate | approved (written by training/gate.py promote)
#    "comment": "..."}
#
# Paths are relative to artifacts/. A policy pins each detector by (detector_id, detector_version,
# capability_sha256); the capability lives at <capabilities_root>/<detector_id>/<detector_version>/capability.json
# and pins every file it loads by sha256 (paths relative to artifacts/). Additional reason codes:
# - `policy_invalid`       the policy file is not a valid ExecutionPolicy, or its id/version disagree with its path
# - `capability_invalid`   capability.json is not a valid CapabilityManifest 2.0
# - PolicyError codes      (capability_sha256_mismatch, kind_mismatch, canary_unsupported, engine_version_incompatible, ...)
# Unlike v1, a v2 bundle is all or nothing: the active policy is served exactly as written or not at all (a bundle
# verdict would not mean anything for a partially loaded bundle).

REGISTRY_SCHEMA_V2 = "sentinel-registry-v2"
REGISTRY_V2_STATUS = ("pending_gate", "approved")


def registry_schema(path: Path) -> str:
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise RegistryError("missing_file", f"registry {path} not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("registry_invalid", f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") not in (REGISTRY_SCHEMA, REGISTRY_SCHEMA_V2):
        raise RegistryError("registry_invalid", f"registry schema must be {REGISTRY_SCHEMA!r} or {REGISTRY_SCHEMA_V2!r}")
    return raw["schema"]


def read_registry_v2(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != REGISTRY_SCHEMA_V2:
        raise RegistryError("registry_invalid", f"registry must have schema {REGISTRY_SCHEMA_V2!r}")
    extra = set(raw) - {"schema", "comment", "active", "previous", "capabilities_root", "status"}
    if extra:
        raise RegistryError("registry_invalid", f"unknown registry keys: {sorted(extra)}")
    out = {"active": raw.get("active"), "previous": raw.get("previous"),
           "capabilities_root": raw.get("capabilities_root", "capabilities"), "status": raw.get("status")}
    for k in ("active", "previous", "capabilities_root"):
        v = out[k]
        if v is None and k != "previous":
            raise RegistryError("registry_invalid", f"{k} is required")
        if v is not None:
            try:
                relative_artifact_path(v)
            except (ValueError, TypeError) as exc:
                raise RegistryError("path_escape", f"{k} {v!r}: {exc}") from exc
    if out["status"] is not None and out["status"] not in REGISTRY_V2_STATUS:
        raise RegistryError("registry_invalid", f"status must be one of {list(REGISTRY_V2_STATUS)}")
    return out


def load_policy_file(artifacts_root: Path, rel: str):
    from contracts.policy import ExecutionPolicy

    path = resolve_inside(Path(artifacts_root), rel, "policy")
    if not path.is_file():
        raise RegistryError("missing_file", f"{path} not found")
    try:
        policy = ExecutionPolicy.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise RegistryError("policy_invalid", str(exc).splitlines()[0] + f" ({rel})") from exc
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[-1] == "policy.json" and (parts[-3], parts[-2]) != (policy.policy_id,
                                                                                       policy.policy_version):
        raise RegistryError("policy_invalid", f"{rel} holds {policy.policy_id}@{policy.policy_version}; the path must "
                                              f"be policies/<policy_id>/<policy_version>/policy.json")
    return policy


def load_capability(artifacts_root: Path, capabilities_root: str, ref) -> Any:
    from contracts.capability import CapabilityManifest, capability_sha256

    rel = f"{capabilities_root}/{ref.detector_id}/{ref.detector_version}/capability.json"
    path = resolve_inside(Path(artifacts_root), rel, "capability")
    if not path.is_file():
        raise RegistryError("missing_file", f"{path} not found")
    try:
        cap = CapabilityManifest.model_validate_json(path.read_bytes())
    except Exception as exc:
        raise RegistryError("capability_invalid", str(exc).splitlines()[0] + f" ({rel})") from exc
    if cap.key != ref.key:
        raise RegistryError("capability_invalid", f"{rel} declares {cap.detector_id}@{cap.detector_version}")
    got = capability_sha256(cap)
    if got != ref.capability_sha256:
        raise RegistryError("capability_sha256_mismatch", f"{rel}: policy pins {ref.capability_sha256}, file is {got}")
    return cap


def load_capability_detector(artifacts_root: Path, cap, *, intra_op_threads: int = 0,
                             installed: InstalledRuntime | None = None, cache: ModelCache | None = None,
                             artifact_dir: str | None = None) -> DetectorSpec:
    """Verify every file a CapabilityManifest pins (path inside artifacts/, exists, sha256) and build the detector."""
    from contracts.capability import capability_sha256

    from .detector import OnnxDebertaDetector, OnnxTfidfDetector

    installed = installed or InstalledRuntime.current()
    root = Path(artifacts_root)
    kind = cap.input.kind
    if kind == "text" and cap.input.tokenizer != TFIDF_TOKENIZER:
        raise RegistryError("tokenizer_mismatch", f"tfidf-v1 counts {TFIDF_TOKENIZER}, capability declares "
                                                  f"{cap.input.tokenizer!r}")
    check_runtime(cap, installed, kind)
    blobs: dict[str, bytes] = {}
    for rel, want in sorted(cap.files.items()):
        path = resolve_inside(root, rel, "file")
        if not path.is_file():
            raise RegistryError("missing_file", f"{path} not found (weights are not committed; see the script named "
                                                f"in serving/README.md)")
        data = path.read_bytes()
        got = sha256_bytes(data)
        if got != want:
            code = "tokenizer_mismatch" if rel == cap.tokenizer_file else "sha256_mismatch"
            raise RegistryError(code, f"{rel}: got {got}, capability {want}")
        blobs[rel] = data
    model_bytes = blobs[cap.model_file]

    def build():
        if kind == "text":
            return OnnxTfidfDetector(model_bytes, detector_id=cap.detector_id, detector_version=cap.detector_version,
                                     threshold=cap.calibration.threshold, injection_index=cap.injection_index,
                                     max_input_tokens=cap.input.max_input_tokens)
        return OnnxDebertaDetector(model_bytes, blobs[cap.tokenizer_file], detector_id=cap.detector_id,
                                   detector_version=cap.detector_version, threshold=cap.calibration.threshold,
                                   injection_index=cap.injection_index, max_len=cap.input.max_input_tokens,
                                   intra_op_threads=intra_op_threads)

    sha = capability_sha256(cap)
    key = ("capability", sha, intra_op_threads)
    try:
        det = cache.get(key, build) if cache is not None else build()
    except Exception as exc:
        raise RegistryError("load_failed", f"{type(exc).__name__}: {exc}") from exc
    return DetectorSpec(detector=det, detector_id=cap.detector_id, detector_version=cap.detector_version,
                        threshold=cap.calibration.threshold, kind=kind, role="stage1" if kind == "text" else "stage2",
                        failure_policy="fail_closed", timeout_ms=cap.limits.recommended_timeout_ms,
                        input_languages=list(cap.input.validated_languages) if cap.input.validated_languages else None,
                        max_input_tokens=cap.input.max_input_tokens, tokenizer=cap.input.tokenizer,
                        manifest_schema=cap.schema_version, artifact_dir=artifact_dir,
                        sha256=sha256_bytes(model_bytes), capability=cap)


@dataclass
class LoadedBundle:
    """What one ClassifierService serves: a validated policy, its capabilities and loaded detectors."""

    policy: Any  # contracts.policy.ExecutionPolicy
    capabilities: dict[tuple[str, str], Any]
    detectors: dict[tuple[str, str], DetectorSpec]  # every non-shadow DetectorRef of the policy
    shadow: DetectorSpec | None
    bundle_sha256: str
    policy_source: Literal["registry_v2", "legacy_registry", "injected"]
    policy_path: str | None = None
    status: str | None = None  # registry v2 status (pending_gate | approved)
    previous: dict | None = None  # {"path", "policy_id", "policy_version", "bundle_sha256"} or {"path", "error"}
    rejections: list[Rejection] = field(default_factory=list)  # v1: per-pointer rejections (served degraded)
    pointers: dict | None = None  # v1 pointers or the v2 registry state


def _policy_detectors(policy, capabilities: dict, artifacts_root: Path, *, intra_op_threads: int,
                      installed, cache) -> tuple[dict, DetectorSpec | None]:
    from contracts.policy import iter_refs

    detectors: dict[tuple[str, str], DetectorSpec] = {}
    # The shadow gets its own session (its own intra-op thread setting), even if another step uses the same detector.
    for ref in iter_refs(policy):
        if ref.key in detectors or (policy.shadow is not None and ref is policy.shadow.detector):
            continue
        detectors[ref.key] = load_capability_detector(artifacts_root, capabilities[ref.key],
                                                      intra_op_threads=intra_op_threads, installed=installed,
                                                      cache=cache)
    shadow = None
    if policy.shadow is not None:
        shadow = load_capability_detector(artifacts_root, capabilities[policy.shadow.detector.key],
                                          intra_op_threads=policy.shadow.intra_op_threads, installed=installed,
                                          cache=cache)
    return detectors, shadow


def load_policy_bundle(artifacts_root: Path, rel: str, capabilities_root: str, *, load_models: bool = True,
                       intra_op_threads: int = 0, installed: InstalledRuntime | None = None,
                       cache: ModelCache | None = None):
    """(policy, capabilities, detectors, shadow) for one policy file. Raises RegistryError / PolicyError."""
    from contracts.policy import PolicyError, iter_refs, validate_policy

    from .policy import check_engine_compat

    policy = load_policy_file(artifacts_root, rel)
    try:
        check_engine_compat(policy)
        capabilities = {r.key: load_capability(artifacts_root, capabilities_root, r) for r in iter_refs(policy)}
        validate_policy(policy, capabilities)
    except PolicyError as exc:
        raise RegistryError(exc.code, exc.detail) from exc
    if not load_models:
        return policy, capabilities, {}, None
    detectors, shadow = _policy_detectors(policy, capabilities, artifacts_root, intra_op_threads=intra_op_threads,
                                          installed=installed, cache=cache)
    return policy, capabilities, detectors, shadow


def load_bundle_v2(registry_path: Path, artifacts_root: Path, *, intra_op_threads: int = 0,
                   installed: InstalledRuntime | None = None, cache: ModelCache | None = None) -> LoadedBundle:
    """Load the active policy (all or nothing: raises RegistryError) and check the previous one (kept loaded so
    rollback is a pointer flip; a broken previous is reported in `previous.error`, the active still serves)."""
    from .policy import bundle_sha256

    state = read_registry_v2(registry_path)
    root = Path(artifacts_root)
    policy, caps, detectors, shadow = load_policy_bundle(root, state["active"], state["capabilities_root"],
                                                         intra_op_threads=intra_op_threads, installed=installed,
                                                         cache=cache)
    prev = None
    if state["previous"] is not None:
        try:
            pp, _, _, _ = load_policy_bundle(root, state["previous"], state["capabilities_root"],
                                             intra_op_threads=intra_op_threads, installed=installed, cache=cache)
            prev = {"path": state["previous"], "policy_id": pp.policy_id, "policy_version": pp.policy_version,
                    "bundle_sha256": bundle_sha256(pp)}
        except RegistryError as exc:
            log.error("registry v2: previous policy %s cannot be loaded: %s", state["previous"], exc)
            prev = {"path": state["previous"], "error": f"{exc.code}: {exc.detail}"}
    return LoadedBundle(policy=policy, capabilities=caps, detectors=detectors, shadow=shadow,
                        bundle_sha256=bundle_sha256(policy), policy_source="registry_v2", policy_path=state["active"],
                        status=state["status"], previous=prev, pointers=state)


def bundle_from_v1(loaded: LoadedRegistry, config, *, registry_sha256: str | None = None) -> LoadedBundle:
    """Convert a loaded v1 registry in memory (rejected pointers are absent, as the legacy gateway served them)."""
    from .convert_registry import CALIBRATION_RECORDS
    from .policy import bundle_sha256, legacy_to_v2

    pointers = {"stage1_mode": loaded.pointers.get("stage1_mode") or "observe"}
    pointers.update({name: spec.artifact_dir for name, spec in loaded.specs.items()})
    manifests = {name: spec.manifest for name, spec in loaded.specs.items()}
    policy, caps = legacy_to_v2(pointers, manifests, config, known_tokenizers=config.known_tokenizers,
                                calibrations=CALIBRATION_RECORDS,
                                policy_version=(registry_sha256 or "unknown")[:12])
    detectors = {}
    for name, spec in loaded.specs.items():
        if name == "shadow":
            continue
        spec.capability = caps[(spec.detector_id, spec.detector_version)]
        detectors[(spec.detector_id, spec.detector_version)] = spec
    shadow = loaded.specs.get("shadow")
    if shadow is not None:
        shadow.capability = caps[(shadow.detector_id, shadow.detector_version)]
    return LoadedBundle(policy=policy, capabilities=caps, detectors=detectors, shadow=shadow,
                        bundle_sha256=bundle_sha256(policy), policy_source="legacy_registry",
                        rejections=list(loaded.rejections), pointers=loaded.pointers)
