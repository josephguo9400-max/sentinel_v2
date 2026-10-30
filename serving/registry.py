"""Manifest-driven detector registry.

`artifacts/registry.json` names the active Stage 1, the active Stage 2 and an optional shadow detector, each
as a directory under `artifacts/` holding a `manifest.json` (contracts.ArtifactManifest v1.1 or v1.2):

    {"schema": "sentinel-registry-v1",
     "stage1": "candidates/stage1-tfidf-v2",
     "stage1_mode": "route",
     "stage2": "baseline/deberta-pi-v2",
     "shadow": "candidates/deberta-pi-v2-int8"}

`stage1_mode` (optional, default "route") says what the Stage 1 pointer may do:
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


def check_runtime(manifest: ArtifactManifest, installed: InstalledRuntime, kind: str) -> None:
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
        return self.pointers.get("stage1_mode") or "route"


def read_registry(path: Path) -> dict[str, str | None]:
    """The three pointers plus "stage1_mode" ("route" when the file omits it)."""
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
    mode = raw.get("stage1_mode", "route")
    if mode not in STAGE1_MODES:
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
                        manifest_schema=manifest.schema_version, artifact_dir=rel_dir, sha256=digest)


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
