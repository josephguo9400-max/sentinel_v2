"""Gateway configuration. Every field can be set with an environment variable SENTINEL_<FIELD_NAME_UPPER>.

Detectors come from manifests named by the registry file (serving/registry.py). Fields marked "injected" apply only
when a detector object is passed to create_app() directly (tests); manifest-loaded detectors take their threshold,
timeout and failure policy from their manifest.
"""
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_POLICIES = ("flag", "quarantine", "allow", "deep_scan")


@dataclass(frozen=True)
class GatewayConfig:
    # --- registry
    registry_path: str = str(ROOT / "artifacts" / "registry.json")
    artifacts_root: str = str(ROOT / "artifacts")  # every manifest pointer must resolve inside this directory
    registry_poll_s: float = 0.0  # > 0: re-read the registry file every N seconds (POST /admin/reload always works)
    # Extra token-model tokenizers "name=sha256,..." on top of serving/registry.py KNOWN_TOKENIZERS (tests, new models).
    extra_tokenizers: str = ""
    # Deprecated, kept so existing callers still construct a config: has no effect. Manifest sha256 is always checked.
    verify_model_sha256: bool = True

    # --- injected detector (tests) / single-detector mode
    threshold: float = 0.5  # injected only
    timeout_ms: int = 5000  # injected only: whole detector budget per request (tokenization + queue + inference)
    failure_policy: Literal["fail_closed", "fail_open"] = "fail_closed"  # injected only
    max_len: int = 512  # injected only: model input length including [CLS] and [SEP]

    # --- chunking (Stage 2 tokenizer units)
    stride: int = 384  # sliding-window step inside a segment longer than a window: 126-token overlap at 510
    structure_aware: bool = True  # pack windows on paragraph / quote / signature / hidden-HTML boundaries
    pack_overlap_tokens: int = 126  # a packed window repeats the previous window's last unit if it is this short

    # --- cascade
    max_chunks: int = 4  # Stage 2 chunk budget per request (first + last + most suspicious)
    stage1_max_chunks: int = 512  # chunks Stage 1 scores; the rest are unscored by Stage 1 and escalate
    stage1_finalize_flag: bool = False  # Stage 1 may finalize FLAG when any chunk scores >= gray_zone high
    # Anti-dilution: Stage 1 also scores overlapping sub-windows of this many regex tokens (0 = off); a chunk's
    # Stage 1 score is the max over the chunk, its segment pieces and these sub-windows. See serving/README.md.
    stage1_subwindow_tokens: int = 64
    stage1_subwindow_stride: int = 32
    # Languages assumed for a Stage 1 manifest that declares no input_languages (v1.1). Comma list; "" = no limit.
    stage1_undeclared_languages: str = "en"
    stage1_timeout_ms: int = 0  # 0 = Stage 1 manifest timeout_ms (the hard wait; see stage1_deadline_margin_ms)
    # Stage 1 is "anytime": it scores chunks in order and stops starting new chunks once (budget - margin) has
    # passed, returning the chunks it scored (status ok, truncated). The margin is the time one more chunk may take;
    # the hard wait at the full budget stays as a backstop for a hung model. Measured per-chunk cost:
    # serving/bench_results/stage1_latency.json. Capped at half the Stage 1 budget.
    stage1_deadline_margin_ms: int = 15
    stage2_timeout_ms: int = 0  # 0 = Stage 2 manifest timeout_ms

    # --- language detection (langdetect, seeded; see serving/language.py)
    lang_seed: int = 0
    lang_min_letters: int = 20  # fewer letters than this -> "unknown" -> escalate
    lang_min_prob: float = 0.80  # top language probability below this -> "unknown" -> escalate
    lang_max_chars: int = 2000  # characters of each chunk given to the detector

    # --- coverage policy: decides when some content was scored by neither stage
    coverage_policy: Literal["flag", "quarantine", "allow", "deep_scan"] = "flag"
    deep_scan_max_chunks: int = 16  # extra Stage 2 chunks for deep_scan
    deep_scan_timeout_ms: int = 10000
    deep_scan_fallback: Literal["flag", "quarantine", "allow"] = "flag"  # content still unscored after deep_scan

    # --- shadow
    shadow_queue: int = 16  # mirrored requests waiting for the shadow; when full, new ones are dropped
    shadow_timeout_ms: int = 0  # 0 = shadow manifest timeout_ms
    shadow_max_chunks: int = 8
    shadow_intra_op_threads: int = 1  # keep the shadow off the second core the active path uses
    shadow_log_path: str = str(ROOT / "serving" / "shadow_logs" / "shadow.jsonl")  # "" disables the JSONL log

    # --- batching, workers, admission (unchanged from gateway-mvp)
    max_batch: int = 1  # 1 disables micro-batching (chosen from the load test, see serving/README.md)
    max_wait_ms: float = 2.0
    workers: int = 1
    max_queue: int = 64
    max_inflight: int = 64
    max_inflight_chars: int = 1_000_000
    tokenize_workers: int = 1
    intra_op_threads: int = 0  # 0 = onnxruntime default (all cores)
    max_text_chars: int = 200_000  # larger bodies get HTTP 413

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.max_len < 3 or not 0 < self.stride <= self.max_len - 2:
            raise ValueError("need max_len >= 3 and 0 < stride <= max_len - 2")
        for name in ("max_chunks", "timeout_ms", "max_batch", "workers", "max_queue", "max_text_chars",
                     "max_inflight", "max_inflight_chars", "tokenize_workers", "stage1_max_chunks",
                     "deep_scan_max_chunks", "deep_scan_timeout_ms", "shadow_queue", "shadow_max_chunks"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("stage1_timeout_ms", "stage2_timeout_ms", "shadow_timeout_ms", "pack_overlap_tokens",
                     "stage1_deadline_margin_ms",
                     "lang_min_letters", "registry_poll_s", "stage1_subwindow_tokens", "shadow_intra_op_threads", "intra_op_threads"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.stage1_subwindow_stride < 1:
            raise ValueError("stage1_subwindow_stride must be >= 1")
        if self.max_inflight_chars < self.max_text_chars:
            raise ValueError("max_inflight_chars must be >= max_text_chars, or a maximal request is never admitted")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if self.failure_policy not in ("fail_closed", "fail_open"):
            raise ValueError("failure_policy must be fail_closed or fail_open")
        if self.coverage_policy not in COVERAGE_POLICIES:
            raise ValueError(f"coverage_policy must be one of {COVERAGE_POLICIES}")
        if self.deep_scan_fallback not in ("flag", "quarantine", "allow"):
            raise ValueError("deep_scan_fallback must be flag, quarantine or allow")
        if not 0.0 <= self.lang_min_prob <= 1.0 or self.lang_max_chars < 1:
            raise ValueError("need 0 <= lang_min_prob <= 1 and lang_max_chars >= 1")

    @property
    def known_tokenizers(self) -> dict[str, str]:
        from .registry import KNOWN_TOKENIZERS
        out = dict(KNOWN_TOKENIZERS)
        for item in filter(None, (x.strip() for x in self.extra_tokenizers.split(","))):
            name, _, sha = item.partition("=")
            out[name.strip()] = sha.strip()
        return out

    @property
    def undeclared_languages(self) -> list[str] | None:
        langs = [x.strip() for x in self.stage1_undeclared_languages.split(",") if x.strip()]
        return langs or None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides) -> "GatewayConfig":
        environ = os.environ if environ is None else environ
        values = {}
        for f in fields(cls):
            raw = environ.get(f"SENTINEL_{f.name.upper()}")
            if raw is None:
                continue
            kind = type(f.default)
            if kind is bool:
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif kind in (int, float):
                values[f.name] = kind(raw)
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)
