"""Gateway configuration. Every field can be set with an environment variable SENTINEL_<FIELD_NAME_UPPER>."""
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GatewayConfig:
    model_dir: str = str(ROOT / "models" / "deberta-pi-v2" / "onnx")
    detector_id: str = "deberta-pi-v2"
    detector_version: str = "2.0.0+90c9989b.onnx.fp32"
    verify_model_sha256: bool = True
    threshold: float = 0.5
    max_len: int = 512  # model input length including [CLS] and [SEP] -> 510 content tokens per chunk
    stride: int = 384  # 126-token overlap between consecutive chunks
    max_chunks: int = 4  # per request; beyond this the middle of the input is dropped and truncated=True
    timeout_ms: int = 5000  # whole detector budget per request: tokenization + queue wait + inference
    failure_policy: Literal["fail_closed", "fail_open"] = "fail_closed"
    max_batch: int = 1  # 1 disables micro-batching (chosen from the load test, see serving/README.md)
    max_wait_ms: float = 2.0  # how long the batcher waits to fill a batch once it has one chunk
    workers: int = 1  # concurrent ONNX calls; a timed-out call keeps its worker until it returns
    max_queue: int = 64  # queued chunks; when full, new requests get the failure policy immediately
    # Admission control, checked before any tokenization. Above either cap a request is rejected at once
    # with reason "overloaded" and the failure policy, like a full chunk queue.
    max_inflight: int = 64  # requests admitted and not finished (tokenizing, queued or running)
    max_inflight_chars: int = 1_000_000  # total characters of admitted, unfinished requests
    tokenize_workers: int = 1  # threads that tokenize; a timed-out tokenization keeps its thread until it returns
    intra_op_threads: int = 0  # 0 = onnxruntime default (all cores)
    max_text_chars: int = 200_000  # larger bodies get HTTP 413

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.max_len < 3 or not 0 < self.stride <= self.max_len - 2:
            raise ValueError("need max_len >= 3 and 0 < stride <= max_len - 2")
        for name in ("max_chunks", "timeout_ms", "max_batch", "workers", "max_queue", "max_text_chars",
                     "max_inflight", "max_inflight_chars", "tokenize_workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_inflight_chars < self.max_text_chars:
            raise ValueError("max_inflight_chars must be >= max_text_chars, or a maximal request is never admitted")
        if self.max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if self.failure_policy not in ("fail_closed", "fail_open"):
            raise ValueError("failure_policy must be fail_closed or fail_open")

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
