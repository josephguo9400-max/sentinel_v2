"""Builds tiny but real ONNX artifacts (and manifests) in a temp artifacts/ tree for registry tests.

- Token model: input_ids/attention_mask -> logits [N, 2]; INJECTION_ID anywhere in a row gives class 1.
- Tokenizer: a WordLevel tokenizers.json over whitespace words, with [CLS]/[SEP]; "INJECT" maps to INJECTION_ID.
- Text model: the committed Stage 1 candidate's model.onnx (1.8 MB), copied.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = ROOT / "artifacts" / "candidates" / "stage1-tfidf-v2"
INJECTION_ID = 3
TOKENIZER_NAME = "hf:test-wordlevel"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_model_bytes() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None, None])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])
    nodes = [
        helper.make_node("Constant", [], ["inj"], value=helper.make_tensor("inj", TensorProto.INT64, [], [INJECTION_ID])),
        helper.make_node("Equal", ["input_ids", "inj"], ["hit_b"]),
        helper.make_node("Cast", ["hit_b"], ["hit_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["hit_f", "mask_f"], ["hit_m"]),
        helper.make_node("ReduceMax", ["hit_m"], ["hit"], axes=[1], keepdims=1),
        helper.make_node("Constant", [], ["scale"], value=helper.make_tensor("scale", TensorProto.FLOAT, [], [10.0])),
        helper.make_node("Constant", [], ["bias"], value=helper.make_tensor("bias", TensorProto.FLOAT, [], [-5.0])),
        helper.make_node("Mul", ["hit", "scale"], ["l1a"]),
        helper.make_node("Add", ["l1a", "bias"], ["l1"]),
        helper.make_node("Neg", ["l1"], ["l0"]),
        helper.make_node("Concat", ["l0", "l1"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(nodes, "fake-token-model", [ids, mask], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def tokenizer_bytes() -> bytes:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "INJECT": INJECTION_ID, "[UNK]": 4}
    for i in range(50):
        vocab[f"w{i}"] = 10 + i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok.to_str().encode("utf-8")


def installed_runtime() -> dict:
    import onnxruntime
    import tokenizers

    return {"python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
            "tokenizers": tokenizers.__version__, "sklearn": None, "skl2onnx": None}


def token_manifest(model: bytes, **over) -> dict:
    m = {"schema_version": "1.2", "detector_id": "tiny-token", "detector_version": "0.0.1", "artifact_format": "onnx",
         "artifact_path": "onnx/model.onnx", "sha256": sha(model), "runtime": installed_runtime(),
         "preprocessing_version": "deberta-v3-window-v1", "label_mapping": {"0": "benign", "1": "injection"},
         "threshold": 0.5, "max_input_tokens": 512, "failure_policy": "fail_closed", "timeout_ms": 5000,
         "cascade_role": "stage2", "gray_zone": None, "tokenizer": TOKENIZER_NAME, "input_languages": None,
         "created_at": "2026-10-29T00:00:00Z"}
    m.update(over)
    return m


def write_token_artifact(root: Path, rel: str, **manifest_over) -> Path:
    d = root / rel
    (d / "onnx").mkdir(parents=True, exist_ok=True)
    model = token_model_bytes()
    (d / "onnx" / "model.onnx").write_bytes(model)
    (d / "onnx" / "tokenizer.json").write_bytes(tokenizer_bytes())
    (d / "manifest.json").write_text(json.dumps(token_manifest(model, **manifest_over)))
    return d


def write_stage1_artifact(root: Path, rel: str, **manifest_over) -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STAGE1_DIR / "model.onnx", d / "model.onnx")
    m = json.loads((STAGE1_DIR / "manifest.json").read_text())
    m["metrics_path"] = None
    m.update(manifest_over)
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def write_registry(root: Path, **pointers) -> Path:
    p = root / "registry.json"
    p.write_text(json.dumps({"schema": "sentinel-registry-v1", **pointers}))
    return p


def known_tokenizers() -> dict[str, str]:
    return {TOKENIZER_NAME: sha(tokenizer_bytes())}


def extra_tokenizers_cfg() -> str:
    return ",".join(f"{k}={v}" for k, v in known_tokenizers().items())


def softmax_check(logits: np.ndarray) -> np.ndarray:  # pragma: no cover - debugging helper
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
