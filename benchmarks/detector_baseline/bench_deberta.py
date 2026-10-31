"""Detector baseline: feasibility check for the Stage 2 detector (protectai/deberta-v3-base-prompt-injection-v2, ONNX fp32).

1. Sanity-checks label semantics on three short examples.
2. Truncation test: a realistic multi-paragraph email with an injection at the end vs at the start,
   scored with standard truncation and with overlapping chunks.
3. Length sweep: how the score of a benign repeated sentence moves with input length.
4. Latency: sequential CPU requests at several lengths and batch sizes. Timing includes tokenization.

Run from the repo root: python scripts/fetch_models.py && python -m benchmarks.detector_baseline.bench_deberta
"""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tokenizers
from tokenizers import Tokenizer

from benchmarks.detector_baseline.encoding import chunk_ids, truncate_ids
from benchmarks.detector_baseline.fixtures import INJECTION, MEETING_RECAP, REPEATED_SENTENCE, SANITY

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "deberta-pi-v2" / "onnx"
OUT = Path(__file__).resolve().parent / "results.json"
MAX_LEN = 512
STRIDE = 384  # 126-token overlap between consecutive windows

cfg = json.loads((MODEL_DIR / "config.json").read_text())
id2label = cfg["id2label"]
INJ = next(int(k) for k, v in id2label.items() if v.upper() == "INJECTION")
tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
CLS, SEP = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
INPUTS = {i.name for i in sess.get_inputs()}


def content_ids(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def run(batch_ids: list[list[int]]) -> np.ndarray:
    width = max(len(x) for x in batch_ids)
    ids = np.zeros((len(batch_ids), width), dtype=np.int64)
    mask = np.zeros_like(ids)
    for r, x in enumerate(batch_ids):
        ids[r, : len(x)], mask[r, : len(x)] = x, 1
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    logits = sess.run(None, {k: v for k, v in feed.items() if k in INPUTS})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, INJ]


def score_truncated(text: str) -> tuple[float, bool]:
    ids, truncated = truncate_ids(content_ids(text), CLS, SEP, MAX_LEN)
    return float(run([ids])[0]), truncated


def score_chunked(text: str) -> tuple[float, int]:
    chunks = chunk_ids(content_ids(text), CLS, SEP, MAX_LEN, STRIDE)
    return float(run(chunks).max()), len(chunks)


def repeat_to(paragraphs: list[str], n_content_tokens: int) -> str:
    text, i = "", 0
    while len(content_ids(text)) < n_content_tokens:
        text += paragraphs[i % len(paragraphs)] + " "
        i += 1
    return text.strip()


def exact_len(text: str, n_content_tokens: int) -> list[int]:
    ids = content_ids(text)[: n_content_tokens]
    assert len(ids) == n_content_tokens
    return ids


def percentile(sorted_ms: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    k = max(0, min(len(sorted_ms) - 1, int(np.ceil(p * len(sorted_ms))) - 1))
    return round(sorted_ms[k], 1)


def bench(batch: int, n_content_tokens: int, iters: int) -> dict:
    # Distinct texts per batch row (rotated paragraphs), all exactly the same length: no padding cost.
    rows = [exact_len(repeat_to(MEETING_RECAP[r % 8:] + MEETING_RECAP[: r % 8], n_content_tokens + 50),
                      n_content_tokens) for r in range(batch)]
    texts = [tok.decode(r) for r in rows]

    def once():
        encoded = [truncate_ids(content_ids(t), CLS, SEP, MAX_LEN)[0] for t in texts]  # tokenization is timed
        run(encoded)

    for _ in range(3):
        once()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return {"batch": batch, "content_tokens": n_content_tokens, "iters": iters,
            "p50_ms": percentile(lat, 0.50), "p95_ms": percentile(lat, 0.95), "p99_ms": percentile(lat, 0.99),
            "max_ms": round(lat[-1], 1), "per_item_p50_ms": round(percentile(lat, 0.50) / batch, 1),
            "p99_is_max": percentile(lat, 0.99) == round(lat[-1], 1)}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    env = {"cpu": cpu_model(), "logical_cores": os.cpu_count(), "python": platform.python_version(),
           "onnxruntime": ort.__version__, "tokenizers": tokenizers.__version__,
           "intra_op_threads": sess.get_session_options().intra_op_num_threads or "default",
           "model_sha256": hashlib.sha256((MODEL_DIR / "model.onnx").read_bytes()).hexdigest(),
           "id2label": id2label, "max_len": MAX_LEN, "stride": STRIDE,
           "timing_includes_tokenization": True}
    print("env:", env)

    sanity = {name: round(float(s), 4) for name, s in zip(SANITY, run([truncate_ids(content_ids(t), CLS, SEP,
                                                                                    MAX_LEN)[0]
                                                                       for t in SANITY.values()]))}
    print("\nsanity:", sanity)

    email = repeat_to(MEETING_RECAP, 690)
    tail, head = f"{email} {INJECTION}", f"{INJECTION} {email}"
    truncation = {
        "email_content_tokens": len(content_ids(email)),
        "benign_email_truncated": round(score_truncated(email)[0], 4),
        "injection_at_end_content_tokens": len(content_ids(tail)),
        "injection_at_end_truncated": round(score_truncated(tail)[0], 4),
        "injection_at_end_chunked": round(score_chunked(tail)[0], 4),
        "injection_at_end_n_chunks": score_chunked(tail)[1],
        "injection_at_start_truncated": round(score_truncated(head)[0], 4),
        "benign_email_chunked": round(score_chunked(email)[0], 4),
    }
    print("\ntruncation test:", truncation)

    sweep_text = tok.decode(content_ids(REPEATED_SENTENCE * 80))
    length_sweep = {n: round(float(run([[CLS, *exact_len(sweep_text, n), SEP]])[0]), 4)
                    for n in (120, 256, 400, 510)}
    print("\nrepeated-sentence length sweep (content tokens -> score):", length_sweep)

    latency = []
    for batch, n, iters in [(1, 64, 100), (1, 128, 100), (1, 256, 100), (1, 510, 50), (8, 256, 30), (32, 128, 20)]:
        r = bench(batch, n, iters)
        latency.append(r)
        print(r)

    OUT.write_text(json.dumps({"env": env, "sanity": sanity, "truncation_test": truncation,
                               "repeated_sentence_length_sweep": length_sweep, "latency": latency}, indent=2))
    print("\nwrote", OUT)
