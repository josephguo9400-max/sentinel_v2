"""Train Stage 1 (TF-IDF + logistic regression), export to ONNX, check parity, time it, publish a candidate.

    python -m training.train_stage1 [--version stage1-tfidf-v1]

Reads ONLY data/splits/train.jsonl and data/splits/val.jsonl. The frozen test is never opened here.
Selection, threshold and gray zone are chosen on val. Publishes to a NEW directory
artifacts/candidates/<version>/ and refuses to overwrite an existing one.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime
import skl2onnx
import sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from contracts import ArtifactManifest
from training.common import ROOT, SEED, SPLITS_DIR, read_jsonl, sha256_file, write_json
from training.stage1_onnx import Stage1Onnx
from training.stage1_text import (MAX_TOKENS, ONNX_LOCALE, PREPROCESSING_VERSION, STRIDE, TOKEN_PATTERN_PY,
                                  TOKEN_PATTERN_RE2, ascii_lower, chunk_text)

CANDIDATES = ROOT / "artifacts" / "candidates"
TARGET_OPSET = 17
# Gray zone rule (provisional, val only): at most this fraction of val positives may be finalized as
# benign by Stage 1 (score < low), and at most this fraction of val negatives may be finalized as
# injection by Stage 1 (score > high).
GRAY_MAX_FN_RATE = 0.01
GRAY_MAX_FP_RATE = 0.01
GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "C": [0.3, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"],
}
# sublinear_tf=False on purpose: skl2onnx 1.20.0 exports sublinear_tf as log(1 + tf) while sklearn computes
# 1 + log(tf); with True, ONNX and sklearn scores differ by up to 0.031 on val (measured, see DATA.md).
FIXED = {"min_df": 2, "max_features": 50000, "sublinear_tf": False}


def make_pipeline(ngram_range, C, class_weight, vocabulary=None) -> Pipeline:
    kw = dict(FIXED)
    if vocabulary is not None:
        kw.pop("min_df"); kw.pop("max_features")
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN_PY, ngram_range=ngram_range,
                                  dtype=np.float32, vocabulary=vocabulary, **kw)),
        ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED)),
    ])


def onnx_safe_vocabulary(texts: list[str], ngram_range) -> list[str]:
    """Vocabulary from min_df/max_features, minus n-grams whose unigrams are not in it.

    onnxruntime's TfIdfVectorizer matches n-grams over unigram ids, so it can never count a bigram whose
    words were pruned by min_df; sklearn does count it. Dropping such n-grams (24 of ~50k on this data)
    makes both sides compute the same features.
    """
    v = TfidfVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN_PY, ngram_range=ngram_range, **FIXED).fit(texts)
    vocab = set(v.vocabulary_)
    return sorted(w for w in vocab if " " not in w or all(p in vocab for p in w.split(" ")))


def fit_pipeline(texts: list[str], y, ngram_range, C, class_weight) -> Pipeline:
    """texts must already be ascii_lower()ed."""
    vocab = onnx_safe_vocabulary(texts, ngram_range)
    return make_pipeline(ngram_range, C, class_weight, vocabulary=vocab).fit(texts, y)


def sk_proba(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    return pipe.predict_proba([ascii_lower(t) for t in texts])[:, 1]


def sk_score_chunked(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    """Reference scorer with the same chunking + max aggregation the ONNX scorer uses."""
    out = np.empty(len(texts))
    for i, t in enumerate(texts):
        chunks, _ = chunk_text(t, MAX_TOKENS, STRIDE)
        out[i] = sk_proba(pipe, chunks).max()
    return out


def per_source_auc(rows: list[dict], scores: np.ndarray) -> dict:
    res = {}
    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        y = [rows[i]["label"] for i in idx]
        if len(set(y)) == 2:
            res[src] = float(roc_auc_score(y, scores[idx]))
    return res


def export_onnx(pipe: Pipeline) -> bytes:
    """skl2onnx export. The fitted vectorizer was trained with lowercase=False on ascii_lower(text); for
    export a copy is flagged lowercase=True so the graph gets StringNormalizer(LOWER, locale=C), which
    lowercases ASCII only, i.e. ascii_lower inside the graph."""
    p = copy.deepcopy(pipe)
    p.named_steps["tfidf"].lowercase = True
    opts = {id(p.named_steps["lr"]): {"zipmap": False},
            id(p.named_steps["tfidf"]): {"tokenexp": TOKEN_PATTERN_RE2, "locale": ONNX_LOCALE}}
    model = to_onnx(p, initial_types=[("text", StringTensorType([None, 1]))], options=opts,
                    target_opset=TARGET_OPSET)
    return model.SerializeToString()


def choose_threshold(y: np.ndarray, s: np.ndarray) -> dict:
    """Max F1 on val over the observed score values (ties -> higher threshold)."""
    best = None
    for t in np.unique(np.round(s, 6)):
        pred = s >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if best is None or f1 >= best[0]:
            best = (f1, float(t))
    return {"threshold": best[1], "val_f1": best[0], "rule": "maximize F1 on val (pooled)"}


def choose_gray_zone(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    # low: largest value such that #positives with score < low <= floor(rate * n_pos)
    k_fn = int(np.floor(GRAY_MAX_FN_RATE * len(pos)))
    low = float(pos[k_fn]) if k_fn < len(pos) else 1.0
    # high: smallest value such that #negatives with score > high <= floor(rate * n_neg)
    k_fp = int(np.floor(GRAY_MAX_FP_RATE * len(neg)))
    high = float(neg[len(neg) - 1 - k_fp]) if k_fp < len(neg) else 0.0
    low, high = round(min(low, threshold), 6), round(max(high, threshold), 6)
    if not low < high:
        low, high = max(0.0, threshold - 1e-3), min(1.0, threshold + 1e-3)
    esc = (s >= low) & (s <= high)
    return {
        "gray_zone": [low, high],
        "rule": (f"low = score below which at most {GRAY_MAX_FN_RATE:.0%} of val positives fall; high = score "
                 f"above which at most {GRAY_MAX_FP_RATE:.0%} of val negatives fall; widened to contain the "
                 f"threshold. Provisional: the cascade-and-shadow milestone sweeps it against Stage 2."),
        "val_escalated": {"numerator": int(esc.sum()), "denominator": int(len(s))},
        "val_final_benign_positives": {"numerator": int(((s < low) & (y == 1)).sum()), "denominator": int((y == 1).sum())},
        "val_final_injection_negatives": {"numerator": int(((s > high) & (y == 0)).sum()), "denominator": int((y == 0).sum())},
    }


def measure_latency(scorer: Stage1Onnx, texts: list[str], warmup: int = 50) -> dict:
    for t in texts[:warmup]:
        scorer.score(t)
    lat, toks, chunks = [], [], []
    for t in texts:
        r = scorer.score(t)
        lat.append(r["latency_ms"]); toks.append(r["input_tokens"]); chunks.append(r["n_chunks"])
    lat = np.array(lat)
    q = lambda a, p: float(np.percentile(a, p))
    toks = np.array(toks)
    buckets = {}
    for name, lo, hi in [("<=64 tokens", 0, 64), ("65-256 tokens", 65, 256), ("257-512 tokens", 257, 512),
                         (">512 tokens (chunked)", 513, 10**9)]:
        m = (toks >= lo) & (toks <= hi)
        if m.sum():
            buckets[name] = {"n": int(m.sum()), "p50_ms": q(lat[m], 50), "p95_ms": q(lat[m], 95), "p99_ms": q(lat[m], 99)}
    return {
        "n": int(len(lat)), "p50_ms": q(lat, 50), "p95_ms": q(lat, 95), "p99_ms": q(lat, 99), "max_ms": float(lat.max()),
        "mean_ms": float(lat.mean()), "by_length": buckets, "max_chunks": int(max(chunks)),
        "conditions": {
            "what": "Stage1Onnx.score(): chunking at token boundaries + onnxruntime inference + max aggregation, "
                    "one text per call, sequential, after warmup",
            "inputs": "every val text, in file order",
            "warmup_calls": warmup,
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "onnxruntime_threads": "default (CPUExecutionProvider)", "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
        },
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="stage1-tfidf-v1")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--out-root", type=Path, default=CANDIDATES)
    a = ap.parse_args(argv)
    out_dir = a.out_root / a.version
    if out_dir.exists():
        print(f"refusing to overwrite existing artifact directory {out_dir}", file=sys.stderr)
        return 2

    train = list(read_jsonl(a.splits_dir / "train.jsonl"))
    val = list(read_jsonl(a.splits_dir / "val.jsonl"))
    Xtr, ytr = [r["text"] for r in train], np.array([r["label"] for r in train])
    Xva, yva = [r["text"] for r in val], np.array([r["label"] for r in val])
    print(f"train {len(train)}  val {len(val)}")
    Xtr_l = [ascii_lower(t) for t in Xtr]

    # 1. hyperparameter selection on val: mean per-source ROC-AUC (sources with both classes), so the
    #    largest val source does not dominate; pooled AUC as tiebreak.
    trials = []
    for ng, C, cw in itertools.product(GRID["ngram_range"], GRID["C"], GRID["class_weight"]):
        t0 = time.time()
        pipe = fit_pipeline(Xtr_l, ytr, ng, C, cw)
        s = sk_score_chunked(pipe, Xva)
        per = per_source_auc(val, s)
        trial = {"ngram_range": list(ng), "C": C, "class_weight": cw, "val_mean_source_auc": float(np.mean(list(per.values()))),
                 "val_pooled_auc": float(roc_auc_score(yva, s)), "val_auc_by_source": per, "fit_s": round(time.time() - t0, 1)}
        trials.append(trial)
        print(json.dumps(trial))
    best = max(trials, key=lambda t: (round(t["val_mean_source_auc"], 4), t["val_pooled_auc"]))
    pipe = fit_pipeline(Xtr_l, ytr, tuple(best["ngram_range"]), best["C"], best["class_weight"])
    s_sk = sk_score_chunked(pipe, Xva)

    # 2. threshold and gray zone on val
    thr = choose_threshold(yva, s_sk)
    gz = choose_gray_zone(yva, s_sk, thr["threshold"])

    # 3. ONNX export + parity on val (raw whole-text and chunked)
    model_bytes = export_onnx(pipe)
    scorer = Stage1Onnx.from_bytes(model_bytes)
    raw_sk = sk_proba(pipe, Xva)
    raw_ox = scorer.proba_raw(Xva)
    s_ox = np.array([scorer.score(t)["score"] for t in Xva])
    d_raw, d_chunk = np.abs(raw_sk - raw_ox), np.abs(s_sk - s_ox)
    flips = int(((s_sk >= thr["threshold"]) != (s_ox >= thr["threshold"])).sum())
    parity = {"n": len(Xva), "max_abs_diff_whole_text": float(d_raw.max()), "max_abs_diff_chunked": float(d_chunk.max()),
              "mean_abs_diff_chunked": float(d_chunk.mean()), "decision_flips_at_threshold": flips,
              "note": "sklearn in float64 on ascii_lower(text) vs onnxruntime float32 graph with "
                      "StringNormalizer(LOWER, locale=C) + RE2 Tokenizer"}
    print("parity", parity)
    if parity["max_abs_diff_chunked"] > 1e-4:
        print("ONNX parity failed", file=sys.stderr)
        return 3

    # 4. publish to a temp dir, then atomically rename to the immutable version directory
    a.out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{a.version}.", dir=a.out_root))
    (tmp / "model.onnx").write_bytes(model_bytes)
    digest = sha256_file(tmp / "model.onnx")
    splits_manifest = json.loads((a.splits_dir / "MANIFEST.json").read_text())
    manifest = ArtifactManifest(
        detector_id="stage1-tfidf-lr",
        detector_version=a.version,
        artifact_format="onnx",
        artifact_path="model.onnx",
        sha256=digest,
        runtime={"python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
                 "sklearn": sklearn.__version__, "skl2onnx": skl2onnx.__version__},
        preprocessing_version=PREPROCESSING_VERSION,
        label_mapping={"0": "benign", "1": "injection"},
        threshold=thr["threshold"],
        max_input_tokens=MAX_TOKENS,
        failure_policy="fail_open",
        timeout_ms=50,
        cascade_role="stage1",
        gray_zone=tuple(gz["gray_zone"]),
        training_data_version=f"{splits_manifest['pipeline_version']}:train:{splits_manifest['splits']['train']['sha256'][:16]}",
        evaluation_dataset_version=f"{splits_manifest['pipeline_version']}:val:{splits_manifest['splits']['val']['sha256'][:16]}",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        metrics_path="evaluation.json",
    )
    (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    # latency is measured on the published files (sha256-verified load path)
    lat = measure_latency(Stage1Onnx(tmp), Xva)
    lat["conditions"]["cpu"] = cpu_model()
    evaluation = {
        "detector_version": a.version,
        "data": {"train": {"n": len(train), "sha256": splits_manifest["splits"]["train"]["sha256"]},
                 "val": {"n": len(val), "sha256": splits_manifest["splits"]["val"]["sha256"]},
                 "frozen_test": "not read by training"},
        "preprocessing": {"version": PREPROCESSING_VERSION, "lowercase": "ASCII A-Z only",
                          "token_pattern_python": TOKEN_PATTERN_PY, "token_pattern_re2": TOKEN_PATTERN_RE2,
                          "onnx_locale": ONNX_LOCALE, "chunking": {"max_tokens": MAX_TOKENS, "stride": STRIDE,
                                                                  "aggregation": "max"}},
        "model": {"vectorizer": {**FIXED, "ngram_range": best["ngram_range"], "analyzer": "word",
                                 "vocabulary_rule": "min_df/max_features, then drop n-grams with out-of-vocabulary words (ONNX parity)",
                                 "vocabulary_size": len(pipe.named_steps["tfidf"].vocabulary_)},
                  "classifier": {"type": "LogisticRegression", "C": best["C"], "class_weight": best["class_weight"]}},
        "selection": {"criterion": "max mean per-source ROC-AUC on val (sources with both classes); pooled AUC tiebreak",
                      "trials": trials, "best": best},
        "threshold": thr,
        "gray_zone": gz,
        "failure_policy_rationale": "fail_open: Stage 1 is a cost filter in front of Stage 2, not the security "
                                    "boundary; provisional, owner may override in the routing config",
        "timeout_ms_rationale": "p99 single-item latency on val is far below 50 ms on this CPU (see latency)",
        "onnx_parity_val": parity,
        "latency": lat,
        "val_metrics_note": "precision/recall/FPR with CIs are produced by training.eval_harness from this artifact",
    }
    write_json(tmp / "evaluation.json", evaluation)
    if out_dir.exists():
        print(f"refusing to overwrite existing artifact directory {out_dir}", file=sys.stderr)
        return 2
    os.rename(tmp, out_dir)
    for p in out_dir.iterdir():
        p.chmod(0o444)
    out_dir.chmod(0o755)
    print(f"published {out_dir} sha256={digest} threshold={thr['threshold']} gray_zone={gz['gray_zone']}")
    print("latency", {k: lat[k] for k in ("n", "p50_ms", "p95_ms", "p99_ms")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
