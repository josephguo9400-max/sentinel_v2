"""Per-item Stage 1 and Stage 2 (DeBERTa) scores for the cascade sweep, cached and resumable.

    flock /home/claude/measure.lock -c 'python -m training.cascade_scores --split val --max-items 200'

Every val item gets one JSON line in training/results/cascade_scores_val.jsonl:

  id, source, lang, label, family (benign_hard only)
  s1          Stage 1 request score: stage1-tfidf ONNX, its own regex-token chunking (512 / stride 384), max
  s1_chunks   number of Stage 1 windows
  deb_tokens  DeBERTa content tokens (no special tokens)
  chunks      one entry per DeBERTa window, in order: [start_char, end_char, deberta_score, stage1_score]
              DeBERTa windows are built like the gateway (serving/encoding.py, not imported here): 510 content
              tokens, stride 384, the last window ending at the tail, each model input [CLS] + window + [SEP],
              batch size 1. stage1_score is Stage 1 on the window's character span (for the chunk-level cascade).
  deb         DeBERTa request score = max over ALL windows (no chunk cap; the gateway's max_chunks cap is
              applied later by the sweep, if at all)

The file is appended one line per item and flushed, so an interrupted run resumes where it stopped
(items already in the file are skipped). Scores are deterministic given the model files, so a rerun
gives the same numbers up to onnxruntime float nondeterminism (not observed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file

DEBERTA_DIR = ROOT / "models" / "deberta-pi-v2" / "onnx"
# Same pin as scripts/fetch_models.py (protectai/deberta-v3-base-prompt-injection-v2 @ 90c9989b, ONNX fp32).
DEBERTA_SHA256 = "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c"
STAGE1_ARTIFACT = ROOT / "artifacts" / "candidates" / "stage1-tfidf-v2"
DEB_MAX_LEN = 512  # model input incl. [CLS] and [SEP]
DEB_WINDOW = DEB_MAX_LEN - 2
DEB_STRIDE = 384
RESULTS = ROOT / "training" / "results"


def window_starts(n: int, window: int = DEB_WINDOW, stride: int = DEB_STRIDE) -> list[int]:
    """Same rule as the gateway: windows every `stride` tokens plus one ending exactly at the tail."""
    if not 0 < stride <= window:
        raise ValueError("stride must be in (0, window]")
    if n <= window:
        return [0]
    return list(range(0, n - window, stride)) + [n - window]


class DebertaScorer:
    def __init__(self, model_dir: Path = DEBERTA_DIR, verify: bool = True, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        if verify:
            got = sha256_file(model_dir / "model.onnx")
            if got != DEBERTA_SHA256:
                raise ValueError(f"DeBERTa model.onnx sha256 {got} != pinned {DEBERTA_SHA256}")
        id2label = json.loads((model_dir / "config.json").read_text())["id2label"]
        self.inj = next(int(k) for k, v in id2label.items() if v.upper() == "INJECTION")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.cls_id, self.sep_id = self.tok.token_to_id("[CLS]"), self.tok.token_to_id("[SEP]")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def windows(self, text: str) -> tuple[list[list[int]], list[tuple[int, int]], int]:
        """(model inputs, char spans, n content tokens). An empty text gives one [CLS][SEP] input."""
        enc = self.tok.encode(text, add_special_tokens=False)
        ids, offs = enc.ids, enc.offsets
        rows, spans = [], []
        for s in window_starts(len(ids)):
            w = ids[s : s + DEB_WINDOW]
            rows.append([self.cls_id, *w, self.sep_id])
            spans.append((offs[s][0], offs[s + len(w) - 1][1]) if w else (0, len(text)))
        return rows, spans, len(ids)

    def infer_one(self, row: list[int]) -> float:
        ids = np.asarray([row], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        logits = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        return float(e[self.inj] / e.sum())


def score_item(row: dict, deb: DebertaScorer, s1) -> dict:
    text = row["text"]
    r1 = s1.score(text)
    rows, spans, n_tok = deb.windows(text)
    chunks = []
    for inp, (a, b) in zip(rows, spans):
        chunks.append([a, b, round(deb.infer_one(inp), 6), round(s1.score(text[a:b])["score"], 6)])
    out = {"id": row["id"], "source": row["source"], "lang": row.get("lang"), "label": row["label"],
           "s1": round(r1["score"], 6), "s1_chunks": r1["n_chunks"], "deb_tokens": n_tok, "chunks": chunks,
           "deb": max(c[2] for c in chunks)}
    if row.get("family"):
        out["family"] = row["family"]
    return out


def cache_path(split: str) -> Path:
    return RESULTS / f"cascade_scores_{split}.jsonl"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # a torn last line from a crash: rewritten below
    return ids


def repair(path: Path) -> None:
    """Drop a torn trailing line left by a crash (only complete JSON lines are kept)."""
    if not path.exists():
        return
    good = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                good.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                pass
    path.write_text("".join(good), encoding="utf-8")


def run(split: str, max_items: int | None, threads: int, out: Path | None = None) -> int:
    if split == "frozen_test":
        raise SystemExit("frozen_test is read by the gate only")
    out = out or cache_path(split)
    repair(out)
    have = done_ids(out)
    todo = [r for r in read_jsonl(SPLITS_DIR / f"{split}.jsonl") if r["id"] not in have]
    todo.sort(key=lambda r: r["id"])
    if max_items is not None:
        todo = todo[:max_items]
    if not todo:
        print(f"{out.relative_to(ROOT)}: complete ({len(have)} items)")
        return 0
    from training.stage1_onnx import Stage1Onnx

    s1 = Stage1Onnx(STAGE1_ARTIFACT)
    deb = DebertaScorer(threads=threads)
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo):
            f.write(json.dumps(score_item(row, deb, s1), ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    print(f"scored {len(todo)} items in {time.time() - t0:.0f}s; {len(have) + len(todo)} in {out.relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max-items", type=int, default=None, help="score at most this many new items, then exit")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    return run(a.split, a.max_items, a.threads)


if __name__ == "__main__":
    sys.exit(main())
