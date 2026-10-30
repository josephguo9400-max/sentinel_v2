"""DeBERTa rescoring with smaller windows (dilution study), cached and resumable.

    flock /home/claude/measure.lock python -m training.window_scores --set val --window 128 --max-items 400

Each request is tokenized once (DeBERTa tokenizer, no special tokens) and split into windows of W content tokens
with the gateway's rule (training.cascade_scores.window_starts: starts every `stride` tokens plus one window ending
exactly at the tail); each model input is [CLS] + window + [SEP], batch size 1; the request score is the max over
ALL windows (no chunk cap). Strides: 384 for the W=510 baseline (the gateway's), W/2 for the smaller windows.

Sets: val (data/splits/val.jsonl), replay (data/replay/replay-v1.jsonl, label None = unlabeled) and probes
(fixtures/probes, probe set v2: label 1 = attack, 0 = matched benign control). The frozen test is never read.

Cache: training/results/window_scores_<set>.jsonl, one JSON line per (id, W):
  id, W, stride, n_tokens, scores (one per window, in order, rounded to 6 places), from
`from` says where the scores came from:
  "model"          computed here
  "cascade_cache"  copied from training/results/cascade_scores_<set>.jsonl (val and replay only). This is used for
                   W=510 (identical windows by construction) and, for W < 510, for requests with n_tokens <= W: those
                   have a single window holding the whole request, which is the same model input as the baseline's
                   single window (n_tokens <= W < 510). tests/training/test_window_scores.py checks both claims
                   against a fresh computation on a sample.
Lines are appended and fsynced one at a time, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from training.cascade_scores import DEB_STRIDE, DEB_WINDOW, REPLAY_FILE, cache_path, repair, window_starts
from training.common import ROOT, SPLITS_DIR, read_jsonl

RESULTS = ROOT / "training" / "results"
WINDOWS = (510, 256, 128, 64)
SETS = ("val", "replay", "probes")
EXIT_MORE = 10  # exit status when items are left (run again; scripts/run_window_scores.sh loops on it)


def stride_for(window: int) -> int:
    """The gateway stride for the baseline window; half the window otherwise."""
    if window == DEB_WINDOW:
        return DEB_STRIDE
    if window < 2:
        raise ValueError("window must be >= 2")
    return window // 2


def window_rows(ids: list[int], window: int, stride: int, cls_id: int, sep_id: int) -> list[list[int]]:
    """Model inputs [CLS] + ids[s:s+window] + [SEP] for every window start (one [CLS][SEP] row for empty ids)."""
    return [[cls_id, *ids[s : s + window], sep_id] for s in window_starts(len(ids), window, stride)]


def max_score(scores: list[float]) -> float:
    if not scores:
        raise ValueError("no window scores")
    return max(scores)


def load_set(name: str) -> list[dict]:
    """Items as {id, label, source, family, lang, text}; label None = unlabeled."""
    if name == "frozen_test":
        raise SystemExit("frozen_test is read by the gate only")
    if name == "probes":
        from redteam.probes import load_probes

        return [{"id": p.probe_id, "label": int(p.kind == "attack"), "source": p.placement, "family": None,
                 "lang": "en", "text": p.body} for p in load_probes()]
    src = {"val": SPLITS_DIR / "val.jsonl", "replay": REPLAY_FILE}[name]
    return [{"id": r["id"], "label": r["label"], "source": r["source"], "family": r.get("family"),
             "lang": r.get("lang"), "text": r["text"]} for r in read_jsonl(src)]


def out_path(name: str) -> Path:
    return RESULTS / f"window_scores_{name}.jsonl"


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                out.add((d["id"], d["W"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def load_scores(name: str, path: Path | None = None) -> dict[int, dict[str, dict]]:
    """{W: {id: record}} from a window-score cache."""
    out: dict[int, dict[str, dict]] = {}
    for d in read_jsonl(path or out_path(name)):
        out.setdefault(d["W"], {})[d["id"]] = d
    return out


def from_cascade_cache(c: dict, window: int) -> list[float] | None:
    """Scores for window W taken from a cascade_scores record, or None when they must be computed."""
    if window == DEB_WINDOW:
        return [ch[2] for ch in c["chunks"]]
    if c["deb_tokens"] <= window:
        return [c["chunks"][0][2]]
    return None


def record(item_id: str, window: int, n_tokens: int, scores: list[float], source: str) -> dict:
    return {"id": item_id, "W": window, "stride": stride_for(window), "n_tokens": n_tokens,
            "scores": [round(float(s), 6) for s in scores], "from": source}


def run(name: str, window: int, max_items: int | None, threads: int) -> int:
    if window not in WINDOWS:
        raise SystemExit(f"window must be one of {WINDOWS}")
    out = out_path(name)
    repair(out)
    have = done_keys(out)
    items = sorted((r for r in load_set(name) if (r["id"], window) not in have), key=lambda r: r["id"])
    cascade = {}
    if name in ("val", "replay"):
        cascade = {r["id"]: r for r in read_jsonl(cache_path(name))}
    out.parent.mkdir(parents=True, exist_ok=True)
    # Cache copies cost nothing; write them all first, then compute up to max_items model records.
    todo = []
    with out.open("a", encoding="utf-8") as f:
        for r in items:
            c = cascade.get(r["id"])
            got = from_cascade_cache(c, window) if c else None
            if got is None:
                todo.append(r)
                continue
            f.write(json.dumps(record(r["id"], window, c["deb_tokens"], got, "cascade_cache"), sort_keys=True) + "\n")
        f.flush()
    n_needed = len(todo)
    if max_items is not None:
        todo = todo[:max_items]
    if not todo:
        print(f"{out.relative_to(ROOT)} W={window}: {n_needed} items need the model, none computed now")
        return 0
    from training.cascade_scores import DebertaScorer

    deb = DebertaScorer(threads=threads)
    t0, n_win = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            ids = deb.tok.encode(r["text"], add_special_tokens=False).ids
            c = cascade.get(r["id"])
            if c is not None and c["deb_tokens"] != len(ids):
                raise SystemExit(f"{r['id']}: {len(ids)} tokens here vs {c['deb_tokens']} in the cascade cache")
            rows = window_rows(ids, window, stride_for(window), deb.cls_id, deb.sep_id)
            scores = [deb.infer_one(x) for x in rows]
            n_win += len(rows)
            f.write(json.dumps(record(r["id"], window, len(ids), scores, "model"), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    left = n_needed - len(todo)
    print(f"W={window} {name}: scored {len(todo)} items ({n_win} windows) in {time.time() - t0:.0f}s; {left} left")
    return EXIT_MORE if left else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--max-items", type=int, default=None, help="compute at most this many new items, then exit")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    return run(a.set, a.window, a.max_items, a.threads)


if __name__ == "__main__":
    sys.exit(main())
