"""Controlled dilution curve: DeBERTa score of a short injection vs the amount of benign context around it.

    flock /home/claude/measure.lock python -m training.dilution_curve score --window 510 --max-items 300
    python -m training.dilution_curve summarize --windows 510,128     # no model; writes the results JSON

Injections (62):
  probe:<id>  the two redteam/probes.py INJECTIONS, as the middle_overlong probes embed them (verb + " " + tail)
  val:<id>    short val injections: label 1, lang en, <= 80 DeBERTa content tokens; up to 20 per source in
              (spml, jackhhao, deepset) by a seeded hash order, spml filling the rest up to 60 (jackhhao has only
              13 eligible items). Items DeBERTa misses even alone are kept; rates conditional on "detected alone"
              are reported separately.
Benign context pools (one 510-token context per injection and pool, seeded by the injection id):
  newsletter  the probe set v2 sentence pool (redteam.probes._pad_to_tokens), exactly 510 tokens
  val_benign  val benign items (label 0, lang en, not benign_hard) that DeBERTa alone scores < 0.5 in
              training/results/cascade_scores_val.jsonl, shuffled and concatenated, cut to 510 tokens
Composition is at the DeBERTa token-id level, so the token counts are exact: with ctx = the first L context tokens,
  start = inj + ctx, middle = ctx[:L//2] + inj + ctx[L//2:], end = ctx + inj      (L = 0: the injection alone)
for L in LENGTHS. A cut can fall inside a word. Context-alone items (ctx with no injection, L > 0) give the
false-positive side. Each composed sequence is scored like the gateway would (training.window_scores.window_rows:
windows of W content tokens, stride 384 for W = 510 and W/2 otherwise, [CLS] + window + [SEP], max over all
windows). With L = 510 the sequence is longer than 510 tokens, so the W = 510 score is a max over two windows,
both of which contain most of the context.

Cache: training/results/dilution_scores.jsonl, one line per (key, W) with the window scores; resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from training.cascade_scores import cache_path, repair
from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file, write_json
from training.metrics import rate
from training.window_scores import EXIT_MORE, max_score, stride_for, window_rows

SEED = "dilution-v1:20261030"
LENGTHS = (0, 32, 64, 128, 256, 384, 510)
POSITIONS = ("start", "middle", "end")
POOLS = ("newsletter", "val_benign")
SOURCES = ("spml", "jackhhao", "deepset")
PER_SOURCE = 20
N_VAL = 60
MAX_INJ_TOKENS = 80
CONTEXT_TOKENS = 510
THRESHOLD = 0.5
CACHE = ROOT / "training" / "results" / "dilution_scores.jsonl"
OUT = ROOT / "training" / "results" / "dilution_curve.json"


def hash_order(key: str) -> str:
    return hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()


def select_val_injections(eligible: list[dict], per_source: int = PER_SOURCE, total: int = N_VAL,
                          sources: tuple[str, ...] = SOURCES, fill: str = "spml") -> list[dict]:
    """Stratified pick: up to per_source per source (hash order), then `fill` tops the sample up to `total`."""
    by = defaultdict(list)
    for r in eligible:
        by[r["source"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: hash_order(r["id"]))
    take = {s: by[s][:per_source] for s in sources}
    short = total - sum(len(v) for v in take.values())
    if short > 0:
        take[fill] = by[fill][: len(take[fill]) + short]
    return [r for s in sources for r in take[s]]


def compose(inj: list[int], ctx: list[int], length: int, position: str) -> tuple[list[int], tuple[int, int]]:
    """(ids, injection token span [a, b)) for the injection placed in the first `length` context tokens."""
    if length > len(ctx):
        raise ValueError("context too short")
    c = ctx[:length]
    if position == "start":
        before, after = [], c
    elif position == "end":
        before, after = c, []
    elif position == "middle":
        before, after = c[: length // 2], c[length // 2 :]
    else:
        raise ValueError(position)
    return [*before, *inj, *after], (len(before), len(before) + len(inj))


def items(injections: list[dict], contexts: dict[tuple[str, str], list[int]]) -> list[dict]:
    """Every composed and context-alone item, in a fixed order. injections: {id, group, ids}."""
    out = []
    for inj in injections:
        for pool in POOLS:
            ctx = contexts[(inj["id"], pool)]
            for length in LENGTHS:
                for pos in (("alone",) if length == 0 else POSITIONS):
                    ids, span = compose(inj["ids"], ctx, length, "start" if pos == "alone" else pos)
                    out.append({"key": f"{inj['id']}|{pool}|{length}|{pos}", "kind": "injection", "inj": inj["id"],
                                "group": inj["group"], "pool": pool, "L": length, "pos": pos, "ids": ids,
                                "span": list(span)})
                if length:
                    out.append({"key": f"ctx|{inj['id']}|{pool}|{length}", "kind": "context", "inj": inj["id"],
                                "group": inj["group"], "pool": pool, "L": length, "pos": None, "ids": ctx[:length],
                                "span": None})
    keys = [x["key"] for x in out]
    assert len(keys) == len(set(keys))
    return out


@lru_cache(maxsize=1)
def _tok():
    from tokenizers import Tokenizer

    from training.cascade_scores import DEBERTA_DIR

    return Tokenizer.from_file(str(DEBERTA_DIR / "tokenizer.json"))


def _ids(text: str) -> list[int]:
    return _tok().encode(text, add_special_tokens=False).ids


def build_inputs() -> tuple[list[dict], dict, dict]:
    """(injections, contexts, provenance). Deterministic; reads val and the val cascade cache, never frozen_test."""
    from redteam.probes import INJECTIONS, _pad_to_tokens, _rng

    cascade = {r["id"]: r for r in read_jsonl(cache_path("val"))}
    val = list(read_jsonl(SPLITS_DIR / "val.jsonl"))
    injections = []
    for spec in INJECTIONS:
        text = spec["verb"] + " " + spec["tail"].format(address=spec["address"])
        injections.append({"id": f"probe:{spec['id']}", "group": "probe", "source": "probe", "ids": _ids(text)})
    eligible = [r for r in val if r["label"] == 1 and r.get("lang") == "en"
                and cascade[r["id"]]["deb_tokens"] <= MAX_INJ_TOKENS]
    for r in select_val_injections(eligible):
        ids = _ids(r["text"])
        assert len(ids) == cascade[r["id"]]["deb_tokens"]
        injections.append({"id": f"val:{r['id']}", "group": f"val_{r['source']}", "source": r["source"], "ids": ids})
    benign = sorted((r for r in val if r["label"] == 0 and r.get("lang") == "en" and r["source"] != "benign_hard"
                     and cascade[r["id"]]["deb"] < THRESHOLD), key=lambda r: r["id"])
    contexts = {}
    for inj in injections:
        news = _ids(_pad_to_tokens(CONTEXT_TOKENS, _rng(SEED, "newsletter", inj["id"])))
        assert len(news) == CONTEXT_TOKENS
        contexts[(inj["id"], "newsletter")] = news
        order = sorted(benign, key=lambda r: hash_order(f"{inj['id']}|{r['id']}"))
        ctx: list[int] = []
        for r in order:
            ctx += _ids(r["text"])
            if len(ctx) >= CONTEXT_TOKENS:
                break
        contexts[(inj["id"], "val_benign")] = ctx[:CONTEXT_TOKENS]
    prov = {"n_injections": len(injections),
            "by_group": {g: sum(i["group"] == g for i in injections) for g in sorted({i["group"] for i in injections})},
            "eligible_val_injections_by_source": {s: sum(r["source"] == s for r in eligible) for s in SOURCES},
            "val_benign_pool_size": len(benign),
            "injection_tokens": {i["id"]: len(i["ids"]) for i in injections}}
    return injections, contexts, prov


# ----------------------------------------------------------------------------------------------- scoring

def _done(path: Path) -> set[tuple[str, int]]:
    out = set()
    if path.exists():
        for d in read_jsonl(path):
            out.add((d["key"], d["W"]))
    return out


def score(window: int, max_items: int | None, threads: int, cache: Path = CACHE) -> int:
    injections, contexts, _ = build_inputs()
    repair(cache)
    have = _done(cache)
    todo = [x for x in items(injections, contexts) if (x["key"], window) not in have]
    n_needed = len(todo)
    if max_items is not None:
        todo = todo[:max_items]
    if not todo:
        print(f"dilution W={window}: {n_needed} left, none computed now")
        return 0 if n_needed == 0 else EXIT_MORE
    from training.cascade_scores import DebertaScorer

    deb = DebertaScorer(threads=threads)
    t0 = time.time()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as f:
        for x in todo:
            rows = window_rows(x["ids"], window, stride_for(window), deb.cls_id, deb.sep_id)
            s = [round(deb.infer_one(r), 6) for r in rows]
            f.write(json.dumps({"key": x["key"], "W": window, "n_tokens": len(x["ids"]), "scores": s},
                               sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    left = n_needed - len(todo)
    print(f"dilution W={window}: scored {len(todo)} in {time.time() - t0:.0f}s; {left} left")
    return EXIT_MORE if left else 0


# ----------------------------------------------------------------------------------------------- summary

def _cell(scores: list[float], alone_ok: list[bool] | None = None) -> dict:
    flags = [s >= THRESHOLD for s in scores]
    out = {"n": len(scores), "detected": rate(sum(flags), len(flags)),
           "median_score": round(statistics.median(scores), 6) if scores else None}
    if alone_ok is not None:
        sel = [f for f, ok in zip(flags, alone_ok) if ok]
        out["detected_given_detected_alone"] = rate(sum(sel), len(sel))
    return out


def summarize(windows: list[int], cache: Path = CACHE) -> dict:
    injections, contexts, prov = build_inputs()
    all_items = items(injections, contexts)
    sc: dict[int, dict[str, list[float]]] = defaultdict(dict)
    for d in read_jsonl(cache):
        sc[d["W"]][d["key"]] = d["scores"]
    res = {"seed": SEED, "threshold": THRESHOLD, "lengths": list(LENGTHS), "positions": list(POSITIONS),
           "pools": list(POOLS), "scores_file_sha256": sha256_file(cache), "inputs": prov, "by_window": {}}
    for w in windows:
        missing = [x["key"] for x in all_items if x["key"] not in sc[w]]
        if missing:
            raise SystemExit(f"W={w}: {len(missing)} items not scored yet (first: {missing[0]})")
        s = {x["key"]: max_score(sc[w][x["key"]]) for x in all_items}
        alone = {(x["inj"], x["pool"]): s[x["key"]] >= THRESHOLD for x in all_items if x["pos"] == "alone"}
        out: dict = {"stride": stride_for(w), "injection": {}, "context_alone": {}, "by_group": {},
                     "probe_injections": {}}
        for pool in POOLS:
            out["injection"][pool], out["context_alone"][pool], out["by_group"][pool] = {}, {}, {}
            for length in LENGTHS:
                for pos in (("alone",) if length == 0 else (*POSITIONS, "any")):
                    sel = [x for x in all_items if x["kind"] == "injection" and x["pool"] == pool
                           and x["L"] == length and (pos == "any" or x["pos"] == pos)]
                    out["injection"][pool][f"{length}|{pos}"] = _cell(
                        [s[x["key"]] for x in sel], [alone[(x["inj"], pool)] for x in sel])
                if length:
                    sel = [x for x in all_items if x["kind"] == "context" and x["pool"] == pool and x["L"] == length]
                    out["context_alone"][pool][str(length)] = _cell([s[x["key"]] for x in sel])
                for g in sorted(prov["by_group"]):
                    sel = [x for x in all_items if x["kind"] == "injection" and x["pool"] == pool
                           and x["L"] == length and x["group"] == g]
                    out["by_group"][pool].setdefault(g, {})[str(length)] = _cell(
                        [s[x["key"]] for x in sel], [alone[(x["inj"], pool)] for x in sel])
            for inj in (i for i in injections if i["group"] == "probe"):
                out["probe_injections"].setdefault(inj["id"], {})[pool] = {
                    f"{x['L']}|{x['pos']}": s[x["key"]] for x in all_items
                    if x["kind"] == "injection" and x["inj"] == inj["id"] and x["pool"] == pool}
        res["by_window"][str(w)] = out
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("score")
    a1.add_argument("--window", type=int, required=True)
    a1.add_argument("--max-items", type=int, default=None)
    a1.add_argument("--threads", type=int, default=2)
    a2 = sub.add_parser("summarize")
    a2.add_argument("--windows", default="510")
    a2.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        return score(a.window, a.max_items, a.threads)
    write_json(a.out, summarize([int(x) for x in a.windows.split(",")]))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
