"""Deterministic, source-aware train / val / frozen_test splits with dedup BEFORE splitting.

    python -m training.sources --fetch      # raw files -> data/raw/ (sha256-verified)
    python -m training.build_splits         # -> data/splits/{train,val,frozen_test}.jsonl, data/frozen_test.sha256

Pipeline (see training/DATA.md for the rationale):
 1. Load every pinned source (sha256-verified) and the synthetic benign-hard generator.
 2. Clean text (CRLF -> LF, strip; nothing else, so stored text is what a gateway would see).
 3. Cluster duplicates: exact (same dedup_key) and near (char-5gram Jaccard >= 0.8 via MinHash LSH),
    joined with union-find. Clusters whose members disagree on the label are dropped entirely.
    Each surviving cluster keeps ONE representative (fixed source priority, then source order).
 4. Assign each representative to a split by source / source group (never by item for held-out sources):
      yanis, gandalf          -> frozen_test (whole held-out sources)
      spml                    -> by system-prompt group: 60% train / 15% val / 25% frozen_test
      deepset                 -> its original train split -> train, original test split -> val
      jackhhao                -> 80% train / 20% val (CONFIRMED in Stage 2 training data; never frozen)
      benign_hard template 0  -> val (dev); templates 1-3 -> frozen_test
 5. Any frozen_test record whose cluster matched a dataset named in the Stage 2 card
    (data/stage2_overlap_keys.txt, produced by training.stage2_overlap) is dropped from frozen_test.
 6. Write JSONL sorted by id, the frozen-test hash and a counts manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from training import benign_hard
from training.common import (FROZEN_TEST_SHA, LSH_CANDIDATE_JACCARD, NEAR_DUP_JACCARD, NUM_PERM, ROOT, SEED,
                             SHINGLE_K, SPLIT_NAMES, SPLITS_DIR, UnionFind, dedup_key, key_hash,
                             near_duplicate_pairs, sha256_file, stable_unit, write_json, write_jsonl)
from training.sources import SOURCES, load_all

PIPELINE_VERSION = "splits-v1"
OVERLAP_KEYS = ROOT / "data" / "stage2_overlap_keys.txt"
# Representative choice inside a duplicate cluster: lower = preferred. Held-out frozen sources win so a
# frozen record is never silently replaced by a training copy of itself; the copy is simply removed.
PRIORITY = {"yanis": 0, "gandalf": 1, "spml": 2, "deepset": 3, "jackhhao": 4, "benign_hard": 5}
SPML_FRACTIONS = (0.60, 0.15)  # train, val; rest frozen_test
JACKHHAO_TRAIN = 0.80


def content_sha256(rows: list[dict]) -> str:
    """Hash over (id, text, label) only: what a model trains on, independent of audit metadata fields."""
    import hashlib
    import json

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["id"]):
        h.update(json.dumps([r["id"], r["text"], r["label"]], ensure_ascii=False).encode("utf-8") + b"\n")
    return h.hexdigest()


def _lang(text: str) -> str:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


def assign_split(rec: dict) -> str:
    src = rec["source"]
    if src in ("yanis", "gandalf"):
        return "frozen_test"
    if src == "spml":
        u = stable_unit(rec["group"], salt=f"{SEED}:spml")
        return "train" if u < SPML_FRACTIONS[0] else "val" if u < sum(SPML_FRACTIONS) else "frozen_test"
    if src == "deepset":
        return "train" if rec["group"] == "deepset:train" else "val"
    if src == "jackhhao":
        return "train" if stable_unit(key_hash(rec["text"]), salt=f"{SEED}:jackhhao") < JACKHHAO_TRAIN else "val"
    if src == "benign_hard":
        return "val" if rec["template"] == benign_hard.DEV_TEMPLATE else "frozen_test"
    raise ValueError(f"no split rule for source {src}")


def collect(limit_per_source: int | None = None) -> list[dict]:
    recs = load_all()
    if limit_per_source:
        per = Counter()
        kept = []
        for r in recs:
            if per[r["source"]] < limit_per_source:
                per[r["source"]] += 1
                kept.append(r)
        recs = kept
    for i, r in enumerate(benign_hard.generate(SEED)):
        r.update(source="benign_hard", source_index=i, stage2_overlap="none_synthetic")
        recs.append(r)
    return recs


def build(out_dir: Path = SPLITS_DIR, frozen_sha_path: Path | None = FROZEN_TEST_SHA,
          limit_per_source: int | None = None, verbose: bool = True) -> dict:
    recs = collect(limit_per_source)
    n = len(recs)
    keys = [dedup_key(r["text"]) for r in recs]

    uf = UnionFind(n)
    first_by_key: dict[str, int] = {}
    n_exact_links = 0
    for i, k in enumerate(keys):
        if k in first_by_key:
            uf.union(first_by_key[k], i)
            n_exact_links += 1
        else:
            first_by_key[k] = i
    # Near-dup search runs on unique keys only (exact duplicates are already linked).
    uniq = list(first_by_key.values())
    near = near_duplicate_pairs([keys[i] for i in uniq])
    for a, b, _ in near:
        uf.union(uniq[a], uniq[b])

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    overlap_keys = set(OVERLAP_KEYS.read_text().split()) if OVERLAP_KEYS.exists() else set()
    dropped_conflict = Counter()
    dropped_overlap = Counter()
    removed_as_dup = Counter()
    kept = []
    for members in clusters.values():
        labels = {recs[i]["label"] for i in members}
        if len(labels) > 1:
            for i in members:
                dropped_conflict[recs[i]["source"]] += 1
            continue
        rep = min(members, key=lambda i: (PRIORITY[recs[i]["source"]], recs[i]["source_index"]))
        for i in members:
            if i != rep:
                removed_as_dup[recs[i]["source"]] += 1
        r = dict(recs[rep])
        r["split"] = assign_split(r)
        r["dup_cluster_size"] = len(members)
        r["dup_cluster_sources"] = sorted({recs[i]["source"] for i in members})
        hit = any(key_hash(recs[i]["text"]) in overlap_keys for i in members)
        r["stage2_named_match"] = hit
        if hit and r["split"] == "frozen_test":
            dropped_overlap[r["source"]] += 1
            continue
        kept.append(r)

    rows = {s: [] for s in SPLIT_NAMES}
    for r in kept:
        out = {
            "id": key_hash(r["text"])[:20],
            "text": r["text"],
            "label": r["label"],
            "source": r["source"],
            "source_label": r["source_label"],
            "group": r["group"],
            "lang": _lang(r["text"]),
            "stage2_overlap": r["stage2_overlap"],
            "stage2_named_match": r["stage2_named_match"],
            "dup_cluster_size": r["dup_cluster_size"],
            "dup_cluster_sources": r["dup_cluster_sources"],
        }
        if r["source"] == "benign_hard":
            out.update(family=r["family"], template=r["template"], synthetic=True, generator=r["generator"])
        rows[r["split"]].append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s in SPLIT_NAMES:
        rows[s].sort(key=lambda x: x["id"])
        p = out_dir / f"{s}.jsonl"
        write_jsonl(p, rows[s])
        by = Counter((x["source"], x["label"]) for x in rows[s])
        summary[s] = {
            "n": len(rows[s]),
            "positives": sum(x["label"] for x in rows[s]),
            "negatives": sum(1 - x["label"] for x in rows[s]),
            "by_source_label": {f"{k[0]}:{k[1]}": v for k, v in sorted(by.items())},
            "sha256": sha256_file(p),
            "content_sha256": content_sha256(rows[s]),
        }
    if frozen_sha_path is not None:
        rel = (out_dir / "frozen_test.jsonl").resolve().relative_to(ROOT)
        frozen_sha_path.write_text(f"{summary['frozen_test']['sha256']}  {rel.as_posix()}\n")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": SEED,
        "limit_per_source": limit_per_source,
        "near_duplicate": {"method": "MinHash LSH (datasketch) + exact Jaccard verification",
                           "shingles": f"character {SHINGLE_K}-grams of dedup_key", "num_perm": NUM_PERM,
                           "lsh_candidate_threshold": LSH_CANDIDATE_JACCARD, "jaccard_threshold": NEAR_DUP_JACCARD},
        "sources": {k: {"repo_id": v.repo_id, "revision": v.revision, "license": v.license,
                        "stage2_overlap": v.stage2_overlap} for k, v in SOURCES.items()},
        "benign_hard_generator": benign_hard.GENERATOR_VERSION,
        "n_input_records": n,
        "n_exact_duplicate_links": n_exact_links,
        "n_near_duplicate_pairs_among_unique": len(near),
        "removed_as_duplicate_by_source": dict(sorted(removed_as_dup.items())),
        "dropped_label_conflict_by_source": dict(sorted(dropped_conflict.items())),
        "dropped_from_frozen_stage2_named_match_by_source": dict(sorted(dropped_overlap.items())),
        "stage2_overlap_keys_sha256": sha256_file(OVERLAP_KEYS) if OVERLAP_KEYS.exists() else None,
        "splits": summary,
    }
    write_json(out_dir / "MANIFEST.json", manifest)
    if verbose:
        import json

        print(json.dumps({k: manifest[k] for k in manifest if k not in ("sources", "near_duplicate")}, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--limit-per-source", type=int, default=None, help="for tests only")
    a = ap.parse_args()
    default = a.out_dir.resolve() == SPLITS_DIR.resolve()
    build(a.out_dir, FROZEN_TEST_SHA if default else None, a.limit_per_source)
