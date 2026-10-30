"""Leakage check across splits. Exits non-zero on ANY exact or near duplicate across splits.

    python -m training.check_leakage [--splits-dir data/splits] [--no-hash-check]

Checks:
  1. Exact: two records in different splits with the same dedup_key (NFKC, casefold, punctuation and
     whitespace collapsed).
  2. Near: char-5gram Jaccard >= 0.8 between records in different splits (MinHash LSH candidates at
     0.6, each verified with the exact Jaccard).
  3. Ids unique across all splits.
  4. data/frozen_test.sha256 matches data/splits/frozen_test.jsonl (default splits dir only).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from training.common import (FROZEN_TEST_SHA, NEAR_DUP_JACCARD, ROOT, SPLIT_NAMES, SPLITS_DIR, dedup_key,
                             near_duplicate_pairs, read_jsonl, sha256_file)


def check(splits_dir: Path, verify_hash: bool = True) -> list[str]:
    errors: list[str] = []
    recs = []
    for s in SPLIT_NAMES:
        p = splits_dir / f"{s}.jsonl"
        if not p.exists():
            errors.append(f"missing split file {p}")
            continue
        recs += [(s, r) for r in read_jsonl(p)]
    if errors:
        return errors

    ids = defaultdict(list)
    for s, r in recs:
        ids[r["id"]].append(s)
    for i, where in ids.items():
        if len(where) > 1:
            errors.append(f"duplicate id {i} in {where}")

    keys = [dedup_key(r["text"]) for _, r in recs]
    by_key = defaultdict(set)
    for (s, r), k in zip(recs, keys):
        by_key[k].add(s)
    n_exact = 0
    for k, where in by_key.items():
        if len(where) > 1:
            n_exact += 1
            errors.append(f"EXACT duplicate across {sorted(where)}: {k[:80]!r}")

    n_near = 0
    for i, j, jac in near_duplicate_pairs(keys):
        (si, ri), (sj, rj) = recs[i], recs[j]
        if si != sj:
            n_near += 1
            errors.append(f"NEAR duplicate (Jaccard {jac:.3f} >= {NEAR_DUP_JACCARD}) {si}:{ri['id']} vs {sj}:{rj['id']}")

    if verify_hash:
        expected = FROZEN_TEST_SHA.read_text().split()[0] if FROZEN_TEST_SHA.exists() else None
        actual = sha256_file(splits_dir / "frozen_test.jsonl")
        if expected != actual:
            errors.append(f"frozen_test sha256 {actual} != {FROZEN_TEST_SHA.relative_to(ROOT)} ({expected})")
    print(f"checked {len(recs)} records in {len(SPLIT_NAMES)} splits: "
          f"{n_exact} exact and {n_near} near cross-split duplicates, {len(errors)} errors")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--no-hash-check", action="store_true")
    a = ap.parse_args(argv)
    default = a.splits_dir.resolve() == SPLITS_DIR.resolve()
    errors = check(a.splits_dir, verify_hash=default and not a.no_hash_check)
    for e in errors[:50]:
        print("ERROR", e, file=sys.stderr)
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
