"""Language mix per source, computed by committed code (the language column of training/DATA.md).

    python -m training.language_mix            # -> data/language_mix.json

Two sections:
  splits      From the committed split files (data/splits/*.jsonl), using each record's `lang` field
              (langdetect 1.0.9, DetectorFactory.seed = 0, first 2,000 characters; set by build_splits).
              This is after dedup, label-conflict removal and the Stage 2 named-match exclusion.
  raw         Only if data/raw/ is present (python -m training.sources --fetch): every raw record of each
              source as loaded by training.sources, BEFORE dedup, with the same langdetect settings. If raw
              data is absent, this section is null; it is never estimated from a sample.

Percentages are of all records of that source, rounded to 0.1. Language tags with under 1% of a source are
summed into `other`, with the full counts kept in `counts`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from training.build_splits import _lang
from training.common import ROOT, SPLIT_NAMES, SPLITS_DIR, read_jsonl, sha256_file, write_json

OUT = ROOT / "data" / "language_mix.json"
MIN_SHARE = 0.01


def _mix(counts: Counter) -> dict:
    n = sum(counts.values())
    pct = {}
    other = 0
    for lang, k in counts.most_common():
        if k / n >= MIN_SHARE:
            pct[lang] = round(100 * k / n, 1)
        else:
            other += k
    if other:
        pct["other"] = round(100 * other / n, 1)
    return {"n": n, "en": counts.get("en", 0), "non_en": n - counts.get("en", 0),
            "pct_en": round(100 * counts.get("en", 0) / n, 1), "pct": pct,
            "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))}


def from_splits(splits_dir: Path = SPLITS_DIR) -> dict:
    per_split: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    combined: dict[str, Counter] = defaultdict(Counter)
    hashes = {}
    for s in SPLIT_NAMES:
        p = splits_dir / f"{s}.jsonl"
        hashes[s] = sha256_file(p)
        for r in read_jsonl(p):
            per_split[s][r["source"]][r["lang"]] += 1
            combined[r["source"]][r["lang"]] += 1
    return {
        "what": "lang field of every committed split record (post-dedup)",
        "split_sha256": hashes,
        "by_source": {src: _mix(c) for src, c in sorted(combined.items())},
        "by_split_and_source": {s: {src: _mix(c) for src, c in sorted(per_split[s].items())} for s in SPLIT_NAMES},
    }


def from_raw() -> dict | None:
    from training.sources import SOURCES

    try:
        for src in SOURCES.values():
            src.verified()
    except (FileNotFoundError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}
    out = {}
    for name, src in SOURCES.items():
        out[name] = _mix(Counter(_lang(r["text"]) for r in src.loader(src)))
    return {"available": True,
            "what": "every raw record per source as loaded by training.sources (sha256-verified), before dedup; "
                    "langdetect 1.0.9, seed 0, first 2,000 characters",
            "by_source": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args(argv)
    res = {"generated_by": "python -m training.language_mix", "langdetect": "1.0.9, DetectorFactory.seed=0",
           "min_share_for_own_column": MIN_SHARE, "splits": from_splits(),
           "raw": None if a.no_raw else from_raw()}
    write_json(a.out, res)
    for src, m in res["splits"]["by_source"].items():
        print(f"splits {src:12s} n={m['n']:6d} en={m['pct_en']:5.1f}%  {m['pct']}")
    if res["raw"] and res["raw"].get("available"):
        for src, m in res["raw"]["by_source"].items():
            print(f"raw    {src:12s} n={m['n']:6d} en={m['pct_en']:5.1f}%  {m['pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
