"""Build the dataset-derived slice of the injection core library (cores_curated.json).

Reads committed train/val injection items (label 1), keeps only short, email-plausible send/forward
instructions that carry an email address, re-targets every address to the reserved placeholder
``{target}``, and writes them as cores with provenance ``dataset:<source>`` and family
``polite_request`` or ``action_line`` (whichever the surface form matches).

Isolation (RFC issue #7 / review n1): reads ONLY data/splits/train.jsonl and val.jsonl. It never
reads data/splits/frozen_test.jsonl or fixtures/scenarios/heldout/. The output carries
``do_not_train: true`` and is a probe/eval artifact, never training input.

    python -m redteam.adaptive.build_cores            # writes fixtures/adaptive/cores_curated.json
    python -m redteam.adaptive.build_cores --dry-run  # counts only; writes nothing

No body text is printed; only counts and ids, so attack payloads never reach stdout/logs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from training.common import SPLITS_DIR, read_jsonl, stable_unit

from ..scenario import _reserved
from .cores import CURATED_PATH

_ADDR = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SEND_VERB = re.compile(r"\b(send|forward|email|e-mail|mail|cc|bcc)\b", re.IGNORECASE)
_ACTION = re.compile(r"\b(send_email|ACTION\s*:|TOOL\s*:)", re.IGNORECASE)
FORBIDDEN = ("frozen_test", "heldout")
MAX_CHARS = 320            # email-plausible, short instruction only
PER_SOURCE_CAP = 4         # keep the curated slice small and auditable


def _retarget(text: str) -> str:
    """Replace every email address with the {target} placeholder (doubling any literal braces first)."""
    text = text.replace("{", "{{").replace("}", "}}")
    return _ADDR.sub("{target}", text)


def _eligible(row: dict) -> bool:
    if row.get("label") != 1 or row.get("lang") not in (None, "en"):
        return False
    text = row.get("text") or ""
    if not (0 < len(text) <= MAX_CHARS):
        return False
    if not _ADDR.search(text):
        return False  # must carry an address so we can re-target it
    if not _SEND_VERB.search(text):
        return False  # must be a send/forward action
    # Single-instruction, email-shaped: one address, no newpage-length prose.
    return len(_ADDR.findall(text)) <= 3


def build(dry_run: bool = False) -> dict:
    for part in CURATED_PATH.parts:
        if part in FORBIDDEN:
            raise RuntimeError("curated cores must not live under a forbidden path")
    per_source: dict[str, list[dict]] = {}
    diag = {"n_label1": 0, "n_with_address": 0, "n_with_send_verb": 0, "n_eligible": 0}
    for split in ("train", "val"):
        src_path = SPLITS_DIR / f"{split}.jsonl"
        for row in read_jsonl(src_path):
            if row.get("label") == 1:
                diag["n_label1"] += 1
                text = row.get("text") or ""
                if _ADDR.search(text):
                    diag["n_with_address"] += 1
                if _SEND_VERB.search(text):
                    diag["n_with_send_verb"] += 1
            if not _eligible(row):
                continue
            diag["n_eligible"] += 1
            source = row.get("source", "unknown")
            bucket = per_source.setdefault(source, [])
            if len(bucket) >= PER_SOURCE_CAP * 4:
                continue  # gather a pool, then down-select deterministically below
            template = _retarget(row["text"]).strip()
            # Re-targeting must leave every address reserved.
            if any(not _reserved(a) for a in _ADDR.findall(template.format(target="x@y.invalid"))):
                continue
            family = "action_line" if _ACTION.search(row["text"]) else "polite_request"
            bucket.append({"src_id": row["id"], "source": source, "family": family,
                           "template": template, "rank": stable_unit(row["id"])})

    cores = []
    for source in sorted(per_source):
        chosen = sorted(per_source[source], key=lambda r: r["rank"])[:PER_SOURCE_CAP]
        for i, r in enumerate(chosen):
            cores.append({
                "core_id": f"ds_{source}_{i}",
                "family": r["family"],
                "template": r["template"],
                "provenance": f"dataset:{source}",
                "attribution": f"{source} train/val injection {r['src_id']}, re-targeted to a reserved address",
            })

    out = {
        "core_set": "adaptive_cores_curated",
        "do_not_train": True,
        "note": ("Email send/forward injection cores re-shaped from committed train/val items and "
                 "re-targeted to a reserved placeholder. Probe/eval use only; never training input "
                 "(RFC issue #7). Frozen test and held-out scenarios are never read."),
        "sources": sorted(per_source),
        "n_cores": len(cores),
        "dataset_diagnostic": {
            **diag,
            "finding": ("Public train/val injection sets are not email-shaped: almost no label-1 item "
                        "carries a recipient address or an email send/forward instruction, so the "
                        "dataset slice contributes few or no email cores. The core library is built "
                        "from repo probe/fixture material and attributed public email-injection "
                        "templates (see redteam/adaptive/cores.py)."),
        },
        "cores": cores,
    }
    if not dry_run:
        CURATED_PATH.parent.mkdir(parents=True, exist_ok=True)
        CURATED_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = build(dry_run=args.dry_run)
    print(json.dumps({"wrote": None if args.dry_run else str(CURATED_PATH),
                      "n_cores": out["n_cores"], "sources": out["sources"],
                      "by_family": {f: sum(1 for c in out["cores"] if c["family"] == f)
                                    for f in sorted({c["family"] for c in out["cores"]})}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
