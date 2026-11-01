"""Turn successful evasions from an adaptive run into a small, committed, labelled probe set.

Input is a run's raw queries.jsonl (gitignored; it holds the full candidate bodies). Output is a
committed JSON probe set of the bodies that evaded the classifier, labelled injection=1, for later
use by the probe runner / regression checks. THIS MILESTONE DOES NOT TRAIN ON THEM (RFC issue #7:
red-team examples may only enter `train` after label review and with strategy/seed isolation).

Safety: every email-like address in a committed body must be a reserved test address (.invalid /
example.*); any body containing a non-reserved address is dropped, so no real email can be committed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..scenario import _reserved

_ADDR = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PROBE_SET_VERSION = 1
MAX_PROBES_DEFAULT = 12
MAX_BODY_CHARS = 20_000  # keep the committed file small; long_dilution bodies are truncated for storage


def addresses_all_reserved(body: str) -> bool:
    return all(_reserved(a) for a in _ADDR.findall(body))


def build(run_dir: Path, *, max_probes: int = MAX_PROBES_DEFAULT,
          require_harm: bool = False) -> dict:
    raw = run_dir / "queries.jsonl"
    summary = json.loads((run_dir / "summary.json").read_text())
    meta = summary.get("meta", {})
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]

    probes = []
    seen_hashes: set[str] = set()
    for r in rows:
        if not r.get("classifier_allowed"):
            continue
        if require_harm and not r.get("mode_b_attack_success"):
            continue
        body = r.get("body") or ""
        if not body or r["body_sha16"] in seen_hashes:
            continue
        if not addresses_all_reserved(body):
            continue  # safety: never commit a body with a non-reserved address
        seen_hashes.add(r["body_sha16"])
        probes.append({
            "probe_id": f"adv_{r['goal_id']}_{r.get('core_id', 'c')}_q{r['query']}",
            "label": 1,
            "injection": True,
            "source_goal": r["goal_id"],
            "core_id": r.get("core_id"),
            "family": r.get("family"),
            "placement": r.get("placement"),
            "evaded_classifier": True,
            "end_to_end_harm_modeB": bool(r.get("mode_b_attack_success")),
            "body_sha16": r["body_sha16"],
            "body": body[:MAX_BODY_CHARS],
            "body_truncated_for_storage": len(body) > MAX_BODY_CHARS,
        })
        if len(probes) >= max_probes:
            break

    return {
        "probe_set": "adaptive_evasions",
        "probe_set_version": PROBE_SET_VERSION,
        "generated_from": str(run_dir),
        "measured_against_bundle_sha256": meta.get("bundle_sha256"),
        "attacker_model": meta.get("attacker_model"),
        "feedback": meta.get("feedback"),
        "do_not_train": True,
        "note": ("Bodies that got past the served D23 classifier (label=injection). Committed for "
                 "regression/probe use only; NOT used for training in milestone adaptive-attacker "
                 "(RFC issue #7). All addresses are reserved test domains."),
        "n_probes": len(probes),
        "probes": probes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="an adaptive run dir (with queries.jsonl + summary.json)")
    ap.add_argument("--out", type=Path, default=Path("fixtures/probes/adaptive/adaptive_evasions.json"))
    ap.add_argument("--max-probes", type=int, default=MAX_PROBES_DEFAULT)
    ap.add_argument("--require-harm", action="store_true", help="keep only evasions that also caused a mode-B send")
    args = ap.parse_args(argv)
    out = build(args.run, max_probes=args.max_probes, require_harm=args.require_harm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "n_probes": out["n_probes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
