"""The version of the policy interpreter (D25.2: manual semantic version).

Bump ENGINE_VERSION whenever a decision-path file changes (the list is in serving/engine_version.json), then run
`python -m serving.engine_version --record`. tests/serving/test_engine_version.py fails while the recorded sha256 of
any decision-path file differs from the file, and `--record` refuses to re-record changed files under an unchanged
version. A new engine version changes every bundle_sha256 (serving/policy.py), so no gate verdict carries over.

Semantics: MAJOR = a policy validated on another major is refused; MINOR = new behaviour (a policy validated on a
newer minor is refused); PATCH = a fix that should not change any decision (still a new bundle, re-gated).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ENGINE_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
RECORD = Path(__file__).with_name("engine_version.json")


def file_hashes(paths: list[str]) -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def load_record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def check() -> list[str]:
    """Problems with the record (empty = consistent)."""
    rec = load_record()
    out = []
    if rec["engine_version"] != ENGINE_VERSION:
        out.append(f"serving/engine_version.json records {rec['engine_version']}, ENGINE_VERSION is {ENGINE_VERSION}")
    now = file_hashes(sorted(rec["files"]))
    for path, sha in sorted(rec["files"].items()):
        if not sha:
            out.append(f"{path} has no recorded hash: run python -m serving.engine_version --record")
        elif now[path] != sha:
            out.append(f"{path} changed since engine {rec['engine_version']} was recorded: bump ENGINE_VERSION and run "
                       f"python -m serving.engine_version --record")
    return out


def record() -> dict:
    """Record the current hashes. Refuses to overwrite a changed file list under an unchanged version, so a
    behaviour change cannot be recorded without a bump. An entry with an empty hash is new (first recording of a
    file) and may be filled in at any version."""
    rec = load_record()
    now = file_hashes(sorted(rec["files"]))
    changed = {k for k, v in rec["files"].items() if v and now[k] != v}
    if rec["engine_version"] == ENGINE_VERSION and changed:
        raise SystemExit(f"decision-path files changed but ENGINE_VERSION is still {ENGINE_VERSION} "
                         f"({', '.join(sorted(changed))}): bump it first")
    rec.update(engine_version=ENGINE_VERSION, files=now)
    RECORD.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args(argv)
    if a.record:
        print(json.dumps(record(), indent=1))
        return 0
    problems = check()
    print("\n".join(problems) or f"engine {ENGINE_VERSION}: decision-path files match the record")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
