"""Fetch the Stage 2 baseline (protectai/deberta-v3-base-prompt-injection-v2, ONNX fp32) into its artifact directory
artifacts/baseline/deberta-pi-v2/ at a pinned revision, and verify the sha256 of every file the gateway loads.

    python scripts/fetch_models.py                       # download from the Hugging Face Hub
    python scripts/fetch_models.py --from-local models/deberta-pi-v2   # reuse an existing download (hard link
                                                                       # or copy), verified the same way

Weights are never committed (artifacts/baseline/deberta-pi-v2/.gitignore). The model hash must also equal the
`sha256` in the committed manifest; the script exits non-zero on any mismatch and deletes nothing it did not write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "protectai/deberta-v3-base-prompt-injection-v2"  # Apache-2.0
REVISION = "90c9989b1a342275dd0d1a95aad283c04e075671"
TARGET = ROOT / "artifacts" / "baseline" / "deberta-pi-v2"
# Files the gateway reads, relative to TARGET, with their pinned sha256.
PINNED = {
    "onnx/model.onnx": "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c",
    "onnx/tokenizer.json": "752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f",
    "onnx/config.json": "3093743035223c46b1497a72e939e56fa0a50afbd7bafbf7eb8aad060b8d23f8",
}
EXTRA = ("LICENSE",)  # copied when present, not hashed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(target: Path) -> list[str]:
    errors = []
    for rel, want in PINNED.items():
        p = target / rel
        if p.is_symlink():
            errors.append(f"{rel} is a symlink; the gateway rejects paths that resolve outside {target}")
        elif not p.is_file():
            errors.append(f"{rel} missing")
        elif (got := sha256(p)) != want:
            errors.append(f"{rel} sha256 {got}, expected {want}")
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest["sha256"] != PINNED[manifest["artifact_path"]]:
        errors.append("manifest.json sha256 differs from the pin in this script")
    return errors


def place(src: Path, dst: Path) -> None:
    """Hard link when possible (same filesystem, no extra 700 MB), else copy. Never a symlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-local", type=Path, default=None,
                    help="an existing snapshot directory (with onnx/ inside) to link or copy from")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    if not args.verify_only:
        if args.from_local is not None:
            for rel, want in PINNED.items():
                src = args.from_local / rel
                if sha256(src) != want:
                    print(f"{src} does not match the pinned sha256; refusing to use it", file=sys.stderr)
                    return 1
                place(src, TARGET / rel)
            for rel in EXTRA:
                if (args.from_local / rel).is_file():
                    place(args.from_local / rel, TARGET / rel)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(REPO_ID, revision=REVISION, allow_patterns=["onnx/*", "LICENSE"], local_dir=TARGET)
    errors = verify(TARGET)
    if errors:
        print("verification FAILED:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    print(f"{REPO_ID}@{REVISION[:8]} in {TARGET.relative_to(ROOT)}: sha256 verified for {', '.join(PINNED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
