"""Download the Stage 2 detector (ONNX build) into models/ at a pinned revision and verify its hash.

Weights are never committed to git.
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "protectai/deberta-v3-base-prompt-injection-v2"  # Apache-2.0
REVISION = "90c9989b1a342275dd0d1a95aad283c04e075671"
MODEL_SHA256 = "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    target = ROOT / "models" / "deberta-pi-v2"
    snapshot_download(REPO_ID, revision=REVISION, allow_patterns=["onnx/*", "LICENSE"], local_dir=target)
    digest = sha256(target / "onnx" / "model.onnx")
    if digest != MODEL_SHA256:
        sys.exit(f"model.onnx sha256 mismatch: got {digest}, expected {MODEL_SHA256}")
    print(f"downloaded {REPO_ID}@{REVISION[:8]} to {target}, sha256 verified")
