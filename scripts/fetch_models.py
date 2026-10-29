"""Download the Stage 2 detector (ONNX build) into models/. Weights are never committed to git."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "protectai/deberta-v3-base-prompt-injection-v2"  # Apache-2.0

if __name__ == "__main__":
    path = snapshot_download(REPO_ID, allow_patterns=["onnx/*", "LICENSE"], local_dir=ROOT / "models" / "deberta-pi-v2")
    print("downloaded to", path)
