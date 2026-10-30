"""Large weights must never be committed: every tracked or committable file under serving-owned paths is < 5 MB,
and generated model directories are ignored."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 5 * 1024 * 1024
OWNED = ["serving", "tests/serving", "artifacts/baseline", "artifacts/candidates/deberta-pi-v2-int8", "scripts"]


def _git(*args) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def test_no_committable_file_over_5mb():
    files = _git("ls-files", "--cached", "--others", "--exclude-standard", "--", *OWNED)
    big = [f for f in files if (ROOT / f).is_file() and (ROOT / f).stat().st_size > LIMIT]
    assert not big, big


def test_weight_dirs_are_ignored():
    for p in ("artifacts/baseline/deberta-pi-v2/onnx/model.onnx",
              "artifacts/candidates/deberta-pi-v2-int8/onnx/model.onnx",
              "artifacts/candidates/deberta-pi-v2-int8/onnx/tokenizer.json"):
        assert _git("check-ignore", p), p
