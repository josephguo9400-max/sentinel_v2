"""Produce the int8 Stage 2 candidate artifacts/candidates/deberta-pi-v2-int8/ from the verified fp32 baseline.

    python scripts/fetch_models.py                       # fp32 baseline, sha256-verified
    flock /home/claude/measure.lock -c 'python scripts/quantize_deberta.py'

Method: onnxruntime.quantization.quantize_dynamic with pinned settings (QUANT_SETTINGS below): int8 weights,
activations quantized dynamically at run time, MatMul/Gemm only (embeddings, LayerNorm and softmax stay fp32),
per-tensor scales, no reduce_range. The fp32 input is verified against the baseline manifest before quantizing.

Determinism: the output bytes depend only on the input model, the settings and the onnxruntime / onnx versions. The
expected output sha256 is the candidate manifest's `sha256`: on a mismatch the script
fails and leaves the new file at <dir>/onnx/model.onnx.mismatch for inspection; with --update-pin it writes the new
hash into the manifest instead (an intentional change, to be committed together with the reason).

Weights are gitignored (artifacts/candidates/deberta-pi-v2-int8/.gitignore); the manifest, this script and the hash
are committed. The tokenizer and config are hard-linked (or copied) from the baseline after verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts" / "baseline" / "deberta-pi-v2"
TARGET = ROOT / "artifacts" / "candidates" / "deberta-pi-v2-int8"
QUANT_SETTINGS = {
    "weight_type": "QInt8",
    "per_channel": False,
    "reduce_range": False,
    "op_types_to_quantize": ["MatMul", "Gemm"],
    "extra_options": {"MatMulConstBOnly": True},
}
# The expected output sha256 is the candidate manifest's `sha256` (single source of truth), produced with:
PINNED_VERSIONS = {"onnxruntime": "1.29.0", "onnx": "1.23.2"}
AUX = ("tokenizer.json", "config.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-pin", action="store_true", help="accept a new output hash and write it to the manifest")
    ap.add_argument("--verify-only", action="store_true", help="only check the existing output against the manifest")
    args = ap.parse_args(argv)

    manifest_path = TARGET / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    out_path = TARGET / manifest["artifact_path"]
    if args.verify_only:
        got = sha256(out_path) if out_path.is_file() else None
        ok = got == manifest["sha256"]
        print(f"{out_path.relative_to(ROOT)} sha256 {got}: {'matches' if ok else 'DOES NOT match'} the manifest")
        return 0 if ok else 1

    versions = {"onnxruntime": onnxruntime.__version__, "onnx": onnx.__version__}
    if versions != PINNED_VERSIONS:
        print(f"warning: versions {versions} differ from the pinned {PINNED_VERSIONS}; the output hash will differ",
              file=sys.stderr)
    base_manifest = json.loads((BASELINE / "manifest.json").read_text())
    fp32 = BASELINE / base_manifest["artifact_path"]
    if sha256(fp32) != base_manifest["sha256"]:
        print(f"{fp32} does not match the baseline manifest sha256; run scripts/fetch_models.py", file=sys.stderr)
        return 1
    for name in AUX:  # tokenizer/config: verified copies of the baseline's
        src = fp32.parent / name
        place(src, out_path.parent / name)
        if sha256(out_path.parent / name) != sha256(src):
            print(f"copy of {name} differs", file=sys.stderr)
            return 1

    tmp = out_path.with_suffix(".onnx.tmp")
    quantize_dynamic(fp32, tmp, op_types_to_quantize=QUANT_SETTINGS["op_types_to_quantize"],
                     per_channel=QUANT_SETTINGS["per_channel"], reduce_range=QUANT_SETTINGS["reduce_range"],
                     weight_type=getattr(QuantType, QUANT_SETTINGS["weight_type"]),
                     extra_options=QUANT_SETTINGS["extra_options"])
    digest = sha256(tmp)
    pinned = manifest["sha256"]
    if digest != pinned and not args.update_pin:
        bad = out_path.with_suffix(".onnx.mismatch")
        os.replace(tmp, bad)
        print(f"output sha256 {digest} != manifest {pinned}; kept as {bad.relative_to(ROOT)} "
              f"(rerun with --update-pin if the change is intended)", file=sys.stderr)
        return 1
    os.replace(tmp, out_path)
    if digest != pinned:
        manifest["sha256"] = digest
        manifest["runtime"]["python"] = platform.python_version()
        manifest["runtime"]["onnxruntime"] = onnxruntime.__version__
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"manifest sha256 updated to {digest}")
    print(f"wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size / 1e6:.1f} MB), sha256 {digest} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
