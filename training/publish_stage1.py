"""Publish stage1-tfidf-v2.1: the stage1-tfidf-v2 model bytes under a manifest schema 1.2.

    python -m training.publish_stage1   # -> artifacts/candidates/stage1-tfidf-v2.1/ (refuses to overwrite)

What changes versus stage1-tfidf-v2 (a config-only version; the model is not retrained):
  - schema_version 1.2 with `tokenizer` = the exact RE2 pattern inside the ONNX graph's Tokenizer node
    (read from the graph and compared with training.stage1_text.TOKEN_PATTERN_RE2), so max_input_tokens and
    DetectorResult.input_tokens are in those units;
  - `input_languages` = ["en"]: val/train hold almost no non-English data and Stage 1 recalls 17.6% of
    non-English frozen-test injections (DATA.md section 6), so the router must escalate everything else;
  - `gray_zone` = the setting chosen by training/cascade_sweep.py on val (training/results/cascade_sweep_val.json).
model.onnx is copied byte for byte and its sha256 is checked against the v2 manifest before and after.
stage1-tfidf-v2 is never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from contracts import ArtifactManifest
from training.common import ROOT, sha256_file, write_json
from training.stage1_text import TOKEN_PATTERN_RE2

CANDIDATES = ROOT / "artifacts" / "candidates"
SOURCE_VERSION = "stage1-tfidf-v2"
VERSION = "stage1-tfidf-v2.1"
SWEEP = ROOT / "training" / "results" / "cascade_sweep_val.json"
INPUT_LANGUAGES = ["en"]


def graph_token_pattern(model_path: Path) -> str:
    import onnx

    m = onnx.load(str(model_path))
    pats = [onnx.helper.get_attribute_value(a).decode() for n in m.graph.node if n.op_type == "Tokenizer"
            for a in n.attribute if a.name == "tokenexp"]
    if len(pats) != 1:
        raise ValueError(f"expected exactly one Tokenizer node with tokenexp, found {pats}")
    return pats[0]


def build_manifest(src: ArtifactManifest, gray_zone: tuple[float, float], tokenizer: str,
                   created_at: str) -> ArtifactManifest:
    d = src.model_dump()
    d.update(schema_version="1.2", detector_version=VERSION, gray_zone=tuple(gray_zone), tokenizer=tokenizer,
             input_languages=list(INPUT_LANGUAGES), created_at=created_at, metrics_path="evaluation.json")
    return ArtifactManifest.model_validate(d)


def publish(candidates: Path = CANDIDATES, sweep_path: Path = SWEEP, created_at: str | None = None) -> Path:
    src_dir, out_dir = candidates / SOURCE_VERSION, candidates / VERSION
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; published artifacts are immutable")
    src = ArtifactManifest.model_validate_json((src_dir / "manifest.json").read_text())
    model = src_dir / src.artifact_path
    if sha256_file(model) != src.sha256:
        raise SystemExit(f"{model} does not match its manifest sha256")
    pattern = graph_token_pattern(model)
    if pattern != TOKEN_PATTERN_RE2:
        raise SystemExit(f"graph tokenizer {pattern!r} != stage1_text.TOKEN_PATTERN_RE2 {TOKEN_PATTERN_RE2!r}")
    sweep = json.loads(sweep_path.read_text())
    gz = sweep["chosen_gray_zone"]
    if gz is None:
        raise SystemExit("the sweep chose no gray zone (no feasible setting)")
    created_at = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(src, tuple(gz), f"regex:{pattern}", created_at)
    chosen = sweep["named_settings"]["chosen"]["slices"]
    evaluation = {
        "detector_version": VERSION,
        "derived_from": {"detector_version": SOURCE_VERSION, "model_sha256": src.sha256,
                         "change": "config only: manifest schema 1.2 (tokenizer, input_languages) and the swept "
                                   "gray zone; model.onnx is byte-identical"},
        "tokenizer": {"value": manifest.tokenizer, "source": "tokenexp attribute of the ONNX Tokenizer node",
                      "python_equivalent": r"[^\W]{2,}",
                      "preprocessing": "ASCII-only lowercasing (StringNormalizer LOWER, locale C) before tokenizing"},
        "input_languages": {"value": INPUT_LANGUAGES,
                            "why": "train/val are >97% English (data/language_mix.json); Stage 1 v2 recalls 120/682 "
                                   "= 17.6% of non-English frozen-test injections (training/results/"
                                   "stage1-tfidf-v2_frozen_test.json), so non-English input must escalate"},
        "gray_zone": {"value": gz, "selected_by": "training/cascade_sweep.py",
                      "results_file": str(sweep_path.relative_to(ROOT)) if ROOT in sweep_path.resolve().parents
                      else str(sweep_path),
                      "results_sha256": sha256_file(sweep_path),
                      "rule": sweep["selection_rule"], "kill_criterion": sweep["kill_criterion"],
                      "val_overall": {k: chosen["overall"][k] for k in ("n", "zones", "escalated", "saved_requests")}
                      | {"cascade_recall": chosen["overall"]["cascade"]["recall"],
                         "cascade_fpr": chosen["overall"]["cascade"]["fpr"],
                         "deberta_alone": chosen["overall"]["deberta_alone"]}},
        "stage1_alone_metrics": f"unchanged from {SOURCE_VERSION}: see artifacts/candidates/{SOURCE_VERSION}/evaluation.json",
    }
    with tempfile.TemporaryDirectory(dir=candidates) as t:
        tmp = Path(t)
        shutil.copyfile(model, tmp / manifest.artifact_path)
        if sha256_file(tmp / manifest.artifact_path) != src.sha256:
            raise SystemExit("copied model.onnx sha256 differs from the source")
        (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        write_json(tmp / "evaluation.json", evaluation)
        if out_dir.exists():
            raise SystemExit(f"{out_dir} appeared while publishing")
        shutil.copytree(tmp, out_dir)
    out_dir.chmod(0o755)
    for p in out_dir.iterdir():
        p.chmod(0o444)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    a = ap.parse_args(argv)
    out = publish(sweep_path=a.sweep)
    m = ArtifactManifest.model_validate_json((out / "manifest.json").read_text())
    print(f"published {out.relative_to(ROOT)} sha256={m.sha256} gray_zone={m.gray_zone} tokenizer={m.tokenizer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
