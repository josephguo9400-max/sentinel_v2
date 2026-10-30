"""T3 harness (training/replay_policy_equivalence.py): --no-cache really means no cache (review r2-m1).

Fakes only: the real registry v1 D23 pointers and committed manifests, with fake detectors in place of the ONNX
models (no weights needed). The test runs the real t3() / build_engines() / run_pair() path on the first few replay
items and counts the rows each engine sent to the model itself.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "serving"))

from fakes import FakeDetector, FakeStage1  # noqa: E402

from training import replay_policy_equivalence as eq  # noqa: E402

N_ITEMS = 3


class CountingDetector(FakeDetector):
    """FakeDetector that counts every row the model was asked to score."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rows_scored = 0

    def infer(self, batch):
        self.rows_scored += len(batch)
        return super().infer(batch)


def fake_specs(pointers: dict, intra_op_threads: int = 0) -> dict:
    """DetectorSpecs as serving.registry.load_detector builds them from the committed manifests, with fakes as the
    models. One stage2 detector object is shared by both engines, as with the real load_specs."""
    from contracts import ArtifactManifest
    from serving.registry import DetectorSpec, TFIDF_TOKENIZER

    out = {}
    for name in ("stage1", "stage2"):
        rel = pointers[name]
        m = ArtifactManifest.model_validate_json((eq.ARTIFACTS / rel / "manifest.json").read_bytes())
        kind = "text" if name == "stage1" else "token"
        det = FakeStage1(threshold=m.threshold) if kind == "text" else CountingDetector(threshold=m.threshold)
        det.detector_id, det.detector_version = m.detector_id, m.detector_version
        out[name] = DetectorSpec(
            detector=det, detector_id=m.detector_id, detector_version=m.detector_version, threshold=m.threshold,
            kind=kind, role=m.cascade_role, failure_policy=m.failure_policy, timeout_ms=m.timeout_ms,
            gray_zone=tuple(m.gray_zone) if m.gray_zone else None,
            input_languages=list(m.input_languages) if m.input_languages else None,
            max_input_tokens=m.max_input_tokens, tokenizer=m.tokenizer or (TFIDF_TOKENIZER if kind == "text" else None),
            manifest_schema=m.schema_version, artifact_dir=rel, sha256=m.sha256, manifest=m)
    return out


@pytest.fixture
def run_t3(tmp_path, monkeypatch):
    holder = {}

    def specs(pointers, intra_op_threads=0):
        holder["specs"] = fake_specs(pointers, intra_op_threads)
        return holder["specs"]

    monkeypatch.setattr(eq, "load_specs", specs)
    real_build = eq.build_engines

    def build(pointers, config, cache, sp):
        holder["cache"] = cache
        out = real_build(pointers, config, cache, sp)
        holder["recs"] = out[2]
        return out

    monkeypatch.setattr(eq, "build_engines", build)

    def run(*flags):
        out = tmp_path / "t3.json"
        rc = eq.main(["t3", "--out", str(out), "--limit", str(N_ITEMS), *flags])
        return rc, json.loads(out.read_text()), holder

    return run


def test_no_cache_gives_each_engine_its_own_model_calls(run_t3):
    rc, rep, h = run_t3("--no-cache")
    assert rc == 0 and rep["complete"] and rep["identical_on_every_item"]
    assert h["cache"] is None  # no RowCache at all, not even an in-memory one
    rows = rep["deberta_rows"]["total_scored_per_engine"]
    assert rows > 0
    legacy, policy = h["recs"]["legacy"], h["recs"]["policy"]
    assert legacy._cache is None and policy._cache is None
    # Every row each engine scored went to the model from that engine: twice the model calls of one engine.
    assert legacy.model_calls == rows and policy.model_calls == rows
    assert h["specs"]["stage2"].detector.rows_scored == 2 * rows
    assert rep["deberta_rows"]["cache"] == "none"
    assert rep["deberta_rows"]["model_calls_this_run"] == {"legacy": rows, "policy": rows}
    assert rep["deberta_rows"]["cache_hits"] is None
    assert rep["provenance"]["this_run"]["no_cache"] is True


def test_default_cache_is_shared_and_labelled(run_t3, tmp_path):
    rc, rep, h = run_t3()
    assert rc == 0 and rep["identical_on_every_item"]
    rows = rep["deberta_rows"]["total_scored_per_engine"]
    det = h["specs"]["stage2"].detector
    # Shared cache: a row is scored once for both engines, so scores are identical by construction.
    assert det.rows_scored < 2 * rows
    assert h["recs"]["legacy"]._cache is h["recs"]["policy"]._cache is h["cache"]
    assert rep["deberta_rows"]["cache"] == "shared"
    assert rep["deberta_rows"]["cache_path"].endswith("t3.rowcache.jsonl")
