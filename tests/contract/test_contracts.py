import pytest
from pydantic import ValidationError

from contracts import ArtifactManifest, DetectorResult, TraceEvent

SHA = "a" * 64


def result(**kw):
    base = dict(request_id="r1", detector_id="pi", detector_version="v1", status="ok", score=0.9,
                threshold=0.5, predicted_injection=True, latency_ms=12.0, input_sha256=SHA, input_tokens=40)
    base.update(kw)
    return DetectorResult(**base)


def manifest(**kw):
    base = dict(detector_id="pi", detector_version="stage1-v1", artifact_format="onnx", artifact_path="model.onnx",
                sha256=SHA, runtime={"python": "3.13", "onnxruntime": "1.29.0"}, preprocessing_version="tfidf-v1",
                label_mapping={"0": "benign", "1": "injection"}, threshold=0.5, max_input_tokens=512,
                failure_policy="fail_closed", timeout_ms=200, cascade_role="stage1", gray_zone=(0.15, 0.85),
                created_at="2026-10-29T00:00:00Z")
    base.update(kw)
    return ArtifactManifest(**base)


def test_ok_result_valid():
    assert result().schema_version == "1.1"


def test_prediction_must_match_threshold():
    with pytest.raises(ValidationError):
        result(score=0.2, predicted_injection=True)


def test_timeout_carries_no_score():
    assert result(status="timeout", score=None, predicted_injection=None).score is None
    with pytest.raises(ValidationError):
        result(status="timeout")


def test_score_range_enforced():
    with pytest.raises(ValidationError):
        result(score=1.5)


def test_manifest_valid():
    assert manifest().cascade_role == "stage1"


@pytest.mark.parametrize("bad", [dict(artifact_format="joblib"), dict(artifact_path="../x.onnx"),
                                 dict(artifact_path="/abs/x.onnx"), dict(gray_zone=None),
                                 dict(gray_zone=(0.9, 0.1)), dict(cascade_role="stage2"),
                                 dict(label_mapping={"0": "benign", "1": "benign"})])
def test_manifest_rejects(bad):
    with pytest.raises(ValidationError):
        manifest(**bad)


def test_trace_requires_mode_and_source():
    ev = dict(event_id="e", run_id="r", attempt_id="a", timestamp="2026-10-29T00:00:00Z", event_type="tool_blocked")
    with pytest.raises(ValidationError):
        TraceEvent(**ev)
    assert TraceEvent(**ev, mode="C", source="scripted").mode == "C"
