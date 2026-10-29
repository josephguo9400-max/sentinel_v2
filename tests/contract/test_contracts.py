import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from contracts import ArtifactManifest, DetectorResult, TraceEvent

SHA = "a" * 64
TS = "2026-10-29T00:00:00Z"


def result(**kw):
    base = dict(request_id="r1", detector_id="pi", detector_version="v1", status="ok", score=0.9,
                threshold=0.5, predicted_injection=True, latency_ms=12.0, input_sha256=SHA, input_tokens=40)
    base.update(kw)
    return DetectorResult(**base)


def stage1(**kw):
    base = dict(detector_id="pi", detector_version="stage1-v1", artifact_format="onnx", artifact_path="model.onnx",
                sha256=SHA, runtime={"python": "3.13", "onnxruntime": "1.29.0", "sklearn": "1.5.2",
                                     "skl2onnx": "1.17.0"},
                preprocessing_version="tfidf-v1", label_mapping={"0": "benign", "1": "injection"}, threshold=0.5,
                max_input_tokens=512, failure_policy="fail_closed", timeout_ms=200, cascade_role="stage1",
                gray_zone=(0.15, 0.85), created_at=TS)
    base.update(kw)
    return ArtifactManifest(**base)


def stage2(**kw):
    base = dict(detector_id="deberta-pi", detector_version="v2-onnx", artifact_format="onnx",
                artifact_path="onnx/model.onnx", sha256=SHA,
                runtime={"python": "3.13", "onnxruntime": "1.29.0", "tokenizers": "0.23.3"},
                preprocessing_version="deberta-v3-tokenizer", label_mapping={"0": "benign", "1": "injection"},
                threshold=0.5, max_input_tokens=512, failure_policy="fail_closed", timeout_ms=1000,
                cascade_role="stage2", created_at=TS)
    base.update(kw)
    return ArtifactManifest(**base)


def event(**kw):
    base = dict(event_id="e", run_id="r", attempt_id="a", timestamp=TS, event_type="tool_blocked", mode="C",
                source="scripted", tool_name="send_email", policy_reason="no user authorization")
    base.update(kw)
    return TraceEvent(**base)


# ---------- DetectorResult ----------

def test_ok_result_valid():
    assert result().schema_version == "1.1"


def test_prediction_must_match_threshold():
    with pytest.raises(ValidationError):
        result(score=0.2, predicted_injection=True)


def test_score_equal_to_threshold_is_injection():
    assert result(score=0.5, threshold=0.5, predicted_injection=True)


def test_float32_producer_near_threshold_is_accepted():
    s, t = np.float32(0.7), np.float32(0.7)
    pred = bool(s >= t)  # producer compares in float32
    assert result(score=float(s), threshold=0.7, predicted_injection=pred).predicted_injection is True


@pytest.mark.parametrize("status", ["timeout", "error", "skipped"])
def test_non_ok_carries_no_score(status):
    assert result(status=status, score=None, predicted_injection=None).score is None
    with pytest.raises(ValidationError):
        result(status=status)


@pytest.mark.parametrize("bad", [dict(score=1.5), dict(latency_ms=float("inf")), dict(predicted_injection="yes"),
                                 dict(n_chunks=3, aggregation="none"), dict(truncate=True),
                                 dict(input_sha256="xyz")])
def test_result_rejects(bad):
    with pytest.raises(ValidationError):
        result(**bad)


# ---------- ArtifactManifest ----------

def test_stage1_stage2_single_manifests_valid():
    assert stage1().cascade_role == "stage1"
    assert stage2().gray_zone is None
    assert stage2(cascade_role="single").cascade_role == "single"


@pytest.mark.parametrize("path", ["model.onnx", "onnx/model.onnx", "model..v2.onnx"])
def test_artifact_path_accepts(path):
    assert stage1(artifact_path=path).artifact_path == path


@pytest.mark.parametrize("path", ["../x.onnx", "a/../../x.onnx", "/abs/x.onnx", "C:\\x\\m.onnx", "C:/x/m.onnx",
                                  "\\\\srv\\share\\m.onnx", "~/m.onnx", "", "./", "./m.onnx", "a//m.onnx",
                                  "model.joblib", "sub\\..\\..\\m.onnx"])
def test_artifact_path_rejects(path):
    with pytest.raises(ValidationError):
        stage1(artifact_path=path)


def test_metrics_path_checked():
    assert stage1(metrics_path="evaluation.json").metrics_path == "evaluation.json"
    with pytest.raises(ValidationError):
        stage1(metrics_path="../../etc/passwd")


@pytest.mark.parametrize("bad", [dict(artifact_format="joblib"), dict(gray_zone=None), dict(gray_zone=(0.9, 0.1)),
                                 dict(gray_zone=(0.5, 0.5)), dict(gray_zone=(-0.1, 0.5)), dict(gray_zone=(0.2, 1.1)),
                                 dict(threshold=0.9), dict(cascade_role="stage2"),
                                 dict(label_mapping={"0": "benign", "1": "benign"}),
                                 dict(label_mapping={"0": "benign", "1": "injection", "2": "injection"}),
                                 dict(label_mapping={"SAFE": "benign", "INJECTION": "injection"}),
                                 dict(created_at="yesterday"), dict(created_at="2026-10-29T00:00:00"),
                                 dict(runtime={"python": "latest", "onnxruntime": "1.29.0"}),
                                 dict(extra_field=1)])
def test_manifest_rejects(bad):
    with pytest.raises(ValidationError):
        stage1(**bad)


def test_gray_zone_not_allowed_on_single():
    with pytest.raises(ValidationError):
        stage2(cascade_role="single", gray_zone=(0.1, 0.9))


# ---------- TraceEvent ----------

@pytest.mark.parametrize("etype", ["tool_proposed", "tool_blocked", "tool_executed", "tool_failed", "model_response",
                                   "detector_prediction", "detector_timeout", "shadow_failed", "content_quarantined",
                                   "attempt_completed"])
def test_every_event_type_constructs(etype):
    assert event(event_type=etype, detector_version="v1").event_type == etype


@pytest.mark.parametrize("bad", [dict(mode="D"), dict(source="fake"), dict(timestamp="nope"),
                                 dict(tool_name=None), dict(policy_reason=None), dict(unknown=1)])
def test_event_rejects(bad):
    with pytest.raises(ValidationError):
        event(**bad)


def test_detector_events_require_version():
    with pytest.raises(ValidationError):
        event(event_type="detector_prediction", detector_version=None)


# ---------- freeze guard ----------

def test_schema_snapshot_unchanged():
    """Contracts are frozen: any schema change must update tests/contract/schema_snapshot.json deliberately."""
    current = {m.__name__: m.model_json_schema() for m in (DetectorResult, ArtifactManifest, TraceEvent)}
    snap = Path(__file__).with_name("schema_snapshot.json")
    assert json.loads(snap.read_text()) == json.loads(json.dumps(current))


# ---------- re-review additions ----------

def test_numpy_bool_accepted_but_strings_and_ints_rejected():
    s = np.float32(0.9)
    assert result(score=float(s), predicted_injection=s >= np.float32(0.5)).predicted_injection is True
    assert result(truncated=np.bool_(True), n_chunks=1).truncated is True  # truncated single-window input is legal
    for bad in ("true", 1, 0):
        with pytest.raises(ValidationError):
            result(predicted_injection=bad)


@pytest.mark.parametrize("ts,normalized", [("2026-10-29T00:00:00Z", "2026-10-29T00:00:00Z"),
                                           ("2026-10-29T00:00:00+00:00", "2026-10-29T00:00:00Z"),
                                           ("2026-10-29T00:00:00.5Z", "2026-10-29T00:00:00.500000Z")])
def test_timestamps_normalized(ts, normalized):
    assert stage1(created_at=ts).created_at == normalized
    assert event(timestamp=ts).timestamp == normalized


@pytest.mark.parametrize("ts", ["2026-10-29 00:00:00Z", "20261029T000000Z", "2026-W44-4T00:00:00Z",
                                "2026-10-29T00:00Z", "2026-10-29T00:00:00-00:00", "2026-10-29T00:00:00+0000",
                                "2026-10-29T00:00:00+01:00", "2026-10-29T00:00:00", "2026-10-29",
                                "2026-10-29T00:00:00z", "2026-10-29T24:00:00Z", "2026-13-01T00:00:00Z"])
def test_timestamps_rejected(ts):
    with pytest.raises(ValidationError):
        stage1(created_at=ts)


@pytest.mark.parametrize("v", ["1.29.0", "3.11.0rc1", "1.17.0.post1", "2.1.0+cpu", "1.0.dev3", "3"])
def test_versions_accepted(v):
    assert stage1(runtime={"python": "3.13", "onnxruntime": v}).runtime.onnxruntime == v


@pytest.mark.parametrize("path", ["m\x00.onnx", ".onnx", "...onnx", "a/.onnx", "m​.onnx", "c:m.onnx",
                                  "m:alt.onnx"])
def test_artifact_path_rejects_edge_cases(path):
    with pytest.raises(ValidationError):
        stage1(artifact_path=path)


@pytest.mark.parametrize("etype", ["model_response", "attempt_completed", "shadow_failed"])
def test_non_tool_events_need_no_tool_fields(etype):
    assert event(event_type=etype, tool_name=None, policy_reason=None).event_type == etype


def test_content_quarantined_requires_policy_reason():
    with pytest.raises(ValidationError):
        event(event_type="content_quarantined", tool_name=None, policy_reason=None)
