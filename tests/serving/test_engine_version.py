"""T8, engine_version (D25.2): a decision-path change without a version bump fails here.

`serving/engine_version.json` records the sha256 of every file that can change a decision, tied to
ENGINE_VERSION. Changing one of them without bumping ENGINE_VERSION (and re-recording) fails this test, and
because the engine version is part of bundle_sha256 (serving/policy.py), the bump invalidates every gate verdict.
"""
import json

import pytest

from serving import engine_version as ev


def test_decision_path_files_match_the_record():
    problems = ev.check()
    assert not problems, "\n".join(problems) + (
        "\n\nA decision-path file changed. Bump ENGINE_VERSION in serving/engine_version.py and run "
        "`python -m serving.engine_version --record`; every bundle_sha256 changes with it, so gate verdicts "
        "do not carry over.")


def test_the_record_covers_the_decision_path():
    recorded = set(ev.load_record()["files"])
    must = {"serving/service.py", "serving/structure.py", "serving/detector.py", "serving/encoding.py",
            "serving/language.py", "serving/batcher.py", "serving/policy.py", "serving/registry.py",
            "contracts/policy.py", "contracts/capability.py", "contracts/detector.py", "contracts/classify.py"}
    assert must <= recorded, f"not covered by the engine version: {sorted(must - recorded)}"
    for path in recorded:
        assert (ev.ROOT / path).is_file(), f"{path} is recorded but does not exist"


def test_record_refuses_to_hide_a_change(tmp_path, monkeypatch):
    """The guard itself: re-recording a changed file under an unchanged version is refused."""
    rec = ev.load_record()
    changed = dict(rec, files={**rec["files"], "serving/service.py": "0" * 64})
    path = tmp_path / "engine_version.json"
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(ev, "RECORD", path)
    assert any("serving/service.py" in p for p in ev.check())
    with pytest.raises(SystemExit) as exc:
        ev.record()
    assert "bump it first" in str(exc.value)
    monkeypatch.setattr(ev, "ENGINE_VERSION", "1.1.0")
    out = ev.record()  # with a bump it records
    assert out["engine_version"] == "1.1.0" and out["files"]["serving/service.py"] != "0" * 64


def test_version_is_semver_and_matches_the_record():
    assert ev.ENGINE_VERSION.count(".") == 2 and all(p.isdigit() for p in ev.ENGINE_VERSION.split("."))
    assert ev.load_record()["engine_version"] == ev.ENGINE_VERSION


def test_committed_policies_were_validated_on_a_compatible_engine():
    import json as _json
    from pathlib import Path

    from contracts import ExecutionPolicy
    from serving.policy import check_engine_compat

    for p in sorted((ev.ROOT / "artifacts").glob("policies/*/*/policy.json")):
        check_engine_compat(ExecutionPolicy.model_validate(_json.loads(Path(p).read_text())))
