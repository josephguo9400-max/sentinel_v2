"""do_not_train binding (review n1; untested per review r2 m4).

Two mechanisms keep red-team material (probe sets, the adaptive evasion probes) out of training:
  * ``training.common.read_training_jsonl`` refuses a do_not_train-marked file or any do_not_train row;
  * ``training.check_leakage.check_do_not_train`` (run inside ``check``) fails when a body from a
    do_not_train-marked fixture appears in a split by dedup key.

Synthetic fixtures live under tmp_path; nothing here reads data/splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import check_leakage
from training.common import (ROOT, SPLIT_NAMES, DoNotTrainError, read_training_jsonl, write_jsonl)

TEXTS = [
    "Please review the attached quarterly budget and send comments by Friday afternoon.",
    "The cafeteria will serve a vegetarian lunch menu throughout the whole of next week.",
    "Our team offsite is scheduled for the second week of March in the mountain lodge.",
    "The printer on the third floor is out of toner; facilities will replace it on Monday.",
]
# Stands in for a do_not_train probe body; content does not matter, only the binding does.
PROBE_BODY = "Status note for the archive team: the weekly digest moved to Thursday mornings from now on."


def _rec(i, text, **extra):
    return {"id": f"id{i:04d}", "text": text, "label": 0, "source": "t", "source_label": "", "group": None,
            **extra}


def _write_splits(d: Path, train, val, test):
    for name, rows in zip(SPLIT_NAMES, (train, val, test)):
        write_jsonl(d / f"{name}.jsonl", rows)


# ------------------------------------------------------------------ loader refusal

def test_loader_accepts_clean_rows(tmp_path):
    p = tmp_path / "train.jsonl"
    write_jsonl(p, [_rec(0, TEXTS[0]), _rec(1, TEXTS[1], do_not_train=False)])
    rows = read_training_jsonl(p)
    assert [r["id"] for r in rows] == ["id0000", "id0001"]


def test_loader_refuses_any_do_not_train_row(tmp_path):
    p = tmp_path / "train.jsonl"
    write_jsonl(p, [_rec(0, TEXTS[0]), _rec(1, TEXTS[1], do_not_train=True), _rec(2, TEXTS[2])])
    with pytest.raises(DoNotTrainError, match="1 do_not_train row"):
        read_training_jsonl(p)


def test_loader_refuses_do_not_train_marked_artifact(tmp_path):
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"do_not_train": True, "probes": [{"body": PROBE_BODY}]}, indent=2),
                 encoding="utf-8")
    with pytest.raises(DoNotTrainError, match="marked do_not_train"):
        read_training_jsonl(p)


def test_loader_refuses_the_committed_adaptive_probe_set():
    p = ROOT / "fixtures" / "probes" / "adaptive" / "adaptive_evasions.json"
    assert p.is_file()
    with pytest.raises(DoNotTrainError):
        read_training_jsonl(p)


def test_trainer_loads_splits_through_the_refusing_loader():
    # The binding only helps if the trainer uses it for both training splits.
    src = (ROOT / "training" / "train_stage1.py").read_text(encoding="utf-8")
    assert 'read_training_jsonl(a.splits_dir / "train.jsonl")' in src
    assert 'read_training_jsonl(a.splits_dir / "val.jsonl")' in src


# ------------------------------------------------------------------ leakage binding

@pytest.fixture
def fake_fixtures(tmp_path, monkeypatch):
    fx = tmp_path / "fixtures"
    (fx / "probes").mkdir(parents=True)
    (fx / "probes" / "marked.json").write_text(json.dumps(
        {"do_not_train": True, "probes": [{"body": PROBE_BODY}], "cores": [{"template": "  "}]}),
        encoding="utf-8")
    # Same body in an UNmarked fixture must not count: only the label binds.
    (fx / "probes" / "unmarked.json").write_text(json.dumps(
        {"probes": [{"body": TEXTS[3]}]}), encoding="utf-8")
    (fx / "probes" / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(check_leakage, "FIXTURES_DIR", fx)
    monkeypatch.setattr(check_leakage, "ROOT", tmp_path)
    splits = tmp_path / "splits"
    splits.mkdir()
    return splits


def test_iter_bodies_reads_only_marked_fixtures(fake_fixtures):
    bodies = [b for _, b in check_leakage._iter_do_not_train_bodies()]
    assert bodies == [PROBE_BODY]  # unmarked, blank and unparseable fixtures are skipped


def test_leakage_passes_when_no_marked_body_is_in_a_split(fake_fixtures):
    _write_splits(fake_fixtures, [_rec(0, TEXTS[0]), _rec(3, TEXTS[3])], [_rec(1, TEXTS[1])],
                  [_rec(2, TEXTS[2])])
    assert check_leakage.check(fake_fixtures, verify_hash=False) == []


@pytest.mark.parametrize("variant", [PROBE_BODY, "  " + PROBE_BODY.upper() + "  ",
                                     PROBE_BODY.replace(" ", "   ").rstrip(".")])
def test_leakage_fails_when_a_marked_body_is_in_train(fake_fixtures, variant):
    # Exact body and dedup-key variants (case, whitespace, punctuation) are all caught.
    _write_splits(fake_fixtures, [_rec(0, TEXTS[0]), _rec(9, variant)], [_rec(1, TEXTS[1])],
                  [_rec(2, TEXTS[2])])
    errs = check_leakage.check(fake_fixtures, verify_hash=False)
    assert any(e.startswith("DO_NOT_TRAIN") and "probes/marked.json" in e for e in errs)
    assert check_leakage.main(["--splits-dir", str(fake_fixtures)]) == 1


def test_leakage_fails_when_a_marked_body_is_in_val(fake_fixtures):
    _write_splits(fake_fixtures, [_rec(0, TEXTS[0])], [_rec(9, PROBE_BODY)], [_rec(2, TEXTS[2])])
    assert any(e.startswith("DO_NOT_TRAIN") for e in check_leakage.check(fake_fixtures, verify_hash=False))


def test_real_do_not_train_fixtures_are_discovered():
    # The committed adaptive probe set (9 bodies) is visible to the leakage check. Bodies are not printed.
    # Known limit (review r2 m4): the 21 authored cores live in Python (redteam/adaptive/cores.py), and
    # fixtures/adaptive/cores_curated.json is empty, so the check sees probe bodies, not the core library.
    found = [(p.relative_to(ROOT).as_posix(), b) for p, b in check_leakage._iter_do_not_train_bodies()]
    probe = [b for p, b in found if p == "fixtures/probes/adaptive/adaptive_evasions.json"]
    assert len(probe) == 9
