"""Cascade sweep (val only), cached scores and the stage1-tfidf-v2.1 artifact."""
import json
import math
from pathlib import Path

import pytest

from contracts import ArtifactManifest
from training import cascade_sweep as cs
from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file

V2 = ROOT / "artifacts" / "candidates" / "stage1-tfidf-v2"
V21 = ROOT / "artifacts" / "candidates" / "stage1-tfidf-v2.1"
SCORES = ROOT / "training" / "results" / "cascade_scores_val.jsonl"
SWEEP = ROOT / "training" / "results" / "cascade_sweep_val.json"
DEBERTA = ROOT / "models" / "deberta-pi-v2" / "onnx" / "model.onnx"


def _row(i, label, s1, deb, lang="en", source="s", chunks=None, family=None):
    r = {"id": f"r{i}", "label": label, "s1": s1, "deb": deb, "lang": lang, "source": source,
         "chunks": chunks or [[0, 10, deb, s1]]}
    if family:
        r["family"] = family
    return r


# ------------------------------------------------------------------ routing and metrics

def test_route_only_in_scope_language_may_be_finalized():
    rows = [_row(0, 1, 0.01, 0.99, lang="de"), _row(1, 0, 0.01, 0.01, lang="en"), _row(2, 1, 0.999, 0.1, lang="fr")]
    for r in rows:
        r["in_scope"] = r["lang"] in cs.INPUT_LANGUAGES
    assert cs.route(rows[0], 0.2, 0.9) == "stage2"  # non-English below low still escalates
    assert cs.route(rows[1], 0.2, 0.9) == "low"
    assert cs.route(rows[2], 0.2, 0.9) == "stage2"  # non-English above high still escalates
    assert cs.decide(rows[0], 0.2, 0.9)["flag"] == 1  # DeBERTa decides


def test_allow_only_high_one_never_finalizes_a_flag():
    r = _row(0, 0, 0.9999999, 0.01)
    r["in_scope"] = True
    assert cs.route(r, 0.3, 1.0) == "stage2"
    assert cs.decide(r, 0.3, 1.0)["flag"] == 0


def test_metrics_counts_saved_and_recall_against_deberta():
    rows = [_row(0, 1, 0.1, 0.9), _row(1, 1, 0.8, 0.9), _row(2, 0, 0.1, 0.9), _row(3, 0, 0.6, 0.1)]
    for r in rows:
        r["in_scope"] = True
    o = cs.metrics(rows, 0.5, 1.0)["overall"]
    assert o["zones"] == {"low": 2, "stage2": 2, "high": 0}
    assert o["saved_requests"]["numerator"] == 2
    assert o["cascade"]["recall"]["numerator"] == 1 and o["deberta_alone"]["recall"]["numerator"] == 2
    assert o["cascade"]["fpr"]["numerator"] == 0 and o["deberta_alone"]["fpr"]["numerator"] == 1
    assert o["stage1_finalized_errors"]["of_which_deberta_would_flag"] == 1
    assert o["cascade"]["recall_minus_deberta"]["discordant"] == {"cascade_only": 0, "deberta_only": 1}


def test_chunk_level_variant_sends_suspicious_first_and_last_windows():
    ch = [[0, 1, 0.1, 0.9], [1, 2, 0.1, 0.05], [2, 3, 0.95, 0.05], [3, 4, 0.1, 0.9], [4, 5, 0.1, 0.1]]
    r = _row(0, 1, 0.9, 0.95, chunks=ch)
    r["in_scope"] = True
    assert cs.chunk_escalation(r, 0.5) == [0, 3, 4]
    d = cs.decide(r, 0.5, 1.0)
    assert d["win"] == 5 and d["win_chunk"] == 3
    assert d["flag"] == 1 and d["flag_chunk"] == 0  # the injected window 2 looked benign to Stage 1
    r["in_scope"] = False
    assert cs.chunk_escalation(r, 0.5) == [0, 1, 2, 3, 4]


def test_newcombe_paired_interval_properties_and_coverage():
    """Not checked against a published table (none available offline): symmetry, containment and a
    simulated coverage check for strongly correlated pairs like cascade-vs-DeBERTa decisions."""
    import random

    d, lo, hi = cs.newcombe_paired(12, 9, 2, 21)
    assert lo < d < hi and math.isclose(d, 7 / 44)
    d2, lo2, hi2 = cs.newcombe_paired(12, 2, 9, 21)
    assert math.isclose(d2, -d) and math.isclose(lo2, -hi) and math.isclose(hi2, -lo)
    d, lo, hi = cs.newcombe_paired(50, 0, 0, 0)
    assert d == 0 and lo < 0 < hi
    rng = random.Random(7)
    n, trials, covered = 300, 1000, 0
    p_both, p_b, p_c = 0.90, 0.01, 0.03  # true difference -0.02
    for _ in range(trials):
        cnt = [0, 0, 0, 0]
        for _ in range(n):
            u = rng.random()
            cnt[0 if u < p_both else 1 if u < p_both + p_b else 2 if u < p_both + p_b + p_c else 3] += 1
        _, lo, hi = cs.newcombe_paired(*cnt)
        covered += lo <= p_b - p_c <= hi
    assert covered / trials >= 0.93


def test_sweep_selects_allow_only_and_respects_constraints():
    # 500 benign items with low Stage 1 scores; 1,000 positives, 3 of which Stage 1 scores low. (With only
    # 100 positives constraint (b) is infeasible: the Wilson lower bound of 100/100 is 0.963 < 1.0 - 0.02.)
    rows = [_row(i, 0, 0.05 + 0.0008 * i, 0.01) for i in range(500)]
    rows += [_row(500 + i, 1, 0.95 if i >= 3 else 0.2 + 0.01 * i, 0.99) for i in range(1000)]
    for r in rows:
        r["in_scope"] = True
    s = cs.sweep(rows)
    ch = s["chosen"]
    assert ch["high"] == 1.0 and ch["constraints"]["all"]
    assert s["strictly_equal_recall_best"]["low"] <= 0.2  # below the lowest-scored positive
    assert ch["low"] >= s["strictly_equal_recall_best"]["low"]
    assert cs.sweep(rows[:500] + rows[500:600])["chosen"] is None  # too few positives -> nothing feasible
    assert all(g["high"] < 1.0 for g in [s["flag_finalizing_variant_best"]] if g)


# ------------------------------------------------------------------ committed val scores and sweep

@pytest.mark.skipif(not SCORES.exists(), reason="cascade scores not computed")
def test_committed_scores_cover_val_exactly_and_are_small():
    ids = {r["id"] for r in read_jsonl(SCORES)}
    val = {r["id"]: r for r in read_jsonl(SPLITS_DIR / "val.jsonl")}
    assert ids == set(val)
    assert SCORES.stat().st_size < 5 * 2**20 and SWEEP.stat().st_size < 5 * 2**20
    for r in read_jsonl(SCORES):
        assert r["label"] == val[r["id"]]["label"] and r["lang"] == val[r["id"]]["lang"]
        assert r["deb"] == max(c[2] for c in r["chunks"]) and len(r["chunks"]) >= 1
        assert 0.0 <= r["s1"] <= 1.0
        break
    assert not any("frozen" in str(p) for p in (ROOT / "training" / "results").glob("cascade_*"))


@pytest.mark.skipif(not SWEEP.exists(), reason="sweep not run")
def test_committed_sweep_is_reproducible_from_committed_scores():
    committed = json.loads(SWEEP.read_text())
    assert committed["scores_file_sha256"] == sha256_file(SCORES)
    fresh = cs.build(SCORES)
    assert fresh["chosen_gray_zone"] == committed["chosen_gray_zone"]
    assert fresh["kill_criterion"] == committed["kill_criterion"]
    assert json.loads(json.dumps(fresh["named_settings"])) == committed["named_settings"]


@pytest.mark.skipif(not DEBERTA.exists(), reason="DeBERTa model not fetched")
@pytest.mark.skipif(not SCORES.exists(), reason="cascade scores not computed")
def test_cached_scores_match_a_fresh_computation_on_a_sample():
    from training.cascade_scores import DebertaScorer, score_item
    from training.stage1_onnx import Stage1Onnx

    cached = {r["id"]: r for r in read_jsonl(SCORES)}
    val = list(read_jsonl(SPLITS_DIR / "val.jsonl"))
    longest = max(val, key=lambda r: len(r["text"]))
    sample = [longest] + val[:3]
    deb, s1 = DebertaScorer(threads=1), Stage1Onnx(V2)
    for row in sample:
        got, want = score_item(row, deb, s1), cached[row["id"]]
        assert got["s1"] == pytest.approx(want["s1"], abs=1e-5)
        assert len(got["chunks"]) == len(want["chunks"])
        for g, w in zip(got["chunks"], want["chunks"]):
            assert g[:2] == w[:2] and g[2] == pytest.approx(w[2], abs=1e-4)
    assert len(cached[longest["id"]]["chunks"]) > 1


def test_window_starts_match_the_gateway_rule():
    from serving.encoding import window_starts as gw
    from training.cascade_scores import window_starts

    for n in (0, 1, 509, 510, 511, 894, 895, 2000, 5003):
        assert window_starts(n) == gw(n, 510, 384)


# ------------------------------------------------------------------ stage1-tfidf-v2.1

@pytest.mark.skipif(not (V21 / "manifest.json").exists(), reason="v2.1 not published")
def test_v21_manifest_schema_12_same_model_bytes_and_swept_gray_zone():
    m = ArtifactManifest.model_validate_json((V21 / "manifest.json").read_text())
    v2 = ArtifactManifest.model_validate_json((V2 / "manifest.json").read_text())
    assert m.schema_version == "1.2" and m.detector_version == "stage1-tfidf-v2.1"
    assert m.sha256 == v2.sha256 == sha256_file(V21 / "model.onnx") == sha256_file(V2 / "model.onnx")
    assert (V21 / "model.onnx").read_bytes() == (V2 / "model.onnx").read_bytes()
    assert m.input_languages == ["en"]
    sweep = json.loads(SWEEP.read_text())
    assert list(m.gray_zone) == sweep["chosen_gray_zone"]
    assert m.threshold == v2.threshold and m.cascade_role == "stage1"
    for k in ("max_input_tokens", "preprocessing_version", "label_mapping", "failure_policy", "timeout_ms", "runtime"):
        assert getattr(m, k) == getattr(v2, k)


@pytest.mark.skipif(not (V21 / "manifest.json").exists(), reason="v2.1 not published")
def test_v21_tokenizer_is_the_exact_regex_in_the_onnx_graph():
    from training.publish_stage1 import graph_token_pattern
    from training.stage1_text import TOKEN_PATTERN_RE2

    m = ArtifactManifest.model_validate_json((V21 / "manifest.json").read_text())
    assert m.tokenizer == f"regex:{graph_token_pattern(V21 / 'model.onnx')}" == f"regex:{TOKEN_PATTERN_RE2}"


def test_v2_is_untouched():
    m = json.loads((V2 / "manifest.json").read_text())
    assert m["schema_version"] == "1.1" and m["gray_zone"] == [0.282715, 0.764614]
    assert sha256_file(V2 / "manifest.json") == "eab9169df67b3142347ab4656b2412f80dd2fe3ef1ea28279fc7241467275d10"


def test_publish_refuses_to_overwrite(tmp_path):
    import shutil

    from training.publish_stage1 import publish

    cand = tmp_path / "candidates"
    shutil.copytree(V2, cand / "stage1-tfidf-v2")
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps({"chosen_gray_zone": [0.3, 1.0], "selection_rule": {}, "kill_criterion": {},
                                 "named_settings": {"chosen": {"slices": {"overall": {
                                     "n": 1, "zones": {}, "escalated": {}, "saved_requests": {},
                                     "cascade": {"recall": {}, "fpr": {}}, "deberta_alone": {}}}}}}))
    out = publish(cand, sweep, created_at="2026-10-30T00:00:00Z")
    m = ArtifactManifest.model_validate_json((out / "manifest.json").read_text())
    assert m.gray_zone == (0.3, 1.0) and m.tokenizer.startswith("regex:")
    with pytest.raises(SystemExit, match="immutable"):
        publish(cand, sweep)


def test_publish_rejects_tampered_source_model(tmp_path):
    import shutil

    from training.publish_stage1 import publish

    cand = tmp_path / "candidates"
    shutil.copytree(V2, cand / "stage1-tfidf-v2")
    p = cand / "stage1-tfidf-v2" / "model.onnx"
    p.chmod(0o644)
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="sha256"):
        publish(cand, Path("unused.json"))
