"""replay_counterfactual: the cascade-vs-DeBERTa comparison counts a Stage 1 allow of a DeBERTa-caught injection as
lost recall, and the equal-recall kill check uses only settings that lose none."""
import json

from training import replay_counterfactual as rc


def _row(i, label, s1, deb, lang="en"):
    return {"id": f"r{i}", "source": "x", "lang": lang, "label": label, "s1": s1, "s1_chunks": 1, "deb_tokens": 10,
            "chunks": [[0, 10, deb, s1]], "deb": deb}


def test_counterfactual_counts_lost_injections(tmp_path, monkeypatch):
    rows = [_row(0, 1, 0.1, 0.9),   # injection Stage 1 allows, DeBERTa catches -> lost
            _row(1, 1, 0.9, 0.9),   # escalated, caught
            _row(2, 0, 0.001, 0.8),  # benign Stage 1 allows; DeBERTa false positive avoided
            _row(3, 0, 0.3, 0.1),
            _row(4, None, 0.2, 0.1)]  # unlabeled: counted in savings, not in rates
    cache = tmp_path / "cache.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps({"id": r["id"], "label": r["label"]}) + "\n" for r in rows))
    monkeypatch.setattr(rc, "REPLAY", replay)
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    rep = rc.build(cache, None)
    assert rep["views"]["deberta_only"]["recall"]["numerator"] == 2
    assert rep["views"]["cascade_offline"]["recall"]["numerator"] == 1
    assert rep["views"]["cascade_offline"]["fpr"]["numerator"] == 0
    assert rep["stage1_misses"]["of_which_deberta_flags"] == 1
    assert rep["kill_criterion"]["served_setting_recall_equal"] is False
    best = rep["kill_criterion"]["best_equal_recall_setting"]
    assert best["deberta_caught_injections_lost"] == 0 and best["low"] <= 0.1


def test_served_view_compares_on_the_same_items_and_splits_flips_by_cause(tmp_path, monkeypatch):
    rows = [_row(0, 1, 0.9, 0.9),   # both flag
            _row(1, 1, 0.9, 0.2),   # served flags by coverage policy, offline DeBERTa allows
            _row(2, 0, 0.9, 0.1),   # served flags by fail_closed
            _row(3, 0, 0.9, 0.1),   # no response (client error): excluded from the same-item denominator
            _row(4, None, 0.9, 0.9)]  # unlabeled
    cache = tmp_path / "cache.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    replay = tmp_path / "replay.jsonl"
    replay.write_text("".join(json.dumps({"id": r["id"], "label": r["label"]}) + "\n" for r in rows))
    base = {"error": None, "stages_run": ["stage2"], "policy_applied": None, "coverage_policy_applied": None}
    served = [{**base, "id": "r0", "decision": "flag"},
              {**base, "id": "r1", "decision": "flag", "coverage_policy_applied": "flag"},
              {**base, "id": "r2", "decision": "flag", "policy_applied": "fail_closed"},
              {**base, "id": "r3", "decision": None, "error": "client_saturated", "stages_run": None},
              {**base, "id": "r4", "decision": "allow"}]
    sp = tmp_path / "requests.jsonl"
    sp.write_text("".join(json.dumps(r) + "\n" for r in served))
    monkeypatch.setattr(rc, "REPLAY", replay)
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    rep = rc.build(cache, sp)
    assert rep["view_kind"].startswith("offline approximation")
    same = rep["views"]["served"]["same_items"]
    assert same["n"] == 3 and same["n_positive"] == 2
    assert same["served"]["recall"]["numerator"] == 2 and same["deberta_only_offline"]["recall"]["numerator"] == 1
    assert same["deberta_only_offline"]["fpr"]["denominator"] == 1  # r3 had no response
    assert same["flips_vs_deberta_only_offline"] == {"coverage_policy:to_flag:label=1": 1,
                                                     "failure_policy:to_flag:label=0": 1}
    assert rep["views"]["served"]["disagrees_with_offline_deberta_only"] == 3  # unlabeled r4 included
