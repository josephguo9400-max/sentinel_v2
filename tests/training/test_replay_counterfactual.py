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
