"""replay_counterfactual_served with fakes: the real ClassifierService routes (route mode), the stub Stage 2 records
which chunks Stage 2 would score without running a model, and phase B decides escalated requests from scores of
exactly those chunks (positive -> flag, else the outcome the service reported, e.g. the coverage policy)."""
import asyncio
import re

from serving.config import GatewayConfig
from serving.registry import DetectorSpec
from serving.service import ClassifierService
from training import replay_counterfactual_served as rcs

CLS, SEP, INJ = 1, 2, 7


class Tok:
    """Stage 2 stand-in: whitespace tokens, the word INJECT is id 7. infer() must never be called in phase A."""
    detector_id, detector_version, threshold, max_len = "fake-s2", "0", 0.5, 8
    cls_id, sep_id = CLS, SEP

    def tokenize(self, text):
        return self.tokenize_with_offsets(text)[0]

    def tokenize_with_offsets(self, text):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        return [INJ if text[a:b] == "INJECT" else 10 + len(text[a:b]) for a, b in spans], spans

    def infer(self, batch):
        raise AssertionError("phase A must not run Stage 2")


def deberta(batch):  # phase B scorer
    return [0.99 if INJ in row else 0.01 for row in batch]


class S1:
    """Stage 1 stand-in: 'SUS' -> 0.9, else 0.01."""
    detector_id, detector_version, threshold, max_input_tokens = "fake-s1", "0", 0.5, 512

    def windows(self, text):
        return [text], len(text.split())

    def infer(self, batch):
        return [0.9 if "SUS" in t else 0.01 for t in batch]


class Lang:
    def describe(self):
        return {"method": "fake"}

    def detect(self, text):
        return ("fr", 0.99) if "bonjour" in text else ("unknown", 0.0) if "zz" in text else ("en", 0.99)


def service(max_chunks=4):
    s2 = DetectorSpec(detector=Tok(), detector_id="fake-s2", detector_version="0", threshold=0.5, kind="token",
                      role="stage2", failure_policy="fail_closed", timeout_ms=60_000, max_input_tokens=8)
    s1 = DetectorSpec(detector=S1(), detector_id="fake-s1", detector_version="0", threshold=0.5, kind="text",
                      role="stage1", failure_policy="fail_open", timeout_ms=60_000, gray_zone=(0.5, 1.0),
                      input_languages=["en"])
    cfg = GatewayConfig(shadow_log_path="", max_len=8, stride=6, max_chunks=max_chunks, stage1_subwindow_tokens=0,
                        stage1_timeout_ms=60_000, stage2_timeout_ms=60_000)
    return ClassifierService(s2, cfg, stage1=s1, language=Lang(), stage1_mode="route")


ITEMS = [
    {"id": "a", "text": "hello there friend", "label": 0, "source": "x"},           # Stage 1 clears -> allow
    {"id": "b", "text": "SUS INJECT now", "label": 1, "source": "x"},               # escalates, DeBERTa flags
    {"id": "c", "text": "bonjour mes amis", "label": 0, "source": "x"},             # language gate escalates
    {"id": "d", "text": "quiet INJECT words", "label": 1, "source": "gandalf"},     # Stage 1 clears: lost
    {"id": "e", "text": "zz short", "label": 0, "source": "x"},                     # unknown language escalates
    # three paragraphs of six tokens, all suspicious, max_chunks 2 -> the middle chunk is unscored -> coverage flag
    {"id": "f", "text": "SUS a b c d e\n\nSUS f g h i j\n\nSUS k l m n o", "label": 0, "source": "x"},
]


def cache_row(it, deb, s1):
    return {"id": it["id"], "deb": deb, "s1": s1, "lang": "en", "in_scope": True, "chunks": [[0, 3, deb, s1]]}


def run(max_chunks=2):
    sim = asyncio.run(rcs.simulate(ITEMS, service(max_chunks)))
    rcs.score_rows(sim, deberta, 0.5)
    return {r["id"]: r for r in sim}, sim


def test_stub_records_stage2_chunks_and_phase_b_decides_from_them():
    by, _ = run()
    assert not by["a"]["escalated"] and by["a"]["stage2_chunks"] == [] and by["a"]["chunk_level_flag"] == 0
    assert by["b"]["escalated"] and [c["index"] for c in by["b"]["stage2_chunks"]] == [0]
    assert by["b"]["stage2_chunks"][0]["deberta"] == 0.99 and by["b"]["chunk_level_cause"] == "stage2_flag"
    assert by["c"]["escalated"] and by["c"]["s1_langs"] == ["fr"] and by["c"]["chunk_level_flag"] == 0
    assert not by["d"]["escalated"] and by["d"]["chunk_level_cause"] == "stage1_allow"
    assert by["e"]["escalated"] and by["e"]["s1_langs"] == ["unknown"]
    f = by["f"]
    assert f["n_chunks"] == 3 and [c["index"] for c in f["stage2_chunks"]] == [0, 2]  # first + last, budget 2
    assert f["stub_coverage_policy_applied"] == "flag" and f["chunk_level_flag"] == 1
    assert f["chunk_level_cause"] == "coverage_policy:flag"


def test_report_counts_lost_injections_served_agreement_and_sweep():
    by, sim = run()
    cache = [cache_row(ITEMS[0], 0.1, 0.01), cache_row(ITEMS[1], 0.99, 0.9), cache_row(ITEMS[2], 0.1, 0.01),
             cache_row(ITEMS[3], 0.99, 0.01), cache_row(ITEMS[4], 0.1, 0.01), cache_row(ITEMS[5], 0.1, 0.9)]
    truth = {it["id"]: it for it in ITEMS}
    served = {i: {"id": i, "decision": "flag" if by[i]["chunk_level_flag"] else "allow",
                  "stages_run": ["stage1", "stage2"] if by[i]["escalated"] else ["stage1"],
                  "statuses": ["ok"], "policy_applied": None, "coverage_policy_applied": None} for i in by}
    served["c"] = {**served["c"], "decision": "flag", "policy_applied": "fail_closed"}  # a served Stage 2 timeout
    rep = rcs.report(sim, cache, truth, served, 0.5, ["en"], {})
    assert rep["stage2_share"]["numerator"] == 4 and abs(rep["saved"] - 2 / 6) < 1e-12
    assert rep["lost_injections"]["n"] == 1 and rep["lost_injections"]["by_source"] == {"gandalf": 1}
    v = rep["views"]
    assert v["deberta_only_offline"]["recall"]["numerator"] == 2
    assert v["cascade_served_faithful_chunk_level"]["recall"]["numerator"] == 1
    assert v["cascade_served_faithful_chunk_level"]["fpr"]["numerator"] == 1  # f: coverage policy
    assert v["cascade_served_faithful_request_level"]["fpr"]["numerator"] == 0  # ignores the coverage policy
    # offline routes c and e low (stored tag en, s1 0.01); the gateway escalates them on its own language
    assert rep["offline_low_vs_gateway"] == {"gateway_language_other": 1, "gateway_language_unknown": 1}
    ag = rep["served_agreement"]
    assert ag["decision_agrees"] == 5 and ag["disagreements_by_served_cause"] == {
        "served_failure_policy:fail_closed": 1}
    sweep = {s["low"]: s for s in rep["low_sweep_descriptive"]}
    assert sweep[0.5]["n_cleared"] == 2 and sweep[0.01]["n_cleared"] == 0
    assert rep["kill_criterion"]["best_equal_recall_setting"]["low"] < 0.5
    assert all("_rows" not in it for it in rep["items"])
