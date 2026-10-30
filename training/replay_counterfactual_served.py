"""Serving-faithful cascade counterfactual on the replay set (review M2).

    flock /home/claude/measure.lock python -m training.replay_counterfactual_served \
        --served reports/cascade-and-shadow/replay_after/requests.jsonl \
        --out reports/cascade-and-shadow/replay_counterfactual_served_faithful.json

training/replay_counterfactual.py is an offline approximation: it scores Stage 1 with its own windows over the whole
text and gates on the stored request-level language tag. The gateway does neither. This script runs every replay
item, one at a time, through the real serving ClassifierService (serving/service.py) in route mode with the default
GatewayConfig (the configuration of the served route-mode run, reports/cascade-and-shadow/run_checkpoint.sh), the
real Stage 1 (artifacts/candidates/stage1-tfidf-v2.1: per chunk, per segment piece, 64/32 sub-windows) and the real
per-chunk language detector (serving/language.py, min_letters / min_prob -> "unknown" -> escalate). Timeouts are
disabled (Stage 1 and Stage 2 budgets of 600 s), so nothing is decided by a deadline.

Phase A: Stage 2 is a stub. The service's real chunk plan (serving/structure.py build_chunks, DeBERTa tokenizer) and
real Stage 2 chunk selection (stage2_order, first max_chunks) run unchanged; the stub records which chunks Stage 2
would score (chunk index, kind and (segment, token start, token end) spans) and returns score 0 for each, so the
service also reports what the outcome is when no chunk is positive (allow, or the coverage policy when some chunk was
scored by neither stage). No DeBERTa inference runs in phase A.

Phase B (the decision for escalated requests), two ways:
  chunk_level     (headline) the real DeBERTa (artifacts/baseline/deberta-pi-v2) scores exactly the chunk rows the
                  stub recorded, one row per call as in serving (max_batch 1). Decision: flag iff any of those
                  scores >= the Stage 2 threshold, otherwise the outcome phase A reported. This is what the route-
                  mode gateway decides without timeouts: a positive chunk always flags, and with no positive chunk
                  the decision does not depend on the scores (coverage policy flag; deep_scan is not configured).
  request_level   escalated -> flag iff the max over ALL offline DeBERTa windows of the text (cache
                  training/results/cascade_scores_replay.jsonl) >= 0.5. Ignores Stage 2's chunk budget and the
                  coverage policy; given for comparison with the offline view.
Requests Stage 1 finalized as allow are allow in both.

Reported: Stage 2 share; recall / FPR of both decision views vs the offline DeBERTa-only reference (every window
scored); DeBERTa-caught injections that Stage 1 finalized as allow ("lost"; the reference is offline DeBERTa-only, as
in the kill criterion); per-item agreement with the served route-mode run (--served); why items the offline view
routes low escalate in the gateway; and a descriptive sweep of Stage 1's gray-zone low using the serving Stage 1's
per-chunk scores and languages (exact for which requests Stage 1 clears at low <= the served low, because the
gateway runs langdetect on every chunk below the served low; the decision of an escalated request in the sweep is
request_level). The sweep is descriptive: the replay set is an evaluation set and no threshold is chosen from it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from training.cascade_sweep import DEBERTA_THRESHOLD, KILL_MIN_SAVED, decide, load
from training.common import ROOT, read_jsonl, sha256_file, write_json
from training.metrics import rate

CACHE = ROOT / "training" / "results" / "cascade_scores_replay.jsonl"
REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"
ARTIFACTS = ROOT / "artifacts"
STAGE1_DIR = "candidates/stage1-tfidf-v2.1"
STAGE2_DIR = "baseline/deberta-pi-v2"
NO_TIMEOUT_MS = 600_000
LOW_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


def make_config():
    from serving.config import GatewayConfig
    return GatewayConfig(shadow_log_path="", stage1_timeout_ms=NO_TIMEOUT_MS, stage2_timeout_ms=NO_TIMEOUT_MS)


def load_specs(intra_op_threads: int = 0):
    """The real Stage 1 and Stage 2 DetectorSpecs, verified and loaded the way the gateway loads them."""
    from serving.registry import load_detector
    s1 = load_detector(ARTIFACTS, "stage1", STAGE1_DIR, intra_op_threads=intra_op_threads)
    s2 = load_detector(ARTIFACTS, "stage2", STAGE2_DIR, intra_op_threads=intra_op_threads)
    return s1, s2


class _Instrument:
    """Wraps one ClassifierService: records Stage 1's per-chunk output and stubs Stage 2 (score 0, recorded)."""

    def __init__(self, service):
        self.service = service
        self.cur: dict | None = None
        run_stage1 = service._run_stage1

        def rec_stage1(cs, deadline=None):
            out = run_stage1(cs, deadline)
            if self.cur is not None:
                self.cur["s1_scores"] = [None if out.scores.get(i) is None else float(out.scores[i])
                                         for i in range(len(cs.chunks))]
                self.cur["s1_langs"] = [out.langs.get(i) for i in range(len(cs.chunks))]
                self.cur["s1_deadline_hit"] = out.deadline_hit
            return out

        async def stub_stage2(cs, idx):
            det = service.stage2.detector
            if self.cur is not None:
                for i in idx:
                    ch = cs.chunks[i]
                    self.cur["stage2_chunks"].append({"index": i, "kind": ch.kind,
                                                      "spans": [list(s) for s in ch.spans]})
                    self.cur["_rows"].append([det.cls_id, *ch.ids, det.sep_id])
            return {i: 0.0 for i in idx}

        service._run_stage1 = rec_stage1
        service._stage2_scores = stub_stage2


async def simulate(items: list[dict], service) -> list[dict]:
    """Phase A: every item through `service` (route mode, stub Stage 2), one at a time."""
    ins = _Instrument(service)
    await service.start()
    out = []
    try:
        for it in items:
            ins.cur = {"id": it["id"], "stage2_chunks": [], "_rows": [], "s1_scores": None, "s1_langs": None,
                       "s1_deadline_hit": None}
            o = await service.classify(it["text"], f"cf-{it['id']}")
            rec = ins.cur
            ins.cur = None
            s1 = next((r for r in o.results if r.stage == "stage1"), None)
            rec.update({"stages_run": list(o.stages_run), "escalated": "stage2" in o.stages_run,
                        "stub_decision": o.decision, "stub_policy_applied": o.policy_applied,
                        "stub_coverage_policy_applied": o.coverage_policy_applied, "path": o.info.get("path"),
                        "n_chunks": o.info.get("n_chunks"), "stage1_status": None if s1 is None else s1.status})
            out.append(rec)
    finally:
        await service.stop()
    return out


def score_rows(sim: list[dict], infer: Callable[[list[list[int]]], list[float]], threshold: float) -> None:
    """Phase B, chunk level: score each recorded Stage 2 row (one row per call, as with max_batch 1)."""
    for r in sim:
        scores = [float(infer([row])[0]) for row in r["_rows"]]
        for ch, sc in zip(r["stage2_chunks"], scores):
            ch["deberta"] = round(sc, 6)
        positive = any(sc >= threshold for sc in scores)
        r["chunk_level_flag"] = int(r["stub_decision"] == "flag" or positive) if r["escalated"] else 0
        r["chunk_level_cause"] = ("stage1_allow" if not r["escalated"] else "stage2_flag" if positive
                                  else f"coverage_policy:{r['stub_coverage_policy_applied']}"
                                  if r["stub_coverage_policy_applied"] else
                                  f"failure_policy:{r['stub_policy_applied']}" if r["stub_policy_applied"]
                                  else "stage2_allow")


def _rates(items: list[tuple[int, int]]) -> dict:
    pos = [f for y, f in items if y == 1]
    neg = [f for y, f in items if y == 0]
    return {"recall": rate(sum(pos), len(pos)), "fpr": rate(sum(neg), len(neg))}


def _cleared(r: dict, low: float, languages: list[str] | None) -> bool:
    """Would the serving Stage 1 clear every chunk at gray-zone low `low` (<= the served low)?"""
    sc, lg = r["s1_scores"], r["s1_langs"]
    if not sc or any(s is None for s in sc):
        return False
    return all(s < low and (languages is None or lg[i] in languages) for i, s in enumerate(sc))


def _served_cause(s: dict) -> str:
    stages, statuses = s.get("stages_run") or [], s.get("statuses") or []
    if stages[:1] == ["stage1"] and statuses[:1] and statuses[0] != "ok":
        return f"served_stage1_{statuses[0]}"
    if s.get("policy_applied"):
        return f"served_failure_policy:{s['policy_applied']}"
    if s.get("coverage_policy_applied"):
        return f"served_coverage_policy:{s['coverage_policy_applied']}"
    return "other"


def report(sim: list[dict], cache_rows: list[dict], truth: dict[str, dict], served: dict[str, dict] | None,
           s1_low: float, languages: list[str] | None, meta: dict) -> dict:
    by_id = {r["id"]: r for r in cache_rows}
    for r in sim:
        c = by_id[r["id"]]
        r["label"] = truth[r["id"]].get("label")
        r["deb_flag"] = int(c["deb"] >= DEBERTA_THRESHOLD)
        r["request_level_flag"] = r["deb_flag"] if r["escalated"] else 0
        r["offline_zone"] = decide(c, s1_low, 1.0)["zone"]
    labeled = [r for r in sim if r["label"] in (0, 1)]
    pos = [r for r in labeled if r["label"] == 1]
    n = len(sim)
    n_s2 = sum(r["escalated"] for r in sim)
    lost = [r for r in pos if r["deb_flag"] and not r["escalated"]]
    views = {
        "deberta_only_offline": _rates([(r["label"], r["deb_flag"]) for r in labeled]),
        "cascade_served_faithful_chunk_level": _rates([(r["label"], r["chunk_level_flag"]) for r in labeled]),
        "cascade_served_faithful_request_level": _rates([(r["label"], r["request_level_flag"]) for r in labeled]),
    }
    views["cascade_served_faithful_chunk_level"]["by_cause"] = dict(sorted(Counter(
        f"{r['chunk_level_cause']}:label={r['label']}" for r in labeled).items()))
    views["cascade_served_faithful_chunk_level"]["disagrees_with_request_level"] = sum(
        r["chunk_level_flag"] != r["request_level_flag"] for r in sim)

    # Why the gateway escalates items the offline view finalizes allow (offline low, gateway stage2).
    why: Counter = Counter()
    for r in sim:
        if r["offline_zone"] == "low" and r["escalated"]:
            sc, lg = r["s1_scores"] or [], r["s1_langs"] or []
            if not sc or any(s is None for s in sc):
                why["stage1_did_not_score_every_chunk"] += 1
            elif any(s >= s1_low for s in sc):
                why["serving_stage1_score_ge_low"] += 1  # sub-windows / segment pieces / per-chunk text
            else:
                bad = [lg[i] for i in range(len(sc)) if languages is not None and lg[i] not in languages]
                why["gateway_language_unknown" if "unknown" in bad else "gateway_language_other"] += 1
        elif r["offline_zone"] == "stage2" and not r["escalated"]:
            why["offline_escalates_gateway_clears"] += 1

    agreement = None
    if served:
        both = [r for r in sim if r["id"] in served]
        pairs = Counter((bool(r["escalated"]), "stage2" in (served[r["id"]]["stages_run"] or [])) for r in both)
        dec = Counter((r["chunk_level_flag"], int(served[r["id"]]["decision"] == "flag")) for r in both)
        diff = [r for r in both if r["chunk_level_flag"] != int(served[r["id"]]["decision"] == "flag")]
        agreement = {
            "n_served_responses": len(both),
            "served_stage2_share": rate(sum("stage2" in (served[r["id"]]["stages_run"] or []) for r in both),
                                        len(both)),
            "escalation_sim_vs_served": {f"sim={a},served={b}": v for (a, b), v in sorted(pairs.items())},
            "decision_agrees": sum(dec[(x, x)] for x in (0, 1)),
            "decision_chunk_level_vs_served": {f"sim={a},served={b}": v for (a, b), v in sorted(dec.items())},
            "disagreements_by_served_cause": dict(sorted(Counter(_served_cause(served[r["id"]])
                                                                 for r in diff).items())),
        }

    sweep = []
    for lo in LOW_GRID + [s1_low]:
        cl = {r["id"]: _cleared(r, lo, languages) for r in sim}
        lost_lo = sum(1 for r in pos if r["deb_flag"] and cl[r["id"]])
        rec = sum(r["deb_flag"] and not cl[r["id"]] for r in pos)
        sweep.append({"low": lo, "saved": sum(cl.values()) / n, "n_cleared": sum(cl.values()),
                      "recall_request_level": rec / len(pos) if pos else None, "n_recall": rec,
                      "deberta_caught_injections_lost": lost_lo})
    check = [s for s in sweep if s["low"] == s1_low][0]
    assert check["n_cleared"] == n - n_s2, "sweep clearing rule disagrees with the service at the served low"
    equal = [s for s in sweep if s["deberta_caught_injections_lost"] == 0]
    best_equal = max(equal, key=lambda s: s["saved"]) if equal else None
    pos_ref = sum(r["deb_flag"] for r in pos)
    within_2pt = [s for s in sweep if pos and (pos_ref - s["n_recall"]) / len(pos) <= 0.02]

    return {
        "view_kind": "serving-faithful: real ClassifierService route mode, real Stage 1 and language detector, "
                     "no timeouts; Stage 2 stubbed in phase A, escalated requests decided with DeBERTa",
        "inputs": meta | {"n_items": n, "n_labeled": len(labeled), "n_positive": len(pos)},
        "served_gray_zone_low": s1_low,
        "stage2_share": rate(n_s2, n),
        "saved": 1 - n_s2 / n,
        "views": views,
        "lost_injections": {
            "reference": "offline DeBERTa-only (every window scored) flags it; Stage 1 finalized allow",
            "n": len(lost), "by_source": dict(sorted(Counter(truth[r["id"]].get("source") for r in lost).items())),
        },
        "offline_low_vs_gateway": dict(sorted(why.items())),
        "served_agreement": agreement,
        "low_sweep_descriptive": sweep,
        "kill_criterion": {
            "min_saved": KILL_MIN_SAVED,
            "served_setting_saved": 1 - n_s2 / n,
            "served_setting_recall_equal": len(lost) == 0,
            "best_equal_recall_setting": best_equal,
            "max_saved_within_2pt_recall": max((s["saved"] for s in within_2pt), default=None),
            "passes_at_equal_recall": bool(best_equal and best_equal["saved"] >= KILL_MIN_SAVED),
        },
        "items": [{k: ([None if x is None else round(x, 6) for x in v] if k == "s1_scores" and v else v)
                   for k, v in r.items() if not k.startswith("_")} for r in sim],
    }


def run(out: Path, served_path: Path | None, cache: Path = CACHE, limit: int | None = None,
        intra_op_threads: int = 0) -> dict:
    from serving.service import ClassifierService

    items = sorted(read_jsonl(REPLAY), key=lambda x: x["id"])
    if limit:
        items = items[:limit]
    truth = {r["id"]: r for r in items}
    cache_rows = [r for r in load(cache) if r["id"] in truth]
    s1, s2 = load_specs(intra_op_threads)
    infer = s2.detector.infer
    # Phase A: the real service with Stage 2 stubbed (the stub replaces the service's Stage 2 scoring call).
    config = make_config()
    service = ClassifierService(s2, config, stage1=s1, stage1_mode="route")
    t0 = time.perf_counter()
    sim = asyncio.run(simulate(items, service))
    t_a = time.perf_counter() - t0
    t0 = time.perf_counter()
    score_rows(sim, infer, s2.threshold)
    t_b = time.perf_counter() - t0
    served = None
    if served_path:
        served = {s["id"]: s for s in read_jsonl(served_path) if s.get("error") is None}
    meta = {"replay_sha256": sha256_file(REPLAY), "cache_sha256": sha256_file(cache),
            "cache": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
            "served": str(served_path) if served_path else None,
            "stage1": f"{s1.detector_id}@{s1.detector_version} ({STAGE1_DIR})", "stage1_sha256": s1.sha256,
            "stage2": f"{s2.detector_id}@{s2.detector_version} ({STAGE2_DIR})", "stage2_sha256": s2.sha256,
            "config": {"max_chunks": config.max_chunks, "coverage_policy": config.coverage_policy,
                       "stage1_subwindow": [config.stage1_subwindow_tokens, config.stage1_subwindow_stride],
                       "lang_min_letters": config.lang_min_letters, "lang_min_prob": config.lang_min_prob,
                       "lang_max_chars": config.lang_max_chars, "stage1_timeout_ms": config.stage1_timeout_ms,
                       "stage2_timeout_ms": config.stage2_timeout_ms},
            "n_deberta_rows_scored": sum(len(r["_rows"]) for r in sim),
            "wall_s": {"phase_a": round(t_a, 1), "phase_b": round(t_b, 1)}}
    languages = service.stage1_languages
    rep = report(sim, cache_rows, truth, served, s1.gray_zone[0], languages, meta)
    write_json(out, rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", type=Path, default=None, help="route-mode replay_runner requests.jsonl")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N items by id (smoke runs only)")
    ap.add_argument("--intra-op-threads", type=int, default=0)
    a = ap.parse_args(argv)
    rep = run(a.out, a.served, a.cache, a.limit, a.intra_op_threads)
    print(json.dumps({k: v for k, v in rep.items() if k != "items" and k != "low_sweep_descriptive"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
