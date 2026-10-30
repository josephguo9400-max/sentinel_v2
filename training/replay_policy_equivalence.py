"""T3 / T4 / T5 (docs/design/capability-policy.md §8): the policy engine decides what the served gateway decided.

    # T3: all 2,000 replay items through the legacy D23 service and the converted policy, real models, in-process,
    # timeouts disabled. Resumable: run it again (same --out) until it reports complete.
    flock /home/claude/measure.lock -c 'python -m training.replay_policy_equivalence t3 \
        --out reports/policy-runtime/t3_d23_equivalence.json --max-seconds 240'

    # T4: the route policy reproduces the committed serving-faithful counterfactual (no DeBERTa inference: the
    # committed per-chunk scores are reused), including its headline 201/306.
    flock /home/claude/measure.lock -c 'python -m training.replay_policy_equivalence t4 \
        --out reports/policy-runtime/t4_cascade_equivalence.json'

    # T5: the T3 policy-engine decisions against the live D23 run.
    python -m training.replay_policy_equivalence t5 --t3 reports/policy-runtime/t3_d23_equivalence.json \
        --out reports/policy-runtime/t5_live_comparison.json

T3, both engines in one process on the same input, one item at a time:
  legacy  serving/legacy_service.py (frozen at milestone/cascade-and-shadow 36ce683) with the committed registry
          v1 state (artifacts/registry.json: DeBERTa decides, Stage 1 observe-only) and the default GatewayConfig
          with both budgets set to 600 s, which is how training/replay_counterfactual_served.py disables timeouts.
  policy  serving/service.py with the policy serving.policy.legacy_to_v2 derives from the same registry state and
          the same config, so both sides use the same no-timeout budgets. The committed bundle
          (artifacts/policies/d23-deberta-decides/1/policy.json) is the same policy with the served budgets
          (50 ms / 5,000 ms); the run records both bundle hashes and asserts that the two policies differ in
          nothing but those three timeout fields.
  Compared per item: decision, decided_by, policy_applied, coverage_policy_applied, stages_run, the DeBERTa rows
  scored (content token ids, in order) and the active scores (within contracts THRESHOLD_TOLERANCE). The observe
  step's status and outcome are compared too; its `role` differs by design (1.2 "shadow" -> 1.3 "observe").

  DeBERTa inference is memoized by row content, shared by both engines and persisted (--cache), so a row is scored
  once instead of twice and a resumed run does not rescore. Identical rows therefore get bit-identical scores; the
  test is about which rows each engine selects and what it decides from them. --no-cache scores every row again and
  compares with THRESHOLD_TOLERANCE instead (slower; the design's answer to thread-count nondeterminism).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from contracts.detector import THRESHOLD_TOLERANCE
from training.common import ROOT, read_jsonl, sha256_file, write_json

REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"
ARTIFACTS = ROOT / "artifacts"
LIVE_D23 = ROOT / "reports" / "cascade-and-shadow" / "d23" / "replay" / "requests.jsonl"
FAITHFUL = ROOT / "reports" / "cascade-and-shadow" / "replay_counterfactual_served_faithful.json"
NO_TIMEOUT_MS = 600_000
DEBERTA_THRESHOLD = 0.5
COMPARED = ("decision", "decided_by", "policy_applied", "coverage_policy_applied", "stages_run")


# ---------------------------------------------------------------------- shared setup

def no_timeout_config():
    from serving.config import GatewayConfig
    return GatewayConfig(shadow_log_path="", stage1_timeout_ms=NO_TIMEOUT_MS, stage2_timeout_ms=NO_TIMEOUT_MS)


def registry_state(path: Path = ARTIFACTS / "registry.json") -> dict:
    return json.loads(path.read_text())


def load_specs(pointers: dict, intra_op_threads: int = 0) -> dict:
    """The real DetectorSpecs, verified and loaded the way the gateway loads them (v1 pointers)."""
    from serving.registry import load_detector
    return {name: load_detector(ARTIFACTS, name, rel, intra_op_threads=intra_op_threads)
            for name, rel in pointers.items() if name in ("stage1", "stage2", "shadow") and rel is not None}


class RowCache:
    """Row content -> score, persisted as JSONL so a resumed run does not rescore."""

    def __init__(self, path: Path | None):
        self.path = path
        self.scores: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        if path is not None and path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.scores[rec["k"]] = rec["v"]
        self._fh = path.open("a", encoding="utf-8") if path is not None else None

    @staticmethod
    def key(row: list[int]) -> str:
        return hashlib.sha256(",".join(map(str, row)).encode()).hexdigest()

    def get(self, row: list[int]):
        k = self.key(row)
        if k in self.scores:
            self.hits += 1
            return self.scores[k]
        self.misses += 1
        return None

    def put(self, row: list[int], score: float) -> None:
        k = self.key(row)
        self.scores[k] = score
        if self._fh is not None:
            self._fh.write(json.dumps({"k": k, "v": score}) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class Recorder:
    """Wraps a token detector: records the rows of the current request and (optionally) memoizes scores."""

    def __init__(self, det, cache: RowCache | None):
        self._det = det
        self._cache = cache
        self.rows: list[list[int]] = []
        for attr in ("detector_id", "detector_version", "threshold", "max_len", "cls_id", "sep_id"):
            setattr(self, attr, getattr(det, attr))

    def tokenize(self, text: str):
        return self._det.tokenize(text)

    def tokenize_with_offsets(self, text: str):
        return self._det.tokenize_with_offsets(text)

    def infer(self, batch: list[list[int]]) -> list[float]:
        out = []
        for row in batch:
            self.rows.append(list(row))
            cached = None if self._cache is None else self._cache.get(row)
            if cached is None:
                cached = float(self._det.infer([row])[0])
                if self._cache is not None:
                    self._cache.put(row, cached)
            out.append(cached)
        return out


def _spec_with(spec, detector):
    import dataclasses
    return dataclasses.replace(spec, detector=detector)


def build_engines(pointers: dict, config, cache: RowCache | None, specs: dict):
    """(legacy service, policy service, {"legacy": recorder, "policy": recorder}, policy, bundle)."""
    from serving.legacy_service import ClassifierService as Legacy
    from serving.policy import bundle_sha256, legacy_to_v2
    from serving.registry import KNOWN_TOKENIZERS, LoadedBundle
    from serving.service import ClassifierService

    from serving.convert_registry import CALIBRATION_RECORDS
    recs = {name: Recorder(specs["stage2"].detector, cache) for name in ("legacy", "policy")}
    mode = pointers.get("stage1_mode")
    legacy = Legacy(_spec_with(specs["stage2"], recs["legacy"]), config, stage1=specs.get("stage1"),
                    shadow=specs.get("shadow"), stage1_mode=mode)
    manifests = {name: spec.manifest for name, spec in specs.items()}
    # Same id, version and rationale as the committed bundle, so the only differences are the raised budgets.
    policy, caps = legacy_to_v2(pointers, manifests, config, known_tokenizers=KNOWN_TOKENIZERS,
                                calibrations=CALIBRATION_RECORDS, policy_id="d23-deberta-decides",
                                policy_version="1", rationale=committed_policy().rationale)
    detectors = {}
    for name, spec in specs.items():
        if name == "shadow":
            continue
        det = recs["policy"] if name == "stage2" else spec.detector
        detectors[(spec.detector_id, spec.detector_version)] = _spec_with(spec, det)
    bundle = LoadedBundle(policy=policy, capabilities=caps, detectors=detectors, shadow=specs.get("shadow"),
                          bundle_sha256=bundle_sha256(policy), policy_source="legacy_registry")
    return legacy, ClassifierService(config=config, bundle=bundle), recs, policy, bundle


def committed_policy(policy_id: str = "d23-deberta-decides", version: str = "1"):
    from contracts import ExecutionPolicy
    path = ARTIFACTS / "policies" / policy_id / version / "policy.json"
    return ExecutionPolicy.model_validate(json.loads(path.read_text()))


def timeout_only_diff(a, b) -> dict:
    """The fields in which two policies differ, as dotted paths (used to prove only budgets were raised)."""
    out = {}

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y:
            out[path] = [x, y]

    walk(a.model_dump(mode="json"), b.model_dump(mode="json"), "")
    return out


# ---------------------------------------------------------------------- T3

def item_view(out, rows: list[list[int]]) -> dict:
    """The comparable part of one response."""
    active = [r for r in out.results if r.role == "active"]
    other = [r for r in out.results if r.role != "active"]
    return {
        **{k: (list(getattr(out, k)) if k == "stages_run" else getattr(out, k)) for k in COMPARED},
        "path": out.info.get("path"),
        "n_chunks": out.info.get("n_chunks"),
        "active": [{"detector_id": r.detector_id, "stage": r.stage, "status": r.status, "score": r.score,
                    "n_chunks": r.n_chunks, "truncated": r.truncated, "failure_reason": r.failure_reason}
                   for r in active],
        "observe": [{"detector_id": r.detector_id, "status": r.status, "score": r.score,
                     "failure_reason": r.failure_reason} for r in other],
        "observe_outcome": (out.info.get("stage1_observe") or {}).get("outcome"),
        "rows": [list(r) for r in rows],
    }


def differences(a: dict, b: dict) -> list[str]:
    """Where two item views disagree. Scores compare within THRESHOLD_TOLERANCE."""
    diffs = []
    for k in (*COMPARED, "path", "n_chunks", "observe_outcome"):
        if a[k] != b[k]:
            diffs.append(f"{k}: {a[k]!r} vs {b[k]!r}")
    if a["rows"] != b["rows"]:
        diffs.append(f"stage2 rows differ: {len(a['rows'])} vs {len(b['rows'])} rows scored")
    for side in ("active", "observe"):
        if len(a[side]) != len(b[side]):
            diffs.append(f"{side} result count: {len(a[side])} vs {len(b[side])}")
            continue
        for i, (x, y) in enumerate(zip(a[side], b[side])):
            for k in x:
                u, v = x[k], y[k]
                if k == "score" and isinstance(u, float) and isinstance(v, float):
                    if abs(u - v) > THRESHOLD_TOLERANCE:
                        diffs.append(f"{side}[{i}].score: {u} vs {v}")
                elif u != v:
                    diffs.append(f"{side}[{i}].{k}: {u!r} vs {v!r}")
    return diffs


async def run_pair(legacy, policy_svc, recs, items, progress_fh, deadline: float | None, done: set[str]) -> int:
    await legacy.start()
    await policy_svc.start()
    n = 0
    try:
        for it in items:
            if it["id"] in done:
                continue
            if deadline is not None and time.perf_counter() > deadline:
                break
            views = {}
            for name, svc in (("legacy", legacy), ("policy", policy_svc)):
                recs[name].rows.clear()
                out = await svc.classify(it["text"], f"eq-{it['id']}")
                views[name] = item_view(out, recs[name].rows)
            rec = {"id": it["id"], "label": it.get("label"), "source": it.get("source"),
                   "diffs": differences(views["legacy"], views["policy"]),
                   "legacy": {k: v for k, v in views["legacy"].items() if k != "rows"},
                   "policy": {k: v for k, v in views["policy"].items() if k != "rows"},
                   "n_rows": len(views["legacy"]["rows"])}
            progress_fh.write(json.dumps(rec, sort_keys=True) + "\n")
            progress_fh.flush()
            n += 1
    finally:
        await legacy.stop()
        await policy_svc.stop()
    return n


def t3(out_path: Path, max_seconds: float | None, cache_path: Path | None, limit: int | None,
       intra_op_threads: int) -> dict:
    items = sorted(read_jsonl(REPLAY), key=lambda x: x["id"])
    if limit:
        items = items[:limit]
    progress = out_path.with_suffix(".progress.jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in progress.read_text().splitlines() if line.strip()} \
        if progress.is_file() else set()
    cache = RowCache(cache_path)
    pointers = registry_state()
    config = no_timeout_config()
    specs = load_specs(pointers, intra_op_threads)
    legacy, policy_svc, recs, policy, bundle = build_engines(pointers, config, cache, specs)
    t0 = time.perf_counter()
    deadline = None if max_seconds is None else t0 + max_seconds
    with progress.open("a", encoding="utf-8") as fh:
        n_new = asyncio.run(run_pair(legacy, policy_svc, recs, items, fh, deadline, done))
    wall = time.perf_counter() - t0
    cache.close()
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    rep = t3_report(rows, items, policy, bundle, specs, config, cache, wall, n_new, cache_path)
    write_json(out_path, rep)
    return rep


def t3_report(rows, items, policy, bundle, specs, config, cache, wall, n_new, cache_path) -> dict:
    from serving import engine_version as ev
    from serving.policy import bundle_sha256

    committed = committed_policy()
    diff = timeout_only_diff(committed, policy)
    expected_timeout_fields = {"decide.members[0].timeout_ms", "observe[0].timeout_ms"}
    with_diffs = [r for r in rows if r["diffs"]]
    by_id = {r["id"]: r for r in rows}
    decisions = Counter(r["policy"]["decision"] for r in rows)
    return {
        "test": "T3 (design §8): D23 replay equivalence, legacy service vs policy engine",
        "complete": len(rows) >= len(items),
        "n_items_total": len(items),
        "n_items_compared": len(rows),
        "n_items_this_run": n_new,
        "identical_on_every_item": not with_diffs,
        "n_items_with_differences": len(with_diffs),
        "differences": [{"id": r["id"], "diffs": r["diffs"]} for r in with_diffs[:50]],
        "inputs": {
            "replay": "data/replay/replay-v1.jsonl", "replay_sha256": sha256_file(REPLAY),
            "registry": "artifacts/registry.json (sentinel-registry-v1)",
            "stage1": f"{specs['stage1'].detector_id}@{specs['stage1'].detector_version}",
            "stage1_model_sha256": specs["stage1"].sha256,
            "stage2": f"{specs['stage2'].detector_id}@{specs['stage2'].detector_version}",
            "stage2_model_sha256": specs["stage2"].sha256,
            "engine_version": ev.ENGINE_VERSION,
            "timeouts_disabled_ms": NO_TIMEOUT_MS,
            "config": {"max_chunks": config.max_chunks, "coverage_policy": config.coverage_policy,
                       "stage1_subwindow": [config.stage1_subwindow_tokens, config.stage1_subwindow_stride],
                       "stage1_max_chunks": config.stage1_max_chunks, "max_batch": config.max_batch},
        },
        "bundle": {
            "run_policy_bundle_sha256": bundle.bundle_sha256,
            "committed_policy": "artifacts/policies/d23-deberta-decides/1/policy.json",
            "committed_bundle_sha256": bundle_sha256(committed),
            "differences_from_committed_policy": diff,
            "only_timeout_fields_differ": set(diff) <= expected_timeout_fields,
            "note": "the run raises both budgets to 600 s so no decision is made by a deadline; everything else, "
                    "including every capability pin, is the committed bundle",
        },
        "decisions": dict(sorted(decisions.items())),
        "paths": dict(sorted(Counter(r["policy"]["path"] for r in rows).items())),
        "observe_outcomes": dict(sorted(Counter(str(r["policy"]["observe_outcome"]) for r in rows).items())),
        "deberta_rows": {"total_scored_per_engine": sum(r["n_rows"] for r in rows),
                         "unique_rows_in_cache": len(cache.scores), "cache_hits": cache.hits,
                         "cache_misses": cache.misses,
                         "cache_path": str(cache_path) if cache_path else None},
        "wall_s_this_run": round(wall, 1),
        "items": [{"id": r["id"], "label": r["label"], "decision": r["policy"]["decision"],
                   "decided_by": r["policy"]["decided_by"], "stages_run": r["policy"]["stages_run"],
                   "policy_applied": r["policy"]["policy_applied"],
                   "coverage_policy_applied": r["policy"]["coverage_policy_applied"],
                   "path": r["policy"]["path"], "observe_outcome": r["policy"]["observe_outcome"],
                   "n_rows": r["n_rows"]} for r in sorted(by_id.values(), key=lambda x: x["id"])],
    }


# ---------------------------------------------------------------------- T4

class _StubStage2:
    """Records the chunk rows the policy would send to the decide member and returns 0 for each (phase A of
    training/replay_counterfactual_served.py)."""

    def __init__(self, det):
        self._det = det
        self.chunks: list[dict] = []
        for attr in ("detector_id", "detector_version", "threshold", "max_len", "cls_id", "sep_id"):
            setattr(self, attr, getattr(det, attr))

    def tokenize(self, text):
        return self._det.tokenize(text)

    def tokenize_with_offsets(self, text):
        return self._det.tokenize_with_offsets(text)

    def infer(self, batch):
        return [0.0] * len(batch)


def t4(out_path: Path, limit: int | None, intra_op_threads: int) -> dict:
    """The route policy (cascade-tfidf-allow-only) through the policy engine, Stage 2 stubbed, compared with the
    committed serving-faithful counterfactual. The committed per-chunk DeBERTa scores are reused, so this needs no
    DeBERTa inference and the headline 201/306 is recomputed from committed numbers."""
    from serving.convert_registry import CALIBRATION_RECORDS
    from serving.policy import bundle_sha256, legacy_to_v2
    from serving.registry import KNOWN_TOKENIZERS, LoadedBundle
    from serving.service import ClassifierService

    faithful = json.loads(FAITHFUL.read_text())
    committed_items = {it["id"]: it for it in faithful["items"]}
    items = sorted(read_jsonl(REPLAY), key=lambda x: x["id"])
    if limit:
        items = items[:limit]
    pointers = registry_state(ROOT / "reports/cascade-and-shadow/registry_integration_route.json")
    config = no_timeout_config()
    specs = load_specs(pointers, intra_op_threads)
    stub = _StubStage2(specs["stage2"].detector)
    manifests = {name: spec.manifest for name, spec in specs.items()}
    policy, caps = legacy_to_v2(pointers, manifests, config, known_tokenizers=KNOWN_TOKENIZERS,
                                calibrations=CALIBRATION_RECORDS, policy_id="cascade-tfidf-allow-only",
                                policy_version="1")
    detectors = {(s.detector_id, s.detector_version): (_spec_with(s, stub) if n == "stage2" else s)
                 for n, s in specs.items() if n != "shadow"}
    bundle = LoadedBundle(policy=policy, capabilities=caps, detectors=detectors, shadow=None,
                          bundle_sha256=bundle_sha256(policy), policy_source="legacy_registry")
    svc = ClassifierService(config=config, bundle=bundle)

    # Instrumentation, as in training/replay_counterfactual_served.py: record the gate's per-chunk output and the
    # chunks the decide member is asked to score. Nothing in serving/ is changed for the test.
    recorded: dict[str, dict] = {}
    chunks_by_item: dict[str, list[dict]] = {}
    cur = {"id": "", "gate": None}
    run_gate, member_scores = svc._run_stage1, svc._member_scores

    def rec_gate(cs, deadline=None):
        out = run_gate(cs, deadline)
        cur["gate"] = {"scores": [None if out.scores.get(i) is None else float(out.scores[i])
                                  for i in range(len(cs.chunks))],
                       "langs": [out.langs.get(i) for i in range(len(cs.chunks))],
                       "deadline_hit": out.deadline_hit}
        return out

    async def rec_member(member, cs, idx):
        rows = chunks_by_item.setdefault(cur["id"], [])
        for i in idx:
            ch = cs.chunks[i]
            rows.append({"index": i, "kind": ch.kind, "spans": [list(sp) for sp in ch.spans]})
        return await member_scores(member, cs, idx)

    svc._run_stage1 = rec_gate
    svc._member_scores = rec_member

    async def scenario():
        await svc.start()
        try:
            for it in items:
                cur["id"], cur["gate"] = it["id"], None
                chunks_by_item.setdefault(it["id"], [])
                out = await svc.classify(it["text"], f"t4-{it['id']}")
                s1 = next((r for r in out.results if r.stage == "stage1"), None)
                gate = cur["gate"] or {}
                recorded[it["id"]] = {
                    "stages_run": list(out.stages_run), "escalated": "stage2" in out.stages_run,
                    "stub_decision": out.decision, "stub_policy_applied": out.policy_applied,
                    "stub_coverage_policy_applied": out.coverage_policy_applied, "path": out.info.get("path"),
                    "n_chunks": out.info.get("n_chunks"), "stage1_status": None if s1 is None else s1.status,
                    "s1_scores": gate.get("scores"), "s1_langs": gate.get("langs"),
                    "s1_deadline_hit": bool(gate.get("deadline_hit")),
                    "escalation_reason": out.escalation_reason,
                }
        finally:
            await svc.stop()

    asyncio.run(scenario())
    return t4_report(out_path, recorded, chunks_by_item, committed_items, items, faithful, policy, bundle, specs)


def t4_report(out_path, recorded, chunks_by_item, committed_items, items, faithful, policy, bundle, specs) -> dict:
    truth = {it["id"]: it for it in items}
    cmp_fields = ("escalated", "stub_decision", "stub_policy_applied", "stub_coverage_policy_applied", "path",
                  "n_chunks", "stage1_status", "s1_langs", "stages_run")
    diffs = []
    chunk_diffs = []
    flags = {}
    for rid, got in recorded.items():
        want = committed_items.get(rid)
        if want is None:
            diffs.append({"id": rid, "diffs": ["not in the committed counterfactual"]})
            continue
        bad = []
        for k in cmp_fields:
            if k in want and want[k] != got[k]:
                bad.append(f"{k}: committed {want[k]!r} vs policy engine {got[k]!r}")
        if got["s1_scores"] is not None and want.get("s1_scores") is not None:
            a = [None if x is None else round(x, 6) for x in got["s1_scores"]]
            if a != want["s1_scores"]:
                bad.append(f"s1_scores: committed {want['s1_scores']} vs policy engine {a}")
        got_chunks = [{k: c[k] for k in ("index", "kind", "spans")} for c in chunks_by_item.get(rid, [])]
        want_chunks = [{k: c[k] for k in ("index", "kind", "spans")} for c in want["stage2_chunks"]]
        if got_chunks != want_chunks:
            chunk_diffs.append({"id": rid, "committed": want_chunks, "policy_engine": got_chunks})
        if bad:
            diffs.append({"id": rid, "diffs": bad})
        # The decision the route bundle serves, from the committed per-chunk DeBERTa scores of those same chunks.
        scores = [c.get("deberta") for c in want["stage2_chunks"]]
        positive = any(s is not None and s >= DEBERTA_THRESHOLD for s in scores)
        flags[rid] = 0 if not got["escalated"] else int(got["stub_decision"] == "flag" or positive)
    labeled = [(truth[r].get("label"), flags[r]) for r in flags if truth[r].get("label") in (0, 1)]
    pos = [f for y, f in labeled if y == 1]
    neg = [f for y, f in labeled if y == 0]
    headline = faithful["views"]["cascade_served_faithful_chunk_level"]
    rep = {
        "test": "T4 (design §8): the cascade route policy reproduces the committed serving-faithful counterfactual",
        "reference": "reports/cascade-and-shadow/replay_counterfactual_served_faithful.json",
        "reference_sha256": sha256_file(FAITHFUL),
        "n_items_compared": len(recorded),
        "identical_on_every_item": not diffs and not chunk_diffs,
        "n_items_with_differences": len(diffs),
        "n_items_with_different_stage2_chunks": len(chunk_diffs),
        "differences": diffs[:50],
        "stage2_chunk_differences": chunk_diffs[:20],
        "recall": {"flagged": sum(pos), "positives": len(pos), "as_fraction": f"{sum(pos)}/{len(pos)}"},
        "fpr": {"flagged": sum(neg), "negatives": len(neg), "as_fraction": f"{sum(neg)}/{len(neg)}"},
        "committed_reference_rates": {"recall": headline["recall"], "fpr": headline["fpr"]},
        "headline_201_306_reproduced": f"{sum(pos)}/{len(pos)}" == "201/306",
        "escalated": sum(1 for r in recorded.values() if r["escalated"]),
        "escalation_reasons": dict(sorted(Counter(str(r["escalation_reason"]) for r in recorded.values()).items())),
        "inputs": {"replay_sha256": sha256_file(REPLAY),
                   "registry": "reports/cascade-and-shadow/registry_integration_route.json",
                   "stage1": f"{specs['stage1'].detector_id}@{specs['stage1'].detector_version}",
                   "stage2": f"{specs['stage2'].detector_id}@{specs['stage2'].detector_version} (stubbed in "
                             f"phase A; the committed per-chunk scores decide, as in the reference)",
                   "timeouts_disabled_ms": NO_TIMEOUT_MS},
        "bundle": {"policy_id": policy.policy_id, "policy_version": policy.policy_version,
                   "bundle_sha256": bundle.bundle_sha256,
                   "committed_policy": "artifacts/policies/cascade-tfidf-allow-only/1/policy.json"},
    }
    write_json(out_path, rep)
    return rep


# ---------------------------------------------------------------------- T5

def t5(out_path: Path, t3_path: Path) -> dict:
    """The T3 policy-engine decisions against the live D23 run. Items whose served response had a failed active
    result are listed, not compared: timing decided them (design §8, T5)."""
    t3_rep = json.loads(t3_path.read_text())
    engine = {it["id"]: it for it in t3_rep["items"]}
    live = [r for r in read_jsonl(LIVE_D23)]
    served = {r["id"]: r for r in live}
    comparable, agree, disagree, skipped = [], 0, [], []
    for rid, s in served.items():
        if rid not in engine:
            continue
        if s.get("error") is not None:
            skipped.append({"id": rid, "why": f"transport error: {s['error']}"})
            continue
        statuses = s.get("statuses") or []
        if any(st != "ok" for st in statuses) or s.get("policy_applied"):
            skipped.append({"id": rid, "why": f"served response had a failed active result: statuses={statuses}, "
                                              f"policy_applied={s.get('policy_applied')}"})
            continue
        comparable.append(rid)
        if s["decision"] == engine[rid]["decision"]:
            agree += 1
        else:
            disagree.append({"id": rid, "served": s["decision"], "policy_engine": engine[rid]["decision"],
                             "served_coverage": s.get("coverage_policy_applied"),
                             "engine_path": engine[rid]["path"]})
    rep = {
        "test": "T5 (design §8): policy engine vs the live D23 run",
        "live_run": "reports/cascade-and-shadow/d23/replay/requests.jsonl",
        "live_run_sha256": sha256_file(LIVE_D23),
        "t3_report": str(t3_path.relative_to(ROOT)) if t3_path.is_relative_to(ROOT) else str(t3_path),
        "n_live_records": len(live),
        "n_compared": len(comparable),
        "n_identical": agree,
        "n_different": len(disagree),
        "identical_on_every_comparable_item": not disagree,
        "differences": disagree,
        "n_not_compared": len(skipped),
        "not_compared": skipped,
        "note": "The live run served under real timeouts on a 2-core host; items it decided by a failure policy "
                "cannot be reproduced by a run with timeouts disabled, so they are listed instead of compared.",
    }
    write_json(out_path, rep)
    return rep


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["t3", "t4", "t5"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seconds", type=float, default=None, help="t3: stop after this long; resumable")
    ap.add_argument("--cache", type=Path, default=None, help="t3: persisted row score cache (JSONL)")
    ap.add_argument("--no-cache", action="store_true", help="t3: score every row again in both engines")
    ap.add_argument("--limit", type=int, default=None, help="first N items by id (smoke runs only)")
    ap.add_argument("--intra-op-threads", type=int, default=0)
    ap.add_argument("--t3", type=Path, default=None, help="t5: the T3 report to compare")
    a = ap.parse_args(argv)
    if a.test == "t3":
        cache = None if a.no_cache else (a.cache or a.out.with_suffix(".rowcache.jsonl"))
        rep = t3(a.out, a.max_seconds, cache, a.limit, a.intra_op_threads)
        print(json.dumps({k: rep[k] for k in ("complete", "n_items_compared", "n_items_total",
                                              "identical_on_every_item", "n_items_with_differences",
                                              "wall_s_this_run")}, indent=1))
        if rep["differences"]:
            print(json.dumps(rep["differences"][:5], indent=1))
        return 0 if rep["identical_on_every_item"] else 1
    if a.test == "t4":
        rep = t4(a.out, a.limit, a.intra_op_threads)
        print(json.dumps({k: rep[k] for k in ("n_items_compared", "identical_on_every_item",
                                              "n_items_with_differences", "n_items_with_different_stage2_chunks",
                                              "recall", "fpr", "headline_201_306_reproduced")}, indent=1))
        return 0 if rep["identical_on_every_item"] else 1
    if not a.t3:
        ap.error("t5 needs --t3")
    rep = t5(a.out, a.t3)
    print(json.dumps({k: rep[k] for k in ("n_compared", "n_identical", "n_different", "n_not_compared",
                                          "identical_on_every_comparable_item")}, indent=1))
    return 0 if rep["identical_on_every_comparable_item"] else 1


if __name__ == "__main__":
    sys.exit(main())
