"""Release gate for a bundle, and the only reader of the frozen test.

The unit of a release is a **bundle**: one ExecutionPolicy, the capability versions it pins and the engine version
that interprets it (docs/design/capability-policy.md §2, owner decision D25). The gate evaluates the bundle **as
served** and writes a report bound to its `bundle_sha256`. It never promotes anything: promotion is a separate
command that needs an explicit `--approved-by`, and rollback is a pointer flip to `previous`.

    # evaluate the bundle a registry serves, on the development split (resumable; never promotes)
    flock /home/claude/measure.lock -c 'python -m training.gate evaluate \
        --registry artifacts/registry.json --split val --out reports/policy-runtime/gate_val.json \
        --max-seconds 240'

    # the owner, after reading the report
    python -m training.gate promote --registry artifacts/registry.json \
        --gate-report reports/policy-runtime/gate_val.json --approved-by "name <email>"
    python -m training.gate rollback --registry artifacts/registry.json --approved-by "name <email>"

Checks, from the RFC table ("Release gate rules") as implemented in training/eval_harness.py:

| check            | rule                                                             | insufficient evidence |
| recall           | paired vs the active bundle: 95% CI lower bound >= -2 pt         | positives < 200       |
| benign_hard_fpr  | Wilson 95% upper bound <= 5%                                     | samples < 100         |
| compute_saved    | share escalated to a decide member <= 50% (only with a gate step)| no gate step          |
| error_rate       | timeouts + exceptions of active steps < 1% (load shedding apart) | -                     |
| p99_latency      | not evaluated here: it needs a fixed load profile (RFC           | always                |
|                  | serving-hardening); this run is sequential and in-process        |                       |
| adaptive_asr     | not evaluated here: RFC adaptive-eval                            | always                |

Any FAIL -> FAIL. No FAIL but some check without evidence -> INSUFFICIENT_EVIDENCE. Everything evaluated and
passing -> PASS, and the owner still has to approve the promotion. The recall check is paired (Newcombe hybrid
score, training/cascade_sweep.newcombe_paired) against the bundle the registry currently serves; when the
candidate *is* the active bundle there is nothing to pair with, so the check reports its rate with
`comparison: null` and counts as INSUFFICIENT_EVIDENCE for a promotion decision.

**The frozen test.** `data/splits/frozen_test.jsonl` is the final split: this script is the only thing that reads
it for a verdict, and only when the owner passes `--split frozen_test --frozen-test-approved-by NAME`. Every other
use of it in the repository predates this gate and is training-side analysis, not a release decision. Each run of
the gate on the frozen test is recorded in the audit log, so repeated looks at it are visible: a "started" entry is
written before the split is opened, then "completed", or "failed" on any exception from then until the report and
the "completed" entry are written (scoring, reading --active-report, building or writing the report). Only a
killed process (or a failure of the audit write itself) leaves "started" alone.

**Promote and rollback record before they act.** The audit line is appended first, then the registry pointer is
moved (written to a temporary file next to the registry, validated, and renamed over it, so the registry is never
left half-written or invalid). If moving the pointer raises, a second line with `"status": "failed"` follows. So
every pointer move has an audit line before it; the one crash-window left is a kill between the two writes, which
leaves an audit line for a move the registry does not show (the registry is the source of truth for what is
served, and the audit log over-reports rather than under-reports).

As served: the bundle runs in-process through serving/service.py with its own budgets, and every response is read
with training/eval_harness.py's rules (`as_served`: a failure-policy or coverage-policy decision counts as the
decision the user got). An in-process sequential run has no queueing, so it cannot produce the p99 or load-shedding
behaviour of a served deployment; the report says so, and the error-rate check counts only what it can observe.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from training.cascade_sweep import newcombe_paired
from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file, write_json
from training.eval_harness import (BENIGN_HARD_FPR_MAX_UPPER, MIN_BENIGN_HARD_N, MIN_POSITIVES_RECALL,
                                   parse_gateway_response, slices_for)
from training.metrics import rate, summarize

AUDIT_LOG = ROOT / "artifacts" / "audit_log.jsonl"
MAX_ESCALATION_SHARE = 0.50  # RFC: compute saved -> fraction escalated <= 50%
MAX_ERROR_RATE = 0.01  # RFC: timeouts + errors < 1%
GATE_SPLITS = ("val", "frozen_test")
PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT_EVIDENCE"
NOT_EVALUATED = "NOT_EVALUATED"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------- running a bundle as served

def load_bundle(registry: Path, artifacts: Path | None = None):
    """The bundle a registry serves (v2), or the one a v1 registry converts to."""
    from serving.config import GatewayConfig
    from serving.registry import (REGISTRY_SCHEMA_V2, bundle_from_v1, load_bundle_v2, load_registry,
                                   registry_schema)

    artifacts = artifacts or ROOT / "artifacts"
    config = GatewayConfig(shadow_log_path="")
    if registry_schema(registry) == REGISTRY_SCHEMA_V2:
        return load_bundle_v2(registry, artifacts), config
    loaded = load_registry(registry, artifacts)
    return bundle_from_v1(loaded, config, registry_sha256=sha256_file(registry)), config


async def _classify_all(service, rows, out_fh, deadline, done: set[str]) -> int:
    n = 0
    await service.start()
    try:
        for row in rows:
            rid = str(row["id"])
            if rid in done:
                continue
            if deadline is not None and time.perf_counter() > deadline:
                break
            t0 = time.perf_counter()
            out = await service.classify(row["text"], f"gate-{rid}")
            payload = out.as_response()
            parsed = parse_gateway_response(payload, payload["request_id"])
            parsed["latency_ms"] = (time.perf_counter() - t0) * 1000
            out_fh.write(json.dumps({"id": rid, "label": row["label"], "source": row.get("source"),
                                     "family": row.get("family"), "stage2_overlap": row.get("stage2_overlap"),
                                     "lang": row.get("lang"), "parsed": parsed}, sort_keys=True) + "\n")
            out_fh.flush()
            n += 1
    finally:
        await service.stop()
    return n


def run_bundle(registry: Path, split: str, out: Path, max_seconds: float | None, limit: int | None,
               splits_dir: Path = SPLITS_DIR) -> tuple[list[dict], dict, object, float, int]:
    """(per-item records, run metadata, bundle, wall seconds, items this run). Resumable via a progress file."""
    from serving.service import ClassifierService

    path = splits_dir / f"{split}.jsonl"
    rows = list(read_jsonl(path))
    if limit:
        rows = rows[:limit]
    progress = out.with_suffix(".progress.jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)
    bundle, config = load_bundle(registry)
    done: set[str] = set()
    if progress.is_file():
        for line in progress.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("bundle_sha256") == bundle.bundle_sha256:
                    done.add(rec["id"])
    service = ClassifierService(config=config, bundle=bundle)
    t0 = time.perf_counter()
    deadline = None if max_seconds is None else t0 + max_seconds
    with progress.open("a", encoding="utf-8") as fh:
        class _Tagging:
            """Writes the bundle hash with every record, so a resumed run cannot mix two bundles."""

            def write(self, line: str) -> None:
                rec = json.loads(line)
                fh.write(json.dumps({**rec, "bundle_sha256": bundle.bundle_sha256}, sort_keys=True) + "\n")

            def flush(self) -> None:
                fh.flush()

        n_new = asyncio.run(_classify_all(service, rows, _Tagging(), deadline, done))
    wall = time.perf_counter() - t0
    records = [json.loads(line) for line in progress.read_text().splitlines()
               if line.strip() and json.loads(line).get("bundle_sha256") == bundle.bundle_sha256]
    meta = {"split": split, "split_path": str(path.relative_to(ROOT)), "split_sha256": sha256_file(path),
            "n_items_in_split": len(rows), "n_items_evaluated": len(records), "complete": len(records) >= len(rows),
            "limit": limit}
    return records, meta, bundle, wall, n_new


# ---------------------------------------------------------------------- metrics and checks

def slice_metrics(records: list[dict]) -> dict:
    """as-served metrics per slice: every non-error item counts with the decision the bundle served."""
    acc: dict[str, dict[str, list]] = {}
    for rec in records:
        p = rec["parsed"]
        if "error" in p:
            continue
        row = {"source": rec.get("source") or "unknown", "family": rec.get("family"),
               "stage2_overlap": rec.get("stage2_overlap"), "lang": rec.get("lang"), "label": rec["label"]}
        for name in slices_for(row):
            a = acc.setdefault(name, {"y": [], "p": []})
            a["y"].append(rec["label"])
            a["p"].append(p["predicted"])
    return {name: summarize(a["y"], a["p"]) for name, a in sorted(acc.items())}


def paired_recall(candidate: list[dict], active: list[dict], label: int = 1) -> dict | None:
    """Paired difference candidate - active over the items both evaluated (Newcombe hybrid score 95%)."""
    by_id = {r["id"]: r for r in active if "error" not in r["parsed"]}
    a = b = c = d = 0
    for r in candidate:
        if "error" in r["parsed"] or r["label"] != label or r["id"] not in by_id:
            continue
        x, y = bool(r["parsed"]["predicted"]), bool(by_id[r["id"]]["parsed"]["predicted"])
        a += x and y
        b += x and not y
        c += y and not x
        d += not x and not y
    out = newcombe_paired(a, b, c, d)
    if out is None:
        return None
    return {"diff": out[0], "ci95": [out[1], out[2]], "only_candidate": b, "only_active": c, "n_pairs": a + b + c + d}


def checks(records: list[dict], slices: dict, paired: dict | None, comparison: str | None, policy=None,
           adaptive_asr: dict | None = None, bundle_sha256: str | None = None) -> dict:
    """One verdict per RFC check. `paired` is None when there is no active bundle to pair with."""
    overall = slices.get("overall", {})
    bh = slices.get("benign_hard")
    out: dict[str, dict] = {}

    positives = overall.get("positives", 0)
    recall = {"name": "recall", "rule": "paired vs the active bundle: 95% CI lower bound >= -2 pt",
              "positives": positives, "min_positives": MIN_POSITIVES_RECALL, "recall": overall.get("recall"),
              "comparison": comparison, "paired": paired}
    if positives < MIN_POSITIVES_RECALL:
        recall["verdict"] = INSUFFICIENT
        recall["why"] = f"{positives} positives < {MIN_POSITIVES_RECALL}"
    elif paired is None:
        recall["verdict"] = INSUFFICIENT
        recall["why"] = ("no active bundle to pair with (the candidate is the active bundle); the rate is reported, "
                         "the paired check needs two bundles")
    else:
        lo = paired["ci95"][0]
        recall["verdict"] = PASS if lo >= -0.02 else FAIL
        recall["why"] = f"paired difference {paired['diff']:+.4f}, 95% CI lower bound {lo:+.4f}"
    out["recall"] = recall

    fpr = {"name": "benign_hard_fpr", "rule": f"Wilson 95% upper bound <= {BENIGN_HARD_FPR_MAX_UPPER}",
           "negatives": (bh or {}).get("negatives", 0), "min_n": MIN_BENIGN_HARD_N, "fpr": (bh or {}).get("fpr")}
    n = fpr["negatives"]
    if n < MIN_BENIGN_HARD_N:
        fpr["verdict"] = INSUFFICIENT
        fpr["why"] = f"{n} benign-hard samples < {MIN_BENIGN_HARD_N}"
    else:
        upper = bh["fpr"]["ci95"][1]
        ok = upper <= BENIGN_HARD_FPR_MAX_UPPER
        fpr["verdict"] = PASS if ok else FAIL
        fpr["why"] = f"Wilson 95% upper bound {upper:.4f} {'<=' if ok else '>'} {BENIGN_HARD_FPR_MAX_UPPER}"
    out["benign_hard_fpr"] = fpr

    served = [r["parsed"] for r in records if "error" not in r["parsed"]]
    escalated = sum(1 for p in served if p.get("escalated"))
    share = rate(escalated, len(served))
    saved = {"name": "compute_saved", "rule": f"share escalated to a decide member <= {MAX_ESCALATION_SHARE}",
             "escalated": escalated, "n": len(served), "escalation_share": share}
    if policy is not None and policy.gate is None:
        # The RFC criterion asks whether a cascade earns its place. A policy with no gate step sends every request
        # to a decide member by design (D23's conclusion), so there is no compute for a gate to save and the check
        # has nothing to measure. It is reported, not failed.
        saved["verdict"] = NOT_EVALUATED
        saved["why"] = ("the policy has no gate step: every request reaches a decide member by design, so there is "
                        "no cascade saving to measure (D23). The escalation share is reported above.")
    else:
        saved["verdict"] = PASS if share["value"] is not None and share["value"] <= MAX_ESCALATION_SHARE else FAIL
        saved["why"] = f"{escalated}/{len(served)} escalated"
    out["compute_saved"] = saved

    reasons = Counter(fr for p in served for fr in (p.get("failure_reasons") or []))
    faults = sum(v for k, v in reasons.items() if k != "overloaded")
    err = rate(faults, len(served))
    out["error_rate"] = {
        "name": "error_rate", "rule": f"timeouts + exceptions of active steps < {MAX_ERROR_RATE} "
                                      f"(load shedding counted apart)",
        "failures": dict(sorted(reasons.items())), "n": len(served), "error_rate": err,
        "shed": reasons.get("overloaded", 0),
        "verdict": PASS if err["value"] is not None and err["value"] < MAX_ERROR_RATE else FAIL,
        "why": f"{faults}/{len(served)} active results failed (excluding {reasons.get('overloaded', 0)} shed)",
    }
    out["p99_latency"] = {"name": "p99_latency", "rule": "no worse than the active bundle under a fixed load "
                                                          "profile", "verdict": NOT_EVALUATED,
                          "why": "needs a load profile and >= 1,000 requests on a host that is not shared with the "
                                 "load generator (RFC serving-hardening); this run is sequential and in-process"}
    aa = {"name": "adaptive_asr", "rule": "mode B ASR no worse than the active bundle",
          "verdict": NOT_EVALUATED, "why": "the adaptive attacker is RFC adaptive-eval"}
    if adaptive_asr is not None:
        # Surface the measured adaptive-ASR (redteam/adaptive). The RFC rule is relative and D23 has no
        # previous bundle to pair against, and the RFC defines no absolute threshold, so this stays
        # report-only (verdict unchanged at NOT_EVALUATED); the owner decides the gate semantics
        # (docs/design/adaptive-attacker.md §8-9).
        measured_sha = adaptive_asr.get("measured_against_bundle_sha256")
        # review m8: a missing bundle hash is NOT a match. applies is True only when both hashes are
        # present and equal; a None on either side is "unknown", treated as a mismatch (does not apply).
        applies = bool(bundle_sha256) and bool(measured_sha) and measured_sha == bundle_sha256
        aa["reported"] = adaptive_asr
        aa["applies_to_this_bundle"] = applies
        if applies:
            detail = "measured on this bundle."
        elif not measured_sha:
            detail = ("WARNING: the adaptive input carries no measured_against_bundle_sha256; the bundle "
                      "it was measured on is unknown, so the number does not apply here.")
        elif not bundle_sha256:
            detail = ("WARNING: this report has no bundle hash to compare against; applicability is "
                      "unknown, so the number does not apply here.")
        else:
            detail = f"WARNING: measured on bundle {measured_sha}, not this one; number does not apply."
        aa["why"] = ("adaptive ASR reported from redteam/adaptive (report-only: the RFC rule is relative, "
                     "D23 has no previous bundle, no absolute threshold is defined). " + detail)
    out["adaptive_asr"] = aa
    return out


def verdict_of(checks_out: dict) -> str:
    verdicts = [c["verdict"] for c in checks_out.values()]
    if FAIL in verdicts:
        return FAIL
    if INSUFFICIENT in verdicts or NOT_EVALUATED in verdicts:
        return INSUFFICIENT
    return PASS


def build_report(records, meta, bundle, wall, n_new, registry: Path, active_records=None,
                 active_bundle=None, frozen_approved_by: str | None = None,
                 adaptive_asr: dict | None = None) -> dict:
    from serving import engine_version as ev
    from serving.policy import describe_bundle

    slices = slice_metrics(records)
    comparison = None
    paired = None
    if active_bundle is not None and active_bundle.bundle_sha256 != bundle.bundle_sha256:
        comparison = active_bundle.bundle_sha256
        paired = paired_recall(records, active_records or [])
    cs = checks(records, slices, paired, comparison, bundle.policy, adaptive_asr=adaptive_asr,
                bundle_sha256=bundle.bundle_sha256)
    return {
        "report": "release gate (training/gate.py)",
        "generated_at": _now(),
        "verdict": verdict_of(cs),
        "promotes": False,
        "bundle": {**describe_bundle(bundle.policy, ev.ENGINE_VERSION), "policy_source": bundle.policy_source,
                   "policy_path": bundle.policy_path, "registry": str(registry.relative_to(ROOT))
                   if registry.is_relative_to(ROOT) else str(registry), "status": bundle.status,
                   "previous": bundle.previous},
        "compared_with_active_bundle": comparison,
        "run": {**meta, "wall_s_this_run": round(wall, 1), "n_items_this_run": n_new,
                "gate_view": "as_served", "mode": "in-process, sequential (no queueing, no load profile)",
                "environment": {"python": platform.python_version(), "machine": platform.machine()},
                "frozen_test_approved_by": frozen_approved_by},
        "gate_minimums": {"recall_min_positives": MIN_POSITIVES_RECALL, "benign_hard_min_n": MIN_BENIGN_HARD_N,
                          "benign_hard_fpr_max_wilson_upper": BENIGN_HARD_FPR_MAX_UPPER,
                          "max_escalation_share": MAX_ESCALATION_SHARE, "max_error_rate": MAX_ERROR_RATE},
        "checks": cs,
        "slices": slices,
        "decisions": dict(sorted(Counter(r["parsed"].get("decision") for r in records
                                         if "error" not in r["parsed"]).items())),
        "policy_decisions": dict(sorted(Counter(str(r["parsed"].get("policy_applied")) for r in records
                                                if "error" not in r["parsed"]).items())),
        "coverage_decisions": dict(sorted(Counter(str(r["parsed"].get("coverage_policy_applied")) for r in records
                                                  if "error" not in r["parsed"]).items())),
        "errors": dict(sorted(Counter(r["parsed"]["error"] for r in records if "error" in r["parsed"]).items())),
        "note": "A gate verdict is bound to this bundle_sha256. Any change to the policy, a pinned capability or "
                "the engine version produces a new hash and this verdict does not carry over. This script never "
                "promotes: `python -m training.gate promote --approved-by ...` does, and appends to "
                "artifacts/audit_log.jsonl.",
    }


# ---------------------------------------------------------------------- audit log, promote, rollback

def append_audit(entry: dict, log: Path = AUDIT_LOG) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_registry_v2_state(registry: Path) -> dict:
    """The validated pointers (active, previous, capabilities_root, status)."""
    from serving.registry import read_registry_v2
    return read_registry_v2(registry)


def write_registry_pointers(registry: Path, **changes) -> dict:
    """Change pointers, keeping every other key of the file (schema, comment) as it is. Crash-safe: the new state is
    written to a temporary file in the same directory, validated, and renamed over the registry, so the registry is
    either the old state or the new one, never a partial or invalid file."""
    raw = json.loads(registry.read_text())
    raw.update(changes)
    tmp = registry.with_name(f".{registry.name}.tmp")
    try:
        tmp.write_text(json.dumps(raw, indent=2) + "\n")
        read_registry_v2_state(tmp)  # the new state must validate before it replaces the registry
        os.replace(tmp, registry)
    finally:
        tmp.unlink(missing_ok=True)
    return raw


def _audited_pointer_move(registry: Path, entry: dict, log: Path, **changes) -> dict:
    """Record first, then act (review r2-n5): append the audit line, then move the pointer. If the move raises, a
    "failed" line for the same action follows and the exception propagates; the registry is unchanged."""
    append_audit(entry, log)
    try:
        write_registry_pointers(registry, **changes)
    except BaseException as exc:
        append_audit({"action": entry["action"], "status": "failed", "at": _now(), "actor": entry["actor"],
                      "bundle_sha256": entry["bundle_sha256"], "registry": entry["registry"],
                      "error": type(exc).__name__}, log)
        raise
    return entry


def promote(registry: Path, gate_report: Path, approved_by: str, candidate_policy: str | None = None,
            log: Path = AUDIT_LOG, allow_not_pass: bool = False) -> dict:
    """Point `active` at the candidate and keep the old active as `previous`. Owner action only.

    Refused, whatever --allow-not-pass says: a report whose run did not complete, ran with --limit or did not use a
    gate split; a report bound to another bundle; a candidate that is already the active bundle (promoting it would
    overwrite `previous` with the active bundle itself and lose the real rollback target)."""
    rep = json.loads(gate_report.read_text())
    state = read_registry_v2_state(registry)
    previous_active = state["active"]
    bundle = _bundle_of_policy(registry, previous_active)  # hashes only: promotion loads no model
    candidate = candidate_policy or rep["bundle"]["policy_path"] or state["active"]
    run = rep.get("run") or {}
    if run.get("split") not in GATE_SPLITS:
        raise SystemExit(f"the gate report ran on split {run.get('split')!r}, not a gate split {GATE_SPLITS}: "
                         f"nothing is promoted")
    if run.get("complete") is not True or run.get("limit") is not None or (
            run.get("n_items_evaluated") is not None and run.get("n_items_in_split") is not None
            and run["n_items_evaluated"] < run["n_items_in_split"]):
        raise SystemExit(f"the gate report's run is incomplete (complete={run.get('complete')}, "
                         f"limit={run.get('limit')}, {run.get('n_items_evaluated')}/{run.get('n_items_in_split')} "
                         f"items): finish the run before promoting; nothing is promoted")
    if rep["verdict"] != PASS and not allow_not_pass:
        raise SystemExit(f"gate verdict is {rep['verdict']}, not {PASS}: nothing is promoted. The owner may "
                         f"override with --allow-not-pass, which is recorded in the audit log.")
    if not approved_by.strip():
        raise SystemExit("--approved-by is required: a promotion names the person who approved it")
    # The report must describe the bundle that would be served.
    candidate_bundle = _bundle_of_policy(registry, candidate)
    if rep["bundle"]["bundle_sha256"] != candidate_bundle.bundle_sha256:
        raise SystemExit(f"the gate report is bound to bundle {rep['bundle']['bundle_sha256']}, but "
                         f"{candidate} is bundle {candidate_bundle.bundle_sha256}: re-run the gate")
    if candidate == previous_active or candidate_bundle.bundle_sha256 == bundle.bundle_sha256:
        raise SystemExit(f"{candidate} (bundle {candidate_bundle.bundle_sha256}) is already the active bundle: "
                         f"nothing to promote, and `previous` ({state['previous']}) is kept as it is")
    return _audited_pointer_move(registry, {
        "action": "promote", "at": _now(), "actor": approved_by,
        "bundle_sha256": candidate_bundle.bundle_sha256,
        "previous_bundle_sha256": bundle.bundle_sha256,
        "active": candidate, "previous": previous_active,
        "gate_report_path": str(gate_report.relative_to(ROOT)) if gate_report.is_relative_to(ROOT)
        else str(gate_report),
        "gate_verdict": rep["verdict"], "gate_split": rep["run"]["split"],
        "override_not_pass": bool(rep["verdict"] != PASS), "registry": str(registry),
    }, log, active=candidate, previous=previous_active, status="approved")


def _bundle_of_policy(registry: Path, rel: str):
    from serving.policy import bundle_sha256
    from serving.registry import load_policy_file

    state = read_registry_v2_state(registry)
    policy = load_policy_file(ROOT / "artifacts", rel)

    class _B:
        pass

    b = _B()
    b.policy = policy
    b.bundle_sha256 = bundle_sha256(policy)
    b.policy_path = rel
    b.status = state["status"]
    b.previous = state["previous"]
    b.policy_source = "registry_v2"
    return b


def rollback(registry: Path, approved_by: str, log: Path = AUDIT_LOG) -> dict:
    """Rollback is a pointer flip: `previous` becomes active and the rolled-back bundle becomes `previous`."""
    state = read_registry_v2_state(registry)
    if not state.get("previous"):
        raise SystemExit("the registry has no `previous` policy: there is nothing to roll back to")
    if not approved_by.strip():
        raise SystemExit("--approved-by is required: a rollback names the person who approved it")
    was_active, target = state["active"], state["previous"]
    before = _bundle_of_policy(registry, was_active)
    after = _bundle_of_policy(registry, target)
    return _audited_pointer_move(registry, {
        "action": "rollback", "at": _now(), "actor": approved_by,
        "bundle_sha256": after.bundle_sha256, "previous_bundle_sha256": before.bundle_sha256,
        "active": target, "previous": was_active, "registry": str(registry)}, log, active=target, previous=was_active)


# ---------------------------------------------------------------------- CLI

def evaluate(args) -> dict:
    if args.split == "frozen_test" and not args.frozen_test_approved_by:
        raise SystemExit("data/splits/frozen_test.jsonl is the final split: pass "
                         "--frozen-test-approved-by NAME to record who decided to look at it, or use --split val")
    audit_log = getattr(args, "audit_log", None) or AUDIT_LOG
    frozen = args.split == "frozen_test"
    if frozen:
        # Recorded BEFORE the split is opened, so a run that crashes, is killed or runs out of memory midway is
        # still visible in the audit log (status "started" without a matching "completed").
        append_audit({"action": "gate_on_frozen_test", "status": "started", "at": _now(),
                      "actor": args.frozen_test_approved_by, "registry": str(args.registry),
                      "gate_report_path": str(args.out), "limit": args.limit}, audit_log)
    # Any exception after "started" (scoring, --active-report, building or writing the report) records "failed",
    # so "started" alone means a killed process (review r2-n1).
    try:
        rep = _evaluate_after_audit(args)
    except BaseException as exc:
        if frozen:
            append_audit({"action": "gate_on_frozen_test", "status": "failed", "at": _now(),
                          "actor": args.frozen_test_approved_by, "gate_report_path": str(args.out),
                          "error": type(exc).__name__}, audit_log)
        raise
    if frozen:
        append_audit({"action": "gate_on_frozen_test", "status": "completed", "at": _now(),
                      "actor": args.frozen_test_approved_by, "bundle_sha256": rep["bundle"]["bundle_sha256"],
                      "verdict": rep["verdict"], "run_complete": rep["run"]["complete"],
                      "n_items_evaluated": rep["run"]["n_items_evaluated"], "gate_report_path": str(args.out)},
                     audit_log)
    return rep


def _evaluate_after_audit(args) -> dict:
    """Score the split, compare with --active-report, build and write the report."""
    records, meta, bundle, wall, n_new = run_bundle(args.registry, args.split, args.out, args.max_seconds,
                                                     args.limit, args.splits_dir)
    active_records = active_bundle = None
    if args.active_report:
        other = json.loads(args.active_report.read_text())
        active_progress = args.active_report.with_suffix(".progress.jsonl")
        if active_progress.is_file():
            want = other["bundle"]["bundle_sha256"]
            active_records = [json.loads(line) for line in active_progress.read_text().splitlines()
                              if line.strip() and json.loads(line).get("bundle_sha256") == want]
            active_bundle = _bundle_of_policy(args.registry, other["bundle"]["policy_path"])
    adaptive_asr = None
    if getattr(args, "adaptive_asr", None):
        adaptive_asr = json.loads(args.adaptive_asr.read_text())
    rep = build_report(records, meta, bundle, wall, n_new, args.registry, active_records, active_bundle,
                       args.frozen_test_approved_by, adaptive_asr=adaptive_asr)
    write_json(args.out, rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Release gate for a bundle. Never promotes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="evaluate the bundle a registry serves; writes a report, promotes nothing")
    ev.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry.json")
    ev.add_argument("--split", default="val", choices=GATE_SPLITS)
    ev.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--max-seconds", type=float, default=None, help="stop after this long; resumable")
    ev.add_argument("--limit", type=int, default=None, help="first N items (smoke runs only)")
    ev.add_argument("--active-report", type=Path, default=None,
                    help="a gate report of the active bundle, for the paired recall check")
    ev.add_argument("--frozen-test-approved-by", default=None,
                    help="required with --split frozen_test: who decided to look at the final split")
    ev.add_argument("--adaptive-asr", type=Path, default=None,
                    help="an adaptive-attacker gate_input.json to surface in the report (report-only; "
                         "no threshold, see docs/design/adaptive-attacker.md §8)")
    ev.add_argument("--audit-log", type=Path, default=AUDIT_LOG,
                    help="where frozen-test runs are recorded (default artifacts/audit_log.jsonl)")

    pr = sub.add_parser("promote", help="owner action: point active at a gated bundle, keeping the old as previous")
    pr.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry.json")
    pr.add_argument("--gate-report", type=Path, required=True)
    pr.add_argument("--approved-by", required=True)
    pr.add_argument("--policy", default=None, help="the policy path to promote (default: the report's bundle)")
    pr.add_argument("--allow-not-pass", action="store_true",
                    help="promote despite a non-PASS verdict; recorded in the audit log")

    rb = sub.add_parser("rollback", help="owner action: flip the pointer back to `previous`")
    rb.add_argument("--registry", type=Path, default=ROOT / "artifacts" / "registry.json")
    rb.add_argument("--approved-by", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "evaluate":
        rep = evaluate(a)
        print(json.dumps({"verdict": rep["verdict"], "complete": rep["run"]["complete"],
                          "n_items_evaluated": rep["run"]["n_items_evaluated"],
                          "bundle_sha256": rep["bundle"]["bundle_sha256"],
                          "checks": {k: v["verdict"] for k, v in rep["checks"].items()}}, indent=1))
        return 0 if rep["run"]["complete"] else 1
    if a.cmd == "promote":
        print(json.dumps(promote(a.registry, a.gate_report, a.approved_by, a.policy,
                                 allow_not_pass=a.allow_not_pass), indent=1))
        return 0
    print(json.dumps(rollback(a.registry, a.approved_by), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
