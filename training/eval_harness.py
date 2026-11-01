"""Evaluation harness: precision / recall / FPR with Wilson 95% CIs and counts, overall and per slice.

Backends
  local  Stage 1 ONNX artifact (sha256-verified), chunked + max-aggregated, decision = score >= threshold
  http   a gateway: POST {url}/classify  {"text": ..., "request_id": ...}
         response {"request_id", "decision": "allow"|"flag", "decided_by", "policy_applied": null|"fail_open"|
         "fail_closed", "results": [DetectorResult dicts]}

    python -m training.eval_harness --backend local --artifact artifacts/candidates/stage1-tfidf-v2 \
        --split val --out training/results/stage1-tfidf-v2_val.json
    python -m training.eval_harness --backend http --url http://127.0.0.1:8000 --split val --out ...

What counts as a model prediction (http backend)
  Every response is validated with contracts.ClassifyResponse (v1.2); anything that does not validate is an
  error ("bad response"). Only a response whose active results all have status "ok", whose policy_applied is
  null and whose coverage_policy_applied is not flag / allow / quarantine is a model prediction (predicted =
  decision == "flag"); deep_scan is a model decision. A response decided by a failure policy (timeout / error /
  skipped -> fail_open or fail_closed) or by a coverage policy is NOT counted in the model-only precision /
  recall / FPR; it is reported under `policy_decisions` (counts by status, by policy_applied and by decision x
  label), overall and per slice, and it IS counted in the as-served view, which is the gate view.
  Per slice, `served` counts stages_run combinations (escalation to Stage 2), coverage_policy_applied (by policy,
  decision and label) and failure_reason of failed active results, with overloaded (load shedding) separated
  from timeouts and exceptions (`served.failures`), plus shadow failure reasons.
  Transport failures (httpx errors, after bounded retries), non-200 responses and malformed responses are
  recorded under `errors` and the run continues. `truncated` (any input content not scored by an active
  detector) is counted over every response that carried results, overall and per slice; truncated model
  predictions stay in the metrics, since that is the decision the gateway made. A truncated observe or shadow
  result (e.g. the observe-only Stage 1 of D23 hitting its deadline) is counted apart, as
  `truncated.n_shadow_truncated`, and never as a truncation of the decision.
  v1.3: every response's `bundle_sha256` is collected, and the run records the bundle(s) it measured under
  `bundles` (a gate verdict is bound to one bundle_sha256, so a run that saw more than one is not a gate run).

Partial output: with --out, the result JSON is rewritten every --checkpoint-every items and once more at
the end, also when the run is interrupted (`complete: false`, `aborted: <reason>`).

Slices: overall, per source, benign_hard (and per benign-hard family), Stage 2 overlap groups, language.
INSUFFICIENT_EVIDENCE is reported per the RFC gate table: recall needs >= 200 positives, benign-hard FPR
needs >= 100 samples. Those minimums are attached to every slice (gate_minimums) so a slice too small for
a gate claim is labeled as such, not silently reported.

Latency fields are client-side wall time per item under whatever else ran on the host at the time; they
are not benchmarks (see `latency_conditions` in the output).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Protocol

from contracts import ClassifyResponse, DetectorResult
from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file, write_json
from training.metrics import rate, summarize

MIN_POSITIVES_RECALL = 200  # RFC: cascade recall check -> insufficient evidence when positives < 200
MIN_BENIGN_HARD_N = 100  # RFC: benign-hard FPR check -> insufficient evidence when samples < 100
BENIGN_HARD_FPR_MAX_UPPER = 0.05  # RFC: Wilson 95% upper bound <= 5%


class Backend(Protocol):
    name: str

    def predict(self, text: str, request_id: str) -> dict: ...

    def describe(self) -> dict: ...


class LocalOnnxBackend:
    name = "local_stage1_onnx"

    def __init__(self, artifact_dir: Path):
        from training.stage1_onnx import Stage1Onnx

        self.dir = Path(artifact_dir)
        self.model = Stage1Onnx(self.dir)
        self.threshold = self.model.manifest.threshold
        self.gray_zone = self.model.manifest.gray_zone

    def predict(self, text: str, request_id: str) -> dict:
        r = self.model.score(text)
        lo, hi = self.gray_zone
        return {"predicted": int(r["score"] >= self.threshold), "score": r["score"], "latency_ms": r["latency_ms"],
                "n_chunks": r["n_chunks"],
                "stage1_zone": "low" if r["score"] < lo else "high" if r["score"] > hi else "gray"}

    def describe(self) -> dict:
        m = self.model.manifest
        return {"backend": self.name, "artifact_dir": str(self.dir.resolve().relative_to(ROOT)) if ROOT in self.dir.resolve().parents else str(self.dir),
                "detector_id": m.detector_id, "detector_version": m.detector_version, "sha256": m.sha256,
                "threshold": m.threshold, "gray_zone": list(m.gray_zone),
                "decision_rule": "score >= threshold (Stage 1 alone; gray zone is reported, not escalated)"}


class GatewayResponseError(ValueError):
    pass


# Coverage-policy outcomes that replace a model decision on unscored content. "deep_scan" means the gateway scored
# the rest of the input, so the decision is still a model decision (counted, and kept in the metrics).
COVERAGE_POLICY_DECISIONS = ("flag", "allow", "quarantine")


def failure_reason(r: DetectorResult) -> str | None:
    """failure_reason of a non-ok result; v1.1 results (no field) map timeout -> timeout, error -> unspecified."""
    if r.status == "ok":
        return None
    if r.failure_reason:
        return r.failure_reason
    return {"timeout": "timeout", "error": "unspecified_error", "skipped": "skipped"}[r.status]


def parse_gateway_response(payload: dict, request_id: str | None = None) -> dict:
    """Validate a /classify response with contracts.ClassifyResponse (v1.2). Raises GatewayResponseError.

    `model_prediction` is True only when every active result has status "ok", no failure policy was applied and
    no coverage policy replaced the decision (deep_scan does not); only then is `predicted` a model prediction.
    """
    if not isinstance(payload, dict):
        raise GatewayResponseError("response is not a JSON object")
    try:
        resp = ClassifyResponse.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError
        raise GatewayResponseError(f"not a valid ClassifyResponse: {str(exc)[:300]}") from exc
    if request_id is not None and resp.request_id != request_id:
        raise GatewayResponseError(f"request_id mismatch: sent {request_id!r}, got {resp.request_id!r}")
    policy = resp.policy_applied
    coverage = resp.coverage_policy_applied
    active = [r for r in resp.results if r.role == "active"]
    # Observe steps (v1.3 role "observe"; "shadow" from a v1.2 gateway) and the shadow mirror never took part in
    # the decision, so they are counted apart and never as a prediction, a failure or a truncation (D23, D25).
    shadow = [r for r in resp.results if r.role in ("observe", "shadow")]
    statuses = [r.status for r in active]
    non_ok = [st for st in statuses if st != "ok"]
    # Item status: "ok" only if every active detector produced a score. A policy without a failed result
    # is inconsistent and is not trusted as a model prediction either.
    status = non_ok[0] if non_ok else ("ok" if policy is None else "inconsistent_policy")
    coverage_decided = coverage in COVERAGE_POLICY_DECISIONS
    if coverage_decided and status == "ok":
        status = f"coverage_{coverage}"  # reported under policy_decisions.by_status, never as a model prediction
    ok_scores = [r.score for r in active if r.status == "ok"]
    # stages_run is authoritative (v1.2); a v1.1 gateway that omits it falls back to the active results' stages.
    stages = list(resp.stages_run) or [r.stage for r in active]
    return {
        "predicted": int(resp.decision == "flag"),
        "model_prediction": status == "ok" and policy is None and not coverage_decided and bool(active),
        "status": status if active else "no_active_result",
        "decision": resp.decision,
        "decided_by": resp.decided_by,
        "policy_applied": policy,
        "coverage_policy_applied": coverage,
        "score": max(ok_scores) if ok_scores else None,
        "stages": stages,
        "stages_run_reported": bool(resp.stages_run),
        "statuses": statuses,
        "failure_reasons": [failure_reason(r) for r in active if r.status != "ok"],
        "shadow_failure_reasons": [failure_reason(r) for r in shadow if r.status != "ok"],
        "n_shadow": len(shadow),
        # v1.3: which bundle served this response (None from a pre-1.3 gateway).
        "policy_id": resp.policy_id,
        "policy_version": resp.policy_version,
        "bundle_sha256": resp.bundle_sha256,
        "unscored_by_any_step": resp.unscored_by_any_step,
        "escalation_reason": resp.escalation_reason,
        "escalated": "stage2" in stages,
        # Only active results count: an observe step (D23's Stage 1) that hit its deadline did not take part in
        # the decision, so it must not mark the item truncated in the release-gate counts.
        "truncated": any(r.truncated for r in active),
        "shadow_truncated": any(r.truncated for r in shadow),
        "n_results": len(resp.results),
    }


RETRY_STATUS = (502, 503, 504)


class HttpGatewayBackend:
    name = "http_gateway"

    def __init__(self, url: str, timeout_s: float = 30.0, transport=None, client=None, retries: int = 2,
                 backoff_s: float = 0.5, sleep: Callable[[float], None] = time.sleep):
        import httpx

        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.client = client or httpx.Client(timeout=timeout_s, transport=transport, trust_env=False)

    def predict(self, text: str, request_id: str) -> dict:
        """Never raises for transport / HTTP / shape problems: they come back as {"error": ...}.

        Transport errors and 502/503/504 are retried up to `retries` times (bounded, linear backoff);
        the same request_id is reused so the gateway logs show the retry.
        """
        import httpx

        t0 = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = self.client.post(f"{self.url}/classify", json={"text": text, "request_id": request_id})
            except httpx.HTTPError as exc:
                if attempts <= self.retries:
                    self.sleep(self.backoff_s * attempts)
                    continue
                return {"error": f"transport: {type(exc).__name__}", "attempts": attempts,
                        "latency_ms": (time.perf_counter() - t0) * 1e3}
            if resp.status_code in RETRY_STATUS and attempts <= self.retries:
                self.sleep(self.backoff_s * attempts)
                continue
            break
        latency = (time.perf_counter() - t0) * 1e3
        if resp.status_code != 200:
            return {"error": f"http {resp.status_code}", "attempts": attempts, "latency_ms": latency}
        try:
            out = parse_gateway_response(resp.json(), request_id)
        except (GatewayResponseError, ValueError) as exc:
            return {"error": f"bad response: {str(exc)[:200]}", "attempts": attempts, "latency_ms": latency}
        out["latency_ms"] = latency
        out["attempts"] = attempts
        return out

    def describe(self) -> dict:
        return {"backend": self.name, "url": self.url, "timeout_s": self.timeout_s, "retries": self.retries,
                "decision_rule": "model prediction = gateway decision == 'flag', counted only when every active "
                                 "result has status 'ok' and policy_applied is null; failure-policy decisions "
                                 "are reported under policy_decisions, not as predictions"}


def slices_for(row: dict) -> list[str]:
    s = ["overall", f"source={row['source']}"]
    if row["source"] == "benign_hard":
        s += ["benign_hard", f"benign_hard_family={row['family']}"]
    s.append(f"stage2_overlap={row.get('stage2_overlap')}")
    s.append("lang=en" if row.get("lang") == "en" else "lang=non_en")
    return s


def evidence(name: str, m: dict) -> dict:
    out = {}
    if m["positives"]:
        out["recall"] = ("OK" if m["positives"] >= MIN_POSITIVES_RECALL else "INSUFFICIENT_EVIDENCE")
    if name == "benign_hard" or name.startswith("benign_hard_family="):
        n = m["negatives"]
        if n < MIN_BENIGN_HARD_N:
            out["fpr"] = "INSUFFICIENT_EVIDENCE"
        else:
            ub = m["fpr"]["ci95"][1]
            out["fpr"] = "PASS" if ub <= BENIGN_HARD_FPR_MAX_UPPER else "FAIL"
            out["fpr_rule"] = f"Wilson 95% upper bound {ub:.4f} <= {BENIGN_HARD_FPR_MAX_UPPER}"
    return out


def as_served(m: dict, slice_name: str = "") -> dict:
    """Metrics over every non-error item using the decision the gateway actually served.

    Model predictions keep their own outcome; failure-policy decisions count as served
    ("flag" -> predicted injection, "allow" -> predicted benign). This is what a user experiences:
    under fail_open a timed-out injection is let through, so it is a miss here even though it is
    excluded from the model-only metrics. Gate checks use this view.
    """
    c = dict(m["confusion"])
    for key, v in m.get("policy_decisions", {}).get("by_decision_and_label", {}).items():
        decision, label = key.split(":label=")
        y = int(label)
        if decision == "flag":
            c["tp" if y else "fp"] += v
        else:
            c["fn" if y else "tn"] += v
    out = {"n": sum(c.values()), "positives": c["tp"] + c["fn"], "negatives": c["fp"] + c["tn"], "confusion": c,
           "precision": rate(c["tp"], c["tp"] + c["fp"]), "recall": rate(c["tp"], c["tp"] + c["fn"]),
           "fpr": rate(c["fp"], c["fp"] + c["tn"])}
    out["evidence"] = evidence(slice_name, out)
    return out


def add_as_served(result: dict) -> dict:
    """Add the as-served view to every slice of an existing results dict (idempotent)."""
    for name, m in result.get("slices", {}).items():
        m["as_served"] = as_served(m, name)
    result["gate_view"] = "as_served"
    return result


class _SliceAcc:
    def __init__(self):
        self.ys: list[int] = []
        self.ps: list[int] = []
        self.policy = Counter()  # (status, policy_applied, decision, label)
        self.errors = 0
        self.with_results = 0
        self.truncated = 0
        self.truncated_predictions = 0
        self.shadow_truncated = 0
        self.stages = Counter()  # "+".join(stages_run)
        self.coverage = Counter()  # (coverage_policy_applied, decision, label)
        self.reasons = Counter()  # failure_reason of failed active results
        self.shadow_reasons = Counter()


def _served_block(a: "_SliceAcc") -> dict | None:
    """v1.2 routing / coverage / failure counts. None for backends that report none of them (local Stage 1)."""
    if not a.with_results:
        return None
    reasons = dict(sorted(a.reasons.items()))
    overloaded = a.reasons.get("overloaded", 0)
    return {
        "n_responses": a.with_results,
        "stages_run": dict(sorted(a.stages.items())),
        "escalated_to_stage2": sum(v for k, v in a.stages.items() if "stage2" in k.split("+")),
        "coverage_policy_applied": {
            "by_policy": dict(sorted(_sum_by(a.coverage, 0).items())),
            "by_policy_decision_and_label": {f"{p}:{d}:label={y}": v for (p, d, y), v in sorted(a.coverage.items(),
                                                                                                 key=str)}},
        "failure_reason": reasons,
        # Load shedding is a capacity signal, not a detector fault: the gate's error-rate check excludes it.
        "failures": {"overloaded": overloaded,
                     "timeout_or_exception": sum(v for k, v in a.reasons.items() if k != "overloaded")},
        "shadow_failure_reason": dict(sorted(a.shadow_reasons.items())),
    }


def _sum_by(c: Counter, idx: int) -> Counter:
    out = Counter()
    for k, v in c.items():
        out[str(k[idx])] += v
    return out


def _policy_block(c: Counter) -> dict:
    by_status, by_policy, by_decision_label = Counter(), Counter(), Counter()
    for (st, pol, dec, y), v in c.items():
        by_status[st] += v
        by_policy[str(pol)] += v
        by_decision_label[f"{dec}:label={y}"] += v
    return {"n": sum(c.values()), "by_status": dict(sorted(by_status.items())),
            "by_policy_applied": dict(sorted(by_policy.items())),
            "by_decision_and_label": dict(sorted(by_decision_label.items()))}


def evaluate(rows: Iterable[dict], backend: Backend, progress: Callable[[int], None] | None = None,
             checkpoint: Callable[[dict], None] | None = None, checkpoint_every: int = 500) -> dict:
    """Score every row. Per-item failures are recorded and the run continues; an interruption (or any
    unexpected exception) stops the loop and the partial result is returned with complete=False."""
    rows = list(rows)
    acc: dict[str, _SliceAcc] = {}
    errors = Counter()
    lat = []
    zones = Counter()
    bundles = Counter()  # bundle_sha256 -> n responses (v1.3); a gate verdict is bound to exactly one
    policies = Counter()
    n_done = 0
    aborted = None

    def snapshot(complete: bool) -> dict:
        return _assemble(rows, n_done, acc, errors, lat, zones, complete, aborted, bundles, policies)

    try:
        for i, row in enumerate(rows):
            rid = f"eval-{row.get('id', i)}-{uuid.uuid4().hex[:8]}"
            try:
                r = backend.predict(row["text"], rid)
            except Exception as exc:  # a backend bug must not lose the whole run
                r = {"error": f"backend exception: {type(exc).__name__}", "latency_ms": 0.0}
            n_done = i + 1
            lat.append(r.get("latency_ms", 0.0))
            if r.get("bundle_sha256"):
                bundles[r["bundle_sha256"]] += 1
            if r.get("policy_id"):
                policies[f"{r['policy_id']}@{r['policy_version']}"] += 1
            names = slices_for(row)
            for s in names:
                a = acc.setdefault(s, _SliceAcc())
                if "error" in r:
                    a.errors += 1
                    continue
                if "truncated" in r:
                    a.with_results += 1
                    a.truncated += int(bool(r["truncated"]))
                    a.shadow_truncated += int(bool(r.get("shadow_truncated", False)))
                if "stages" in r:
                    a.stages["+".join(r["stages"]) or "none"] += 1
                    a.coverage[(r.get("coverage_policy_applied"), r.get("decision"), row["label"])] += 1
                    for fr in r.get("failure_reasons", []):
                        a.reasons[fr] += 1
                    for fr in r.get("shadow_failure_reasons", []):
                        a.shadow_reasons[fr] += 1
                if r.get("model_prediction", True):
                    a.ys.append(row["label"]); a.ps.append(r["predicted"])
                    a.truncated_predictions += int(bool(r.get("truncated", False)))
                    if "stage1_zone" in r:
                        zones[(s, r["stage1_zone"], row["label"])] += 1
                else:
                    a.policy[(r.get("status"), r.get("policy_applied"), r.get("decision"), row["label"])] += 1
            if "error" in r:
                errors[r["error"]] += 1
            if progress and i % 500 == 0:
                progress(i)
            if checkpoint and n_done % checkpoint_every == 0:
                checkpoint(snapshot(False))
    except (Exception, KeyboardInterrupt) as exc:
        aborted = f"{type(exc).__name__}: {str(exc)[:200]}"
    return snapshot(aborted is None and n_done == len(rows))


def _assemble(rows, n_done, acc, errors, lat, zones, complete, aborted, bundles=None, policies=None) -> dict:
    import numpy as np

    slices = {}
    for name in sorted(acc, key=lambda k: (k != "overall", k)):
        a = acc[name]
        m = summarize(a.ys, a.ps)
        m["evidence"] = evidence(name, m)
        if zones:
            # Stage 1 routing if this artifact ran as cascade Stage 1: low -> final allow, gray -> Stage 2,
            # high -> final flag. Counts per (zone, true label).
            m["stage1_zones"] = {f"{z}:label={y}": zones[(name, z, y)] for z in ("low", "gray", "high")
                                 for y in (0, 1) if zones[(name, z, y)]}
        m["n_items"] = len(a.ys) + sum(a.policy.values()) + a.errors
        m["n_errors"] = a.errors
        m["policy_decisions"] = _policy_block(a.policy)
        m["as_served"] = as_served(m, name)
        if a.with_results:
            m["truncated"] = {"n_responses_with_results": a.with_results, "n_truncated": a.truncated,
                              "n_truncated_model_predictions": a.truncated_predictions,
                              "n_shadow_truncated": a.shadow_truncated}
        served = _served_block(a)
        if served and a.stages:
            m["served"] = served
        slices[name] = m
    la = np.array(lat) if lat else np.zeros(1)
    overall = slices.get("overall")
    bundles = bundles or Counter()
    return {
        "gate_view": "as_served",
        "complete": complete,
        # v1.3: the bundle(s) these metrics describe. A gate verdict is bound to one bundle_sha256; more than one
        # here (a reload mid-run) means the run does not describe a single bundle.
        "bundles": {"by_bundle_sha256": dict(sorted(bundles.items())),
                    "bundle_sha256": next(iter(bundles)) if len(bundles) == 1 else None,
                    "policies": dict(sorted((policies or Counter()).items())),
                    "single_bundle": len(bundles) == 1},
        "aborted": aborted,
        "n_items": len(rows),
        "n_processed": n_done,
        "n_model_predictions": overall["n"] if overall else 0,
        "n_policy_decisions": overall["policy_decisions"]["n"] if overall else 0,
        "n_errors": sum(errors.values()),
        "errors": dict(errors),
        "slices": slices,
        "client_latency_ms": {"n": len(lat), "p50": float(np.percentile(la, 50)), "p95": float(np.percentile(la, 95)),
                              "p99": float(np.percentile(la, 99))},
    }


LATENCY_CONDITIONS = ("client-side wall time per item, sequential, on a shared host where other CPU-heavy "
                      "jobs may have been running; not a benchmark")


def run(split: str, backend: Backend, splits_dir: Path = SPLITS_DIR, out: Path | None = None,
        checkpoint_every: int = 500, latency_conditions: str = LATENCY_CONDITIONS) -> dict:
    path = splits_dir / f"{split}.jsonl"
    header = {
        "split": split, "split_sha256": sha256_file(path), "backend": backend.describe(),
        "gate_minimums": {"recall_min_positives": MIN_POSITIVES_RECALL, "benign_hard_min_n": MIN_BENIGN_HARD_N,
                          "benign_hard_fpr_max_wilson_upper": BENIGN_HARD_FPR_MAX_UPPER},
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "command": "python -m training.eval_harness " + " ".join(sys.argv[1:]),
        "latency_conditions": latency_conditions,
    }

    def write(res: dict) -> None:
        if out is not None:
            write_json(out, {**header, **res})

    res = evaluate(read_jsonl(path), backend, checkpoint=write, checkpoint_every=checkpoint_every)
    write(res)
    return {**header, **res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "http"])
    ap.add_argument("--add-as-served", type=Path, metavar="RESULTS_JSON",
                    help="add the as-served view to an existing results file in place and exit")
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--timeout-s", type=float, default=30.0, help="http: client timeout per attempt")
    ap.add_argument("--retries", type=int, default=2, help="http: retries per item on transport errors / 502-504")
    ap.add_argument("--split", default="val", choices=["train", "val", "frozen_test"])
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--latency-conditions", default=LATENCY_CONDITIONS,
                    help="free-text description of the host load during this run, stored with the latency fields")
    a = ap.parse_args(argv)
    if a.add_as_served:
        res = add_as_served(json.loads(a.add_as_served.read_text()))
        write_json(a.add_as_served, res)
        o = res["slices"].get("overall", {}).get("as_served", {})
        print(json.dumps({"as_served_recall": o.get("recall"), "as_served_fpr": o.get("fpr")}, indent=1))
        return 0
    if not a.backend:
        ap.error("--backend is required unless --add-as-served is given")
    if a.backend == "local":
        if not a.artifact:
            ap.error("--artifact is required for --backend local")
        backend: Backend = LocalOnnxBackend(a.artifact)
    else:
        if not a.url:
            ap.error("--url is required for --backend http")
        backend = HttpGatewayBackend(a.url, timeout_s=a.timeout_s, retries=a.retries)
    res = run(a.split, backend, a.splits_dir, a.out, a.checkpoint_every, a.latency_conditions)
    o = res["slices"].get("overall", {})
    print(json.dumps({"split": a.split, "complete": res["complete"], "n_model_predictions": res["n_model_predictions"],
                      "n_policy_decisions": res["n_policy_decisions"], "errors": res["n_errors"],
                      "precision": o.get("precision"), "recall": o.get("recall"), "fpr": o.get("fpr")}, indent=1))
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
