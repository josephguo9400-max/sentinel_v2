"""Evaluation harness: precision / recall / FPR with Wilson 95% CIs and counts, overall and per slice.

Backends
  local  Stage 1 ONNX artifact (sha256-verified), chunked + max-aggregated, decision = score >= threshold
  http   a gateway: POST {url}/classify  {"text": ..., "request_id": ...}
         response {"decision": "allow"|"flag", "results": [DetectorResult dicts]}; predicted = decision == "flag"

    python -m training.eval_harness --backend local --artifact artifacts/candidates/stage1-tfidf-v1 \
        --split val --out training/results/stage1-tfidf-v1_val.json
    python -m training.eval_harness --backend http --url http://127.0.0.1:8000 --split val --out ...

Slices: overall, per source, benign_hard (and per benign-hard family), Stage 2 overlap groups.
INSUFFICIENT_EVIDENCE is reported per the RFC gate table: recall needs >= 200 positives, benign-hard FPR
needs >= 100 samples. Those minimums are attached to every slice (gate_minimums) so a slice too small for
a gate claim is labeled as such, not silently reported.
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

from contracts import DetectorResult
from training.common import ROOT, SPLITS_DIR, read_jsonl, sha256_file, write_json
from training.metrics import summarize

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


def parse_gateway_response(payload: dict) -> dict:
    """Validate the gateway response shape. Raises GatewayResponseError on anything unexpected."""
    if not isinstance(payload, dict):
        raise GatewayResponseError("response is not a JSON object")
    decision = payload.get("decision")
    if decision not in ("allow", "flag"):
        raise GatewayResponseError(f"decision must be 'allow' or 'flag', got {decision!r}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise GatewayResponseError("results must be a list")
    parsed = []
    for i, r in enumerate(results):
        try:
            parsed.append(DetectorResult.model_validate(r))
        except Exception as exc:  # pydantic.ValidationError
            raise GatewayResponseError(f"results[{i}] is not a valid DetectorResult: {exc}") from exc
    active = [r for r in parsed if r.role == "active"]
    ok_scores = [r.score for r in active if r.status == "ok"]
    return {
        "predicted": int(decision == "flag"),
        "decision": decision,
        "score": max(ok_scores) if ok_scores else None,
        "stages": [r.stage for r in active],
        "statuses": [r.status for r in active],
        "escalated": any(r.stage == "stage2" for r in active),
        "truncated": any(r.truncated for r in parsed),
        "n_results": len(parsed),
    }


class HttpGatewayBackend:
    name = "http_gateway"

    def __init__(self, url: str, timeout_s: float = 30.0, transport=None, client=None):
        import httpx

        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout_s, transport=transport, trust_env=False)

    def predict(self, text: str, request_id: str) -> dict:
        t0 = time.perf_counter()
        resp = self.client.post(f"{self.url}/classify", json={"text": text, "request_id": request_id})
        latency = (time.perf_counter() - t0) * 1e3
        if resp.status_code != 200:
            return {"error": f"http {resp.status_code}", "latency_ms": latency}
        try:
            out = parse_gateway_response(resp.json())
        except (GatewayResponseError, ValueError) as exc:
            return {"error": f"bad response: {exc}", "latency_ms": latency}
        out["latency_ms"] = latency
        return out

    def describe(self) -> dict:
        return {"backend": self.name, "url": self.url,
                "decision_rule": "gateway decision == 'flag' counts as predicted injection"}


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


def evaluate(rows: Iterable[dict], backend: Backend, progress: Callable[[int], None] | None = None) -> dict:
    rows = list(rows)
    per_slice: dict[str, tuple[list[int], list[int]]] = {}
    errors = Counter()
    lat = []
    zones = Counter()
    for i, row in enumerate(rows):
        rid = f"eval-{row.get('id', i)}-{uuid.uuid4().hex[:8]}"
        r = backend.predict(row["text"], rid)
        lat.append(r.get("latency_ms", 0.0))
        if "error" in r:
            errors[r["error"]] += 1
            continue
        for s in slices_for(row):
            ys, ps = per_slice.setdefault(s, ([], []))
            ys.append(row["label"]); ps.append(r["predicted"])
            if "stage1_zone" in r:
                zones[(s, r["stage1_zone"], row["label"])] += 1
        if progress and i % 500 == 0:
            progress(i)
    slices = {}
    for name in sorted(per_slice, key=lambda k: (k != "overall", k)):
        ys, ps = per_slice[name]
        m = summarize(ys, ps)
        m["evidence"] = evidence(name, m)
        if zones:
            # Stage 1 routing if this artifact ran as cascade Stage 1: low -> final allow, gray -> Stage 2,
            # high -> final flag. Counts per (zone, true label).
            m["stage1_zones"] = {f"{z}:label={y}": zones[(name, z, y)] for z in ("low", "gray", "high")
                                 for y in (0, 1) if zones[(name, z, y)]}
        slices[name] = m
    import numpy as np

    la = np.array(lat) if lat else np.zeros(1)
    out = {
        "n_items": len(rows),
        "n_errors": sum(errors.values()),
        "errors": dict(errors),
        "slices": slices,
        "client_latency_ms": {"n": len(lat), "p50": float(np.percentile(la, 50)), "p95": float(np.percentile(la, 95)),
                              "p99": float(np.percentile(la, 99))},
    }
    return out


def run(split: str, backend: Backend, splits_dir: Path = SPLITS_DIR) -> dict:
    path = splits_dir / f"{split}.jsonl"
    res = evaluate(read_jsonl(path), backend)
    return {
        "split": split, "split_sha256": sha256_file(path), "backend": backend.describe(),
        "gate_minimums": {"recall_min_positives": MIN_POSITIVES_RECALL, "benign_hard_min_n": MIN_BENIGN_HARD_N,
                          "benign_hard_fpr_max_wilson_upper": BENIGN_HARD_FPR_MAX_UPPER},
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "command": "python -m training.eval_harness " + " ".join(sys.argv[1:]),
        **res,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "http"], required=True)
    ap.add_argument("--artifact", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--split", default="val", choices=["train", "val", "frozen_test"])
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.backend == "local":
        if not a.artifact:
            ap.error("--artifact is required for --backend local")
        backend: Backend = LocalOnnxBackend(a.artifact)
    else:
        if not a.url:
            ap.error("--url is required for --backend http")
        backend = HttpGatewayBackend(a.url)
    res = run(a.split, backend, a.splits_dir)
    if a.out:
        write_json(a.out, res)
    o = res["slices"]["overall"]
    print(json.dumps({"split": a.split, "n": o["n"], "precision": o["precision"], "recall": o["recall"],
                      "fpr": o["fpr"], "errors": res["n_errors"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
