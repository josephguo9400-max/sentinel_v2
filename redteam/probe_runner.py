"""Send every probe to a gateway's /classify and report, per placement, how the two attack placements
fare as served -- before and after the cascade fix the serving agent is building.

The runner never re-thresholds: it uses redteam.gateway_client.GatewayClient, so every response is
validated against contracts.ClassifyResponse and the gateway's own `decision` is trusted. It works
against any gateway:

  * the current gateway, whose decision ignores `truncated` (an injection in dropped content is
    silently allowed);
  * the new cascade gateway, which applies a coverage policy to unscored content.

For each placement it reports, with Wilson 95% intervals and raw counts:

  * detection rate "as served": the fraction of attack probes the gateway returned `decision=flag`
    for, whatever the reason -- a coverage-policy flag counts here, because as served it does stop
    the email;
  * coverage-policy flags: the subset of those detections that came only from the coverage policy
    (`coverage_policy_applied` set and no scored chunk was positive), reported separately so a
    coverage-policy catch is never mistaken for the classifier scoring the injection;
  * classifier-score flags: detections where a scored chunk was actually positive;
  * benign-control flag rate: the false-positive rate on the matched controls;
  * truncated count and the distribution of `stages_run`.

Run it against the fake gateway in tests, or against a live gateway URL (the owner does the live
run against a real model; this milestone only wires it up). Fallback verdicts (gateway unreachable)
are reported under `unavailable` and excluded from the flag-rate denominators, so a dead gateway is
never read as perfect detection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .classifier import Classifier, ClassifierVerdict
from .gateway_client import GatewayClient
from .probes import Probe, load_probes


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials. (0, 0) for n == 0."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class ProbeOutcome:
    probe_id: str
    placement: str
    kind: str
    offset_label: str
    total_tokens: int
    decision: str
    gateway_status: str
    fallback: bool
    flag_cause: str | None
    coverage_policy_applied: str | None
    truncated: bool
    stages_run: tuple[str, ...]
    scored_positive: bool  # any scored chunk predicted injection
    # per-stage detail, for diagnosing a decision: {stage: {"score", "n_chunks", "truncated", "status"}}
    stage_scores: dict = field(default_factory=dict)

    @classmethod
    def from_verdict(cls, probe: Probe, v: ClassifierVerdict) -> "ProbeOutcome":
        scored_positive = any(r.status == "ok" and r.predicted_injection for r in v.results)
        return cls(probe.probe_id, probe.placement, probe.kind, probe.offset_label, probe.total_tokens,
                   v.decision, v.gateway_status, v.fallback, v.flag_cause, v.coverage_policy_applied,
                   v.truncated, tuple(v.stages_run), scored_positive,
                   {(r.stage or r.detector_id): {"score": r.score, "n_chunks": r.n_chunks, "truncated": r.truncated,
                                                 "status": r.status} for r in v.results})


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}


@dataclass
class PlacementReport:
    placement: str
    outcomes: list[ProbeOutcome] = field(default_factory=list)

    def summary(self) -> dict:
        attacks = [o for o in self.outcomes if o.kind == "attack"]
        controls = [o for o in self.outcomes if o.kind == "control"]
        a_usable = [o for o in attacks if not o.fallback]
        c_usable = [o for o in controls if not o.fallback]
        a_flag = [o for o in a_usable if o.decision == "flag"]
        coverage = [o for o in a_flag if o.flag_cause == "coverage_policy"]
        score_flag = [o for o in a_flag if o.scored_positive]
        stages = {}
        for o in a_usable:
            key = ",".join(o.stages_run) or "none"
            stages[key] = stages.get(key, 0) + 1
        return {
            "placement": self.placement,
            "n_attack": len(attacks), "n_control": len(controls),
            "detection_rate_as_served": _rate(len(a_flag), len(a_usable)),
            "coverage_policy_flags": _rate(len(coverage), len(a_usable)),
            "classifier_score_flags": _rate(len(score_flag), len(a_usable)),
            "benign_control_flag_rate": _rate(sum(o.decision == "flag" for o in c_usable), len(c_usable)),
            "attack_truncated": _rate(sum(o.truncated for o in a_usable), len(a_usable)),
            "stages_run": stages,
            "unavailable": {"attack": sum(o.fallback for o in attacks),
                            "control": sum(o.fallback for o in controls)},
        }


def run_probes(classifier: Classifier, probes: list[Probe] | None = None,
               attempt_id: str = "probe") -> dict:
    """Classify every probe and return a report grouped by placement. `classifier` is any
    redteam.classifier.Classifier (a GatewayClient, or a fake in tests)."""
    probes = probes if probes is not None else load_probes()
    reports: dict[str, PlacementReport] = {}
    outcomes: list[ProbeOutcome] = []
    for p in probes:
        v = classifier.classify(p.body, f"{attempt_id}:{p.probe_id}")
        o = ProbeOutcome.from_verdict(p, v)
        outcomes.append(o)
        reports.setdefault(p.placement, PlacementReport(p.placement)).outcomes.append(o)
    from .probes import PROBE_SET_VERSION

    return {
        "probe_set_version": PROBE_SET_VERSION,
        "n_probes": len(probes),
        "placements": {name: rep.summary() for name, rep in sorted(reports.items())},
        "outcomes": [o.__dict__ for o in outcomes],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run middle/boundary probes against a gateway.")
    p.add_argument("--gateway-url", required=True, help="base URL of the gateway (POST {url}/classify)")
    p.add_argument("--gateway-timeout", type=float, default=None)
    p.add_argument("--on-unavailable", choices=["flag", "allow", "error"], default="allow",
                   help="how to treat an unusable gateway; default allow so a dead gateway is not "
                        "scored as a detection")
    p.add_argument("--label", default=None, help="a free-text label stored in the report (e.g. 'before')")
    p.add_argument("--out", type=Path, help="write the full report JSON here")
    p.add_argument("--full", action="store_true", help="print per-probe outcomes too")
    args = p.parse_args(argv)

    with GatewayClient(args.gateway_url, timeout_s=args.gateway_timeout,
                       on_unavailable=args.on_unavailable) as c:
        report = run_probes(c)
        report["gateway"] = {"url": args.gateway_url, "timeout_s": c.timeout_s,
                             "timeout_source": c.timeout_source, "on_unavailable": c.on_unavailable,
                             "label": args.label}
    if not args.full:
        report = {k: v for k, v in report.items() if k != "outcomes"}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
