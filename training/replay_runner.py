"""Replay the traffic replay set against a gateway at a fixed rate and summarize it per time window.

    python -m training.replay_runner --url http://127.0.0.1:8000 --rps 5 --mode open \\
        --window-s 20 --out-dir training/results/replay_runs/<name>

Modes
  open    item i is sent at its scheduled time whether or not earlier requests have finished (arrivals do not
          slow down when the gateway does). --arrivals schedule (default) uses the set's Poisson offsets scaled
          by nominal_rps / rps; --arrivals uniform sends every 1 / rps s. If --max-inflight requests are
          already outstanding, the item is not sent and is counted as error "client_saturated" (the run stays
          open-loop instead of silently turning closed-loop).
  closed  --concurrency workers each send the next item as soon as their previous request finished, paced so
          item i is never sent before start + i / rps (a rate cap; the achieved rate is reported).

Every response is validated with contracts.ClassifyResponse (training.eval_harness.parse_gateway_response). One
attempt per item, no retries (a retry would change the offered rate); failures are recorded as errors:
transport:<ExceptionName>, http_<status>, invalid_response, client_timeout, client_saturated.

Windows are wall-clock intervals of --window-s seconds from the start of the run; a request belongs to the
window in which it was sent. A window is written to windows.jsonl once its end has passed and all its requests
have finished, so the file grows during the run and drift is visible while it runs. Each window reports:
  n_sent, n_responses, errors (by kind), flag_rate, escalation_rate (stages_run contains stage2), stages_run
  counts, decided_by counts, policy_applied counts, coverage_policy_applied counts, failure_reason counts of
  failed active results (overloaded separate from timeout / exception), shadow: n_ok, agreement with the served
  decision, shadow failure reasons (shadow detectors other than the observe-only Stage 1); observe_stage1: the
  observe-only Stage 1 of D23 (role "observe" in v1.3, "shadow" in v1.2; stage="stage1"), which can only ever
  clear (allow), so agreement
  with the decision means nothing for it. Reported instead: n_ok, n_partial (anytime deadline, some chunks
  unscored -> could not clear), below_low_share (fully scored and every chunk below gray-zone low = the Stage 1
  threshold: the share Stage 1 would allow if the per-chunk language gate passes, an upper bound on its
  would-finalize-allow share; the language is not in the response), below_low_but_served_flag (of those, how
  many the active path flagged: injections or failure / coverage-policy flags Stage 1 would have let through,
  split by the policy that applied, "None" = a Stage 2 score), and failure reasons; client latency (n, p50, p95,
  p99, max); non_en_share (replay metadata, not seen by the gateway).
Delayed labels: an item's label may be used only from send_time + label_available_after_s * label_delay_scale
(default scale nominal_rps / rps). Window w reports metrics over labels revealed in (end_{w-1}, end_w] and
cumulatively up to end_w. Labels never revealed (null delay or null label) are never used. Labels revealed after
the last window are reported as pending unless --drain-labels waits for them.

Latency is client-side wall time on whatever host runs this; it is not a benchmark.
Outputs in --out-dir: windows.jsonl, requests.jsonl (per item: id, seq, window, send offset, latency, decision,
stages, statuses, failure reasons, error; no text), summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from training.common import ROOT, read_jsonl, sha256_file, write_json
from training.eval_harness import GatewayResponseError, parse_gateway_response
from training.metrics import rate

REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"
MANIFEST = ROOT / "data" / "replay" / "MANIFEST.json"
LATENCY_CONDITIONS = "client-side wall time on a shared 2-core host; not a benchmark"


@dataclass
class Rec:
    item: dict
    window: int
    sent_at: float  # seconds since run start
    label_at: float | None  # seconds since run start when the label may be used; None = never
    latency_ms: float | None = None
    parsed: dict | None = None
    error: str | None = None
    done: bool = False

    def row(self) -> dict:
        p = self.parsed or {}
        return {"id": self.item["id"], "seq": self.item.get("seq"), "window": self.window,
                "sent_at_s": round(self.sent_at, 4), "latency_ms": None if self.latency_ms is None
                else round(self.latency_ms, 3), "error": self.error, "decision": p.get("decision"),
                "decided_by": p.get("decided_by"), "stages_run": p.get("stages"), "statuses": p.get("statuses"),
                "policy_applied": p.get("policy_applied"), "coverage_policy_applied": p.get("coverage_policy_applied"),
                "failure_reasons": p.get("failure_reasons"), "shadow": p.get("shadow"),
                # v1.3: the bundle that served this request.
                "bundle_sha256": p.get("bundle_sha256"), "policy_id": p.get("policy_id"),
                "policy_version": p.get("policy_version"), "unscored_by_any_step": p.get("unscored_by_any_step"),
                "escalation_reason": p.get("escalation_reason")}


def _shadow(payload: dict) -> list[dict]:
    """Results that did not take part in the decision: an observe step (v1.3 role "observe"; "shadow" from a v1.2
    gateway) or the shadow mirror."""
    return [r for r in payload.get("results", []) if isinstance(r, dict)
            and r.get("role") in ("observe", "shadow")]


def _is_observe_stage1(s: dict) -> bool:
    """An observe-only Stage 1 (D23): stage="stage1", role "observe" (v1.3) or "shadow" (v1.2). Older rows
    without `stage` are not."""
    return s.get("stage") == "stage1"


def _bundle_block(ok: list[Rec]) -> dict:
    """Which bundle(s) served the responded requests (v1.3; empty from a pre-1.3 gateway)."""
    by_bundle = Counter(r.parsed["bundle_sha256"] for r in ok if r.parsed.get("bundle_sha256"))
    policies = Counter(f"{r.parsed['policy_id']}@{r.parsed['policy_version']}" for r in ok
                       if r.parsed.get("policy_id"))
    return {"by_bundle_sha256": dict(sorted(by_bundle.items())),
            "bundle_sha256": next(iter(by_bundle)) if len(by_bundle) == 1 else None,
            "policies": dict(sorted(policies.items())), "single_bundle": len(by_bundle) == 1,
            "n_without_bundle": sum(1 for r in ok if not r.parsed.get("bundle_sha256"))}


def _observe_stage1_block(ok: list[Rec]) -> dict:
    """Observe-only Stage 1 (D23) over responded records; see the module docstring."""
    n_ok = n_partial = n_below = below_flag = 0
    by_policy: Counter = Counter()
    reasons: Counter = Counter()
    for r in ok:
        p = r.parsed
        for s in p.get("shadow", []):
            if not _is_observe_stage1(s):
                continue
            if s["status"] != "ok":
                reasons[s.get("failure_reason") or s["status"]] += 1
            elif s.get("truncated"):
                n_ok += 1
                n_partial += 1  # anytime deadline: unscored chunks stay uncleared
            else:
                n_ok += 1
                if s.get("predicted_injection") is False:
                    n_below += 1
                    if p["decision"] == "flag":
                        below_flag += 1
                        by_policy[str(p["policy_applied"] or p["coverage_policy_applied"])] += 1
    return {"n_ok": n_ok, "n_partial": n_partial, "below_low_share": rate(n_below, n_ok),
            "below_low_but_served_flag": below_flag,
            "below_low_and_served_flag_by_policy": dict(sorted(by_policy.items())),
            "failure_reason": dict(sorted(reasons.items()))}


def _latency(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs)
    return {"n": len(xs), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def _labeled(recs: list[Rec]) -> dict:
    """Detection metrics over records whose label is usable, as served (the decision the caller received)."""
    ok = [r for r in recs if r.parsed is not None]
    pos = [r for r in ok if r.item["label"] == 1]
    neg = [r for r in ok if r.item["label"] == 0]
    return {"n_labels": len(recs), "n_with_response": len(ok), "n_error": len(recs) - len(ok),
            "recall_as_served": rate(sum(r.parsed["decision"] == "flag" for r in pos), len(pos)),
            "fpr_as_served": rate(sum(r.parsed["decision"] == "flag" for r in neg), len(neg))}


def summarize_window(w: int, t0: float, t1: float, recs: list[Rec], all_recs: list[Rec], prev_end: float) -> dict:
    """Summary of the requests sent in [t0, t1) and of the labels revealed in (prev_end, t1]."""
    mine = [r for r in recs if r.window == w]
    ok = [r for r in mine if r.parsed is not None]
    n = len(ok)
    stages, decided, pol, cov, reasons, sh_reasons = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    sh_ok = sh_agree = 0
    for r in ok:
        p = r.parsed
        stages["+".join(p["stages"]) or "none"] += 1
        decided[p["decided_by"]] += 1
        pol[str(p["policy_applied"])] += 1
        cov[str(p["coverage_policy_applied"])] += 1
        for fr in p["failure_reasons"]:
            reasons[fr] += 1
        for s in p.get("shadow", []):
            if _is_observe_stage1(s):
                continue  # reported by _observe_stage1_block
            if s["status"] == "ok":
                sh_ok += 1
                sh_agree += int(bool(s["predicted_injection"]) == (p["decision"] == "flag"))
            else:
                sh_reasons[s.get("failure_reason") or s["status"]] += 1
    revealed_now = [r for r in all_recs if r.label_at is not None and prev_end < r.label_at <= t1]
    revealed_cum = [r for r in all_recs if r.label_at is not None and r.label_at <= t1]
    return {
        "window": w, "t_start_s": round(t0, 3), "t_end_s": round(t1, 3),
        "n_sent": len(mine), "n_responses": n,
        "errors": dict(sorted(Counter(r.error for r in mine if r.error).items())),
        "flag_rate": rate(sum(r.parsed["decision"] == "flag" for r in ok), n),
        "escalation_rate": rate(sum("stage2" in r.parsed["stages"] for r in ok), n),
        "stages_run": dict(sorted(stages.items())),
        "decided_by": dict(sorted(decided.items())),
        "policy_applied": dict(sorted(pol.items())),
        "coverage_policy_applied": dict(sorted(cov.items())),
        "failure_reason": dict(sorted(reasons.items())),
        "failures": {"overloaded": reasons.get("overloaded", 0),
                     "timeout_or_exception": sum(v for k, v in reasons.items() if k != "overloaded")},
        "shadow": {"n_ok": sh_ok, "agreement_with_served_decision": rate(sh_agree, sh_ok),
                   "failure_reason": dict(sorted(sh_reasons.items()))},
        "observe_stage1": _observe_stage1_block(ok),
        "client_latency_ms": _latency([r.latency_ms for r in mine if r.latency_ms is not None]),
        "non_en_share": rate(sum(r.item.get("lang") != "en" for r in mine), len(mine)),
        "delayed_labels": {"revealed_in_window": _labeled(revealed_now), "cumulative": _labeled(revealed_cum)},
    }


@dataclass
class Config:
    url: str
    rps: float
    mode: str = "open"
    arrivals: str = "schedule"
    concurrency: int = 4
    window_s: float = 10.0
    timeout_s: float = 10.0
    max_inflight: int = 256
    label_delay_scale: float | None = None  # None -> nominal_rps / rps
    drain_labels: bool = False
    nominal_rps: float = 10.0
    extra: dict = field(default_factory=dict)


class Runner:
    def __init__(self, items: list[dict], cfg: Config, out_dir: Path | None = None, transport=None,
                 clock: Callable[[], float] = time.monotonic):
        if cfg.rps <= 0 or cfg.window_s <= 0:
            raise ValueError("rps and window_s must be > 0")
        if cfg.mode not in ("open", "closed") or cfg.arrivals not in ("schedule", "uniform"):
            raise ValueError("mode must be open|closed and arrivals schedule|uniform")
        self.items = sorted(items, key=lambda x: x.get("seq", 0))
        self.cfg = cfg
        self.out_dir = out_dir
        self.transport = transport
        self.clock = clock
        self.scale = cfg.nominal_rps / cfg.rps
        self.label_scale = cfg.label_delay_scale if cfg.label_delay_scale is not None else self.scale
        self.recs: list[Rec] = []
        self.windows: list[dict] = []
        self.inflight = 0
        self.t_start = 0.0

    # ---------------------------------------------------------------- timing
    def now(self) -> float:
        return self.clock() - self.t_start

    def scheduled(self, i: int, item: dict) -> float:
        if self.cfg.mode == "closed" or self.cfg.arrivals == "uniform":
            return i / self.cfg.rps
        return float(item.get("arrival_offset_s", i / self.cfg.nominal_rps)) * self.scale

    async def _sleep_until(self, t: float) -> None:
        d = t - self.now()
        if d > 0:
            await asyncio.sleep(d)

    # ---------------------------------------------------------------- requests
    async def _send(self, client, rec: Rec) -> None:
        import httpx

        rid = f"replay-{rec.item['id']}"
        t0 = self.clock()
        try:
            resp = await client.post(f"{self.cfg.url}/classify", json={"text": rec.item["text"], "request_id": rid})
            rec.latency_ms = (self.clock() - t0) * 1e3
            if resp.status_code != 200:
                rec.error = f"http_{resp.status_code}"
            else:
                try:
                    payload = resp.json()
                    p = parse_gateway_response(payload, rid)
                    p["shadow"] = [{"status": s.get("status"), "predicted_injection": s.get("predicted_injection"),
                                    "failure_reason": s.get("failure_reason"), "detector_id": s.get("detector_id"),
                                    "stage": s.get("stage"), "role": s.get("role", "shadow"),
                                    "truncated": s.get("truncated")}
                                   for s in _shadow(payload)]
                    rec.parsed = p
                except (GatewayResponseError, ValueError):
                    rec.error = "invalid_response"
        except httpx.TimeoutException:
            rec.latency_ms = (self.clock() - t0) * 1e3
            rec.error = "client_timeout"
        except httpx.HTTPError as exc:
            rec.latency_ms = (self.clock() - t0) * 1e3
            rec.error = f"transport:{type(exc).__name__}"
        finally:
            rec.done = True
            self.inflight -= 1

    def _new_rec(self, item: dict) -> Rec:
        sent = self.now()
        d = item.get("label_available_after_s")
        label_at = None if d is None or item.get("label") is None else sent + float(d) * self.label_scale
        rec = Rec(item=item, window=int(sent // self.cfg.window_s), sent_at=sent, label_at=label_at)
        self.recs.append(rec)
        return rec

    # ---------------------------------------------------------------- windows
    def _flush(self, final: bool = False) -> None:
        """Write every window whose end has passed and whose requests have all finished, in order."""
        w = len(self.windows)
        while True:
            t0, t1 = w * self.cfg.window_s, (w + 1) * self.cfg.window_s
            if not final and self.now() < t1:
                return
            mine = [r for r in self.recs if r.window == w]
            if any(not r.done for r in mine):
                return
            if final and not mine and t0 > self.last_send:
                return
            prev_end = self.windows[-1]["t_end_s"] if self.windows else float("-inf")
            s = summarize_window(w, t0, t1, self.recs, self.recs, prev_end)
            self.windows.append(s)
            if self.out_dir:
                with (self.out_dir / "windows.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(s, sort_keys=True) + "\n")
            w += 1

    async def _flusher(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self._flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=min(0.5, self.cfg.window_s / 4))
            except TimeoutError:
                pass

    # ---------------------------------------------------------------- run
    async def run(self) -> dict:
        import httpx

        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "windows.jsonl").write_text("")
        limits = httpx.Limits(max_connections=max(self.cfg.max_inflight, self.cfg.concurrency))
        self.last_send = 0.0
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s, transport=self.transport, trust_env=False,
                                     limits=limits) as client:
            self.t_start = self.clock()
            stop = asyncio.Event()
            flusher = asyncio.create_task(self._flusher(stop))
            tasks = []
            if self.cfg.mode == "open":
                for i, item in enumerate(self.items):
                    await self._sleep_until(self.scheduled(i, item))
                    rec = self._new_rec(item)
                    self.last_send = rec.sent_at
                    if self.inflight >= self.cfg.max_inflight:
                        rec.error, rec.done = "client_saturated", True
                        continue
                    self.inflight += 1
                    tasks.append(asyncio.create_task(self._send(client, rec)))
                await asyncio.gather(*tasks)
            else:
                queue = list(enumerate(self.items))
                queue.reverse()

                async def worker():
                    while queue:
                        i, item = queue.pop()
                        await self._sleep_until(self.scheduled(i, item))
                        rec = self._new_rec(item)
                        self.last_send = max(self.last_send, rec.sent_at)
                        self.inflight += 1
                        await self._send(client, rec)

                await asyncio.gather(*(worker() for _ in range(self.cfg.concurrency)))
            t_last = self.now()
            if self.cfg.drain_labels:
                pending = [r.label_at for r in self.recs if r.label_at is not None]
                if pending:
                    await self._sleep_until(max(pending))
            # Close the windows up to the current time (draining labels can add empty, label-only windows).
            stop.set()
            await flusher
            self.last_send = max(self.last_send, self.now() - 1e-9)
            self._flush(final=True)
        return self._summary(t_last)

    def _summary(self, t_last: float) -> dict:
        end = self.windows[-1]["t_end_s"] if self.windows else 0.0
        ok = [r for r in self.recs if r.parsed is not None]
        used = [r for r in self.recs if r.label_at is not None and r.label_at <= end]
        pending = [r for r in self.recs if r.label_at is not None and r.label_at > end]
        reasons = Counter(fr for r in ok for fr in r.parsed["failure_reasons"])
        out = {
            "config": {k: v for k, v in self.cfg.__dict__.items() if k != "extra"} | self.cfg.extra,
            "time_scale": self.scale, "label_delay_scale": self.label_scale,
            "n_items": len(self.items), "n_sent": len(self.recs), "n_responses": len(ok),
            "duration_s": round(t_last, 3),
            "achieved_rps": round(len(self.recs) / t_last, 3) if t_last > 0 else None,
            "errors": dict(sorted(Counter(r.error for r in self.recs if r.error).items())),
            "flag_rate": rate(sum(r.parsed["decision"] == "flag" for r in ok), len(ok)),
            "escalation_rate": rate(sum("stage2" in r.parsed["stages"] for r in ok), len(ok)),
            "failure_reason": dict(sorted(reasons.items())),
            # v1.3: the bundle(s) this run measured. More than one means the gateway reloaded mid-run, so the run
            # does not describe a single bundle (a gate verdict is bound to one bundle_sha256).
            "bundles": _bundle_block(ok),
            "observe_stage1": _observe_stage1_block(ok),
            "client_latency_ms": _latency([r.latency_ms for r in self.recs if r.latency_ms is not None]),
            "latency_conditions": LATENCY_CONDITIONS,
            "labels": {"used": _labeled(used), "pending_at_end": len(pending),
                       "never_revealed": sum(r.label_at is None for r in self.recs)},
            "windows": [{k: w[k] for k in ("window", "n_sent", "n_responses")}
                        | {"flag_rate": w["flag_rate"]["value"], "escalation_rate": w["escalation_rate"]["value"],
                           "non_en_share": w["non_en_share"]["value"], "errors": sum(w["errors"].values()),
                           "p95_ms": w["client_latency_ms"].get("p95")} for w in self.windows],
        }
        if self.out_dir:
            write_json(self.out_dir / "summary.json", out)
            with (self.out_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
                for r in self.recs:
                    f.write(json.dumps(r.row(), sort_keys=True) + "\n")
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--replay", type=Path, default=REPLAY)
    ap.add_argument("--rps", type=float, required=True)
    ap.add_argument("--mode", choices=["open", "closed"], default="open")
    ap.add_argument("--arrivals", choices=["schedule", "uniform"], default="schedule")
    ap.add_argument("--concurrency", type=int, default=4, help="closed loop: number of workers")
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--timeout-s", type=float, default=10.0, help="client timeout per request")
    ap.add_argument("--max-inflight", type=int, default=256, help="open loop: cap on outstanding requests")
    ap.add_argument("--label-delay-scale", type=float, default=None)
    ap.add_argument("--drain-labels", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="replay only the first N items (by seq)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    items = list(read_jsonl(a.replay))[: a.limit]
    nominal = json.loads(MANIFEST.read_text())["timing"]["nominal_rps"] if MANIFEST.exists() else 10.0
    cfg = Config(url=a.url.rstrip("/"), rps=a.rps, mode=a.mode, arrivals=a.arrivals, concurrency=a.concurrency,
                 window_s=a.window_s, timeout_s=a.timeout_s, max_inflight=a.max_inflight,
                 label_delay_scale=a.label_delay_scale, drain_labels=a.drain_labels, nominal_rps=nominal,
                 extra={"replay_file": str(a.replay), "replay_sha256": sha256_file(a.replay), "limit": a.limit,
                        "python": platform.python_version(), "command": "python -m training.replay_runner "
                        + " ".join(sys.argv[1:])})
    res = asyncio.run(Runner(items, cfg, a.out_dir).run())
    print(json.dumps({k: res[k] for k in ("n_sent", "n_responses", "achieved_rps", "errors", "failure_reason")}
                     | {"flag_rate": res["flag_rate"]["value"], "escalation_rate": res["escalation_rate"]["value"],
                        "windows": len(res["windows"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
