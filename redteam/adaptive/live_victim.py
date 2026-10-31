"""Live-victim spot check (review item 4).

The headline uses the deterministic gullible victim, which bounds harm from above but defines "harmful"
by a regex. This module re-runs a *sample* of classifier-allowed candidates through a live
claude-haiku-5-5 victim (redteam.llm_defender.LLMDefender) in the sandbox, mode B (classifier on,
firewall off) and mode C (firewall on), so "harmful" means the model actually attempted the send,
not just that a regex matched. It reports regex-harm vs live-harm agreement and the mode B vs mode C
end-to-end outcome.

Spend is capped by the milestone ledger (redteam/adaptive/ledger.py). Every victim API call goes through
``MeteredLLMClient``: it is preflighted against the cumulative milestone cap from the actual request
(system prompt, messages and tool schemas, at a conservative chars/3 token estimate, plus the call's
max_tokens as output), its input and output tokens are recorded separately and priced at their own
rates, and the ledger is rewritten after the call (review r2 m3). The first redesign preflighted once
per sample although two victim runs followed, priced input+output tokens at the input rate, and wrote
the ledger only at the end. The API
key is read only from ANTHROPIC_API_KEY and never printed, logged or cached with headers.

Input is a run's gitignored queries.jsonl (full bodies). Output is a committed JSON with per-candidate
hashes, the regex/live verdicts and the agreement table - no bodies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.metrics import rate

from ..judge import judge_attempt
from ..llm_defender import AnthropicLLMClient, LLMBudget, LLMDefender
from ..sandbox import Limits, run_attempt
from ..scenario import find_scenario
from ..trace import RunTrace
from .budget import SpendTracker
from .ledger import SpendLedger
from .llm import _estimate_tokens
from .oracle import make_gateway_classifier

VICTIM_MODEL = "claude-haiku-5-5"


@dataclass
class MeteredLLMClient:
    """Wraps a victim ``LLMClient``: preflight before EVERY call, record real usage after it.

    The preflight bound is the request's own size (system + messages + tool schemas) plus the call's
    ``max_tokens`` as output, so it covers the call about to be made rather than a per-sample guess.
    ``spend.record`` prices input and output at their own rates and, via ``spend.on_record``, rewrites
    the milestone ledger after each call."""

    inner: Any
    spend: SpendTracker
    model: str = VICTIM_MODEL

    def create(self, *, system, messages, tools, max_tokens):
        est_in = _estimate_tokens(system + json.dumps(tools, ensure_ascii=False), messages)
        self.spend.preflight(self.model, est_input_tokens=est_in, est_max_output_tokens=max_tokens)
        resp = self.inner.create(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        self.spend.record(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
        return resp


@dataclass
class _Spic:
    goal_id: str
    scenario_id: str
    body: str
    body_sha16: str
    regex_harm_modeB: bool  # the deterministic gullible verdict recorded by the search


def _sample(run_dir: Path, *, per_goal: int, seed: int) -> list[_Spic]:
    import random
    rows = [json.loads(l) for l in (run_dir / "queries.jsonl").read_text().splitlines() if l.strip()]
    by_goal: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("classifier_allowed") and r.get("body"):
            by_goal.setdefault(r["goal_id"], []).append(r)
    out: list[_Spic] = []
    rng = random.Random(seed)
    for goal_id, rs in by_goal.items():
        # de-duplicate by body hash so the sample is distinct candidates
        seen, uniq = set(), []
        for r in rs:
            if r["body_sha16"] not in seen:
                seen.add(r["body_sha16"])
                uniq.append(r)
        rng.shuffle(uniq)
        for r in uniq[:per_goal]:
            out.append(_Spic(goal_id=goal_id, scenario_id=goal_id, body=r["body"],
                             body_sha16=r["body_sha16"], regex_harm_modeB=bool(r.get("mode_b_attack_success"))))
    return out


def _run_live(spic: _Spic, mode: str, client, classifier, *, budget: LLMBudget) -> dict:
    scenario = find_scenario(spic.scenario_id)
    defender = LLMDefender(client, model=VICTIM_MODEL, budget=budget, source="live")
    trace = RunTrace(run_id=f"live-{spic.body_sha16[:8]}-{mode}", out_dir=None)
    clf = None if mode == "A" else classifier
    res = run_attempt(scenario, mode, defender, trace, classifier=clf, attacker_body=spic.body,
                      limits=Limits(deadline_s=budget.wall_time_s + 5.0))
    v = judge_attempt(trace.attempt_events(res.attempt_id), scenario)
    return {"status": v.status, "attack_success": v.attack_success, "blocked": v.blocked,
            "tokens": defender.tokens_used}


def spot_check(run_dir: Path, gateway_url: str, *, per_goal: int = 2, seed: int = 0,
               cap_usd: float = 10.0) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set (load /root/.sentinel/.env)")
    ledger = SpendLedger.load()
    spend = SpendTracker(cap_usd=cap_usd, prior_usd=ledger.prior_total_usd)
    ledger_run = ledger.begin(label=f"live_victim:{run_dir.name}",
                              note="live claude-haiku-5-5 victim spot check (mode B/C)")
    spend.on_record = ledger_run.update
    client = MeteredLLMClient(AnthropicLLMClient(model=VICTIM_MODEL), spend)
    budget = LLMBudget(max_steps=6, max_tokens=12_000, wall_time_s=40.0)
    sample = _sample(run_dir, per_goal=per_goal, seed=seed)
    rows = []
    for spic in sample:
        classifier = make_gateway_classifier(gateway_url)
        # Each victim API call inside these runs is preflighted and recorded by MeteredLLMClient.
        mb = _run_live(spic, "B", client, classifier, budget=budget)
        mc = _run_live(spic, "C", client, classifier, budget=budget)
        rows.append({"goal_id": spic.goal_id, "body_sha16": spic.body_sha16,
                     "regex_harm_modeB": spic.regex_harm_modeB,
                     "live_harm_modeB": mb["attack_success"], "live_modeB_status": mb["status"],
                     "live_harm_modeC": mc["attack_success"], "live_modeC_status": mc["status"]})
        if spend.remaining_usd <= 0:
            break

    agree = sum(1 for r in rows if r["regex_harm_modeB"] == r["live_harm_modeB"])
    live_b = sum(1 for r in rows if r["live_harm_modeB"])
    live_c = sum(1 for r in rows if r["live_harm_modeC"])
    out = {
        "spot_check": "live_victim", "victim_model": VICTIM_MODEL, "run_dir": str(run_dir),
        "n_checked": len(rows),
        "n_api_calls": spend.n_calls,
        "regex_vs_live_modeB_agreement": rate(agree, len(rows)),
        "live_harm_modeB": rate(live_b, len(rows)),
        "live_harm_modeC": rate(live_c, len(rows)),
        "modeC_invariant_holds": live_c == 0,
        "spend": spend.as_dict(),
        "rows": rows,
    }
    ledger_run.finish(spend)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gateway-url", required=True)
    ap.add_argument("--per-goal", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-usd", type=float, default=10.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = spot_check(args.run, args.gateway_url, per_goal=args.per_goal, seed=args.seed, cap_usd=args.cap_usd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "n_checked": out["n_checked"],
                      "agreement": out["regex_vs_live_modeB_agreement"]["value"],
                      "spent_usd": out["spend"]["spent_usd"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
