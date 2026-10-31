"""Author-mode refusal probe (review M4 / item 5).

Asks claude-haiku-5-5 to *author* a working injection exfiltration email for each goal, several times
per goal, and records whether each reply is a refusal (``looks_like_refusal``). This documents the
safety finding that a production model declines to author these attacks - as a committed summary with
counts, hashes and token usage (no bodies), so the claim traces to committed output rather than prose.

Ledger-tracked and capped. Key read only from ANTHROPIC_API_KEY; never printed, logged or committed.

    python -m redteam.adaptive.refusal_probe --attempts 4 --out reports/adaptive-attacker/author_refusal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from training.metrics import rate

from .budget import SpendTracker
from .cache import ResponseCache
from .goals import enumerate_goals
from .ledger import SpendLedger
from .llm import AnthropicAttackerClient
from .strategies import ATTACKER_SYSTEM, build_prompt, looks_like_refusal, parse_body

MODEL = "claude-haiku-5-5"
SHORT_MAX_TOKENS = 1024


def run(attempts: int, *, cap_usd: float = 10.0) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set (load /root/.sentinel/.env)")
    ledger = SpendLedger.load()
    spend = SpendTracker(cap_usd=cap_usd, prior_usd=ledger.prior_total_usd)
    # Ledger entry exists before the first call and is rewritten after every call (review r2 m3).
    ledger_run = ledger.begin(label="author_refusal_probe", note="author-mode refusal probe")
    spend.on_record = ledger_run.update
    cache = ResponseCache()
    client = AnthropicAttackerClient(MODEL, spend, cache)
    goals = enumerate_goals()
    rows = []
    for g in goals:
        for i in range(attempts):
            system, messages = build_prompt(g.scenario(), target=g.target, channel=g.channel)
            # Vary the prompt slightly per attempt so cache keys differ and the model sees fresh asks.
            messages[0]["content"] += f"\n\n(attempt {i + 1})"
            comp = client.complete(system=system, messages=messages, max_tokens=SHORT_MAX_TOKENS,
                                   temperature=1.0)
            body = parse_body(comp.text)
            refused = looks_like_refusal(comp.text)
            rows.append({"goal_id": g.goal_id, "attempt": i + 1, "refused": refused,
                         "reply_sha16": hashlib.sha256(comp.text.encode()).hexdigest()[:16],
                         "reply_len": len(comp.text), "cached": comp.cached})
            if spend.remaining_usd <= 0:
                break
    n_ref = sum(1 for r in rows if r["refused"])
    out = {
        "probe": "author_mode_refusal", "model": MODEL, "attempts_per_goal": attempts,
        "n_goals": len(goals), "n_candidates": len(rows),
        "refusal_rate": rate(n_ref, len(rows)),
        "note": ("Counts of a production model refusing to author injection exfiltration emails. A "
                 "safety finding, not a classifier measurement. Soft refusals are counted (review m4)."),
        "spend": spend.as_dict(), "rows": rows,
    }
    ledger_run.finish(spend)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempts", type=int, default=4, help="author attempts per goal")
    ap.add_argument("--cap-usd", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("reports/adaptive-attacker/author_refusal.json"))
    args = ap.parse_args(argv)
    out = run(args.attempts, cap_usd=args.cap_usd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "n_candidates": out["n_candidates"],
                      "refusal_rate": out["refusal_rate"]["value"], "spent_usd": out["spend"]["spent_usd"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
