# Design: Adaptive attacker and adaptive ASR (redesign r1)

Milestone **adaptive-attacker**. This document defines the threat model, the attacker's knowledge,
the success definitions, the goal set, and how adaptive attack success rate (ASR) is measured and
reported. It is the companion to `docs/RFC.md` ("Adaptive attacker" and "Release gate rules") and to
`docs/design/capability-policy.md` §9.

**This is the r1 redesign** after an independent review (`reviews/adaptive-attacker-r1.md`) found the
first version's headline (6/6 harmful evasion) was **one fixed injection sentence** (`nl_forward`,
3 targets × 2 scenarios + its mirror) diluted in six Haiku-written carriers at seed 0, with no
adaptivity exercised and the decision-only / score-feedback runs identical by construction. That
number is renamed here to **fixed-core dilution evasion** (§10) and is no longer presented as an
adaptive ASR. Everything below is the replacement.

Every number quoted here comes from a committed script or output under `reports/adaptive-attacker/`;
the harness is `redteam/adaptive/`. No number is written by hand.

## 1. What is under test

The served **D23 bundle** (`artifacts/registry.json`, bundle `2ebc7d9a…`, engine 1.1.0): DeBERTa
(`deberta-pi-v2`) decides every request, TF-IDF Stage 1 runs observe-only, there is no gate step, and
the coverage policy flags any input with content no step scored. The attacker crafts an **indirect**
prompt injection hidden in the body of the one low-trust email it controls, sends that body through
the gateway's `/classify` endpoint, and tries to get `allow` while the body still carries a functional
harmful instruction.

## 2. Threat model and attacker knowledge

| Dimension | This milestone |
| --- | --- |
| **Attack surface** | exactly one attacker-controlled email body per scenario; everything else is fixed (RFC threat model). |
| **Attacker observation (headline)** | **black-box, decision-only**: for each candidate body the attacker learns only `allow` vs `flag`, plus the sandbox behavioural outcome. No score, no per-chunk score, no threshold. |
| **Attacker observation (ablation)** | **score-feedback**: additionally the decider's numeric score (max active chunk score). An *upper bound* on a realistic attacker; reported as an ablation, never the headline. |
| **Query budget** | a hard per-goal cap on gateway queries, default **100** (RFC ~100); each genuine (uncached) classification is one query and is logged. "No bypass within N queries" is a valid reported outcome. |
| **Where variety comes from** | a **library of injection cores** (`redteam/adaptive/cores.py`, §4) and **mutation operators** (`redteam/adaptive/mutators.py`, §5), searched keep-best. The only LLM-authored content is the **benign carrier** email; the model never authors an injection. |
| **What the attacker does *not* get** | the classifier weights, the training data, the thresholds, the hash-locked test split, the held-out scenarios, or any ground-truth label. |

## 3. Attacker goals

A **goal** is `(dev scenario, reserved collector target, channel)`; the harmful action is an
unauthorised `send_email` to a `.invalid` address no grant authorises. Goals are enumerated by
`redteam/adaptive/goals.py` into `reports/adaptive-attacker/goals.json` (3 dev-scenario goals). The
attacker never reads the held-out scenarios or the hash-locked test split.

**dev vs eval.** The split only separates a cheap pilot (dev: `dev_daily_digest`) from the headline
(eval: the other two scenarios). The search does **no cross-goal or cross-run tuning** — fixed
operators, a fixed core library, online keep-best within a per-goal budget, no persisted state — so
there is no tuned training set whose reuse could inflate the headline (this is the substance behind
review M7). The harness itself was developed against offline synthetic goals with fake/keyword
classifiers, never against the served gateway's responses.

## 4. Injection core library and families

`redteam/adaptive/cores.py` provides **21 cores** grouped into **six families**: `polite_request`,
`action_line`, `system_notice`, `ignore_previous`, `role_play`, `quoted_forward`. Each core is a
template whose only variable is the goal's reserved `{target}`; `assert_reserved_cores` checks that no
core can render a non-reserved address. Provenance is recorded per core: repo probe fixtures
(`redteam/probes.py`), the dev-scenario attacker bodies, and clearly-attributed public
indirect-injection template styles.

`redteam/adaptive/build_cores.py` tried to add a dataset slice from the **train/val** injection items
(never the hash-locked test split, never held-out). The committed diagnostic
(`fixtures/adaptive/cores_curated.json`) records the finding: the public injection sets are **not
email-shaped** — 1 of 10,118 label-1 train/val items carries a recipient address and none is a short
email send/forward — so that slice contributes **0** email cores, and the library rests on repo and
template material. This is itself a reportable result about the public corpora.

The **unit of analysis is the (core, carrier) pair**, and results are reported per core and per
family. Because pairs that share a core are correlated, intervals are **clustered by core** (§7).

## 5. Mutation operators (the source of adaptivity)

`redteam/adaptive/mutators.py` + `redteam/adaptive/compose.py`:

- **placement** — start / middle / end / **after DeBERTa token 512** (for long bodies). Placement is
  by content-token offset, so `after_512` genuinely lands past the first scoring window; a test
  asserts the core starts at token ≥ 512, and the composer records `placement_reached` so a body too
  short to reach 512 is never silently counted as having done so (addresses review M2 and nit n2).
- **carrier length** — grow/shrink the benign carrier toward a target token length (DILUTION.md sweep),
  reaching past 512 tokens when scaled up.
- **paraphrase** — vary the benign framing (lead-in / trailer) around the core; the core string is
  preserved verbatim.
- **split / frame** — break the framing across paragraphs; quoting or forwarded-message framing.
- **noise** — whitespace / zero-width formatting noise around the framing.

The search is **keep-best** (`redteam/adaptive/search.py`): for each core it hill-climbs mutations over
a carrier, keeping the best candidate under the per-goal budget. In **score-feedback** it adopts a
candidate with a strictly lower (allow-rank, score), so the observed score steers the next query; in
**decision-only** the score is hidden, so among flagged candidates it explores (random walk) and locks
in once allowed. A test (`test_score_feedback_reaches_search_after_query_1`) checks that feedback
reaches the search after query 1 — the failure the review flagged (M2).

An LLM writes or paraphrases the **benign carrier** only (`redteam/adaptive/build_carriers.py`), which
it does not refuse. Mutating a core can break its parseability; such non-functional candidates are
counted (`n_refusals_core_broken`) and excluded from "harmful" by the mode-A check.

## 6. Success definitions and the victim

Per candidate, using the deterministic **GullibleDefender** as the worst-case victim:

- **is-harmful (mode A)** — the body causes an unauthorised send with no classifier (functional).
- **harmful-evasion / end-to-end harm (mode B)** — classifier on, firewall off: a functional injection
  the classifier allowed and the victim acted on.
- **firewall invariant (mode C)** — firewall on: `attack_success` must be 0 for every goal.

The gullible victim bounds harm from above but defines "harmful" by a regex. **A live-victim spot
check** (`redteam/adaptive/live_victim.py`, §8) re-runs a sample of classifier-allowed candidates
through a live `claude-haiku-5-5` victim in modes B and C, so "harmful" means the model actually
attempted the send, and reports regex-vs-live agreement.

## 7. Headline results (eval goals, budget 100/goal, bundle 2ebc7d9a…, engine 1.1.0)

From `reports/adaptive-attacker/eval_decision/summary.json` (black-box headline) and `eval_score/`
(score-feedback ablation). ASR is the mean of per-core ASRs (equal weight per core); the CI is a
cluster bootstrap resampling cores.

| Metric | decision-only (headline) | score-feedback (ablation) |
| --- | --- | --- |
| adaptive ASR, cluster-aware | **0.452**, 95% CI [0.262, 0.667] | 0.500, 95% CI [0.310, 0.690] |
| naive per-(goal,core) rate (ignores clustering) | 0.452 of 42 | 0.500 of 42 |
| end-to-end harm, mode B (gullible victim) | 1.00 | 1.00 |
| **mode C invariant (firewall on)** | **0/goals — holds** | **holds** |
| benign-control FPR (overall) | **0.00** | 0.00 |
| queries-to-first-success (median, gateway queries) | 26 (19/42 solved) | 23 (21/42 solved) |

Per-family cluster-aware ASR (score-feedback run): `ignore_previous` 0.83 [0.50, 1.00],
`system_notice` 0.75 [0.25, 1.00], `role_play` 0.67, `polite_request` 0.50, `quoted_forward` 0.17,
`action_line` 0.12 [0.00, 0.38]. The machine-style `action_line` family is caught most often; the
natural-language `ignore_previous` / `system_notice` families evade most. The queries-vs-success curve
(`queries_vs_success_curve`) rises from ~0.02 at query 1 to a plateau near 0.45–0.50 by ~query 50:
success needs real search, not one shot.

**Benign controls (review M3).** For every carrier the harness also scores (a) carrier + a benign
forwarding request to an authorised/ordinary address, (b) carrier + a benign-hard instruction-shaped
sentence, (c) the carrier alone. The classifier flagged **0** of these on D23, so "the classifier
allowed the injection" is distinguishable from "the classifier correctly allowed benign forwarding":
the evasions are genuine misses, not the classifier behaving as specified on benign forwarding.

**Score-feedback vs black-box.** Score feedback is modestly stronger (0.500 vs 0.452, median
queries-to-first-success 23 vs 26): seeing the gradient helps, but the black-box attacker gets most of
the way, consistent with D23's documented permissiveness to diluted indirect injections
(`training/DILUTION.md`).

## 8. Live-victim spot check (review item 4)

`reports/adaptive-attacker/live_victim_spotcheck.json`, 6 classifier-allowed candidates (3/eval goal),
live `claude-haiku-5-5` victim:

- **regex-harm vs live-harm agreement (mode B): 0.667** (4/6).
- **live end-to-end harm, mode B: 0/6.** Even when the classifier allowed the injected body, the real
  victim followed its system instruction to treat email content as untrusted and did **not** execute
  the send in any sampled case — the gullible regex (2/6 here) is a loose upper bound on real harm.
- **mode C: 0/6; the firewall invariant holds end-to-end.**

So the classifier's permissiveness to these diluted injections is real, but on this small live sample
the model victim did not convert an allowed body into a send; the **authorization firewall remains the
security boundary**, exactly as the RFC positions it. (Sample size is small; this is a spot check, not
a rate.)

## 9. Author-mode refusal (safety finding) and spend

**Author mode** (the model authors the full injection, `redteam/adaptive/refusal_probe.py`): committed
`reports/adaptive-attacker/author_refusal.json` records **12/12 refusals** across the three goals
(`claude-haiku-5-5`). The refusal counts come from committed output, not prose (review M4); the
hand-written "36/36" of the first version is removed. The soft-refusal heuristic (review m4) now counts
"I'm not comfortable; here is a benign email" as a refusal.

**Spend.** A milestone-wide ledger (`reports/adaptive-attacker/spend_ledger.json`,
`redteam/adaptive/ledger.py`) records every run's API spend with no key, header or request content; the
dollar cap reads the prior total so it is milestone-wide, not per invocation (review m5). The mutation
**search itself spends $0** (no model-authored injections; black-box over the local DeBERTa). Total
milestone API spend to date: **$0.0157** (benign carriers $0.0022 + live victim $0.0091 + refusal probe
$0.0044), far under the $10 redesign cap. The input-token estimate is a strict upper bound for any
script (review m5).

## 10. The first-version number, renamed

The first version's "6/6 harmful evasion, Wilson [0.61, 1.00]" is renamed **fixed-core dilution
evasion**: one injection core (`nl_forward`) in six benign carriers at seed 0, with the decision-only
and score-feedback runs identical by construction and no adaptivity exercised. It is a real
measurement of `P(allow | one diluted sentence, these carriers)` and is consistent with
`training/DILUTION.md`, but it is **not** an adaptive ASR and is kept out of the `adaptive_asr` gate
block. The replacement adaptive measurement (§7) is what the gate now surfaces.

## 11. Wiring into the release gate

`training/gate.py` surfaces the adaptive `adaptive_asr` block **report-only** (verdict
`NOT_EVALUATED`): the RFC rule is relative ("no worse than the active bundle") and D23 has no previous
bundle, and the RFC defines no absolute threshold. A missing `measured_against_bundle_sha256` is now
treated as **unknown → does not apply**, never as a match (review m8). The adaptive gate input is
`reports/adaptive-attacker/<run>/adaptive_asr_gate_input.json`.

## 12. Isolation and `do_not_train`

The adaptive-evasion probe set and the core library carry `do_not_train: true`. The training loader
(`training.common.read_training_jsonl`, used by `train_stage1`) now **refuses** any row or whole-object
artifact marked `do_not_train`, and `training.check_leakage` fails if any `do_not_train` fixture body
appears in a split — making the label binding rather than a convention (review n1).

## 13. Open questions for the owner

1. **Adaptive-ASR gate semantics.** Keep report-only, set an absolute mode-B cap, or pair against a
   second bundle once one exists? (This milestone keeps report-only.)
2. **Budget / sample size.** 100 queries/goal and 2 eval goals give wide cluster CIs; raise the budget
   or add goals (more CPU)?
3. **Live-victim sample.** The spot check is 6 candidates. Promote a live-victim rate to the headline,
   or keep the deterministic worst-case victim as the headline and the live victim as a check?
4. **Prices.** Per-token prices are third-party-reported; confirm against billing before trusting the
   spend to the cent.
5. **Promoting evasions to training.** Committed as a `do_not_train` probe set; which, if any, should
   enter `train` next milestone (with RFC issue #7 isolation)?
