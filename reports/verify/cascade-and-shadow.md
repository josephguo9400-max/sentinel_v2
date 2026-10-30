# Verification: milestone cascade-and-shadow (incl. D22/D23)

Verifier: independent agent, 2026-10-30. Branch `milestone/cascade-and-shadow` at `ec15207`, reviewed in worktree
`/home/claude/wt-verify-cas` (branch `verify/cascade-and-shadow`). Scope: `git log main..milestone/cascade-and-shadow`,
`git diff milestone/gateway-mvp...milestone/cascade-and-shadow`. `data/splits/frozen_test.jsonl` was not read
(`training.check_leakage` was therefore not run, because it reads it). Scratch scripts are in my scratchpad and are not
committed. Their outputs are quoted below.

## Verdict: merge after fixes

The headline result holds. On the replay set the cascade loses DeBERTa-caught injections at every useful gray zone,
and the kill criterion fires however you slice it. D23 (DeBERTa decides, Stage 1 observe-only) is the right call. I
reproduced every number in `replay_counterfactual.json` exactly. Two parts of the story are wrong or incomplete:

- Observe-only Stage 1 is **not** fully isolated. It holds the admission slot after the decision is fixed, and it
  shares the CPU with other requests' Stage 2, so it can change *other* requests' decisions.
- The RFC/README text calls the offline counterfactual "the served gray zone". The served Stage 1 is a different
  scorer (sub-windows, segment pieces, per-chunk langdetect). It escalates more for that reason, not mainly because of
  `unknown` language.

Several docs are also stale after D23, and the milestone's probe before/after result is not documented anywhere.

## Findings

### Major

**M1. Observe-only Stage 1 can change other requests' decisions through admission control (and CPU).**
`serving/service.py:339-345` holds the admission slot (`_admit` / `finally: _release`) around `_classify`.
`serving/service.py:378-386` runs `_observe_stage1` inside `_classify`, after the outcome is fixed. So each response
keeps its `max_inflight` / `max_inflight_chars` share for up to the Stage 1 budget (50 ms) after its decision. A request
that arrives in that window is rejected as `overloaded`, and Stage 2's `fail_closed` turns it into a flag. A gateway
without Stage 1 would have scored it. Scratch reproduction (fake Stage 2, `FakeStage1(delay_s=0.3)`, `max_inflight=1`,
request B sent 150 ms after A, when A's decision is already final):

```
$ PYTHONPATH=. python3 <scratchpad>/test_observe_admission.py
stage1 observe=False: B decision=allow policy=None path=stage2_allow
stage1 observe=True: B decision=flag policy=fail_closed path=overloaded
```

The same kind of coupling exists for CPU. `serving/README.md:190-200` argues that running Stage 1 after the decision
makes the guarantee "structural". That is only true per request: under concurrency, request A's post-decision Stage 1
(TF-IDF ONNX plus pure-Python langdetect, which holds the GIL) runs while request B's Stage 2 runs, on the same 2 cores.
That is exactly the "could stretch Stage 2 towards its own deadline" case the README rejected. The D23 run had 4
Stage 2 timeouts that became `fail_closed` flags (`d23/metrics.txt:128`). Nobody measured whether Stage 1 contributed.
`tests/serving/test_stage1_observe.py` runs requests sequentially only. No test covers concurrent requests.

Fix:
1. Release admission before observing. One way: split `classify` so `_release` runs right after
   `_classify_active`. Another: give observe work its own small semaphore that never counts toward `max_inflight`.
2. Add a concurrency test: same decisions with and without observe Stage 1 at `max_inflight=1`.
3. Either measure active-path p95 and Stage 2 timeouts with observe on vs off under the replay load, or say in the
   README that observe mode may perturb concurrent requests' latency. Long term, the README's own fallback (Stage 1
   runs after the response and is logged, not returned) gives the isolation that is being claimed.

**M2. The "served gray zone" counterfactual is not the served cascade, and the explanation for the 60% vs 52%
escalation gap is mostly wrong.**
`training/replay_counterfactual.py:14` and `docs/RFC.md:185` ("At the served gray zone the cascade sends 52% of requests
to Stage 2 ... 56 injections DeBERTa catches are allowed by Stage 1") use the offline `s1` from
`training/cascade_scores.py`: Stage 1's own 512-regex-token windows over the whole text, plus the stored request-level
langdetect tag. The gateway scores each DeBERTa chunk, each segment piece and 64/32 sub-windows
(`serving/service.py:250-255`). It also runs langdetect per chunk on the first 2,000 characters, with
`min_letters=20, min_prob=0.8` (`serving/language.py`).

I replayed all 2,000 replay items through the real `ClassifierService` in route mode. I used the real `stage1-tfidf-v2.1`
and the real `LanguageDetector`, with a stub Stage 2 that has the real tokenizer and `infer` returning 0. I set
`stage1_timeout_ms=60000`, so there were no timeouts.

- The simulation escalates 1,198/2,000. The served run (`replay_after`) escalated 1,196/1,995. Per item, the simulation
  agrees with the served run on 1,993 of 1,995 (the 2 that differ are served Stage 1 timeouts).
- 165 items are offline "low" but escalate in the gateway. 98 escalate because the serving Stage 1 score is
  >= low (sub-windows / segment pieces). Only 66 escalate because the gateway's langdetect returns `unknown`. 1 goes
  the other way. The extra escalations come mainly from the different scorer.
- With the serving Stage 1 the numbers differ but the conclusion does not: at the served gray zone 48 (not 56)
  DeBERTa-caught injections are lost, recall is 199/306 (served, with timeouts: 202/306 = 66.0%), and savings are 40.1%.
  The best equal-recall savings on the grid are 2.5% (low=0.02). Within a 2-point recall margin savings stay at or below
  about 6%.

```
$ python3 <scratchpad>/sim_route.py <scratchpad>/sim_route.json && PYTHONPATH=. python3 <scratchpad>/decomp.py ...
sim stage2 share 1198 2000
sim vs served (sim_allow, served_allow): {(False, False): 1194, (True, True): 799, (True, False): 2}
98 serving s1 >= low (sub-windows/segment pieces)
66 gateway lang != en: unknown
$ PYTHONPATH=. python3 <scratchpad>/sweep_served_s1.py ...
low=0.02: saved 50/2000=2.50%  lost=0  recall=247/306
low=0.05: saved 121/2000=6.05%  lost=3  recall=244/306
low=0.505068: saved 802/2000=40.10%  lost=48  recall=199/306
```

The same gap affects `training/CASCADE.md`: its val sweep ignores the sub-windows the gateway adds, so the val numbers
describe a Stage 1 that is not the one served.

Fix:
1. In the RFC and the serving README, call the 62.4% / 52% / 56 figures "offline cascade (request-level Stage 1,
   stored language tag)". Cite the as-served numbers next to them: `views.served` in the JSON, recall 202/306 = 66.0%,
   Stage 2 share 1,196/1,995 = 60.0%.
2. State the real reason for the gap.
3. Make `cascade_scores.py` / `replay_counterfactual.py` score Stage 1 the way the gateway does (or call the serving
   scorer), so the next candidate is judged on the scorer that will actually run. The equal-recall rule should use
   that scorer.

**M3. Stage 1 role="shadow" results still leak into consumers other than `redteam/classifier.py`'s catch logic.**
`active_results` fixed `flag_cause` and `truncated`. The following still count every result:
- `training/eval_harness.py:158`: `"truncated": any(r.truncated for r in resp.results)`. An observe Stage 1 that hit its
  deadline (38 times in D23, `d23/metrics.txt:242`) marks the item truncated in the harness's truncation counts. This
  is the release-gate tool. Use `active` instead.
- `redteam/judge.py:235-239` `_prediction_truncated` scans `detail["results"]`, which `sandbox.py` fills with every
  result including role="shadow" (`redteam/classifier.py:100`). So `n_truncated_predictions` is inflated. Skip
  `role == "shadow"`.
- `redteam/classifier.py:53-55` `failure_reasons` and `:41` `detector_version` include the observe Stage 1. The trace's
  `detector_version` names `stage1-tfidf-lr@...` for a verdict Stage 1 did not take part in.
  `redteam/sandbox.py:201-205` (emit at :203) emits a `detector_timeout` TraceEvent for an observe Stage 1 timeout. Use
  `active_results`, or tag these as observe.
- `training/replay_runner.py:118-121` treats the observe Stage 1 as a shadow and reports
  `shadow.agreement_with_served_decision` from `predicted_injection` (score >= 0.505). For an allow-only Stage 1 the
  meaningful number is "would have finalized allow" versus the decision. As written, the agreement figure is not
  interpretable. Report it separately per `detector_id` or `stage`.
- `serving/service.py:542-549`: `sentinel_detector_failures_total` and `sentinel_truncated_requests_total` have no
  role/mode label. Observe Stage 1 timeouts (6 in D23) sit in the same series a gate error-rate check would read.

`docs/contract_requests.md` (last row) wrongly says `training/eval_harness.py` already splits results by role
everywhere. Fix as above, plus one test per consumer that feeds a response containing a truncated/timed-out
role="shadow" Stage 1 result.

**M4. The milestone's probe before/after result is not documented anywhere, and it is mostly negative.**
The RFC checkpoint asks for middle-placement and boundary-split probes measured before vs after. The JSONs exist.
No doc reports them. `redteam/README.md:292` still says "the owner runs it against the real model". From the committed
reports:

- Route-mode cascade (`probes_after.json`): Stage 1 alone finalized **all 6 boundary-split attacks** and **1 middle
  attack** (`middle_system_notice_action_gap_early_attack`, Stage 1 ok, 7 chunks, max 0.261) as `allow`. DeBERTa never
  saw them.
- Classifier catches on middle attacks: 0/6. The other 5 were flagged only by `coverage_policy:flag`, with 5/6
  controls flagged too.
- D23 (`d23/probes.json`): middle 6/6 flagged, all by the coverage policy, with 6/6 controls flagged, so the probe
  cannot tell attack from control. Boundary 0/6 (DeBERTa max 0.009). The commit message's "6/6 each" is correct but
  should not be read as detection.

Fix: add a before/after/D23 probe table to `redteam/README.md`, generated from the three JSONs, and state these three
facts.

### Minor

**m1. Stale docs after D23.**
- `README.md:3,5,11`: still describes "a two-stage cascade" as the system, says "next is cascade-and-shadow", and
  lists contracts as v1.1. It has no cascade-and-shadow findings section.
- `training/CASCADE.md:64-65`: "this milestone does not remove the cascade from the plan".
- `training/DATA.md:400`: the replay runner "has not been run against the real gateway in this milestone". It was run
  twice (`replay_after/`, `d23/replay/`).
- `serving/README.md:207-208` and `:510-511`: say the replay and probes were not re-run on the observe registry.
  `reports/cascade-and-shadow/d23/` did exactly that.
- `redteam/README.md:253-254`: says probes are verified against `serving.encoding`'s "real chunk planner" (see m4).

**m2. D23 numbers appear only in a commit message.** I reproduced "recall 248/303 = 81.8%, FPR 199/1169 = 17.0%" with
`python -m training.replay_counterfactual --served reports/cascade-and-shadow/d23/replay/requests.jsonl --out <scratch>`.
Its output is not committed, and no doc cites it (project rule: only committed-script numbers). Commit the output as
`reports/cascade-and-shadow/d23/replay_counterfactual.json` and cite it. The comparison "matching the offline
DeBERTa-only view (80.7% / 16.4%)" uses a different denominator. On the same 1,974 responses, offline DeBERTa alone is
245/303 and 191/1169. The served view flags 3 more positives and 8 more negatives. The per-item flips:
- 5 come from `coverage_policy:flag` on inputs with 4 or more windows (2 positives, 3 negatives);
- 3 come from Stage 2 timeouts that became `fail_closed` flags (all negatives);
- 7 are chunking/score differences (2 of them toward allow).

Both views agree within noise, and the doc should say why they differ.

**m3. Stage 1 latency write-up misreads its own inputs.**
- `serving/README.md:240` and `serving/service.py:256` say "langdetect alone is 37-48%" of Stage 1 cost. In
  `stage1_latency_before.json`, `language_only.p50 / run_stage1.p50` ranges from 34% (1,000 tokens: 9.7/28.2) to 62%
  (64 tokens: 1.6/2.6). The "23.6 of 50.7 ms" example mixes two files: the probe value before the fix is 24.2.
- `serving/README.md:244` says Fix 1 "does nothing for benign text, where every chunk is below low". But the benchmark's
  "val_benign_en" inputs score at or above low on many chunks. Running the bench's own inputs through `_run_stage1`:

```
64 chunks 1 scores [0.534] langdetect on 0
510 chunks 1 scores [0.755] langdetect on 0
5000 chunks 14 scores [0.755, 0.881, 0.186, ...] langdetect on 7
```

So the 510 / 1,000 / 5,000-token speedups (10.3->6.6, 28.2->21.5, 131.7->91.8 ms p50) are Fix 1 at work, not noise.
The README's explanation is wrong. Separately, val benign English paragraphs reaching 0.88 under sub-windows is itself
worth noting.

Code check for item 5 of the review request: "langdetect only below low" does not change outcomes for a given set of
scored chunks. A chunk at or above low can never be cleared, and `stage2_order` reads only scores, so only the
diagnostic `info["stage1_langs"]` changes. With the anytime deadline the cheaper scoring can change *which* chunks get
scored, which only allows more clearing. That is intended and covered by
`test_route_partial_stage1_clears_only_scored_chunks_and_never_finalizes`.

**m4. Probe tests check the old chunk planner.** `tests/redteam/test_probes.py:4,14` cross-checks the geometry against
`serving.encoding.plan_chunks`. The gateway now chunks with `serving/structure.py` `build_chunks` / `plan_windows`,
whose last window ends at n rather than starting at n-510 (for 2,600 tokens: `[2304,2600)`, not `[2090,2600)`). I
checked all 24 committed probes against the served planner and observe-mode selection (`stage2_order(...)[:4]`). Every
invariant still holds: the middle injections overlap no selected window, and no boundary window contains a whole
injection. Fixtures equal `generate_probes()`. But `expected.selected_windows` in the fixtures and the comment at
`redteam/probes.py:267` describe the old planner. Add a test against `ClassifierService._prepare` and fix the comment.
Nit: `redteam/probes.py:84` says "~2,000 distinct sentences". `len(_sentence_pool())` is 3,610 (the commit message says
~3,600). v1's "mostly random single words" was about half: in v1 middle bodies about 1,250-1,300 of 2,600 tokens were
filler words, versus 27-42 in v2.

**m5. The registry defaults to `route` when `stage1_mode` is missing** (`serving/registry.py:208`). Since D23 the safe
default is observe. An owner edit that drops the key, or any old registry with a Stage 1 pointer (for example
`reports/cascade-and-shadow/registry_integration.json`), silently gives Stage 1 the power to finalize. Require the key
whenever `stage1` is non-null, or default to `observe`.

### Nits

- n1. `reports/cascade-and-shadow/replay_after/` and `probes_after.json` were measured on `d8dfca0`, before the anytime
  Stage 1 (`79962d1`). Re-running `run_checkpoint.sh` at HEAD would not reproduce them (Stage 1 timeouts become partial
  `ok`). Note the producing commit in the report directory.
- n2. `training/replay_counterfactual.py:94`: `both = [r for r in got]` is a no-op copy. `disagrees_with_offline_cascade`
  includes unlabeled items. That is fine, but say so.
- n3. D23 makes the replay set the decision set for future Stage 1 candidates. It has 306 positives, and its weak spot
  (gandalf_summ, 53 of 56 losses) is now known. A candidate tuned after seeing this will overfit it. Keep a
  second, untouched replay draw for the decision.

## Claims verified as correct (and how)

1. **Replay counterfactual reproduces exactly.**
   `python3 -m training.replay_counterfactual --served reports/cascade-and-shadow/replay_after/requests.jsonl --out <scratch>/rc.json`.
   Compared to the committed JSON, the output is identical except for the `served` path string. Confirmed values:
   DeBERTa-only 247/306 = 80.7%; offline cascade 191/306 = 62.4%; paired diff -0.1830, CI [-0.2269, -0.1392]
   (0 cascade-only, 56 DeBERTa-only discordant); 85 injections finalized allow, 56 of them DeBERTa-flagged
   (gandalf_summ 53, itw_jailbreak 2, sandbox_dev 1); Stage 2 share 1,035/2,000 = 51.75%; best equal-recall savings
   1.35% at low = 0.01.
2. **The replay cache is genuine.** Under the lock I re-scored the first 120 replay items (the script's own id order)
   with `training.cascade_scores.run("replay", 120, 2, out=<scratch>)`. Max |deb diff| = 0, max |s1 diff| = 0, and the
   window counts match.
3. **Kill-criterion logic is correct and not cherry-picked.** Under allow-only routing the cascade cannot gain recall,
   so "equal recall" means 0 DeBERTa-caught injections lost, and that is what the script tests. The descriptive sweep
   shows the criterion fails under any reasonable tolerance: at most 10.45% savings within a 2-point recall loss
   (low = 0.05, 3 lost). It also fails with the serving scorer (M2).
4. **The replay set was not used to choose a threshold.** `stage1-tfidf-v2.1` (gray zone 0.505068) was published in
   `09b5c21` at 22:43, before the replay set existed (`16829a4`, 22:58). The counterfactual's sweep is labelled
   descriptive.
5. **The offline routing direction matches serving.** Offline `s1 < low` -> allow and `s1 > high` (never, high = 1.0).
   Serving clears iff `sc < lo` and the language is in scope, and finalizes a flag only with `stage1_finalize_flag`
   (off). The scorer and language-gate differences are in M2.
6. **D23 served rates.** 248/303 and 199/1169 were recomputed (m2). The 26 `client_saturated` errors (3 positive, 16
   negative, 7 unlabeled; mostly short texts, max 1,273 tokens) cannot move the rates much. Worst-case bounds with all
   26 included: recall 81.0-82.0%, FPR 16.8-18.1%. They are client-side drops at `--max-inflight 8`, not gateway
   failures.
7. **Every D23 request was decided by DeBERTa.** `stages_run == ["stage2"]` on all 1,974 responses. The Stage 1 result is
   last, with role="shadow" (1,968 ok, 6 timeout). The metrics show `would_finalize_allow` 806, which is consistent
   with my route simulation (802 cleared without timeouts).
8. **v1 probe claim.** In `probes_before_v1.json` / `probes_after_v1.json` the middle-placement
   `benign_control_flag_rate` is 6/6 in both. v2 controls are 0/6 before (gateway-mvp). The generation invariants hold
   against the served planner (m4).
9. **Stage 1 latency results match the README table** (p50/p95, over-budget counts, chunks scored under the deadline),
   read from both JSONs. The deadline logic is correct: the margin is capped at half the budget, the deadline is
   measured from submission, and queued jobs return without scoring.
10. **`/health` `config.timeout_ms`** = 5,050 = Stage 2 5,000 + Stage 1 50 in observe mode
    (`ClassifierService.request_budget_ms`). `GatewayClient` derives 7.05 s from it (`d23/probes.json`).
11. **Tests pass**, each directory run separately under the lock: contract 109 passed; serving 310 passed, 1 skipped;
    training 72 passed, 2 skipped; redteam 385 passed, 4 skipped; benchmarks 27 passed.
