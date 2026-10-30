# Verification round 2: milestone cascade-and-shadow (fixes + dilution study)

Verifier: independent agent, 2026-10-30. Branch `milestone/cascade-and-shadow` at `620d944`, reviewed in worktree
`/home/claude/wt-verify2` (branch `verify/cascade-and-shadow-r2`). Scope: the round-1 findings
(`origin/verify/cascade-and-shadow:reports/verify/cascade-and-shadow.md`) against the fix merges `7f476c7`,
`2400888`, `9ce1d7f`; the dilution work (`75b779a`, `2668a83`, merge `620d944`); doc consistency with the committed
JSON outputs. `data/splits/frozen_test.jsonl` was not read. Scratch scripts are in my scratchpad and are not
committed. No long model scoring was rerun. Everything was re-derived from cached outputs, plus a 12-record fresh-model
spot check under the lock.

## Verdict: ready for owner review: **yes**

All five round-1 Majors are fixed, with code and tests. M2 has one doc gap left (serving README header, m-b below).
The dilution study is reproducible: both result JSONs regenerate byte-identically from the committed caches. Every
number I checked in `training/DILUTION.md` (all tables and all inline figures in sections 1-4) matches a committed JSON.
The two-pass parameters are chosen on val only. The frozen test is never opened. The paired intervals are correct
Newcombe hybrid-score intervals on the right discordant counts. I found no Majors. The Minors below are doc precision
fixes. The most useful one is m-a: the two-pass headline depends on a tie-break that val cannot see. None of them
changes a conclusion.

## Round-1 findings: status

| Finding | Status | Evidence |
| --- | --- | --- |
| M1 observe Stage 1 holds admission / CPU | **Fixed** (admission). The CPU coupling is now measured and disclosed, not removed. | `serving/service.py:351-358` releases admission in `finally` before `_observe_guarded`. `:389-407` adds its own slot bound (`stage1_observe_concurrency`, default 1). When the slot is busy the observation is dropped, not queued. `app.py:102` retires on `busy` (inflight or observe slots). Tests: `tests/serving/test_stage1_observe.py::test_observe_stage1_does_not_hold_admission_against_other_requests` (my round-1 repro, now B=`allow`), `::test_observe_work_is_bounded_and_drops_instead_of_queueing`. `serving/bench_results/observe_overhead.json` matches the README table: decision p50 652.0 -> 730.5 ms, p99 4,395.2 -> 4,706.2 ms, 0/360 decision diffs, 0 dropped. The per-repeat p50s (none 643-674, observe 713-750) do not overlap. The README now says "Not guaranteed across requests through the CPU". |
| M2 "served gray zone" was offline | **Fixed**, apart from one doc (m-b) | New `training/replay_counterfactual_served.py` runs the real `ClassifierService` in route mode with the real Stage 1, language detector and chunk plan, then DeBERTa on exactly the recorded rows. I recomputed from the committed `items`: 1,198/2,000 escalated, 48 lost (gandalf_summ 45), chunk-level recall 201/306, FPR 191/1,185, offline-low->gateway-escalate 98 score + 66 `unknown` = 164, decisions agree with the served run 1,991/1,995. These match my round-1 independent simulation (1,198, 48, 2.5% / 6.05%). RFC (`docs/RFC.md:185`), README and `training/CASCADE.md:69` now cite these numbers and give the real reason for the gap. `replay_counterfactual.py:12-16` and the JSON `view_kind` label the old view as an offline approximation. The serving README header still uses the offline figures unlabelled (m-b). |
| M3 shadow results leak into consumers | **Fixed** | `training/eval_harness.py:160-163` (truncated over active only; `shadow_truncated` kept apart). `redteam/judge.py:239-241` (check at :240) skips role="shadow". `redteam/sandbox.py:201-203` iterates `active_results`. `redteam/classifier.py` `detector_version` / `failure_reasons` are active-only, with `observe_*` counterparts. `training/replay_runner.py:92-122` adds an `observe_stage1` block in place of the meaningless agreement rate. `serving/metrics.py:20-34` adds a `role` label. `docs/contract_requests.md` is corrected. There is one test per consumer (`tests/redteam/test_observe_stage1.py`, `tests/training/test_harness_v12.py`, `tests/training/test_replay_runner.py`, `test_observe_failures_are_labelled_shadow_in_metrics`). `predicted_injection` of the Stage 1 result is `score >= 0.505068` = gray-zone low (manifest), so `below_low_share` is computed correctly. |
| M4 probe results undocumented | **Fixed** | `redteam/README.md:299-343` gives a v1/v2 x gateway-mvp/route/D23 table. I recomputed every cell from `probes_{before,after}{,_v1}.json` and `d23/probes.json`: all 10 rows match, including stages_run. Also verified: Stage 1 max 0.261 over 7 chunks on `middle_system_notice_action_gap_early_attack`, its control allowed the same way, D23 boundary max 0.0091. |
| M5 registry defaults to route | **Fixed** | `serving/registry.py:213-216, 328-333`: `stage1` without `stage1_mode` is rejected with `stage1_mode_missing`. `LoadedRegistry.stage1_mode` falls back to `observe` (`:186-189`). `registry_integration.json` is kept byte-identical, and `registry_integration_route.json` + `run_checkpoint.sh` use the explicit copy. Tests: `test_stage1_without_stage1_mode_is_rejected_not_routed` (startup degraded, reload 409, last good kept), `test_historical_integration_registry_is_rejected_and_its_route_copy_loads`. |
| m1 stale docs | **Fixed** | README status/layout/v1.2, `CASCADE.md:64-71`, `DATA.md:399-403`, `serving/README.md:238-240, 585-590`, `redteam/README.md:253-258`. |
| m2 D23 numbers only in a commit message | **Partially fixed** | `reports/cascade-and-shadow/d23/replay_counterfactual.json` is committed. Its `views.served.same_items` gives 248/303, 199/1,169 vs offline 245/303, 191/1,169 and the flip causes (coverage 5, failure policy 3, other 7). **No doc cites it** (see m-c). |
| m3 latency write-up | **Fixed** | `serving/README.md:301-313` now says 34-59% (1.55/2.62 = 59.2% is correct; my round-1 "62%" was rounding) and 24.17 of 50.69 ms, and explains Fix 1 correctly. Nit n-e on citing an uncommitted check. |
| m4 probe tests used old planner | **Fixed** | `tests/redteam/test_probes.py:146-189` (`test_probe_geometry_holds_for_the_served_planner`, :181) checks every attack against `ClassifierService._prepare` + `stage2_order(...)[:4]`. Comments in `redteam/probes.py:84-86, 246-249, 270-275` are corrected. The pool count is now 3,610. |
| n1 producing commit | Fixed | `run_checkpoint.sh:4-6`, `serving/README.md:585-587`, `redteam/README.md:301-304`. |
| n2 no-op copy / unlabeled | Fixed | `replay_counterfactual.py:133-137`. |
| n3 fresh replay draw | Fixed (as policy) | `docs/RFC.md:185` "on a fresh replay draw". |

## Dilution study: checks done

- **Reproducibility.** I re-ran `training.window_eval.build()` and `training.dilution_curve.summarize([510,128,64])` into
  scratch from the committed caches. Both are byte-identical to `training/results/window_eval.json` and
  `dilution_curve.json`.
- **Frozen test never read.** I ran both builds under a `sys.addaudithook` that raises on any `open` of a path
  containing `frozen_test`. It did not fire. Files opened: val, replay, the val cascade cache, the window/dilution caches
  and the timing JSONs. `window_scores.load_set` refuses `frozen_test` (`training/window_scores.py:64-65`). No new
  script names it otherwise.
- **Cache integrity.** I re-scored 12 records with a fresh `DebertaScorer` under the lock: replay W=64/128/256 model
  records, probe W=128/510, and five dilution keys including the headline `ignore_previous_forward` middle at L=128/510
  and a W=64 val_benign context. Max |diff| = 5e-7. The committed test only spot-checks 4 val records.
- **Every DILUTION.md number traces to a committed JSON.** Every cell of the section-1 tables, the W=64/128 table,
  the medians and the per-source counts (68/81, 0/60, 1/39, 7/20, 11/13) come from `dilution_curve.json`. The probe
  score sequences come from there too (1.000/0.987/0.135/0.005/0.001/0.002; start >= 0.972; `system_notice_action`
  <= 0.047 at W=510; >= 0.812 at W=64). All section-2 rows come from `window_eval.json`: recall/FPR, paired diffs,
  discordant counts, token ratios, benign_hard, long replay emails, per source. The counts 45 val / 120 replay items
  over 510 tokens and 870/3,214 over 128 also check out. The timing table and the batched comparisons match
  `window_timing_*.json`. The section-3 table, the 7/25 split by source, the derived ms, and the 27%/20% shares are in
  `window_eval.json`. I found no number that is missing from the outputs.
- **Two-pass chosen on val only.** `window_eval.py:233-234`: `grid(data["val"])` / `select(g)`. Replay and probes are
  only passed to `apply_two_pass` at the chosen setting. I re-derived the selection from `grid_val`: 56/80 admissible,
  max admissible val recall 2,169, chosen (128, 128, 0.9). There is no code path from replay/probes into the choice.
  The rule and the replay results landed in the same commit (`75b779a`), so git cannot show that the rule was fixed
  before replay was looked at. I also found nothing that suggests it was not. The doc reports honestly that the replay
  FPR rise (+2.11, UB +2.99) breaks the +2-point margin.
- **Paired CIs.** `cascade_sweep.newcombe_paired` is Newcombe's method 10, with the continuity-corrected phi for
  AD > BC, the correct dl/du terms and the sign p1 - p2. `window_eval.paired(x, y)` passes b = x-only, c = y-only, so the
  diff is two-pass/W minus 510. The `(+gained/-lost)` labels are `only_first/only_second`, as printed. The intervals
  are over the same items. They are correct.
- **Window construction vs serving.** `window_rows` uses `cascade_scores.window_starts`, which equals the *original*
  `serving/encoding.py` rule (`test_510_windows_are_the_gateway_chunks` compares against `encoding.chunk_ids`). The
  gateway now chunks with `serving/structure.py` (`build_chunks`/`plan_windows`: per-segment tokenization, paragraph
  boundaries, kinds kept apart, last window ending at n). I measured the gap on the replay items, using the
  serving-faithful counterfactual's recorded per-chunk DeBERTa scores (`items[].stage2_chunks[].deberta`, real
  planner). On the 1,139 escalated items whose every chunk was scored, the max served-chunk score equals the W=510
  window max exactly on 1,074. The window count equals the chunk count on 1,120. **5 items flip at 0.5** (2 labelled
  positive, 2 negative, 1 unlabeled). So W=510 is a close but not exact proxy for the served Stage 2 (m-d).

## Findings

### Major

None.

### Minor

**m-a. The two-pass headline rests on a tie-break that val cannot see. The doc presents t2 = 0.9 as val-chosen.**
`training/DILUTION.md:143-148`. In `window_eval.json` `two_pass.grid_val`, the settings (128, 128, t2) for
t2 in {0.5, 0.6, 0.7, 0.8, 0.9} are identical on val: recall 2,169, the same FPR, the same pass-2 tokens. t2 = 0.9 wins
only through the last rule, "prefer the larger t2". The val evidence for W2 = 128 over 510 is itself +4/2,240
(+0.18 [-0.06, +0.44], not significant). On replay the val-equivalent t2 = 0.5 gives 257/306 and FPR 227/1,185
(+2.78 pts, UB +3.78) (`two_pass.descriptive_t2_0.5`), against 254/306 and 219/1,185 at t2 = 0.9. The doc never quotes
these. Nothing leaks. But a reader of "Every parameter was chosen on val only. Chosen: ... t2 = 0.9" would take 0.9
for a val-tuned threshold, when val was indifferent between 0.5 and 0.9.
Fix: add one sentence after `:148`: "On val, t2 in [0.5, 0.9] are indistinguishable and the val recall gain is
4 items (CI includes 0); t2 = 0.9 is the tie-break. At t2 = 0.5 the replay result is 257/306, FPR 227/1,185
(+2.78 [+1.84, +3.78])." This strengthens the doc's own recommendation (do not block on it).

**m-b. The serving README header still states the offline counterfactual as the D23 result.**
`serving/README.md:8-11`: "Stage 1 finalizing `allow` cut recall from 80.7% to 62.4% (56 DeBERTa-caught injections
allowed, 53 of them from `gandalf_summ`) ... saved 1.35%". These are the figures that round-1 M2 asked to label as
offline. RFC/README/CASCADE now lead with the serving-faithful numbers (201/306, 48 lost, 45 gandalf_summ, 2.5%).
Fix: replace them with the serving-faithful figures and cite `replay_counterfactual_served_faithful.json`, or label
them "offline approximation".

**m-c. The D23 served replay result is committed but cited nowhere. DATA.md points to docs that do not contain it.**
`training/DATA.md:399-403` says the two replay runs' "results and their reading are in `training/CASCADE.md` and
`docs/RFC.md` (D23)". Neither contains the D23 deployment's measured rates. The served recall 248/303 = 81.8%
[77.1, 85.8] and FPR 199/1,169 = 17.0% [15.0, 19.3] appear in no `.md` (`grep -rn "248/303"` finds nothing). Nor
does the same-denominator comparison with offline DeBERTa-only (245/303, 191/1,169) or the flip causes. All are in
`reports/cascade-and-shadow/d23/replay_counterfactual.json` `views.served.same_items`.
Fix: add 2-3 lines to `CASCADE.md`'s D23 paragraph (or the RFC row) quoting these and the 5/3/7 flip causes. Or
point DATA.md at the JSON.

**m-d. DILUTION.md says W = 510 is DeBERTa "as the gateway runs it (`serving/encoding.py`)", and the test's name
says the same. The gateway now uses the structure-aware planner.**
`training/DILUTION.md:25`, `training/window_eval.py:8`, `training/window_scores.py:6`,
`tests/training/test_window_scores.py:39-42` (`test_510_windows_are_the_gateway_chunks` compares with
`serving.encoding.chunk_ids`, the pre-structure planner, as round-1 m4 found for the probes). Measured above: 5/1,139
fully scored replay items flip at 0.5 between the W=510 proxy and the served planner, and 65 differ in score. That is
small, but it is not "as the gateway runs it". The study also omits the `max_chunks = 4` cap and the coverage policy.
The doc admits this in `:206`. Under D23, every replay item with more than 4 chunks is coverage-flagged anyway, so the
two-pass gain on those items is partly already "won" by the coverage policy.
Fix: reword to "the original sliding-window planner (`serving/encoding.py`); the served structure-aware planner
(`serving/structure.py`) agrees on 1,074/1,139 fully scored replay items, 5 flips at 0.5". Rename the test, or add a
second test that compares with `build_chunks` on a plain single-paragraph input. Optionally, report the two-pass
replay gain restricted to items with at most 4 chunks.

**m-e. The served-faithful headline has no paired interval, although the offline one did.**
The RFC row used to give "-18.3 pt, 95% CI [-22.7, -13.9]". The new row (`docs/RFC.md:185`) and
`replay_counterfactual_served_faithful.json` give no paired difference. I recomputed it from `items` with the
repo's `newcombe_paired`. Chunk-level cascade minus offline DeBERTa-only: recall **-15.0 pt [-19.4, -10.7]** (3
cascade-only, 49 DeBERTa-only). FPR **-0.25 pt [-0.91, +0.40]** (5 / 8). The FPR result is new information: simulated
the way the gateway routes, the cascade also brings no FPR benefit. The offline view claimed -3.6 pt.
Fix: add a `paired` block to `replay_counterfactual_served.report()`, regenerate (no model needed if it is computed
from the existing `items`), and quote the recall CI in the RFC.

**m-f. The RFC demo script is stale after D23 and contradicted by the dilution study.**
`docs/RFC.md:284` step 2 ("Measured cascade numbers: share of requests resolved in Stage 1 ...") describes a cascade
that no longer decides. Step 3 (`:285`) is "after the chunking fix, the same payload is caught". DILUTION.md section 4
shows that a fully covered middle injection is usually still missed: 20/62 detected at L = 510 in newsletter context,
probes 1/12 at W = 510. The RFC's step-3 caveat covers the attacker-run case, not this one. DILUTION.md is linked from
neither the README nor the RFC.
Fix: update steps 2-3 (for example, step 2 becomes the D23 result: "the cascade failed its kill criterion; here is
the measured loss"; step 3 becomes coverage + the dilution limit). Add a README findings bullet linking
`training/DILUTION.md`: "coverage is necessary but not sufficient".

### Nits

- **n-a.** `README.md:30`: "served 202/306 = 66.0% while still sending 59.9% of requests to DeBERTa". The served
  share is 1,196/1,995 = 59.95%, which rounds to 60.0%. 59.9% (1,198/2,000) is the simulation's. The section header
  promises "Wilson 95% intervals", but none are given.
- **n-b.** The probe paired CIs (`DILUTION.md:109-111, 156`, for example "+41.7 [+5.1, +67.2]") treat the 12 attack
  probes as independent. They are 2 payloads x 6 placements. Say so, or drop the intervals for probes.
- **n-c.** `DILUTION.md:135-137`: "batched helped only at W = 64". Val W = 128 also improved slightly (226.5 -> 215.6
  ms) and replay W = 256 barely moved (506.6 -> 502.6). Write "materially only at W = 64".
- **n-d.** The "cond." column (`DILUTION.md:45`) conditions on the injections detected alone *at that W*. That is
  46/62 at every W, but not the same 46 (W = 64: probe 2/2 vs 1/2, jackhhao 10/13 vs 11/13). Across-W comparisons of
  "cond." are therefore over slightly different sets. Mention it once.
- **n-e.** `serving/README.md:309` cites "verifier's check of the bench inputs, not a committed output" to explain
  Fix 1. That is a number from an uncommitted script. Either commit a tiny script that prints the per-chunk scores of
  the bench inputs, or keep the sentence qualitative.
- **n-f.** With observations dropped under load (`outcome="dropped"`), `would_finalize_allow` and
  `observe_stage1.below_low_share` are rates over non-dropped observations. Drops cluster in time with other
  observations, so they are not uniformly random. Add one line to `serving/README.md` "Observe mode" saying the savings
  rate is conditional on not being dropped, and that the dropped count must be reported next to it. (In the measured
  run there were 0 drops, so this has no effect yet.)
- **n-g.** `window_eval.two_pass_ms` adds two separately timed passes, each including its own tokenization. A real
  two-pass run would tokenize once, so the derived two-pass ms overstates the cost slightly. The doc already says
  "derived".
- **n-h.** `tests/training/test_window_scores.py:177-192` fresh-model spot check covers only 4 val records (W = 128
  model, W = 64 cache copy). Add 1-2 replay and dilution records. My 12-record check above passed with max |diff| 5e-7.

## Tests (each directory separately, under `flock /home/claude/measure.lock`)

- `tests/training`: 118 passed, 2 skipped. Run with the DeBERTa model available (an untracked `models` symlink to the
  main checkout's gitignored `models/`, removed afterwards), so `test_cached_window_scores_match_a_fresh_computation_on_a_sample`
  ran. `tests/training/test_window_scores.py` alone: 41 passed, 0 skipped.
- `tests/serving`: 312 passed, 4 skipped.
- `tests/redteam`: 402 passed, 4 skipped.
- The tests/contract and benchmarks directories were not rerun this round. They are not touched by these merges,
  apart from the README.

## Scratch checks used (not committed)

- `sf_check.py`: recount of served-faithful numbers and the paired CIs in m-e.
- `planner_fidelity.py`: W = 510 proxy vs served-planner chunk scores (m-d).
- `regen.py`: regenerate `window_eval.json` / `dilution_curve.json` under a frozen-test audit hook.
- `dil_check.py`, `we_check.py`: print every DILUTION.md figure from the JSONs.
- `spot.py`: 12-record fresh-model check of the window and dilution caches.
