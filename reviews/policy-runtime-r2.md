# Review: policy-runtime round-1 fixes (r2)

2026-10-30 · Reviewer: independent review agent (round 2) · Head `916eceb` (`feat/pr-r1-fixes`, 15 commits on
`milestone/policy-runtime` `bc49454`) · Branch `verify/policy-runtime-r2` · Round 1:
`reviews/policy-runtime-r1.md` on `verify/policy-runtime` (3cacec5)

Written for this review (nothing in product code changed):
- `reviews/policy-runtime-r2/test_reviewer_probes_r2.py`: 14 probe tests with fakes. `fixed_*` tests assert that a
  round-1 finding no longer reproduces. `r2_*` tests assert a round-2 observation, so a pass means it is reproduced.
- `reviews/policy-runtime-r2/t3_subset_nocache.py` + `t3_subset_nocache_engine110.json`: the round-1 uncached subset
  script re-run unchanged at engine 1.1.0 (only the header and a provenance field are new).
- `reports/policy-runtime/t3_d23_equivalence_engine110_2ebc7d9a.json`: the full T3 at engine 1.1.0.
- `reports/policy-runtime/gate_val_d23_engine110_2ebc7d9a.json`: the val gate dry run for bundle `2ebc7d9a…`.

Nothing read `data/splits/frozen_test.jsonl`.

## Verdict

**Mergeable.** Both Majors are fixed, and so are 9 of the 10 Minors. m7 is left to the owner by design, and the
fixer's extra `contracts/policy.py` edits are logged for sign-off. All 9 nits are fixed. The merge condition holds at
the fix-branch HEAD: the full T3 at engine 1.1.0 found 0 differences on 2,000/2,000 items, and every item record
equals the committed engine-1.0.0 record. The val gate result for the new bundle is identical to the 1.0.0 one.

The fixes themselves are sound. I found no new Major. I found one new Minor about evidence (r2-m1): `t3 --no-cache`
still shares an in-memory score cache between the two engines, so the fixer's "first 300 items, no cache" report is
not uncached, and four docs cite it as uncached evidence. My truly uncached re-run (193 items, below) gives that
evidence instead. Fix the docs wording before or right after merge. The owner should still sign off m7 and the
`scope_languages_required` validator.

## Test results (HEAD `916eceb`, each directory under `flock /home/claude/measure.lock`)

| Directory | Result |
| --- | --- |
| `tests/contract` | 170 passed |
| `tests/serving` | 568 passed, 1 skipped (`test_quantize_script.py:57`, int8 weights not generated) |
| `tests/training` | 160 passed, 2 skipped (raw sources not fetched) |
| `tests/redteam` | 417 passed, 4 skipped (`test_firewall.py:122`, by design) |
| `tests/benchmarks` | 27 passed |
| `reviews/policy-runtime-r2` (probes) | 14 passed |

Round-1 probes (`reviews/policy-runtime-r1/test_reviewer_probes.py`) re-run at HEAD: all 5 now fail, which is what a
fix should do. Four fail on the corrected behaviour. `test_r3_…` fails earlier, because the committed val report is
bound to the engine-1.0.0 bundle and no longer matches. So the active-bundle refusal was re-probed separately with a
report bound to the current bundle (`test_fixed_m2_promoting_the_active_bundle_is_refused_and_previous_kept`).

Worktree setup: `models` symlink as in the main checkout, and hard links of
`artifacts/baseline/deberta-pi-v2/onnx/{model.onnx,tokenizer.json,config.json}` from `/home/claude/wt-pr-fix`.
None of these are committed.

## Round-1 findings

| # | Status | Evidence |
| --- | --- | --- |
| M1 T3 not reproducible | **FIXED** | `--registry` defaults to `reports/policy-runtime/registry_v1_d23.json`. The full T3 ran at HEAD in one invocation (git `916eceb`, clean tree, engine 1.1.0, recorded per item). The report records the registry sha256, a repo-relative cache path and per-run provenance. The probe `test_fixed_m1_t3_reads_the_preserved_v1_registry` passes. |
| M2 parallel FAIL_OPEN skips coverage | **FIXED** | `serving/service.py:996` `coverage_applies`. The r1 scenario now returns `flag`, `coverage_policy:flag`, `policy_applied=null`. The fixer's T7 property test (48 cells) and the 16-cell multi-chunk T6 matrix pass. Review of the carve-out is below. |
| m1 band without scope | **FIXED** | `contracts/policy.py:410-415` `scope_languages_required` (gate and observe bands with `clear_below > 0`). A flag-only band is still accepted. All three committed policies still load (probe). |
| m2 promote active overwrites previous | **FIXED** | `training/gate.py:380-382` refuses by path or by bundle hash, also with `--allow-not-pass`. The registry and audit log are untouched (probe). |
| m3 promote ignores completeness/split | **FIXED** (see note) | `training/gate.py:360-368` refuses `complete != true`, a `--limit` run, items < split, and a split outside `GATE_SPLITS`. Note: a complete **val** report is still promotable (probe `test_r2_m3_…`). The split is recorded in the audit entry (`gate_split`), and the RFC does not require the frozen test for promotion, so this is an owner policy question, not a defect. |
| m4 frozen-test crash unaudited | **FIXED** | A "started" entry is written before `run_bundle`, and "failed" is written on any `BaseException` (probe, KeyboardInterrupt). A gap remains after the run: r2-n1. |
| m5 docs overstate T3 / understate benign-hard | **FIXED** | README:31-32, serving/README:781-790, RFC row 3.5 and design §11a now say the scores are identical by construction and that `stages_run`/`policy_applied` are constant. Benign-hard: 33/92 = 35.9% [26.8, 46.1], "will FAIL once it can be evaluated", which matches the committed report. The new "uncached" wording is wrong for the 300-item subset: r2-m1. |
| m6 milestone numbering | **FIXED** | README says "Milestone 3.5 … numbered as in the RFC". No other "milestone 4 = policy-runtime" remains. |
| m7 contract ownership / snapshot | **Left for owner (as intended)** | `tests/contract/schema_snapshot.json` is unchanged on the branch. The fixer's further edits to `contracts/policy.py` (the `scope_languages_required` validator, sorted `scope_languages`, the docstring §11a) are logged in `docs/contract_requests.md` (last row, "Pending owner sign-off, together with review r1 m7"). Confirmed. |
| m8 v1.3 breaks v1.2 readers | **FIXED** | RFC "Contracts after v1.1" compatibility paragraph, README layout row, serving/README compatibility bullet. |
| m9 T4 not bound to committed policy | **FIXED** | The T4 report `bundle.differences_from_committed_policy` lists only the 3 timeout fields, `only_timeout_fields_differ: true`, and the committed bundle is recorded (`b12b7ebc…` at 1.1.0). Re-run at 1.1.0: 201/306, 191/1,185, 0 differences. |
| m10 two FALLBACK clocks undocumented | **FIXED** | Design §11a now lists both clocks, the gate terminal overriding members after a gate failure, and deep scan skipped after a member fallback. The parallel stage after a gate failure now runs under the gate-fallback budget (`service.py:921-922`). Review below. |
| n1 legacy "byte-identical" | FIXED | The header is precise, and `test_legacy_service_is_the_frozen_pre_refactor_file` pins the sha256. |
| n2 contract_requests row | FIXED | The `escalation_reason` row now has its columns restored and the Decision filled. |
| n3 §12 → §11a | FIXED | `contracts/policy.py:13` |
| n4 frozen rationale | FIXED | `serving/convert_registry.py:66-70` comment. `engine_version` is pinned per source. |
| n5 benign-hard "why" | FIXED | `training/gate.py:228-230` |
| n6 frozen-test guard | FIXED | Git-free pinned set, plus a git check over added and modified lines with a base fallback. |
| n7 probe_runner overwrite | FIXED | `_stage_scores` keys duplicates as `<key>:<detector_id>`, with a test. |
| n8 "three timeout fields" | FIXED | The docstring says two. |
| n9 scope order hashes | FIXED | Sorted in the validator (probe `test_fixed_n9_…`). Committed hashes are unchanged (all `["en"]`). |

## Review of the fixes

**M2 and its carve-out (`service.py:987-1015`).** The coverage rule now applies when the aggregated vote is allow,
content is uncovered, and *some* member (or its fallback) scored a chunk. In that case a fail-open vote cannot hide
unscored content: the failed member would have scored the same `chosen` chunks as the healthy one, so the unscored set
is the same. I walked every combination of {any_flag, all_flag} × {ok/failed member} × {FAIL_OPEN, FAIL_CLOSED,
FALLBACK ok/fails} × {nothing uncovered, something uncovered}. The outcome is never more permissive than the same vote
succeeding, and a flag is never turned into an allow.

The carve-out says that when no member scored anything, the failure policy alone decides, so FAIL_OPEN allows. That is
the same outcome a single FAIL_OPEN decide member gives on the same multi-chunk text (probe
`test_r2_carveout_matches_the_single_member_fail_open_semantics`, and T6 multi-chunk row `decide/FAIL_OPEN`). It is
consistent with §4: FAIL_OPEN is the policy explicitly saying "allow when the detector did not answer", and
`lint_policy` warns on every FAIL_OPEN decide member. Applying coverage here would instead make FAIL_OPEN unreachable
for any multi-chunk text, which is a different policy. It is safe for what is served: D23's decide member is
`FAIL_CLOSED` and shape B is not served.

Two small inaccuracies in how the fix is documented:
- **r2-n2.** Design §11a row "Engine 1.1.0" (`docs/design/capability-policy.md:513`) says the single-member FAIL_OPEN
  semantics are kept because "T3 depends on it for D23". The committed D23 policy's decide member is `FAIL_CLOSED`,
  and T3 has 0 failure-decided items (`policy_applied` is null on 2,000/2,000). T3 does not depend on it. The
  rationale should be legacy semantics, not T3.
- **r2-n3.** With `coverage: allow`, a failed FAIL_OPEN member plus a healthy member gives `decided_by:
  coverage_policy:allow` and `policy_applied: null` (probe `test_r2_coverage_allow_after_a_failed_member_…`). The
  failure is visible only in `results[].status` and `info.members`, and a consumer that counts failure-policy
  decisions from `policy_applied` will miss it. This is disclosed in §11a ("the failure visible in the member's
  result") and no decision is wrong, so it is a nit.

**m10 gate-fallback budget (`service.py:917-928`).** After a gate failure the parallel members now run until
`t0 + gate.fallback.timeout_ms`, as the single-member path does (`:764`). That is correct for the member the gate's
FALLBACK names. It also applies to the *other* member, which the gate's FALLBACK does not name (probe
`test_r2_gate_fallback_budget_also_bounds_the_member_the_gate_fallback_does_not_name`: B, own budget 2 s, times out
under a 150 ms gate-fallback budget). §11a documents this deliberately ("the decide stage *is* the gate's FALLBACK"),
and `request_budget_ms` already takes the max of both, so no request outlives its advertised budget. I accept it. It
does mean the gate FALLBACK's `detector` field has no effect in shape B beyond the validator's "must be a member"
check. **r2-n4** (a note in §11a would help).

**gate.py promote refusals and audit ordering.** The refusals are checked in this order: split, completeness,
verdict, binding, already-active. All of them come before `write_registry_pointers`, so a refusal writes nothing. The
fixer's tests and my probes confirm the registry bytes and the log are untouched. For a frozen-test evaluate:
"started" is written before the split is opened, and "failed" is written on any exception inside `run_bundle`.
Remaining gaps:
- **r2-n1.** `training/gate.py:446-452`: the try covers only `run_bundle`. An exception after the whole frozen split
  was scored leaves just "started" in the log. That includes a missing or bad `--active-report` (`:456-462`), as well
  as `build_report` and `write_json` (`:463-466`). The docstring (`:38-40`) reads "started" alone as "a killed
  process". The look at the frozen test is still recorded, so the audit property holds, but the status misleads
  (probe `test_r2_m4_an_error_after_the_split_was_read_leaves_only_started`). Widening the try to the end of the
  function would fix it.
- **r2-n5.** `promote` (`:383-384`) and `rollback` (`:426-427`) write the registry pointer and only then append the
  audit line. If the append fails (disk full, permissions), the pointer has moved with no audit record. This is the
  opposite of the "record before acting" order adopted for m4. It is pre-existing and owner-only.

**`scope_languages_required` (`contracts/policy.py:410-415`).** The logic is correct. It refuses only when the
capability declares languages, the band has no scope, and the band clears something (`clear_below > 0`). Together with
the existing `undeclared_languages` rule, a clearing band now always needs an explicit scope, and therefore a
`language_detector`. That matches the legacy behaviour (the gateway always gated on `input_languages`) and §4. It
applies to the observe what-if band too, which is consistent, though it is a behaviour-free band. All three
committed policies validate. Sorting `scope_languages` in a field validator changes the hash only for unsorted
inputs, and none are committed. The registry's file pin is on raw bytes and is unaffected. No issue found. It still
needs owner sign-off (m7).

### New findings

**Major:** none.

**Minor**

- **r2-m1. `t3 --no-cache` still memoizes scores across the two engines, so the "uncached" 300-item report at engine
  1.1.0 is not uncached evidence.** `training/replay_policy_equivalence.py:355` (`cache = RowCache(cache_path)`,
  built even when `cache_path` is None), `:217` (both Recorders get the same cache), `:703`. The bug is
  pre-existing since 9ac0e92; round 1 missed it because its own script passed `cache=None` to `build_engines`. Failure
  scenario: `--no-cache` gives an in-memory `RowCache(None)` shared by both engines. The legacy engine scores a row
  and the policy engine gets the memoized value, so scores are identical by construction exactly as in the cached
  run. The committed `t3_d23_first300_nocache_engine110.json` shows it: `cache_hits 343`, `cache_misses 341` for 342
  rows per engine, which is about one model call per row, not two. The probe
  `test_r2_t3_no_cache_still_shares_an_in_memory_cache_between_both_engines` shows one model call for two engines.
  Yet README:31 and :33, serving/README:795-797, RFC row 3.5 and design §11a (:529-530) present that report as
  uncached evidence that "the scores agree too". The module docstring (`:38-40`) also promises `--no-cache` "scores
  every row again in each engine and compares with THRESHOLD_TOLERANCE". Row selection is still compared exactly, so
  the decision-equivalence claim is unaffected, but the score-agreement claim for that subset is unsupported. Truly
  uncached evidence at 1.1.0 now exists: `reviews/policy-runtime-r2/t3_subset_nocache_engine110.json` (below). Fix:
  pass `None` to `build_engines` when `cache_path` is None (or give each engine its own cache), then re-run or relabel
  the 300-item report.

**Nit**

- **r2-n1 … r2-n5** above.
- **r2-n6.** README:33 and serving/README:793-795 cite "T4 re-run at 1.1.0: identical on all 2,000 items" as evidence
  that "the served D23 path did not change". T4 checks the cascade route policy (`cascade-tfidf-allow-only`), not
  D23. It shows the gate/band/language code is unchanged, not the D23 path. The D23 evidence is the full T3 at 1.1.0
  (below).
- **r2-n7.** README:31, the RFC row 3.5 and design §11a cite `t3_d23_equivalence.json` without saying it is the
  engine-1.0.0 run (bundle `17fac609…`). The report itself says so, and design :527 labels T5, the smoke and the val
  gate as 1.0.0, but T3 is not named there. Docs should point at the engine-1.1.0 reports committed here.

## T3 at engine 1.1.0 (the merge condition, re-run in full)

`reports/policy-runtime/t3_d23_equivalence_engine110_2ebc7d9a.json`, produced by
`python -m training.replay_policy_equivalence t3 --out reports/policy-runtime/t3_d23_equivalence_engine110_2ebc7d9a.json --max-seconds 900`
(the script's entry point, defaults otherwise), under the lock.

- **2,000/2,000 items compared, 0 items with differences, `identical_on_every_item: true`.**
- One invocation, 439 s. Provenance: git `916eceb…`, `git_dirty: false`, engine 1.1.0, engine record
  `c4dc5e43…`, all 2,000 items under that one code version.
- Bundle: committed `artifacts/policies/d23-deberta-decides/1/policy.json` = **`2ebc7d9ae676e800…`** at 1.1.0. The run
  policy differs only in `decide.members[0].timeout_ms` (5,000 → 600,000) and `observe[0].timeout_ms` (50 → 600,000)
  (`only_timeout_fields_differ: true`).
- Decisions: allow 1,247 / flag 753. `decided_by`: `deberta-pi-v2` 1,994 / `coverage_policy:flag` 6. Observe:
  would_escalate 1,198 / would_finalize_allow 802. `stages_run` and `policy_applied` are constant (as disclosed).
- DeBERTa rows: 2,314 per engine, 2,285 unique.
- Row cache: no existing cache was available to reuse (the 1.0.0 run's cache lived in another session's scratch
  directory and is gone), so a fresh one was built for this run
  (`reports/policy-runtime/t3_d23_equivalence_engine110_2ebc7d9a.rowcache.jsonl`, gitignored). As disclosed, scores
  in this run are therefore shared by construction. Rows are compared exactly.
- Every per-item record (decision, decided_by, policy_applied, coverage_policy_applied, stages_run, path,
  observe_outcome, n_rows) equals the committed engine-1.0.0 record on all 2,000 items (compared by id).

**Uncached subset at 1.1.0 (truly uncached).** `reviews/policy-runtime-r2/t3_subset_nocache_engine110.json`: the
round-1 script with `cache=None` per engine. **193 items, 459 rows per engine, 0 engine differences, 0 differences
from the committed records, maximum active-score gap 0.0**, 349 s.

## Gate dry run on val, bundle `2ebc7d9a…` (engine 1.1.0)

`reports/policy-runtime/gate_val_d23_engine110_2ebc7d9a.json`, produced by
`python -m training.gate evaluate --registry artifacts/registry.json --split val --out … --max-seconds 900`.
All 3,214 items in one invocation (563 s), `complete: true`, bound to `2ebc7d9ae676…`, `promotes: false`.

| Check | Result | Verdict |
| --- | --- | --- |
| Recall | 2,164/2,240 = 96.6% [95.8, 97.3] | INSUFFICIENT_EVIDENCE (no active bundle to pair with) |
| Benign-hard FPR | 33/92 = 35.9% [26.8, 46.1] | INSUFFICIENT_EVIDENCE (n < 100); certain to FAIL once n reaches 100 |
| Compute saved | 3,214/3,214 escalated | NOT_EVALUATED (no gate step) |
| Error rate | 0/3,214 | PASS |
| p99 / adaptive ASR | not run | NOT_EVALUATED |
| **Overall** | | **INSUFFICIENT_EVIDENCE** |

Overall FPR is 88/974. Decisions: allow 962 / flag 2,252, with 1 coverage decision. Every slice's confusion matrix
(overall, per language, per source, per benign-hard family) is identical to the engine-1.0.0 report
`gate_val_d23.json`. That report stays committed and describes bundle `17fac609…`.

## Docs numbers spot-check

These match the committed outputs: T3 1.0.0 (2,000 / 0 differences, 6 coverage), T4 (201/306, 191/1,185, 1,198
escalated, re-run at 1.1.0), gate val (2,164/2,240, 88/974, 33/92 [26.8, 46.1], 1 coverage), first-300 report (342
rows per engine), the 193-item reviewer subset (459 rows), the "48 cells" (2×3×2×2×2) and "16 more" (2×2×4) test
counts, the bundle hashes `17fac609…` / `2ebc7d9a…` / `b12b7ebc…`, and "33/660 = 5%".

Engine/bundle labelling is clear for the gate val, smoke and T5 (1.0.0) and for T4 and the 300 subset (1.1.0). The
exceptions are r2-n6 (T4 cited for D23), r2-n7 (T3 not labelled 1.0.0) and r2-m1 ("uncached").
