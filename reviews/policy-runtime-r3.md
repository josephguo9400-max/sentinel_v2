# Review: policy-runtime round-2 fixes (r3)

2026-10-30 · Reviewer: independent review agent (round 3) · Head `3df8c4b` (`feat/pr-r2-fixes`: 4d533ca, a709b95,
3df8c4b on `milestone/policy-runtime` `22ec311`) · Branch `verify/policy-runtime-r3` · Round 2:
`reviews/policy-runtime-r2.md` on `verify/policy-runtime-r2` (7f29bd1)

Written for this review (no product code changed):
- `reviews/policy-runtime-r3/test_reviewer_probes_r3.py`: 2 probes with fakes. `fixed_r2_m1_…` checks behaviour only
  (model call count, no new report fields). `r3_m1_…` reproduces the new Minor below, so a pass means it reproduces.

Nothing read `data/splits/frozen_test.jsonl`. No model run was needed: every check below uses fakes or committed reports.

## Verdict

**Mergeable.** r2-m1 is fixed in code: under `--no-cache` each engine now calls the model itself for every row. The
new test fails on the 22ec311 script at the behavioural assertion, and the doc relabelling is accurate. r2-n1 and
r2-n5 are fixed. Promote and rollback now write the audit line before the pointer moves, and the registry is replaced
atomically. r2-n2, n4, n6 and n7 are fixed in the four docs. Every changed doc number matches its cited file. No
decision-path file changed, and the served bundle is still `2ebc7d9ae676e800…`.

I found one new Minor, about evidence labelling (r3-m1). A `--no-cache` run that resumes an older progress file labels
the whole report as uncached. No committed report is affected, so it does not block the merge. There are also three
nits.

## Tests run (HEAD `3df8c4b`, each under `flock /home/claude/measure.lock`, one directory at a time)

| Target | Result |
| --- | --- |
| `tests/training` | 169 passed, 2 skipped (`test_replay.py:176`, `test_training.py:86`: raw sources not fetched). That is 9 more than r2's 160, all of them new tests. |
| `tests/serving/test_engine_version.py` | 5 passed |
| `tests/training/test_replay_policy_equivalence.py` with the **22ec311** script swapped in (then restored with `git checkout`) | 2 failed. The no-cache test fails at `assert h["cache"] is None` (a `RowCache(None)` is shared). The default-cache test fails on the new `deberta_rows.cache` key. |
| r2 probes (`7f29bd1:reviews/policy-runtime-r2/test_reviewer_probes_r2.py`, temp copy) | 12 passed, 2 failed. The 2 failures are exactly the r2 observations that were fixed: `test_r2_m4_an_error_after_the_split_was_read_leaves_only_started` and `test_r2_t3_no_cache_still_shares_an_in_memory_cache_between_both_engines`. |
| r3 probes at HEAD | 2 passed |
| r3 probes with the 22ec311 script | 2 failed. `fixed_r2_m1`: 3 model rows for 3 rows per engine, not 6. That is the old bug, seen through behaviour alone. |

Bundle hash recomputed from `artifacts/policies/d23-deberta-decides/1/policy.json`: `2ebc7d9ae676e800c12c90075db5ba509fa7aa5263572dec0298519415500240`.

Worktree setup: a `models` symlink as in the main checkout, and hard links of
`artifacts/baseline/deberta-pi-v2/onnx/{model.onnx,tokenizer.json,config.json}` from `/home/claude/wt-pr-fix2`.
None of these are committed.

## Round-2 findings

| # | Status | Evidence |
| --- | --- | --- |
| r2-m1 `--no-cache` shared cache | **FIXED** | `replay_policy_equivalence.py:365`: `cache = None` when `cache_path` is None. `build_engines` (`:225`) gives each `Recorder` `cache=None`, and `Recorder.infer` (`:196-207`) calls `self._det.infer` for every row. Both recorders wrap the same detector object, and nothing below it memoizes: there is no `lru_cache` or memo in `serving/`, and `ClassifierService` and `legacy_service` keep no score cache. Scores are compared with `THRESHOLD_TOLERANCE` (`differences`, `:301-321`). Measured with fakes: 6 model rows for 3 rows per engine (HEAD) against 3 (22ec311). The report records `cache: "none"`, `model_calls_this_run` and `this_run.no_cache`. The default (cached) path is unchanged. |
| r2-m1 docs | **FIXED** | README:31, RFC row 3.5, design §11a (:529-537) and serving/README (:800-815, first-300 at :810-813) now cite the two reviewer subsets as the uncached evidence. They say that `t3_d23_first300_nocache_engine110.json` supports row and decision equality only. No remaining "uncached" claim cites that report (grep over `*.md`, `*.py` and `*.sh`). The `run_equivalence.sh` T3 uses `--cache`, so it is unaffected. |
| r2-n1 frozen-test audit after scoring | **FIXED** | `gate.py` `evaluate`: the try now wraps `_evaluate_after_audit`, which covers `run_bundle`, `--active-report`, `build_report` and `write_json`. "completed" is appended after it, outside the try. So "started" alone means either the process was killed or the "completed" write itself failed, as the docstring (`:39-41`) says. The new parametrized test (3 stages) would fail on the old code. |
| r2-n5 record before acting | **FIXED** | `_audited_pointer_move` (`:364-376`): the audit line comes first, then `write_registry_pointers`. On any `BaseException` a `"status": "failed"` line follows and the exception is re-raised. If the first append fails, nothing moves. `write_registry_pointers` (`:348-361`) writes to `.<name>.tmp` via `with_name`, so the temp file is in the same directory and on the same filesystem. It validates the temp file with `read_registry_v2` (schema, keys, path syntax), runs `os.replace`, and in `finally` unlinks the temp file. An invalid state (`active="../escape.json"`) leaves the registry bytes unchanged (test). |
| `--approved-by` | **Holds** | `promote` and `rollback` are the only callers of `_audited_pointer_move`, which is the only caller of `write_registry_pointers`. The CLI has `required=True` (`:534`, `:541`), and both functions refuse blank names before any write. All other refusals (split, completeness, verdict, binding, already active) still come before the audit line, so a refused promote writes nothing. |
| r2-n2 T3 does not depend on FAIL_OPEN | **FIXED** | Design §11a. Confirmed: the D23 decide member is `FAIL_CLOSED`, and `policy_applied` is `null` on 2,000/2,000 items in the 1.1.0 T3. |
| r2-n4 gate-fallback `detector` no-op in shape B | **FIXED** | Design §11a. The cited validator rule exists (`contracts/policy.py:303`). |
| r2-n6 T4 is not D23 | **FIXED** | README, serving/README and design. The T4 report's bundle is `cascade-tfidf-allow-only` / `b12b7ebc…`, at 2,000 items and identical. |
| r2-n7 label the 1.0.0 T3 | **FIXED** in the four docs (see r3-n3 for the registry comment). |
| Frozen-test guard | **Holds** | `test_no_committed_gate_report_used_the_frozen_test`, `test_no_new_file_mentions_the_frozen_test` and `test_the_gate_is_the_only_new_reader_of_the_frozen_test` pass. The new gate tests replace `run_bundle` and use temp registries and audit logs. |

## Docs numbers checked against the files

| Claim | File | Matches |
| --- | --- | --- |
| T3 1.1.0: 2,000/2,000, one invocation at `916eceb`, 0 differences, bundle `2ebc7d9ae676…` | `reports/policy-runtime/t3_d23_equivalence_engine110_2ebc7d9a.json` | yes (`this_run.n_items` 2,000, one code version, clean tree, 439 s) |
| "every item record equal to the engine-1.0.0 record" | same file vs `t3_d23_equivalence.json` | yes: same 2,000 ids, 0 records differ |
| 1.0.0 T3 is bundle `17fac609a8f2…` | `t3_d23_equivalence.json` | yes |
| 6 coverage items, `stages_run` and `policy_applied` constant | both T3 reports | yes (`coverage_policy:flag` 6, `policy_applied` null on all 2,000) |
| r1 subset: 193 items, 459 rows per engine, 0 differences, gap 0.0, branch `verify/policy-runtime` 3cacec5 | `3cacec5:reviews/policy-runtime-r1/t3_subset_nocache.json` | yes (branch tip = 3cacec5) |
| r2 subset at 1.1.0: same numbers, branch `verify/policy-runtime-r2` 7f29bd1 | `7f29bd1:reviews/policy-runtime-r2/t3_subset_nocache_engine110.json` | yes (engine 1.1.0, git `916eceb`, branch tip = 7f29bd1) |
| first-300: 342 rows per engine, 0 differences | `t3_d23_first300_nocache_engine110.json` | yes |
| Val gate 1.1.0: 3,214 items, `INSUFFICIENT_EVIDENCE`, slices identical to 1.0.0 | `gate_val_d23_engine110_2ebc7d9a.json` vs `gate_val_d23.json` | yes (`slices` equal, complete, bound to `2ebc7d9a…`) |

## New findings

**Major:** none.

**Minor**

- **r3-m1. A `--no-cache` run that resumes a progress file labels every item as uncached, including items scored with
  the shared cache.** `training/replay_policy_equivalence.py:361` (the `done` set is read from
  `<out>.progress.jsonl` regardless of cache mode), `:383`, and `:444`/`:450-453` (`deberta_rows.cache` and `note`
  describe the whole report from this invocation's mode). Per-item records carry `code` but not the cache mode.
  Failure scenario: someone re-runs `t3 --no-cache --out X` where `X.progress.jsonl` was written by a cached run, or
  by any pre-fix `--no-cache` run. For example, they re-create the first-300 report under its own name to "fix" it.
  The run scores nothing new and writes `cache: "none"` and "no cache: each engine called the model for every row it
  scored". That is the r2-m1 overclaim again, now with the fix's own label on it. The only hint is
  `model_calls_this_run: {legacy: 0, policy: 0}`. A mixed resume (part cached, part not) is mislabelled the same way.
  Reproduced by `test_r3_m1_no_cache_resume_relabels_cached_items_as_uncached`. No committed report is affected:
  the progress files are gitignored, and the first-300 report predates the fix and is relabelled in the docs. Fix
  options: record the cache mode in each progress record and report counts per mode, or refuse to resume when the
  modes differ.

**Nit**

- **r3-n1. "Crash-safe" registry write is safe against a killed process, not against power loss, and has a few edge
  cases.** `training/gate.py:354-358`. There is no `fsync` of the temp file or the directory before or after
  `os.replace`. The temp file name is fixed (`.registry.json.tmp`), so two concurrent promote or rollback runs share
  it, and one's `finally` can unlink the other's file. There was no lock before either. The replaced file gets
  umask permissions, not the original mode. If the registry is a symlink, it is replaced by a regular file, and the
  temp file is created next to the link, not next to its target. None of these apply to the committed layout, where
  `artifacts/registry.json` is a regular file.
- **r3-n2. The first promote or rollback audit line carries no `status`, and the `failed` line has no correlation
  id.** `training/gate.py:367-372`. Anyone counting promotions from the log has to look ahead for a following
  `failed` line with the same action and bundle. Frozen-test runs use `started`/`completed`. A
  `"status": "started"` (or `"intent"`) on the first line would make the log self-describing. The over-reporting
  direction is disclosed in the docs and the docstring.
- **r3-n3. Two remaining unlabelled 1.0.0 references, outside the four docs.** The `comment` in
  `artifacts/registry.json` (line 3) still cites `reports/policy-runtime/t3_d23_equivalence.json` as the T3 evidence
  without naming the engine. The committed `t3_d23_first300_nocache_engine110.json` itself still says
  `this_run.no_cache: true` in a file named "nocache", with no note in the file. A reader of the JSON alone, without
  the docs, is misled. The registry comment is not part of the bundle hash, so fixing it is safe. Neither changes a
  decision.

## Decision path

`git diff --name-only 22ec311 3df8c4b` touches only `training/gate.py`, `training/replay_policy_equivalence.py`, two
tests and four docs. Nothing under `serving/`, `contracts/` or `artifacts/` changed.
`tests/serving/test_engine_version.py` passes (5), and the bundle hash is `2ebc7d9ae676e800…`, as before.
