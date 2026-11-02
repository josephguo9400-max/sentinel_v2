# Review: milestone streaming, round 2

2026-11-01 · Reviewer: independent (did not write the branch) · Branch `feat/streaming` @ `86d9004`. The fix diff
is `7d060fe..86d9004` (4 commits), read in the context of the whole branch against `milestone/streaming`.
Worktree: `verify/streaming-r2`. Round 1: `reviews/streaming-r1.md` on `verify/streaming` (`0af3c53`).

## Verdict

**Merge after M1-r2 is fixed or its claims are narrowed.** Every round-1 finding is addressed except n2, which is
harmless. Every `FAIL_CLOSED` configuration now releases only text that a check passed with `failed=False`. A
4,000-case randomized fuzz over every failure kind found no violation. The new measurements (injection offset,
paired McNemar counts, the live coalescing run) are committed as scripts, re-derive byte for byte from the cache,
and every number checked in §5.1, §5.2, §5.3 and the live paragraph traces to them.

The M1 fix itself has a problem. Its re-covering window has no bound. Under windowed `FAIL_OPEN` with a deadline,
one failure can therefore turn into permanent loss of the guard. This happens with a benign input and a transient
2 s classifier stall, not only with "an input that always times out". It is a regression against round 1 in that
regime, and doc §2 says the opposite. As in round 1, no committed number is affected, and the recommended
`FAIL_CLOSED` setting is not affected.

## Status of round-1 findings

| # | Finding | Status | Evidence |
| --- | --- | --- | --- |
| M1 | Windowed `FAIL_OPEN` / failed-allow leaves released text permanently unscored | **FIXED** (the hole). The fix introduces **M1-r2** | `serving/streaming.py:240`: the window starts at `min(p - W, scored_to)`, and `scored_to` advances only on a non-failed allow (`:284-288`). r1 probe P3b now stops `classifier_failure` at (64, 80) with nothing unscored released. The new single-failure tests re-cover the failed span. With the old window start restored in a scratch copy, 11 of the new tests fail, including `test_fail_open_window_rescores_a_timed_out_injection_window`, as the doc says. Fuzz (below): a stream that ends `Done` has every token either scored or released and counted `checked=False`. The coverage test now counts only `allow`/`flag` spans |
| m1 | "Stall cannot release unchecked text" was stronger than the code | **FIXED** | P1 (stall `FAIL_CLOSED`, failed-allow on every call) now stops at the first check, with 0 released. P2 (`FAIL_OPEN` stall plus an exception) still releases 8 unchecked tokens, and that is now documented: the `stalls` docstring (`:84-86`), the §2 table and guarantee paragraph (`docs/design/streaming.md:73-87`), and `serving/README.md:623-628`. Grid case `test_failed_check_under_each_setting` added |
| m2 | Boundary alignment flatters small B | **FIXED** | `offset` subcommand committed (`serving/bench/streaming.py:393`), output in `streaming_offset.json`. Re-run with `--cache-only`: `configs` and `paired` identical, `aligned_rows_match_sweep` 16/16. All 12 rows of the §5.2 table (zero-leak with Wilson %, leaked mean/max, caught at 128/120/136) match the JSON. "k = 64 with B = 0 leaks nothing" is withdrawn. Mechanism wording: see m2-r2 |
| m3 | Live check validated only the no-timing regime | **FIXED** | `live_check_coalescing.json` (full prefix, 100 tok/s): 280 vs 278 checks, same check count on 20/24 (each off by ±1), same stop check on 23/24, same released tokens on 24/24, +28 ms p50 first visible. All of this matches §5 text `:170-173`. The note now calls the check "scheduling and decisions, not timing under load" and lists what is not validated |
| m4 | Injection streams stopped before the injection were unreported | **FIXED** | `stopped_before_injection` and `caught_excluding_stopped_before` are in `aggregate`, with a test (`tests/serving/test_streaming_bench.py:56`). The §5.1 "stopped before inj." and "caught of streams reaching inj." columns match `streaming_sweep.json` in all 14 rows. The definition note is at `:179-184` |
| m5 | Unattributed industry claim | **FIXED** | §4 `:117-122` restates it as "this design's own premise". §1 `:31-32` states the verbatim-repeat claim as an assumption |
| m6 | Frozen-test guard did not tie audit entries to the approved run | **FIXED** | `tests/training/test_gate.py:196-203`. Mutations of `artifacts/audit_log.jsonl`, restored after each: both paths → OTHER, started path only, both actors, completed actor only, completed bundle changed or removed, `run_complete` false, order swapped, statuses swapped, an extra pair for another report. **All FAIL.** Unmodified: PASS. Still unchecked: timestamp order and the started entry's `limit` (n3-r2) |
| n1 | Lag budget not validated against B under `FAIL_OPEN` | **FIXED** | `:76-80` rejects it. P6 now raises `ValueError` |
| n2 | Frozen-test guard commits are unrelated to streaming | **NOT FIXED** | `7d060fe` and `04b1e0c` are still on this branch. Harmless, same as round 1 |
| n3 | Paired claims without a paired test | **FIXED** | `paired` subcommand and `streaming_paired.json`. k = 8 vs 64: 4/1, p = 0.375. Full prefix vs checked once (newsletter pool): 4/0, p = 0.125. W = 64 vs full: 20/3 and 12/0, p = 0.0005. All re-derive identically. I checked the exact binomial p-values by hand. `_mcnemar_p` has a unit test |
| n4 | Single latency measurement per text | **FIXED** | `:158-160` |
| n5 | Units are detector tokens | **FIXED** | `:96-98` and the module docstring `:35-36` |

### r1 probes re-run at HEAD (`probes.py`, virtual clock)

| Probe | r1 (`7d060fe`) | HEAD (`86d9004`) |
| --- | --- | --- |
| P1 stall `FAIL_CLOSED`, every check a failed-allow | `Done`, 64 tokens released unchecked | `Stop/classifier_failure`, 0 released |
| P2 `FAIL_OPEN` stall, one exception | `Done`, 8 unchecked | the same (now documented) |
| P3 `FAIL_OPEN`, deadline, W = 16, every check holding the injection slow | `Done`, injection unscored, **16** unchecked | `Done`, injection unscored, **136** unchecked (every later window re-covers 64.. and times out). See M1-r2 |
| P3b stall `FAIL_CLOSED`, W = 16, failed-allow on the injection window | injection released unchecked | `Stop/classifier_failure` at (64, 80), 0 unchecked |
| P4 non-aligned injection, full prefix | as before | unchanged |
| P5 coalescing with W ∈ {1, 4, 16} | windows stretch, none skipped | unchanged: (4, 92), (92, 184) |
| P6 `FAIL_OPEN` lag 40 < B 64 | 17 injection tokens leaked | `ValueError` at config time |

## Major

### M1-r2. The re-covering window is unbounded: one failure under windowed `FAIL_OPEN` plus a deadline can disable the guard for the rest of the stream

`serving/streaming.py:240` together with `:284-291`. After a check that did not score, `scored_to` stays put, and
every later window is `(scored_to, p)`. It grows by k per check, for as long as checks keep failing. Classifier
latency grows with text length (in the committed cache: 81 ms p50 below 300 characters, 332 ms at 1,200 or more).
So once the re-covering span is longer than what fits under `check_timeout_s`, every later check times out, and
nothing brings the window back to W. This is a ratchet. It does not need an adversary or "an input that always
times out".

**Failure scenario** (scratchpad `outage_sweep.py`; virtual clock; benign text with an injection at tokens
300-304; k = 16, B = 16, `FAIL_OPEN`, `check_timeout_s = 0.25`; latency 5 ms + 2.5 ms per token, so 64 tokens take
0.165 s and windows above 98 tokens time out; the classifier stalls from t = 1.0 s for `d` seconds):

| W | stall d | r1 scheduler (`7d060fe`) | HEAD |
| --- | --- | --- | --- |
| 64 | 1.5 s | `Stop/flag`, 48 unchecked, injection withheld | `Stop/flag`, 48 unchecked, injection withheld |
| 64 | 2.0 s | `Stop/flag`, 80 unchecked, injection withheld | **`Done`, 352/400 unchecked, injection released**, 22 timeouts, longest window 352 |
| 64 | 3.0 s | `Stop/flag`, 128 unchecked | **`Done`, 352/400 unchecked, injection released** |
| 16 | 2.0 s | `Stop/flag`, 96 unchecked | **`Done`, 352/400 unchecked, injection released** |

Up to the threshold the fix is as good as or better than round 1. Past it, a windowed guard becomes a full-prefix
guard that never succeeds. That is strictly worse than round 1, which lost one window's worth of coverage and then
recovered. W also stops bounding the cost of a check. With D23 a re-covering window past about 1,660 tokens hits
the coverage rule (4 chunks), which flags, and the stream stops `classifier_failure`. That stop is safe, but it is
another way the "re-cover" turns into something else.

**Contradicted text:**
- `docs/design/streaming.md:77-78` (the §2 table) and `serving/streaming.py:23-25`: "the next check re-covers it
  and can still stop the stream".
- `docs/design/streaming.md:84-86`: the stream is lost only "if every check over some text keeps failing (an input
  that always times out)". In the scenario above the input is benign and the failure is transient; it is the
  growth of the window that keeps it failing.
- `test_fail_open_window_keeps_rescoring_while_checks_keep_failing` pins the 136-unchecked P3 outcome as intended.
  That is fine as a statement of behaviour, but the doc presents it as needing an adversarial input.

**Scope.** No committed number changes: the harness sets no `check_timeout_s`, there are no failed rows in 11,707,
and lag-only `FAIL_OPEN` never fails a check, so `scored_to == sched_to` at every scheduling. The `FAIL_CLOSED`
recommendation (§7) is unaffected. `FAIL_OPEN` with a deadline is still a documented row.

**Fix options (author's choice):**
- Bound the re-cover. Score the unscored span as separate W-sized windows, or cap the window at W + k·m and, past
  the cap, give up on the oldest unscored span. Mark it permanently unscored (an `abandoned` outcome or a counter)
  and resume W-sized windows from `p - W`. That restores round 1's "lose one span, keep guarding" property and keeps
  the hole explicit.
- Or keep the behaviour and state it: under `FAIL_OPEN` with a deadline and W set, a failure lasting longer than
  about (timeout − base latency) / per-token latency − W tokens of generation disables all later checks.

**Verification:** add the outage case above as a test. With W = 64 and a 2 s stall, the injection at 300 must be
withheld (or the abandoned span must be reported).

## Minor

### m1-r2. `service_check` marks a successful deep scan as a failure

`serving/streaming.py:342`: `failed = out.policy_applied is not None or out.coverage_policy_applied is not None`.
The service sets `coverage_policy_applied = "deep_scan"` when the deep scan scored every remaining chunk (allow,
`unscored_chunks == 0`), or found a positive score (flag), at `serving/service.py:887-888`. Both of those are scored
decisions. Probe (scratchpad `deepscan_probe.py`, using `tests/serving/test_cascade.py` fakes):

- deep-scan full-coverage allow → `CheckResult(decision='allow', failed=True)`. Every `FAIL_CLOSED` stream stops
  `classifier_failure` on fully scored benign text, and `FAIL_OPEN` releases it marked unchecked.
- deep-scan positive flag (score 0.99) → `failed=True`. The stop reason is `classifier_failure`, not `flag`, so the
  harness `aggregate` would count it under `stopped_by_failure_or_lag`, not as caught.

The docstring (`:335-336`, "a coverage rule decided content no step scored") and doc `:68` describe the intended
semantics, and the code does not match them for `deep_scan`. For the other coverage values (`flag`, `allow`,
`quarantine`, and a deep scan that fell back, where the field holds the fallback value) `failed=True` is correct:
some content was unscored.

**Effect on committed numbers and the decision path: none.** D23's coverage mode is `flag` with `deep_scan: null`.
On D23, a coverage flag means no positive chunk, because `positive` returns before coverage (`service.py:899`). So
the only change on the served bundle is the stop-reason label of a coverage flag (`flag` → `classifier_failure`).
Both stop the stream. No `check_cache.jsonl` row has `coverage` set. No decision-path file changed.

**Fix:** `out.coverage_policy_applied not in (None, "deep_scan")`, or `failed = out.policy_applied is not None or
bool(out.unscored_by_any_step)`. Add the two deep-scan cases to `test_service_check_adapter_reads_decision_score_and_failure`.

Related, nit-level: the harness's `CachedScorer` still derives `failed` from `policy_applied` only
(`serving/bench/streaming.py:157`, used at `:187`). If a coverage row ever appeared, the simulator and the live
path (`service_check`) would disagree on the stop reason. Today that is unreachable.

### m2-r2. The §7 mechanism for "B = 16 covers any alignment" is mis-stated

`docs/design/streaming.md:374-375`: "an offset of o tokens needs B >= o, and at k = 16 the offset can be up to 15".
With o defined as in §5.2 (tokens into the check window), the passed check before the flag exposes k − o injection
tokens, so the condition is B ≥ k − o. The note's own §5.2 data contradicts B ≥ o. At k = 64 and start 120 (o = 56),
B = 8 gives 50/50 leak-free catches at full prefix (`:271`, confirmed in `streaming_offset.json`).

The conclusion survives: the maximum of k − o is k − 1 = 15, so B = 16 at k = 16. But it covers only the alignment
component. Detector delay over more than one check still leaks at B = 16 even when aligned (W = 64: 60 zero-leak of
68 catches). The note's caveat "measured only at offset 8" is present, and the B = 16 to 32 recommendation is
otherwise supported by the committed numbers. Please fix the formula and say "covers the alignment, not the
detector's own delay".

## Nit

- **n1-r2.** In §5.2 (and §5.1) the Wilson intervals are percentages, but in §5.2 they sit next to bare counts:
  "45 [28, 45]" reads as a point at its own upper bound. Label them "%" or give counts out of 124. All 36 intervals
  in the §5.2 table match the JSON when read as percent.
- **n2-r2.** `check_latency_ms_all_cached` and `_by_chars` are computed over the whole cache, not over the sweep's
  checks. A `run --cache-only` today reports n = 11,707, p50 171.8, p95 318.5, against the committed 9,732 / 169.8 /
  314.4, because `offset` added 1,975 rows. The doc's "over the sweep's 9,732 checks" (`:157`) is true of the
  committed file only. All result blocks (`sweep`, `rates`, `backpressure`, `whole_response_once`, per-stream) still
  re-derive identically.
- **n3-r2.** `tests/training/test_gate.py:196-203` does not check that `completed.at` ≥ `started.at`, or that the
  started entry has `limit: null`. Both mutations pass. Low value, since the report's `run.complete` is asserted.
- **n4-r2.** The McNemar p-values on 124 streams treat the 62 injections × 2 pools as independent pairs. §5.6
  (`:351-352`) says this for the intervals only. The newsletter-only comparisons (n = 62) are not affected.
- **n5-r2.** `serving/streaming.py:285` is a production `assert`, stripped under `-O`. The invariant does hold by
  construction: one check is in flight, and `scored_to` changes only when it completes.

## What was verified

| Check | Result |
| --- | --- |
| `python -m serving.engine_version` | `engine 1.1.0: decision-path files match the record` |
| Decision-path files touched by `7d060fe..86d9004` | none (only `docs/`, `serving/README.md`, `serving/streaming.py`, `serving/bench/streaming.py`, `serving/bench_results/streaming/*`, `tests/serving/test_streaming*.py`, `tests/training/test_gate.py`) |
| Bundle | all 11,707 cache rows `bundle = 2ebc7d9ae676`, `decision ∈ {allow, flag}`, `failed = False`, `coverage = None`. The live files carry `2ebc7d9ae676e800…0240` |
| `flock … pytest tests/serving` | 986 passed, 9 skipped (72.4 s). The 8 new skips are the `timeout` × no-deadline cells of `test_failed_check_under_each_setting`, by design |
| `flock … pytest tests/training` | 182 passed, 2 skipped (91.7 s) |
| `flock … pytest tests/contract` | 170 passed |
| `run --cache-only`, `offset --cache-only`, `paired` into scratch | sweep, rates, backpressure, whole_response_once, streams, the per-stream file, offset `configs` / `paired` / `aligned_rows_match_sweep`, and paired `comparisons` all identical to committed. Only the cache-wide stats differ (n2-r2) |
| Docs §5.1 (14 rows, new columns), §5.2 (12 rows × 3 starts), §5.3 McNemar, live coalescing paragraph | all trace to the committed JSON |
| Randomized fuzz (scratchpad `probes_r2.py` R1): 4,000 cases, both modes, k ∈ {1, 4, 8, 16}, B ∈ {0..32}, W ∈ {full, 1..64}, deadline and lag-budget variants, checks that randomly raise, failed-allow, failed-flag, are slow, or flag the marker | 0 violations. `FAIL_CLOSED` (1,918 cases): no unchecked release, and every released token lies inside an allow check that ended before its release. `FAIL_OPEN` (2,082): release spans contiguous, `released_unchecked` equals the sum of `checked=False` spans, and on `Done` every token is scored or counted unchecked (451 streams end with some unscored). Every case terminates with at most ⌈n/k⌉ + 1 checks (no unbounded re-scheduling). Windows reached 88 × W (the M1-r2 growth) |
| R4: `FAIL_OPEN`, W = 16, two exceptions on the injection window, then recovery | the re-covering check (48, 96) flags and stops the stream. The 3 injection tokens released during the failure stay released (the documented `FAIL_OPEN` cost) |
| Frozen test | not read. All streams come from val via `training/dilution_curve.build_inputs` |

Probe scripts (scratchpad, not committed): `probes.py` (r1), `probes_r2.py`, `outage_sweep.py`,
`deepscan_probe.py`, `mut_gate.py`.
