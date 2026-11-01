# Review: milestone streaming, round 1

2026-11-01 · Reviewer: independent (did not write the branch) · Branch `feat/streaming` @ `7d060fe` against
`milestone/streaming` = `origin/main` @ `495e163` · Worktree `verify/streaming`

## Verdict

**Merge after the Major is fixed or its claims are narrowed.** The guard is sound where the committed measurement
uses it. Stall and `FAIL_CLOSED` (with the served D23 bundle) never release unchecked text, a flag withholds
everything not yet shown, the hold-back accounting is exact, and coalescing never skips a token. Every number in
the design note traces to the committed JSON, and a `--cache-only` re-run reproduces it byte for byte.

The problem is the stated semantics. The `FAIL_OPEN` contract ("a later check still runs over it") and the
"every token is in at least one scored window" guarantee are false once a windowed check times out, raises, or
gets a failed-allow from the bundle. Two docstring/table claims about stall mode are also stronger than the code.
The boundary-aligned injection flatters small B a lot at B = 0 and modestly at B = 16. That is now estimated
below and should be put in the note.

## What was verified

| Check | Result |
| --- | --- |
| `python -m serving.engine_version` | `engine 1.1.0: decision-path files match the record` |
| Files touched | `docs/design/streaming.md`, `serving/README.md`, `serving/streaming.py`, `serving/bench/{streaming,vclock}.py`, `serving/bench_results/streaming/*`, `tests/serving/test_streaming*.py`, `tests/training/test_gate.py`. None is in `serving/engine_version.json`'s decision-path list |
| Bundle | all 9,732 cache rows `bundle = 2ebc7d9ae676`; `live_check.json` bundle `2ebc7d9ae676e800…0240` |
| `flock … pytest tests/serving` | 923 passed, 1 skipped (66.8 s) |
| `flock … pytest tests/training` | 182 passed, 2 skipped (87.7 s) |
| `run --cache-only --out <scratch>` | `sweep`, `rates`, `backpressure`, `whole_response_once` and the whole per-stream file identical to the committed ones |
| Doc tables §5.1–5.4, §5.2 pool numbers, k = 8 zero-leak series (12/30/42/48/51), FAIL_OPEN 50.5 leaked tokens, 210 / 27,629 / 11,166 unchecked, live 24/24 and +22 ms | all match `streaming_sweep.json` / `live_check.json` |
| Cache composition | 8,996 allow + 736 flag, `failed` and `coverage` never set; longest check text 1,447 chars (no stream reaches the 510-token chunking) |
| Frozen test | not read. `build_streams` → `training/dilution_curve.build_inputs` reads val and the val cascade cache only |

Probes, all in the reviewer's scratchpad, none committed:
- `probes.py`: fake-classifier adversarial cases on the virtual loop.
- A lag-only coverage sweep: 216 `FAIL_OPEN` configurations, W ∈ {1, 8, 64}, k ∈ {1, 4, 16}, lag ∈ {16, 40, 64}, three latencies, three rates.
- `nonaligned.py`: the real D23 service under the lock, a scratch copy of the cache, 1,389 new checks.

## Major

### M1. Windowed `FAIL_OPEN` (and any failed-allow) leaves released text permanently unscored, against the stated contract

`serving/streaming.py:262-269` together with `:223`. When a check times out or raises under `FAIL_OPEN`, its span
counts as resolved: `checked_to` advances, its tokens stay `checked=False`, and `sched_to` has already moved past
it. The next window starts at `min(p - W, sched_to)`. So with W set, every token of the failed span earlier than
`p - W` is never scored by any check that returns. The same happens in any guard mode when the bundle answers
allow with `failed=True` (`rec.outcome = "failed_open"`, lines 265-268).

**Failure scenario** (`probes.py` P3). The setup is k = 16, W = 16, `FAIL_OPEN`, `check_timeout_s = 0.2`, and a
classifier that is slow on the window holding the injection (an adversary can push latency with long or odd
text). The check (64, 80) times out. Injection tokens 70–74 are released `checked=False`. The next windows are
(80, 96), (96, 112)… and the stream ends `Done` with all 5 injection tokens shown and never scored. P3b shows the
same hole with stall `FAIL_CLOSED`, W = 16 and one failed-allow result for the injection window.

**Contradicted text:**
- `docs/design/streaming.md:69`: "a later check still runs over it and can still stop the stream".
- `docs/design/streaming.md:50-51` and `serving/streaming.py:13-15`: "every token is in at least one scored window".
- `tests/serving/test_windows_cover_every_token` (`tests/serving/test_streaming.py:237-245`) counts spans whatever their outcome, so a timed-out span counts as "covered".

**Scope.** The committed numbers are unaffected. The harness never sets `check_timeout_s`, and the D23 cache has
no failed rows. The lag-only `FAIL_OPEN` path is sound: across all 216 lag-only configurations, every token ended
up inside an `allow`-scored span, because lag release does not move `sched_to`. But the §7 recommendation is
W = 64, and the doc presents `FAIL_OPEN` + timeout as a supported row.

**Fix options (author's choice):** on a failed span, leave `sched_to` at the previous resolved check (or set
`s = min(p - W, first_unscored)`) so the next window re-covers it; or document that windowed `FAIL_OPEN` with a
timeout gives no after-the-fact coverage. Either way, change the coverage test to count only `allow`/`flag`
spans and add a timeout + window case.

## Minor

### m1. "Stall cannot release unchecked text" is stronger than the code

- `serving/streaming.py:22-24` says stall is "the only setting that cannot release unchecked text even under
  FAIL_OPEN". `probes.py` P2 (`FAIL_OPEN`, no timeout, no lag budget, `cfg.stalls == True`, one exception): 8
  tokens are released `checked=False`. An exception is not "slowness", but the sentence says "cannot".
- `docs/design/streaming.md:67` ("stall … nothing unchecked is ever released") and the §2 invariant (`:72-75`)
  hold only while the bundle never answers allow with `failed=True`. In P1 (stall `FAIL_CLOSED`, a classifier
  whose own policy fails open), all 64 tokens, injection included, are released unchecked and the stream ends
  `Done`.

This is consistent with "the bundle's decision is never overridden", and D23's decide member is `FAIL_CLOSED`, so
it is unreachable on the served bundle. But it should be said, and the 324-cell grid
(`tests/serving/test_streaming.py:84-108`) never feeds a failed-allow result or a `FAIL_OPEN` exception.

**Verification:** add those two cases to the grid and assert the documented behaviour.

### m2. Boundary alignment flatters small B: estimated

The note says this is not measured (§5.1, §5.5). I measured it with the committed harness, injection streams
only, k = 16, 50 tok/s, stall. The injection starts at token 136, 8 tokens into a check window, instead of 128.
The pre=128 rows reproduce the committed sweep exactly, which validates the probe.

| W | B | zero-leak catches, start 128 (committed) | start 136 | mean leaked tokens per catch, 128 → 136 |
| --- | --- | --- | --- | --- |
| full | 0 | 29/124 | **8/124** [3, 12] | 8.71 → 10.51 |
| full | 8 | 29 | 39 | 5.33 → 3.98 |
| full | 16 | 46 | **39** [24, 40] | 1.88 → 2.51 |
| full | 32 | 50 | 47 | 0.31 → 0.65 |
| 64 | 0 | 42 | **24** [13, 27] | 8.85 → 11.94 |
| 64 | 8 | 42 | 55 | 5.88 → 7.17 |
| 64 | 16 | 60 | **55** [36, 53] | 2.82 → 5.72 |
| 64 | 32 | 64 | 61 | 0.94 → 3.12 |

Total catches move only slightly: 51 → 49 at full prefix and 68 → 72 at W = 64. The shift also moves 8 more
benign tokens in front of the injection, so this is not a pure alignment effect.

**Reading.**
- At B = 0 alignment roughly halves to triples the zero-leak count.
- At B = 16 it adds 5–7 of 124 zero-leak catches.
- At B = 32 it adds 3.
- B = 8 and B = 16 coincide at offset 8. B = 8 already covers the misalignment, and the rest is detector delay.
- The §5.1 bullet "k = 64 with B = 0 leaks nothing … the cadence acts as a buffer" is an artifact of alignment.
  With a random start, k = 64 / B = 0 would leak up to 63 tokens of a passed window.

Please add a non-aligned (or random-offset) row to the sweep. At minimum, qualify the B = 0 and k = 64 claims and
the §7 "B = 16 to 32" advice with these numbers.

### m3. The live cross-check validates the regime where timing does not matter

`serving/bench/streaming.py:375-377`; `docs/design/streaming.md:129-132`.

- The 24 live streams are the first 6 injections × 2 pools at W = 64, k = 16, 50 tok/s. In that configuration the
  classifier is busy about 25% of the time and never coalesces: every benign stream that completes runs exactly
  16 checks, both live and simulated.
- Agreement on stop check, released tokens and check count therefore follows from deterministic scheduling plus
  identical per-text decisions. It says little about the simulator's timing model where it drives results:
  full-prefix coalescing (busy 0.80 at 100 tok/s), lag stops, and the slowdown rows.
- The streams are also not a random sample.

**Suggestion:** add a live run at full prefix and 100 tok/s (coalescing), compare per-stream check counts, and
call the current check a scheduling/decision cross-check, not a timing validation.

### m4. Injection streams stopped before the injection are inside the "caught" denominator but unreported

`aggregate` (`serving/bench/streaming.py:228, 241`) computes `stopped_by_flag_outside_injection`: 16–23 of 124 per
config (18 at full / k = 16, 20 at W = 64). These are benign-prefix flags. The stream never reaches the injection
(live row `inj|val:2927…|val_benign`, stopped at check end 32), so these streams cap "caught" at about 101–108 of
124. They are false stops on injection streams.

The note's metric list (`docs/design/streaming.md:134-138`) and tables omit them. A reader cannot tell "caught
51/124" from "missed 73": in fact about 18 were stopped by benign text and are never shown the injection. Please
report the column. Related: with W = full, any flag from a check ending after token 128 counts as caught. The note
says this for `val_benign` (`:191-192`); a one-line note on the definition would help.

### m5. Unattributed industry claim in §4

`docs/design/streaming.md:95-96`: "Production output-side classifiers are generally described as scoring a
model's output as it streams and halting generation when a threshold is crossed". The section correctly declines
to attribute specifics, but this sentence is still a claim about industry practice with no source ("generally
described" by whom?). Either cite a retrievable source or rephrase it as the design's own premise. A related
softer case is `:31`, "a hijacked assistant usually does not repeat the injection verbatim": reasonable, unsourced,
and better phrased as an assumption.

### m6. The frozen-test guard does not tie the audit entries to the approved run

`tests/training/test_gate.py:194-195` checks only the status sequence of `gate_on_frozen_test` entries.

**Probe:** I rewrote both audit entries' `gate_report_path` to `reports/OTHER.json` and `actor` to `someone-else`.
The test still passed (file restored with `git checkout`).

**What the rest of the test does correctly:**
- A copy of the approved report under a new name (`gate_probe_x.json`) fails with "unapproved gate run on the
  frozen test", so an unapproved frozen-test report still fails.
- A non-frozen report with `frozen_test_approved_by` set still fails.
- A missing approved report fails.

**Status:** not weakened in what it checks about reports, only weaker than its docstring ("a started + completed
pair in the audit log" for the approved run).

**Suggestion:** for each approved path, assert one `started` and one `completed` entry whose `gate_report_path`
equals it, `actor` equals the approver, and `completed.bundle_sha256` equals the pinned bundle. Both entries in
`artifacts/audit_log.jsonl` already carry these fields.

## Nit

- **n1.** `serving/streaming.py:65-67`: `max_unchecked_tokens` is validated against k but not B. Under
  `FAIL_OPEN`, a lag budget below B makes B meaningless. In `probes.py` P6 (B = 64, lag 40), all 167 released
  tokens went out before any check covered them, and 17 injection tokens leaked. Validate it, or say in §2 that
  under `FAIL_OPEN` the lag budget, not B, bounds what is held back.
- **n2.** The commit `7d060fe` (frozen-test guard) is unrelated to streaming. It fixes main's red training suite
  and would be easier to audit as its own PR to main. Harmless here.
- **n3.** §5.2 "full prefix catches more than checking once (27 vs 23 of 62)" and §5.1 "catches change little with
  k": these are paired comparisons on the same streams, stated without a paired test. 4 of 62 is within noise.
  Say "not significantly different", or give McNemar counts.
- **n4.** Latency is one sample per distinct text, measured on first scoring (the second scoring run, 74 rows, ran
  under a 1.4–1.8 load average). Fine for the purpose; the note could say "single measurement per text".
- **n5.** A token is a DeBERTa piece, so B, k and W are in detector tokens, not generator tokens. With a real LLM the
  re-tokenized window size differs, and with W = k the window edges split words. The note mentions
  re-tokenization (`:122`); one sentence on units would complete it.

## Notes on what is right

- Hold-back: release is only to `checked_to - B` after a non-failed allow (`:269-272`), and the final check is
  required before the tail goes out (`final = source_done and checked_to == len(toks)`). On a flag, `stop()`
  releases nothing further, cancels the in-flight check and closes the source. The ordering "check result before
  token" in one wake-up makes a simultaneous flag win.
- Coalescing: `p = sched_to + k·⌊(n − sched_to)/k⌋` and `s ≤ sched_to` mean the jump to the newest boundary never
  leaves a gap. `probes.py` P5 (W = 1/4/16, 0.9 s checks, 100 tok/s) shows windows stretching to 88 tokens rather
  than skipping.
- `FAIL_OPEN` leak accounting: `released_unchecked` is counted at release time from `st.checked`, so text released
  by the lag rule before its check returns is counted as unchecked even if that check later allows it. That is the
  right definition for "shown before checked". `leaked` counts released injection tokens independently of it.
- The doc's threat-model section (§1) separates input-side and output-side clearly. It calls DeBERTa a stand-in,
  says the streams are synthetic, and states the D25 versioning gap (§3) and the no-endpoint decision plainly.
