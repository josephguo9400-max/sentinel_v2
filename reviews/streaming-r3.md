# Review: milestone streaming, round 3

2026-11-01 · Reviewer: independent (did not write the branch) · Branch `feat/streaming` @ `97b63b6`. The fix diff
is `86d9004..97b63b6` (4 commits: `b26438b`, `a97c36d`, `dff84a3`, `97b63b6`). Worktree: `verify/streaming-r3`.
Round 2: `reviews/streaming-r2.md` on `verify/streaming-r2` (`b39550e`).

## Verdict

**Merge after M1-r3 is fixed or its claims are narrowed.** The narrowing is a doc change; the recommended
`FAIL_CLOSED` setting and every committed number are unaffected.

The round-2 findings are addressed. m1-r2 and n1-r2 to n5-r2 are fixed. The specific M1-r2 ratchet is gone: no
check is now longer than a regular window, and the reviewer's 2 s stall row catches the injection and gives up
nothing. Given-up spans are recorded, counted and never overlap scored text. The stated check-count bound held in
20,000 independent fuzz cases. `FAIL_CLOSED` still releases only text that a check passed with `failed=False`.

The fix still leaves one way for a transient outage to disable preventive checking (M1-r3). The cause is that
retries always run before regular checks. After an outage the guard clears the retry backlog back to back. The next
regular window then holds every token that arrived meanwhile, so it is longer than the deadline allows and times
out. That failure creates new backlog, and the cycle can last for the rest of the stream. Every later token is then
released unchecked and scored only afterwards by a retry. This needs windowed `FAIL_OPEN`, a deadline, and a
classifier near its capacity. Round 1's scheduler recovers in every such case I ran. The docstring says the opposite
("a transient outage cannot grow the window until every check times out"). One §7 sentence also has an off-by-one
(m2-r3), which round 2's own wording introduced.

## Status of round-2 findings

| # | Finding | Status | Evidence |
| --- | --- | --- | --- |
| M1-r2 | Unbounded re-covering window | **FIXED as stated; M1-r3 remains** | `serving/streaming.py:278-292`: a regular window is the last W tokens extended over the new (coalesced) tokens. Failed text that leaves it becomes retry debt, retried oldest first in pieces of at most L = max(W, k) (`:286-289`) and given up after `retry_attempts` failures (`:339-346`). r2 `outage_sweep.py` at HEAD: every W ∈ {16, 64}, stall ∈ {0.5, 1, 1.5, 2, 3} s ends `Stop/flag` with the injection withheld; the longest check is 64 (W = 64) or 48 (W = 16, coalesced). The two new outage tests fail on `86d9004`: I swapped in its `streaming.py` and 9 of 9 failed (7 on behaviour, 2 on the missing `abandoned` attribute) |
| m1-r2 | Deep-scan success marked failed | **FIXED** | `outcome_failed` (`:391-399`): `policy_applied`, or `coverage_policy_applied ∉ {None, deep_scan}`. Checked against `serving/service.py:873-889`. The service sets `deep_scan` only when the deep scan leaves no unscored chunk, or a score crossed the threshold. A deep scan that fell back carries the fallback value (`:905-913`, and `_finish` `:855-859`). So `deep_scan` is always a scored decision, and `flag`/`allow`/`quarantine` always mean unscored content was decided by rule. The live path (`service_check`, `:412`) and the simulator (`CachedScorer`, `serving/bench/streaming.py:157`, then `simulate` `:187`) now share the function. New tests cover a deep-scan allow and a deep-scan flag through the guard (stop reason `flag`), and a coverage flag/allow (`classifier_failure`). The cache has no `coverage` row, so the change cannot move a number |
| m2-r2 | §7 mechanism mis-stated | **FIXED, with an off-by-one** | B ≥ k − o is now stated, with "covers the alignment, not the detector's own delay". All quoted numbers match the §5.2 table and `streaming_offset.json`. The worst-case sentence is off by one: see m2-r3 |
| n1-r2 | Wilson units | **FIXED** | §5 definition and §5.2 column header. All 36 cells now carry `%`. Example check: 51/124 gives Wilson [32.9, 49.9] % |
| n2-r2 | Cache-wide latency n | **FIXED** | §5 `:181-185`. My `run --cache-only` reports n = 11,707, p50 171.77, p95 318.53, exactly as stated. The committed figures (9,732 / 169.78 / 314.43; 81.44 and 331.56 by length) match the "170 / 314 / 81 / 332" in the text |
| n3-r2 | Gate audit order and limit | **FIXED** | `tests/training/test_gate.py:205-208`. Mutations below |
| n4-r2 | McNemar independence | **FIXED** | §5.6 now covers the p-values too. Newsletter-only comparisons are correctly excluded from the caveat |
| n5-r2 | Production `assert` | **FIXED** | It is now a `RuntimeError` (`:331-332`) |

## Major

### M1-r3. After an outage, back-to-back retries can keep every regular check timing out for the rest of the stream (windowed `FAIL_OPEN` with a deadline)

`serving/streaming.py:282` and `:336-346`. Two facts combine:

- A retry is chosen whenever there is debt, unless the previous retry *failed* (`prefer_regular` is set only on a
  failed retry, `:346`). After an outage the retries succeed, so the whole backlog is retried back to back with no
  regular check in between.
- The regular window extends back over every token that arrived since the previous regular check (`:280`). It is
  therefore as long as the backlog took to clear, times the generation rate.

Once that window takes longer than `check_timeout_s` to score, it times out. Its text becomes the next backlog, and
the cycle repeats. Each retry is at most L tokens and succeeds. Each regular check carries all the fresh text and
fails. The guard has turned from preventive into after-the-fact: every new token goes out `checked=False`, and a
retry scores it seconds later.

**Trace** (scratchpad `r3/trace.py`; virtual clock; 100 tok/s; latency 200 ms + 4 ms per token, which is
the cache's ~50 ms + 1 ms/token at `slowdown` 4; a 64-token window takes 0.456 s; deadline 0.5 s; W = 64, k = B = 16;
the classifier stalls for t ∈ [1, 2) s; injection at token 1,050):

```
  2.21-  2.71 reg   (112, 208) len=96 timeout      <- first regular check after the stall
  2.71-  3.16 retry (64, 128)  len=64 allow
  3.16-  3.62 retry (128, 192) len=64 allow
  3.62-  3.88 retry (192, 208) len=16 allow
  3.88-  4.38 reg   (208, 384) len=176 timeout     <- 1.2 s of new text in one window
  4.38-  5.69 retry ... x3 allow
  5.69-  6.19 reg   (384, 560) len=176 timeout     ... the same cycle until the end
 12.12- 12.56 retry (992, 1056) len=64 flag        <- the injection, already released
unchecked 944 of 1088, abandoned []
```

The same configuration without the stall releases 0 unchecked tokens, and every regular check passes.

**It does not decay.** I put the injection at 5, 15, 30 or 60 s after a 1 s stall at t = 1 s (scratchpad
`r3/srun_long.py`). At HEAD every regular check in the rest of the stream times out, 82 to 90% of released text is
unchecked, the injection is released every time, and a retry catches it 2.1 to 2.6 s later. Round 1's scheduler
(`7d060fe`) releases 80 tokens unchecked in total, withholds the injection, and catches it 0.8 to 1.0 s after it
arrives. At an idle-host latency and 600 tok/s, HEAD's detection delay grows with the stream: 2.9, 5.1, 7.8, then
11.4 s.

**Onset** (scratchpad `r3/onset.py`; cache-like latency 50 ms + 1 ms/token × slowdown; W = 64, k = B = 16,
`FAIL_OPEN`; 30 s stream, injection at 25 s; stall at t = 1 s of 0 / 1 / 3 s; % of released text unchecked, `!` =
injection released):

| deadline, rate, slowdown | HEAD `97b63b6` | round-1 scheduler `7d060fe` |
| --- | --- | --- |
| 0.5 s, 50 tok/s, ×1 / ×2 / ×4 | ≤ 13% (recovers) | ≤ 11% |
| 0.5 s, 100 tok/s, ×4 | 0% / **89%!** / **91%!** | 0% / 3% / 12% |
| 1.0 s, 50 tok/s, ×8 | 0% / **88%!** / **90%!** | 0% / 1% / 11% |
| 1.0 s, 100 tok/s, ×4 | 0% / 1% / **93%!** | 0% / 2% / 10% |
| 1.0 s, 100 tok/s, ×6 | 0% / **95%!** / **96%!** | 0% / 3% / 10% |
| 0.5 s, 150 tok/s, ×4 (already marginal) | 89%! / 95%! / 95%! | 49% / 51%! / 59% |

HEAD is fine at the latencies the note measures (idle host and ×2, 50 to 150 tok/s). The ratchet starts when one
W-window takes a large share of the deadline, or when the rate is high relative to classifier throughput. A busy
gateway at 100 tok/s with a 0.5 s deadline is in that regime. Note that the units are detector tokens.

**What is still true.** No check is longer than a regular window. Nothing is given up in these runs. Every span is
eventually scored, and the stream ends `Stop/flag`, not `Done`. So this is not a return of M1-r2's permanent
blindness. It is the loss of prevention, which is the only thing `FAIL_OPEN` with a deadline still offers over
monitoring. `released_unchecked` counts all of it. The doc's "after-the-fact coverage only" is accurate for each
span, but it does not say that the mode can persist for the whole stream after a 1 s blip.

**Contradicted text.**
- `serving/streaming.py:18-20`: "no check is ever longer than a regular one (W, or the coalesced new tokens), and
  a transient outage cannot grow the window until every check times out". The regular window itself grows: 176
  tokens in the trace, and up to 1,792 at 600 tok/s.
- `docs/design/streaming.md:104-106`: a span is "lost for good exactly when …". That is true of *lost* spans, but it
  leaves out that the visible stream can stay unchecked after recovery.

**Cause check.** I tried a scratch variant that sets `prefer_regular` after every retry, so at most one retry runs
between regular checks (scratchpad `r3/var_alt`). It fixes the high-rate rows (400 to 600 tok/s: 0 regular timeouts
after recovery, injection withheld). It does not fix the near-deadline rows (×4 at 100 tok/s, ×8 at 50 tok/s). There,
any retry pushes the next coalesced regular window past the deadline. A second sketch also caps the regular window
at L under `FAIL_OPEN` and leaves the overflow to the retry queue (scratchpad `r3/var_cap`). It recovers in the
near-deadline rows, but at 600 tok/s its detection delay grows with the stream (7.1 s at 30 s, 11.3 s at 60 s). Neither variant is a drop-in fix. The underlying
trade-off is that full re-coverage competes with fresh text for a saturated classifier.

**Fix options (author's choice).**
- Give fresh text priority. Run regular checks first and retries only with spare capacity, for example at most one
  retry per regular check. Cap the regular window at L, so text coalesced past L becomes debt. Abandon debt older than
  a bound (age or size), reported as now. That keeps the visible stream checked and makes the loss explicit, which is
  round 1's property plus the reporting.
- Or keep the scheduler and narrow the claims. Delete "a transient outage cannot grow the window…". In §2, state
  that under windowed `FAIL_OPEN` with a deadline, when a W-window's latency is near the deadline or the rate is
  near the classifier's throughput, an outage can leave every later regular check timing out. The rest of the
  stream then goes out unchecked and is scored only afterwards. Pin the trace above as a test of intended behaviour.

**Verification:** add the trace configuration as a test. W = 64, k = B = 16, deadline 0.5 s, 100 tok/s, latency
0.2 + 0.004·len, 1 s stall: either the injection is withheld, or the test asserts the documented degraded mode.

## Minor

### m2-r3. §7: the worst-case alignment is o = 0, not o = 1

`docs/design/streaming.md:409-410`: "The worst case is o = 1, so B = k − 1 = 15 at k = 16". With o defined in §5.2
as tokens into the check window, o ranges over 0..k−1, and k − o is largest at o = 0, the aligned case. The first
check that sees the injection then ends k tokens into it, so the bound is B ≥ k = 16. The committed aligned rows show
this: full prefix, k = 16, start 128 gives 29 zero-leak at B = 8 and 46 at B = 16. W = 64 gives 42 → 60. The
conclusion "B = 16 covers any alignment at k = 16" stands, but B = 15 does not. Also, `:413` "measured only at o = 8
and o = 56" leaves out o = 0, which the aligned start 128 measured. *(Round 2's m2-r2 wrote "k − 1 = 15" itself; the
error is the reviewer's, copied into the note.)*

## Nit

- **n1-r3.** An abandoned span can include tokens that were retried fewer than `retry_attempts` times. Between two
  attempts, the regular window start s moves forward, so the second retry from the same head is longer (`:287`).
  Scratchpad `r3/under.py`: W = 16, k = 8, calls 3 to 7 raise. The retries are (16, 24) and then (16, 32), and
  (16, 32) is given up, so tokens 24 to 31 had one retry. The fuzz counted 4,706 such tokens. This is harmless,
  because those tokens were also in failed regular windows. But the docstring (`:20`) and §2 `:56-57` ("given up
  when its retry has failed `retry_attempts` times") describe the head, not every token.
- **n2-r3.** `CheckRecord.outcome = "abandoned"` (`:146`, `:246`) means "cancelled when the stream stopped", while
  `StreamStats.abandoned` (`:160`) means "given up after failed retries". Two meanings for one word in the same
  record set.
- **n3-r3.** §2 `:56` says "A failed retry lets one regular check run before the next retry". That is accurate, but
  it is the only scheduling rule stated. Successful retries running back to back, which is the M1-r3 mechanism,
  is implied only by "before the next regular check". Say it either way.

## Probes at HEAD

| Probe | Result |
| --- | --- |
| r2 `outage_sweep.py` (W ∈ {16, 64}, 0.5 to 3 s stall at t = 1, 50 tok/s, 0.25 s deadline) | all `Stop/flag`, injection withheld, longest check 64 / 48 |
| O1: the same, stall 2 to 8 s | stall up to 6 s: `Stop/flag`, injection withheld, abandoned 80 to 208 (W = 64) or 32 to 112 (W = 16) tokens, all reported. 8 s (covers the injection at t = 6 and its retries): W = 64 ends `Done` with 352 abandoned and the injection released; W = 16 stops later with the injection released, abandoned 160 |
| O2: stall aligned with the injection (t0 5.5 to 6.2 s, 0.5 to 2 s) | W = 16 loses the injection after a 1.0 s stall, W = 64 after a 2.0 s stall: span (288, 304) or (288, 320) abandoned and reported. Shorter stalls release 4 to 5 injection tokens unchecked, then stop `flag` (documented `FAIL_OPEN` cost). "An outage longer than about `retry_attempts` checks" (`:105-106`) is in the right range: about (W/k + 2·`retry_attempts`) check periods |
| I1: every p-th call raises, p ∈ {2, 3}, every phase, no deadline | all `Stop/flag`, 0 unchecked, nothing abandoned. With W = 16 and p = 2, 18 of 19 regular checks fail and every retry rescues them |
| I2: only first-time texts fail (3 calls) | `Stop/flag`, nothing abandoned |
| S1, S2, long and onset runs | M1-r3 |
| F: independent fuzz, 20,000 cases (seed 7; `r3/probes_r3.py F 20000`): both modes, k ∈ {1, 2, 4, 8, 16, 32}, B ∈ {0..32}, W ∈ {full, 1, 3, 8, 16, 24, 64}, deadline ∈ {none, 0.05, 0.2, 0.25}, lag budgets, `retry_attempts` ∈ {1, 2, 3}, rates 20 to 500 tok/s, five failure patterns (random raise / failed-allow / failed-flag; periodic; fail only on injection text; stall aligned with the injection; latency ∝ length plus raises) | **0 violations.** `FAIL_CLOSED` (9,936 cases): nothing released unchecked, no retries, nothing abandoned, every released token inside an allow check that ended before its release, marker never released. `FAIL_OPEN` (10,064): releases contiguous, a `checked=True` release is scored; abandoned spans sorted, disjoint, never overlapping any allow/flag span, each with `retry_attempts` failed retries at its start; on `Done` with W set, unscored set == abandoned set and `unscored_tokens == abandoned_tokens`; a released marker is always `checked=False`. Retries ≤ L tokens. Regular checks ≤ ⌈n/k⌉ + 1, retries ≤ `retry_attempts`·(⌈n/L⌉ + 2·regular + 1). The bound held in every case, with the tightest case at 0.96 of it. 2,663 cases abandoned something and 4,950 ran retries |

## Frozen-test guard mutations (`artifacts/audit_log.jsonl`, restored after each, byte-identical afterwards)

`unmodified`: PASS. **FAIL**: both paths → OTHER; started path only; both actors; completed actor only; completed
bundle changed; completed bundle removed; `run_complete` false; order swapped; statuses swapped; completed `at`
before started; completed `at` equal to started; started `at` in 2099; started `limit` = 10, = 0, or removed; an
extra pair for another report. PASS (correctly): completed `at` written as `+00:00`, 1 s after started.

## What was verified

| Check | Result |
| --- | --- |
| `python -m serving.engine_version` | `engine 1.1.0: decision-path files match the record` |
| Files touched by `86d9004..97b63b6` | `docs/design/streaming.md`, `serving/README.md`, `serving/streaming.py`, `serving/bench/streaming.py`, `tests/serving/test_streaming.py`, `tests/training/test_gate.py`. No decision-path file and no `serving/bench_results/` file changed |
| Bundle | all 11,707 cache rows `bundle = 2ebc7d9ae676`, `decision ∈ {allow, flag}`, `failed = False`, `coverage = None` |
| `flock … pytest tests/serving` | 1,017 passed, 9 skipped (76.5 s) |
| `flock … pytest tests/training` | 182 passed, 2 skipped (90.4 s) |
| `flock … pytest tests/contract` | 170 passed |
| `run --cache-only`, `offset --cache-only`, `paired` into scratch (0 cache misses, cache file unchanged) | `sweep`, `rates`, `backpressure`, `whole_response_once`, `streams`, the per-stream file, offset `configs` / `paired` / `aligned_rows_match_sweep`, and paired `comparisons` all identical to committed. Only `cache`, the cache-wide latency stats, timestamps and wall time differ, as §5 now says. McNemar: 4/1 p 0.375, 4/0 p 0.125, 3/20 and 0/12 p 0.0005 |
| `serving/README.md:623-628` | accurate: bounded retries, `StreamStats.abandoned`, `Done.unscored_tokens`, engine 1.1.0 unchanged |
| Doc §2 numbers (r2 row: 352 tokens, injection at 300; HEAD catches and gives nothing up) | reproduced |
| Frozen test | not read. No probe touches `data/splits/` |

Probe scripts (scratchpad, not committed): `r3/probes_r3.py` (F, O, S, I), `r3/srun.py`, `r3/srun_long.py`,
`r3/onset.py`, `r3/trace.py`, `r3/under.py`, `r3/mut_gate_r3.py`, plus scheduler copies `r3/var_{head,alt,cap,r1,r2}`.
