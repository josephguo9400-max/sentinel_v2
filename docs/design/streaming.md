# Design: Streaming Output Classification (milestone streaming)

2026-11-01 · Status: implemented as a library and a measurement, not served; owner questions in §8 · Base: `main` @
`495e163` · Code: `serving/streaming.py` · Harness: `serving/bench/streaming.py` · Results:
`serving/bench_results/streaming/`

## 1. A different threat model

Sentinel today is **input-side**: it classifies an untrusted email body before it reaches the agent's context
(RFC "The classifier's role"; mode B quarantines flagged emails). Streaming classification is **output-side**: it
classifies the text a model is *generating*, while it streams to the user or to a tool, so the gateway can stop
the assistant mid-response, for example before it emits an instruction it picked up from injected content, or a
link or address that exfiltrates data.

| | Input side (served today) | Output side (this milestone) |
| --- | --- | --- |
| What is classified | a complete untrusted document | a growing prefix of the model's own output |
| When the decision is needed | before the document enters the context | before each piece of text is shown or acted on |
| Latency budget | one request (D23 decide timeout 5,000 ms) | per token: the visible stream waits for the classifier |
| Unit of a decision | one request | one stream, decided by many checks |
| A miss means | injected text enters the context | injected or exfiltrating text reaches the user or a tool |
| What a flag does | quarantine the email | stop generation, withhold the unshown buffer |

**What transfers:** the serving machinery (chunking, timeouts, the policy engine, `FAIL_OPEN` / `FAIL_CLOSED`
semantics, the bundle), the measurement discipline (Wilson intervals, committed outputs), and the dilution
finding (`training/DILUTION.md`), which applies with more force because a stream prefix keeps growing.

**What does not:** the detector. `protectai/deberta-v3-base-prompt-injection-v2` was trained to recognize
injection *inputs*. Running it on an output is a stand-in for an output classifier, not a validated one: nobody
has measured it on model outputs, the streams below are synthetic (benign text with an injection spliced in, not
real assistant output). This note also assumes, without a measurement, that a hijacked assistant often does not
repeat the injection verbatim but acts on it (a tool call, a summary that contains a link). A real output classifier is trained on outputs, and
its positive class is "the assistant is doing something it should not", not "this text looks like an injection".
Every detection number below therefore measures the **streaming mechanics** with a weak stand-in model.

## 2. Mechanism (`serving/streaming.py`)

`guard_stream(tokens, check, config)` consumes an async iterator of generated token strings and yields
`Release(text, start, end, checked)` events, then exactly one `Stop` or `Done`.

- **Hold-back buffer, B tokens.** Only text a passed check covered is released, and the last B checked tokens are
  held back as well. B is what makes a flag useful: an injection that starts near the end of a passed check's
  window is often recognized only by the next check, and only text still in the buffer can be withheld. The user
  sees text at least B tokens plus one pending check behind generation. At the end of the stream a final check
  covers the tail and everything left is released.
- **Check every k tokens, one at a time.** A check is scheduled at the next k-token boundary. If the classifier is
  slower than the stream, the next check jumps to the newest boundary that has arrived (boundaries coalesce, the
  effective k grows); checks never queue. Smaller k means earlier detection and more classifier calls.
- **Window.** `window_tokens=None` re-scores the full prefix on every check. W > 0 scores the last W tokens,
  extended back over every token that arrived since the previous check, so coalescing never skips text. A token
  counts as scored only when a check covering it returned allow with `failed=False`, or flag
  (`test_windows_cover_every_token`). Text a check did not score (below; this happens only under `FAIL_OPEN`) is
  re-covered by the regular windows while it is still inside the last W tokens. Once it falls out of them it
  becomes *retry debt*: before the next regular check the guard retries it, oldest first, as one unscored run of
  at most L = max(W, k) tokens, so no check is ever longer than a regular one. A failed retry lets one regular check
  run before the next retry of the same span. **A span is given up when its retry has failed `retry_attempts`
  times (default 2)**, after the regular windows covering it failed too; at the end of the stream retries run back
  to back. A given-up span is recorded in `StreamStats.abandoned`, its tokens stay `checked=False` (they have been
  or will be released marked unchecked, or are withheld by a later stop), `Done.unscored_tokens` counts every token
  no check scored, and checking goes on with the next span. Nothing else is ever given up, and nothing scored is
  given up. The number of checks stays bounded: at most ceil(n/k) + 1 regular checks and `retry_attempts` x
  (ceil(n/L) + 2 x regular + 1) retries for n tokens (`test_randomized_invariants`).
  Review history: round 1 found that a failed window was counted as covered and the next window started after it,
  so its text could be released and never scored (`test_fail_open_window_rescores_a_timed_out_injection_window`).
  The round-1 fix extended the next window back to the first unscored token without a bound. Round 2 showed that
  this is a ratchet: classifier latency grows with window length, so after a transient 2 s stall (W = 64, k = 16,
  0.25 s deadline) every re-covering window was too long to finish in time, the window grew to 352 tokens, and the
  rest of the stream went out unchecked with the injection at token 300 released. With bounded retries the same
  run catches that injection right after the stall and gives nothing up
  (`test_transient_outage_w64_two_seconds_catches_the_injection_after_recovery`; this test and
  `test_transient_outage_does_not_disable_a_windowed_fail_open_guard` fail on `86d9004`). The full prefix has no
  retries: every check re-covers all failed text, but every check also grows with the prefix, so with a deadline
  the full prefix fails every check once the prefix takes longer than the deadline to score, outage or not.
  The full prefix sees the whole context, which is
  exactly what dilutes a short injection; a small window dilutes less but sees less context, so it flags more
  benign text, and an injection longer than the overlap can be split across windows
  (`test_window_misses_an_injection_split_across_checks_only_if_window_is_too_small`). Through the gateway a full
  prefix longer than 510 tokens is chunked by the D23 policy; past 4 chunks (about 1,660 tokens) its coverage rule
  flags it, so full-prefix rescoring cannot serve long responses with that bundle as it is.
- **Flag.** A check returning `flag` stops the stream: the source is closed (`aclose`), every generated but
  unreleased token is withheld, and `Stop(reason="flag", retracted_tokens=...)` is the last event. Released text
  cannot be unsaid by this guard; a UI that wants to retract it needs the stop event, which carries the counts.
- **A check that did not score is handled by the guard's failure mode; a flag from the bundle is never
  overridden.** "Did not score" means the check raised, passed the guard's deadline (`check_timeout_s`), or came
  back allow with `failed=True` (the classifier's own failure policy allowed text it did not score; with D23 this
  cannot happen, its decide member is `FAIL_CLOSED`; `service_check` also sets `failed` when a coverage rule decided
  content no step scored, i.e. `coverage_policy_applied` is `flag`, `allow` or `quarantine`; a `deep_scan` result
  is a scored decision and is not a failure, so its flag stops the stream with reason `flag`). The lag budget
  (`max_unchecked_tokens`: generated tokens waiting for a check) is the other trigger. A check result of `flag`
  stops the stream even when the classifier's failure policy produced it. Treating a failed allow as "not scored"
  is stricter than the bundle, never more permissive.

| Guard setting | When a check is slow | When a check raises or comes back failed-allow |
| --- | --- | --- |
| `FAIL_CLOSED`, no timeout, no lag budget (**stall**) | the visible stream waits; latency grows without bound | the stream stops (`classifier_failure`); nothing it did not score is released |
| `FAIL_CLOSED` + timeout and/or lag budget | the stream stops (`classifier_failure` or `lag`); a benign response is cut off | the stream stops (`classifier_failure`) |
| `FAIL_OPEN`, no timeout, no lag budget | the visible stream waits | the failed check's text is released `checked=False` and counted; a later window or a retry (W set: at most L tokens, up to `retry_attempts` times, then the span is given up and reported) re-covers it and can still stop the stream |
| `FAIL_OPEN` + timeout and/or lag budget | the oldest waiting text is released `checked=False` and counted; later checks re-cover it as in the row above and can still stop the stream | as the row above |

**Which configurations guarantee that no unchecked text is released:** every `FAIL_CLOSED` configuration, with or
without a deadline or lag budget. A token is released only after a check that covered it returned allow with
`failed=False`. **No `FAIL_OPEN` configuration guarantees it,** stall included: no deadline and no budget only
remove slowness as a cause; an exception or a failed-allow still releases text. Under `FAIL_OPEN` the re-covering
check gives after-the-fact coverage only. With W set, a span is lost for good exactly when the regular windows over
it and all `retry_attempts` retries of it failed (for example an input that always times out, or an outage longer
than about `retry_attempts` checks); that span is reported (`StreamStats.abandoned`, `Done.unscored_tokens`) and
the guard keeps checking the text after it
(`test_fail_open_window_gives_up_a_span_whose_retry_fails_and_keeps_guarding`,
`test_permanent_outage_under_fail_open_terminates_and_reports_everything_unscored`). Under `FAIL_OPEN` a lag
release also releases the held-back buffer, so a lag budget below B is rejected (it would make B meaningless;
review n1).

`FALLBACK` (a cheaper check when the main one is late) and `ASYNC_ONLY` (release everything, classify afterwards,
which is monitoring rather than prevention) are not implemented. Invariant, tested over a 324-cell grid of B, k,
W, latency and injection position, plus a timeout / exception / failed-allow at each of the four settings above
(`test_failed_check_under_each_setting`): in `FAIL_CLOSED` every released token was inside a check that returned
allow, released after that check returned, and while the stream runs at least B checked tokens stay held back
(`tests/serving/test_streaming.py::test_fail_closed_never_releases_unchecked_text`).

**Units.** B, k, W and the lag budget count tokens of the stream given to the guard. In the harness these are
DeBERTa pieces (detector tokens). A real generator's tokens differ, and the classifier re-tokenizes the joined text.
With W = k the window edges can split words.

## 3. Where it sits: no engine or bundle change

`serving/streaming.py` calls the classifier service the way any client calls `/classify`
(`service_check(service)`), so no decision-path file changed: `python -m serving.engine_version` reports
"engine 1.1.0: decision-path files match the record", and every one of the 11,707 checks in
`check_cache.jsonl` was decided by bundle `2ebc7d9ae676…`, the served D23 bundle at engine 1.1.0
(`serving/README.md`). **Engine version and bundle hash are unchanged.**

That is also the gap: the stream's own settings (B, k, W, failure mode, budgets) decide what a user sees, and they
are in no `ExecutionPolicy`, so a stream decision is not bound to a gated bundle. Per D25 ("a policy change is a
deployment") they must be before anything is served, which is why there is **no SSE endpoint**: it would have been
cheap to write but would serve decisions no bundle hash covers. Adding a `stream` section (or a second
`entrypoint: classify_stream`) to `ExecutionPolicy` changes `contracts/policy.py` and the engine, so it needs a minor
engine bump and a new bundle hash (§8 question 1).

## 4. Industry practice (briefly)

The repository references Anthropic's Constitutional Classifiers++ (arXiv 2601.04603, `docs/RFC.md`) only for its
cascade. The shape built here (score the output as it streams, halt generation on a flag) is this design's own
premise, not a claim about how any production system works; I could not retrieve the original
Constitutional Classifiers paper (arXiv 2501.18837) from this environment to check its streaming details, so this
note attributes no specific mechanism, buffer size or rate to it or to any other system (§8 question 4). The
general trade-offs (buffer vs latency, check frequency vs compute, prefix vs window) are the ones measured below.

## 5. Measurement

**Command** (all numbers below are in the committed JSON):

```bash
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming run'     # scores cache misses (~28 min)
python -m serving.bench.streaming run --cache-only                              # re-derives the JSON, no model
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming live --n 24'
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming live --n 24 --rate 100 --window 0 \
    --out-live serving/bench_results/streaming/live_check_coalescing.json'
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming offset'  # injection position (§5.2)
python -m serving.bench.streaming paired                                        # McNemar counts, no model
```

Outputs: `streaming_sweep.json` (aggregates), `streaming_sweep_per_stream.json` (one row per stream and config),
`check_cache.jsonl` (decision, score and latency of every distinct check text), `scoring_runs.jsonl` (three
scoring runs: 9,658 checks in 1,650.7 s, then 74 more for the sweep; 1,975 more for `offset`), `live_check.json`,
`live_check_coalescing.json`, `streaming_offset.json` (injection position), `streaming_paired.json` (paired
comparisons). A `--cache-only` re-run reproduced `sweep`, `rates`, `backpressure` and `whole_response_once`
exactly, before and after the round-1 guard fix (no committed run has a failed check, so the fix does not change
them). The aligned rows of `streaming_offset.json` equal the sweep's injection metrics (`aligned_rows_match_sweep`).

**Streams** (`build_streams`, from `training/dilution_curve.py` `build_inputs()`: the val split and the probe
generator; the frozen test is not read). 62 short injections (2 redteam probes, 60 val injections; 13 to 78
DeBERTa tokens, median 43), each with a 510-token benign context from two pools: `newsletter` (the probe sentence
pool) and `val_benign` (val benign items DeBERTa alone scores below 0.5). **Injection stream** = 128 context tokens,
the injection (starts at token 128), 64 more context tokens: 124 streams. **Benign stream** = the first 256 tokens
of the same context: 124 streams. A token is a DeBERTa piece; the guard joins pieces into text and the service
re-tokenizes it.

**Classifier and clock.** Every check is one `ClassifierService.classify` on the served D23 bundle (DeBERTa fp32
decides, Stage 1 observe-only). The guard and a fixed-rate token source run on a virtual-time event loop
(`serving/bench/vclock.py`); each check is charged the wall latency the service measured for that exact text on
the idle 2-core host, one stream at a time (p50 170 ms, p95 314 ms over the sweep's 9,732 checks; 81 ms p50 below
300 characters, 332 ms p50 at 1,200 characters or more). Each latency is a **single measurement per distinct
text**, taken when the text was first scored; the second sweep scoring run (74 rows) ran at a load average of 1.4
to 1.8 (`scoring_runs.jsonl`). `slowdown` multiplies it to stand in for a busy gateway.

**Simulator check: scheduling and decisions, not timing under load.** The guard on a real event loop against the
real service, compared stream by stream with the simulation. The streams are the first 12 injection and first 12
benign streams in build order (6 injections x 2 pools), not a random sample.
- `live_check.json`, B = k = 16, W = 64, 50 tok/s: same outcome on 24/24 [86, 100], the same stopping check, the
  same released tokens and the same check count on 24/24. First visible text is 22 ms later live (p50; 8 to 38 ms
  over 22 streams), event-loop and service overhead the simulator does not charge. In this configuration the
  classifier is busy about a quarter of the time and never coalesces, so agreement follows from deterministic
  scheduling plus identical per-text decisions; it does not test the timing model.
- `live_check_coalescing.json`, the full prefix at 100 tok/s, the same 24 streams: here checks coalesce (14
  instead of 16 checks on a completed benign stream). Same outcome on 24/24 [86, 100], the same released tokens on
  24/24, the same stopping check on 23/24, and the same check count on 20/24 (each of the four differs by one;
  280 checks live vs 278 simulated). First visible text is 28 ms later live (p50; 5 to 65 ms).
- Not validated live: lag stops and `FAIL_OPEN` lag releases (§5.5), and the `slowdown` rows, where timing drives
  the result and a few tens of ms of overhead per check can move a lag decision.

**Metrics.** *Caught*: stopped by a flag from a check whose window contains at least one injection token.
*Zero-leak catch*: caught with no injection token released. *Leak*: injection tokens released before the stop.
*Delay*: tokens generated after the injection start when the stream stopped. *Stopped before inj.*: an injection
stream stopped by a check that ended at or before the injection start (a flag on the benign prefix; every one in
the sweep is a flag). That stream never reached the injection, so it can be neither caught nor missed. *Caught of
streams reaching inj.* leaves them out of the denominator. With W = full, every check ending after the injection
start contains the injection, so any later flag counts as caught, including one the benign context caused (see
§5.3 on `val_benign`). *False stop*: a benign stream
stopped. *First visible, extra*: p50 time to the first released token minus 1/rate (the unguarded stream's first
token). *Busy*: p50 per stream of the fraction of time a check was in flight. Intervals: Wilson 95%.

### 5.1 B x k x W at 50 tok/s, stall mode (`streaming_sweep.json` `sweep`)

| W | k | B | caught | stopped before inj. | caught of streams reaching inj. | zero-leak catch | leaked tok. (mean, caught) | delay tok. p50 / p90 | benign false stop | first visible, extra | checks / 1k tok. | busy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 8 | 0 | 51/124 [33, 50] | 20/124 | 51/104 [40, 59] | 12/124 [6, 16] | 12.0 | 32 / 54 | 20/124 [11, 24] | 0.20 s | 107.0 | 0.80 |
| full | 8 | 16 | 51/124 [33, 50] | 20/124 | 51/104 [40, 59] | 42/124 [26, 43] | 2.67 | 32 / 54 | 20/124 [11, 24] | 0.51 s | 107.0 | 0.80 |
| full | 8 | 64 | 51/124 [33, 50] | 20/124 | 51/104 [40, 59] | 51/124 [33, 50] | 0.0 | 32 / 54 | 20/124 [11, 24] | 1.52 s | 107.0 | 0.80 |
| full | 16 | 0 | 51/124 [33, 50] | 18/124 | 51/106 [39, 58] | 29/124 [17, 32] | 8.71 | 27 / 44 | 18/124 [9, 22] | 0.35 s | 62.2 | 0.54 |
| full | 16 | 16 | 51/124 [33, 50] | 18/124 | 51/106 [39, 58] | 46/124 [29, 46] | 1.88 | 27 / 44 | 18/124 [9, 22] | 0.68 s | 62.2 | 0.54 |
| full | 16 | 64 | 51/124 [33, 50] | 18/124 | 51/106 [39, 58] | 51/124 [33, 50] | 0.0 | 27 / 44 | 18/124 [9, 22] | 1.71 s | 62.2 | 0.54 |
| full | 64 | 0 | 48/124 [31, 48] | 16/124 | 48/108 [35, 54] | 48/124 [31, 48] | 0.0 | 76 / 80 | 17/124 [9, 21] | 1.35 s | 16.0 | 0.16 |
| 128 | 16 | 16 | 60/124 [40, 57] | 18/124 | 60/106 [47, 66] | 50/124 [32, 49] | 5.38 | 26 / 57 | 26/124 [15, 29] | 0.68 s | 62.1 | 0.44 |
| 64 | 8 | 0 | 70/124 [48, 65] | 23/124 | 70/101 [60, 77] | 22/124 [12, 25] | 11.84 | 20 / 44 | 34/124 [20, 36] | 0.20 s | 124.0 | 0.49 |
| 64 | 8 | 16 | 70/124 [48, 65] | 23/124 | 70/101 [60, 77] | 58/124 [38, 56] | 3.9 | 20 / 44 | 34/124 [20, 36] | 0.51 s | 124.0 | 0.49 |
| 64 | 16 | 0 | 68/124 [46, 63] | 20/124 | 68/104 [56, 74] | 42/124 [26, 43] | 8.85 | 20 / 51 | 32/124 [19, 34] | 0.35 s | 62.3 | 0.25 |
| 64 | 16 | 16 | 68/124 [46, 63] | 20/124 | 68/104 [56, 74] | 60/124 [40, 57] | 2.82 | 20 / 51 | 32/124 [19, 34] | 0.68 s | 62.3 | 0.25 |
| 64 | 16 | 64 | 68/124 [46, 63] | 20/124 | 68/104 [56, 74] | 68/124 [46, 63] | 0.0 | 20 / 51 | 32/124 [19, 34] | 1.66 s | 62.3 | 0.25 |
| 64 | 64 | 0 | 70/124 [48, 65] | 16/124 | 70/108 [55, 73] | 69/124 [47, 64] | 0.91 | 68 / 68 | 26/124 [15, 29] | 1.35 s | 15.9 | 0.07 |
| *whole response, checked once* | — | — | 52/124 [34, 51] | 0/124 | 52/124 [34, 51] | 52/124 [34, 51] | 0.0 | 113 / 139 | 5/124 [2, 9] | 5.43 s | 4.1 | 0.06 |

All 60 cells are in the JSON. What the sweep shows:

- **B buys zero leak, linearly in latency.** At W = full, k = 16, zero-leak catches go from 29/124 (B = 0) to 46/124
  (B = 16) to 51/124 (B = 64, every catch), and mean leaked tokens per catch from 8.71 to 1.88 to 0. Each 16
  tokens of B cost about 0.33 s of first-visible latency at 50 tok/s. B has to cover the detector's own delay, not
  just k: DeBERTa usually needs a good part of the injection before it flags, so at k = 8 zero-leak catches are
  12, 30, 42, 48 and 51 of the 51 catches for B = 0, 8, 16, 32 and 64.
- **k trades compute for delay.** Checks per 1,000 tokens follow 1,000/k when the classifier keeps up (W = 64:
  124.0, 62.3, 31.3, 15.9 for k = 8, 16, 32, 64); the full prefix at k = 8 gets 107.0 instead of 125 because its
  checks coalesce (busy 0.80). Detection delay p50 at full prefix: 32 tokens at k = 8 (coalescing again), 27 at
  k = 16, 76 at k = 64. Catches change little with k (48 to 51 of 124). The difference is not significant: at
  B = 16, k = 8 and k = 64 differ on 5 streams (4 caught only at k = 8, 1 only at k = 64; exact McNemar p = 0.375,
  `streaming_paired.json`).
- **About 16 to 23 of the 124 injection streams are stopped by benign text before the injection starts**
  (column "stopped before inj.": 18 at full prefix / k = 16, 20 at W = 64 / k = 16). They are false stops on
  injection streams, and they cap "caught" at 101 to 108 of 124. Of the streams that reach the injection, the full
  prefix at k = 16 catches 51/106 [39, 58] and W = 64 68/104 [56, 74].
- **The injection starts on a check boundary, the best case for small B.** Token 128 is a multiple of every k
  swept, so the last check before the injection ends exactly where it starts and releases none of it. The B = 0
  rows (and k = 64 with B = 0 leaking nothing at full prefix) are flattered by this alignment. §5.2 measures
  injections that start inside a check window.
- **Streaming multiplies false stops.** A benign 256-token stream that completes is checked 4 to 32 times, and any
  flag stops it.
  Checked once at the end, 5/124 [2, 9] benign streams are flagged; checked every 16 tokens on the full prefix,
  18/124 [9, 22]; with W = 64, 32/124 [19, 34]. The per-check false-positive rate is what matters for streaming,
  and a threshold chosen for one decision per document (D23's 0.5) is too low for 16 decisions per response.

### 5.2 Injection position: aligned vs non-aligned (`streaming_offset.json`)

`python -m serving.bench.streaming offset` re-runs the 124 injection streams with the injection starting at token
120 or 136 instead of 128. The contexts and injections are the same, with 50 tok/s and stall mode. Both shifted
starts are 8 tokens into a k = 16 check window; for k = 64 they are 56 (start 120) and 8 (start 136) tokens in. Start
120 has 8 benign tokens fewer before the injection and 136 has 8 more, so the pair brackets the effect of the
prefix length; it is not a pure alignment effect. The 128 rows reproduce the sweep (`aligned_rows_match_sweep`:
16/16 equal). Only these offsets were measured, not a random or uniform offset. Zero-leak catches of 124
(Wilson 95%), mean / max leaked injection tokens per catch, and catches:

| W | k | B | zero-leak, 128 (aligned) | zero-leak, 120 | zero-leak, 136 | leaked mean / max, 128 → 120 → 136 | caught 128 / 120 / 136 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 16 | 0 | 29 [17, 32] | **8** [3, 12] | **8** [3, 12] | 8.71 / 44 → 9.36 / 40 → 10.51 / 40 | 51 / 53 / 49 |
| full | 16 | 8 | 29 [17, 32] | 45 [28, 45] | 39 [24, 40] | 5.33 / 40 → 2.72 / 32 → 3.98 / 35 | 51 / 53 / 49 |
| full | 16 | 16 | 46 [29, 46] | 45 [28, 45] | 39 [24, 40] | 1.88 / 32 → 1.51 / 24 → 2.51 / 35 | 51 / 53 / 49 |
| full | 16 | 32 | 50 [32, 49] | 52 [34, 51] | 47 [30, 47] | 0.31 / 16 → 0.15 / 8 → 0.65 / 24 | 51 / 53 / 49 |
| full | 64 | 0 | 48 [31, 48] | **10** [4, 14] | 47 [30, 47] | 0.0 / 0 → 6.4 / 8 → 1.17 / 56 | 48 / 50 / 48 |
| full | 64 | 8 | 48 [31, 48] | 50 [32, 49] | 47 [30, 47] | 0.0 / 0 → 0.0 / 0 → 1.0 / 48 | 48 / 50 / 48 |
| 64 | 16 | 0 | 42 [26, 43] | **19** [10, 23] | **24** [13, 27] | 8.85 / 48 → 11.12 / 72 → 11.94 / 72 | 68 / 73 / 72 |
| 64 | 16 | 8 | 42 [26, 43] | 60 [40, 57] | 55 [36, 53] | 5.88 / 40 → 5.75 / 64 → 7.17 / 64 | 68 / 73 / 72 |
| 64 | 16 | 16 | 60 [40, 57] | 60 [40, 57] | 55 [36, 53] | 2.82 / 32 → 4.56 / 56 → 5.72 / 56 | 68 / 73 / 72 |
| 64 | 16 | 32 | 64 [43, 60] | 63 [42, 59] | 61 [41, 58] | 0.94 / 16 → 2.19 / 40 → 3.12 / 49 | 68 / 73 / 72 |
| 64 | 64 | 0 | 69 [47, 64] | **21** [11, 25] | 66 [44, 62] | 0.91 / 64 → 7.38 / 72 → 1.65 / 56 | 70 / 65 / 68 |
| 64 | 64 | 8 | 69 [47, 64] | 63 [42, 59] | 66 [44, 62] | 0.8 / 56 → 1.97 / 64 → 1.41 / 48 | 70 / 65 / 68 |

All 48 cells (B = 16 and 32 at k = 64 too) and the per-stream paired counts are in the JSON. What it shows:

- **At B = 0 the aligned sweep overstates zero-leak catches by a factor of about 1.8 to 5.** At k = 16 an injection
  8 tokens into a window gets 8 zero-leak catches instead of 29 (full prefix) and 19 or 24 instead of 42 (W = 64).
  Paired, full prefix, start 136: 22 streams are leak-free only when aligned, 1 only when shifted.
- **"k = 64 with B = 0 leaks nothing" was an alignment artefact; it is withdrawn.** With the injection 56 tokens
  into a 64-token window (start 120), the check at 128 sees its first 8 tokens and usually passes them. Zero-leak
  catches fall from 48 to 10 at full prefix, and from 69 to 21 at W = 64. Leaks at start 120 are at most 8 tokens
  because only 8 injection tokens precede the boundary. A larger offset would leak more, up to k - 1 tokens.
- **At these offsets, B = 8 absorbs the misalignment.** At offset 8, B = 8 brings zero-leak catches back to at least
  the aligned B = 8 value in every k = 16 row (full: 45 and 39 vs 29; W = 64: 60 and 55 vs 42). At k = 64, start
  120 (offset 56, 8 injection tokens before the boundary), B = 8 gives 50/50 leak-free catches at full prefix. That
  is a statement about these offsets only: B covers misalignment only up to B tokens, and an offset uniform in
  [0, k) would need B close to k - 1 for the same effect. That was not measured.
- **Above that, B buys less than the aligned sweep says.** At start 136 (8 more benign tokens), B = 16 gives 39
  zero-leak catches at full prefix vs 46 aligned (paired: 7 vs 0) and 55 vs 60 at W = 64. B = 32 gives 47 vs 50 and
  61 vs 64. Start 120 is at or near the aligned values. So the remaining loss at 136 is detector delay with more
  benign text in front, not alignment.
- **Catches barely move** (full: 51 / 53 / 49; W = 64, k = 16: 68 / 73 / 72). Position changes when the stream is
  stopped, not whether it is.

### 5.3 Dilution: full prefix vs window

The `newsletter` pool has no benign false stop in any configuration (0/62 [0, 6] every time; `by_pool`), so its
catches are clean detections. At B = k = 16 (50 tok/s): full prefix 27/62 [32, 56], W = 128 30/62 [36, 61], W = 64
39/62 [50, 74]; checked once at the end, 23/62 [26, 50]. That is the dilution curve (`training/DILUTION.md`) in
streaming form: the full prefix around a short injection is mostly benign text, a 64-token window is mostly the
injection. Streaming the full prefix catches 27 vs 23 of 62 for checking the finished response once. This is not
a significant difference: the 4 discordant streams are all caught only by streaming, exact McNemar p = 0.125
(`streaming_paired.json`). The direction fits a prefix that ends just after the injection having less benign text
after it than the whole response. W = 64 vs the full prefix on the same pool is significant: 12 streams caught
only with the window, 0 only with the full prefix, p = 0.0005.
The `val_benign` pool shows the price: its benign streams are stopped 18/62 [19, 41] at full prefix and 32/62
[39, 64] at W = 64 (5/62 [3, 18] checked once), because val benign items are instruction-like prompts; its
"caught" counts include benign flags inside the window and are not clean detections. A smaller window is not
turned on here for the same reason the gateway does not turn it on for input (`serving/README.md`, Stage 1
anti-dilution): it trades dilution for false positives, and the threshold was never calibrated per window.

### 5.4 Token rate (`rates`, B = k = 16, stall)

| W | rate | delay tok. p50 / p90 | first visible, extra | token delay p50 / p95 | busy | checks / 1k |
| --- | --- | --- | --- | --- | --- | --- |
| full | 20 tok/s | 20 / 36 | 1.61 s | 1.34 / 1.75 s | 0.23 | 62.8 |
| full | 50 tok/s | 27 / 44 | 0.68 s | 0.65 / 0.88 s | 0.54 | 62.2 |
| full | 100 tok/s | 46 / 71 | 0.37 s | 0.46 / 0.79 s | 0.80 | 54.2 |
| 64 | 20 tok/s | 17 / 49 | 1.61 s | 1.23 / 1.63 s | 0.10 | 62.8 |
| 64 | 100 tok/s | 25 / 55 | 0.37 s | 0.32 / 0.39 s | 0.49 | 61.7 |

Catches and zero-leak catches are the same at every rate within a window mode (full: 51 or 50 of 124, 46 zero-leak;
W = 64: 68 and 60). Faster streams make B cheaper in seconds and make delay in tokens longer, because more tokens
are generated during each check; at 100 tok/s full-prefix checks are in flight 80% of the time and start to
coalesce (54.2 checks per 1,000 tokens instead of 62).

### 5.5 Backpressure (`backpressure`, 100 tok/s, B = k = 16, lag budget 64 tokens)

| W | slowdown | mode | caught / zero-leak | benign false stop (reasons) | unchecked tokens released | token delay p50 / p95 / max |
| --- | --- | --- | --- | --- | --- | --- |
| full | 1 | stall | 50 / 46 of 124 | 18/124 [9, 22] (flag 18) | 0 | 0.46 / 0.79 / 1.07 s |
| full | 1 | FAIL_CLOSED | 50 / 46 | 70/124 [48, 65] (lag 52, flag 18) | 0 | 0.43 / 0.75 / 1.07 s |
| full | 1 | FAIL_OPEN | 50 / 46 | 18/124 [9, 22] | 210 in 61 streams (4.5 per 1k released) | 0.46 / 0.79 / 1.07 s |
| full | 4 | stall | 45 / 45 | 17/124 [9, 21] | 0 | 1.40 / 2.61 / 3.44 s |
| full | 4 | FAIL_CLOSED | 0 / 0 (every stream stopped: lag 114 + flag 10 of each kind) | 124/124 [97, 100] | 0 | 0.56 / 0.71 / 0.80 s |
| full | 4 | FAIL_OPEN | 45 / 0 [0, 3] | 17/124 [9, 21] | 27,629 in 228 streams (539 per 1k) | 0.64 / 2.52 / 3.44 s |
| 64 | 1 | all three identical | 68 / 60 | 32/124 [19, 34] | 0 | 0.32 / 0.39 / 0.54 s |
| 64 | 4 | stall | 69 / 65 | 28/124 [16, 31] | 0 | 0.73 / 0.94 / 1.25 s |
| 64 | 4 | FAIL_CLOSED | 0 / 0 (every stream stopped: lag 114 + flag 10) | 124/124 [97, 100] | 0 | 0.56 / 0.71 / 0.80 s |
| 64 | 4 | FAIL_OPEN | 69 / 28 [16, 31] | 28/124 [16, 31] | 11,166 in 228 streams (244 per 1k) | 0.64 / 0.84 / 1.19 s |

- **Stall** (`FAIL_CLOSED`, no deadline, no budget) never releases unchecked text here, and pays in latency: with a 4x slower classifier the full-prefix stream
  shows text 1.40 s behind generation at p50 and up to 3.44 s, and detection delay grows to 113 tokens p50
  (zero leak holds because the visible stream is so far behind).
- **FAIL_CLOSED with a lag budget** turns slowness into stopped responses: already at slowdown 1 with the full
  prefix, 52 of 124 benign streams are cut off for lag; at slowdown 4 every stream is.
- **FAIL_OPEN** keeps streams flowing and releases unchecked text: at slowdown 4 a caught injection has leaked 50.5
  tokens on average with the full prefix (the injections are 43 tokens median, so most of it), and no catch is
  leak-free. Later checks still run, so catches stay at 45/124.
- A small window keeps the classifier ahead of the stream (busy 0.49 at slowdown 1 vs 0.80 for the full prefix),
  which is the most effective backpressure measure here: at slowdown 1 the three modes do not differ at all.

### 5.6 Limits of this measurement

- The detector is an input classifier used as an output stand-in (§1); the streams are synthetic splices, one
  injection position in the sweep (token 128; 120 and 136 in §5.2) and one length (256 benign / 192 + injection tokens).
- Latency is from one stream at a time on an idle host, charged per distinct text from the cache; concurrent streams
  share the CPU (2 cores) and would be slower, which `slowdown` approximates without measuring.
- One check in flight per stream; no batching across streams or across a stream's windows.
- The sweep's injection always starts at a multiple of every k swept, the most favourable alignment for small B.
  §5.2 measures two other starts (offsets 8 and 56), not a distribution of offsets.
- Latency is a single measurement per distinct text. Comparisons between configurations are on the same streams
  (paired). Only the ones in `streaming_paired.json` have a paired test; the others are point differences.
- B, k and W are in detector tokens (DeBERTa pieces), not generator tokens (§2).
- The 124 injection streams are 62 injections in two contexts, so they are not 124 independent draws; intervals
  treat them as independent and are narrower than they should be.

## 6. What is implemented

| Item | Where |
| --- | --- |
| Streaming guard, `StreamGuardConfig`, `CheckResult`, events, `service_check` adapter | `serving/streaming.py` |
| Virtual-time loop for tests and the harness | `serving/bench/vclock.py` |
| Tests: invariant grid, flag/retract, B = 0 leak, failure flag never overridden, timeout, lag, FAIL_OPEN release, exceptions, k = 1, B longer than the stream, tail check, empty stream, window coverage, window split, bounded retries and given-up spans, transient and permanent outages, a 4,000-case seeded fuzz of the invariants, validation, real loop, early consumer close, adapter (including deep-scan results) | `tests/serving/test_streaming.py` |
| Harness bookkeeping tests | `tests/serving/test_streaming_bench.py` |
| Harness and committed outputs | `serving/bench/streaming.py`, `serving/bench_results/streaming/` |

Not implemented: an SSE endpoint (§3), `FALLBACK` / `ASYNC_ONLY` stream modes, retracting already-released text,
batching checks across streams, a per-check threshold.

## 7. Recommendation (for discussion, not a decision)

If a stream guard were served with this detector: W = 64, k = 16, B = 16 to 32, on a recalibrated threshold.
Use `FAIL_CLOSED` (the only mode that never releases unscored text, §2): stall, or a generous lag budget kept only as
a safety valve. B = 0 is not an option: with the injection inside a check window it leaves 8 to 24 of 124 zero-leak
catches (§5.2). In the measured offsets, B = 8 absorbs an 8-token misalignment. With W = 64 / k = 16, B = 16 gives
55 to 60 zero-leak catches and B = 32 gives 61 to 64, over the three injection starts. Real injections start
anywhere in a window. Let o be the injection's offset into a check window (§5.2): the first check that sees any of
the injection ends k - o tokens into it. If that check passes and the next one flags, those k - o tokens are what
B must still hold back, so a zero leak needs B >= k - o. That matches the committed rows: at k = 16, o = 8, B = 8
(= k - o) already gives at least the aligned B = 8 values (full prefix 45 and 39 vs 29; W = 64 60 and 55 vs 42),
and at k = 64, o = 56 (start 120), B = 8 (= k - o) gives 50/50 leak-free catches at full prefix. The worst case is
o = 1, so B = k - 1 = 15 at k = 16, and B = 16 covers any alignment there. This bound covers the alignment only,
not the detector's own delay: when the detector needs more than one check after the first one that sees the
injection, more injection text passes, and even aligned injections leak at B = 16 (W = 64, k = 16: 60 zero-leak of
68 catches). B >= k - o was measured only at o = 8 and o = 56. The numbers say the window and the
threshold matter more than k or B. B buys zero leak at about 0.33 s per 16 tokens at 50 tok/s, while the
false-stop rate is what users would notice: 17 to 34 of 124 benign streams, plus 16 to 23 of 124 injection streams
stopped on their benign prefix.

## 8. Open questions for the owner

1. **Versioning.** Put the stream settings into `ExecutionPolicy` (a `stream` section or `entrypoint:
   classify_stream`), with a minor engine bump and a new bundle hash, before any endpoint serves streams? Until then
   the guard stays a library.
2. **Calibration.** A stream makes many decisions per response. Who owns a per-check threshold (detector author:
   new capability version; or policy owner: a stream-level band), and on what data, given that no output-side labels
   exist in the repo?
3. **Default failure mode.** Every `FAIL_CLOSED` setting never releases unchecked text (§2). Stall is the only one
   that also never cuts off a response for slowness, at the cost of unbounded latency. FAIL_CLOSED with a lag
   budget cuts off benign answers under load (52/124 at slowdown 1, full prefix). Which one is
   the default, and what lag budget?
4. **External references.** Which published description of streaming output classifiers should the write-up cite?
   I could not retrieve the Constitutional Classifiers paper text from this environment, so §4 attributes nothing
   specific.
5. **Retraction UX.** Is withholding the unshown buffer enough, or should the stop event let a client retract
   already-shown text (which changes what B is for)?
