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
real assistant output), and an assistant that has been hijacked usually does not repeat the injection verbatim; it
acts on it (a tool call, a summary that contains a link). A real output classifier is trained on outputs, and
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
- **Window.** `window_tokens=None` re-scores the full prefix on every check; W > 0 scores the last W tokens,
  extended back to the first token no check has scored yet: after coalescing made the gap longer than W, and after
  a check that did not score (below), the next window re-covers the gap. A token counts as scored only when a check
  covering it returned allow with `failed=False`, or flag (`test_windows_cover_every_token`,
  `test_fail_open_rescores_a_failed_window`). Round 1 of the review found that a timed-out, raised or failed-allow
  window was counted as covered and the next window started after it, so with W set its text could be released and
  never scored (`test_fail_open_window_rescores_a_timed_out_injection_window` is the reviewer's probe; it fails on
  the old scheduler). The full prefix sees the whole context, which is
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
  cannot happen, its decide member is `FAIL_CLOSED`, and `service_check` also sets `failed` when a coverage rule
  decided). The lag budget (`max_unchecked_tokens`: generated tokens waiting for a check) is the other trigger. A
  check result of `flag` stops the stream even when the classifier's failure policy produced it. Treating a failed
  allow as "not scored" is stricter than the bundle, never more permissive.

| Guard setting | When a check is slow | When a check raises or comes back failed-allow |
| --- | --- | --- |
| `FAIL_CLOSED`, no timeout, no lag budget (**stall**) | the visible stream waits; latency grows without bound | the stream stops (`classifier_failure`); nothing it did not score is released |
| `FAIL_CLOSED` + timeout and/or lag budget | the stream stops (`classifier_failure` or `lag`); a benign response is cut off | the stream stops (`classifier_failure`) |
| `FAIL_OPEN`, no timeout, no lag budget | the visible stream waits | the failed check's text is released `checked=False` and counted; the next check re-covers it and can still stop the stream |
| `FAIL_OPEN` + timeout and/or lag budget | the oldest waiting text is released `checked=False` and counted; later checks re-cover it and can still stop the stream | as the row above |

**Which configurations guarantee that no unchecked text is released:** every `FAIL_CLOSED` configuration, with or
without a deadline or lag budget. A token is released only after a check that covered it returned allow with
`failed=False`. **No `FAIL_OPEN` configuration guarantees it,** stall included: no deadline and no budget only
remove slowness as a cause; an exception or a failed-allow still releases text. Under `FAIL_OPEN` the re-covering
check gives after-the-fact coverage only. If every check over some text keeps failing (an input that always times
out), the rest of the stream goes out unchecked and the stream ends `Done`; the tokens are marked and counted
(`test_fail_open_window_keeps_rescoring_while_checks_keep_failing`). Under `FAIL_OPEN` a lag release also releases
the held-back buffer, so a lag budget below B is rejected (it would make B meaningless; review n1).

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
"engine 1.1.0: decision-path files match the record", and every one of the 9,732 checks in
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
cascade. Production output-side classifiers are generally described as scoring a model's output as it streams and
halting generation when a threshold is crossed, which is the shape built here; I could not retrieve the original
Constitutional Classifiers paper (arXiv 2501.18837) from this environment to check its streaming details, so this
note attributes no specific mechanism, buffer size or rate to it or to any other system (§8 question 4). The
general trade-offs (buffer vs latency, check frequency vs compute, prefix vs window) are the ones measured below.

## 5. Measurement

**Command** (all numbers below are in the committed JSON):

```bash
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming run'     # scores cache misses (~28 min)
python -m serving.bench.streaming run --cache-only                              # re-derives the JSON, no model
flock /home/claude/measure.lock -c 'python -m serving.bench.streaming live --n 24'
```

Outputs: `streaming_sweep.json` (aggregates), `streaming_sweep_per_stream.json` (one row per stream and config),
`check_cache.jsonl` (decision, score and latency of every distinct check text), `scoring_runs.jsonl` (the two
scoring runs: 9,658 checks in 1,650.7 s, then 74 more), `live_check.json`. A `--cache-only` re-run reproduced
`sweep`, `rates`, `backpressure` and `whole_response_once` exactly.

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
the idle 2-core host, one stream at a time (p50 170 ms, p95 314 ms over 9,732 checks; 81 ms p50 below 300
characters, 332 ms p50 at 1,200 characters or more). `slowdown` multiplies it to stand in for a busy gateway.
**Simulator check:** the guard on a real event loop against the real service, 24 streams (12 injection, 12 benign),
B = k = 16, W = 64, 50 tok/s: same outcome on 24/24 [86, 100], the same stopping check and the same number of
released tokens on 24/24; first visible text 22 ms later live than simulated (p50; 8 to 38 ms over 22 streams),
event-loop and service overhead the simulator does not charge.

**Metrics.** *Caught*: stopped by a flag from a check whose window contains at least one injection token.
*Zero-leak catch*: caught with no injection token released. *Leak*: injection tokens released before the stop.
*Delay*: tokens generated after the injection start when the stream stopped. *False stop*: a benign stream
stopped. *First visible, extra*: p50 time to the first released token minus 1/rate (the unguarded stream's first
token). *Busy*: p50 per stream of the fraction of time a check was in flight. Intervals: Wilson 95%.

### 5.1 B x k x W at 50 tok/s, stall mode (`streaming_sweep.json` `sweep`)

| W | k | B | caught | zero-leak catch | leaked tok. (mean, caught) | delay tok. p50 / p90 | benign false stop | first visible, extra | checks / 1k tok. | busy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 8 | 0 | 51/124 [33, 50] | 12/124 [6, 16] | 12.0 | 32 / 54 | 20/124 [11, 24] | 0.20 s | 107.0 | 0.80 |
| full | 8 | 16 | 51/124 [33, 50] | 42/124 [26, 43] | 2.67 | 32 / 54 | 20/124 [11, 24] | 0.51 s | 107.0 | 0.80 |
| full | 8 | 64 | 51/124 [33, 50] | 51/124 [33, 50] | 0.0 | 32 / 54 | 20/124 [11, 24] | 1.52 s | 107.0 | 0.80 |
| full | 16 | 0 | 51/124 [33, 50] | 29/124 [17, 32] | 8.71 | 27 / 44 | 18/124 [9, 22] | 0.35 s | 62.2 | 0.54 |
| full | 16 | 16 | 51/124 [33, 50] | 46/124 [29, 46] | 1.88 | 27 / 44 | 18/124 [9, 22] | 0.68 s | 62.2 | 0.54 |
| full | 16 | 64 | 51/124 [33, 50] | 51/124 [33, 50] | 0.0 | 27 / 44 | 18/124 [9, 22] | 1.71 s | 62.2 | 0.54 |
| full | 64 | 0 | 48/124 [31, 48] | 48/124 [31, 48] | 0.0 | 76 / 80 | 17/124 [9, 21] | 1.35 s | 16.0 | 0.16 |
| 128 | 16 | 16 | 60/124 [40, 57] | 50/124 [32, 49] | 5.38 | 26 / 57 | 26/124 [15, 29] | 0.68 s | 62.1 | 0.44 |
| 64 | 8 | 0 | 70/124 [48, 65] | 22/124 [12, 25] | 11.84 | 20 / 44 | 34/124 [20, 36] | 0.20 s | 124.0 | 0.49 |
| 64 | 8 | 16 | 70/124 [48, 65] | 58/124 [38, 56] | 3.9 | 20 / 44 | 34/124 [20, 36] | 0.51 s | 124.0 | 0.49 |
| 64 | 16 | 0 | 68/124 [46, 63] | 42/124 [26, 43] | 8.85 | 20 / 51 | 32/124 [19, 34] | 0.35 s | 62.3 | 0.25 |
| 64 | 16 | 16 | 68/124 [46, 63] | 60/124 [40, 57] | 2.82 | 20 / 51 | 32/124 [19, 34] | 0.68 s | 62.3 | 0.25 |
| 64 | 16 | 64 | 68/124 [46, 63] | 68/124 [46, 63] | 0.0 | 20 / 51 | 32/124 [19, 34] | 1.66 s | 62.3 | 0.25 |
| 64 | 64 | 0 | 70/124 [48, 65] | 69/124 [47, 64] | 0.91 | 68 / 68 | 26/124 [15, 29] | 1.35 s | 15.9 | 0.07 |
| *whole response, checked once* | — | — | 52/124 [34, 51] | 52/124 [34, 51] | 0.0 | 113 / 139 | 5/124 [2, 9] | 5.43 s | 4.1 | 0.06 |

All 60 cells are in the JSON. What the sweep shows:

- **B buys zero leak, linearly in latency.** At W = full, k = 16, zero-leak catches go from 29/124 (B = 0) to 46/124
  (B = 16) to 51/124 (B = 64, every catch), and mean leaked tokens per catch from 8.71 to 1.88 to 0. Each 16
  tokens of B cost about 0.33 s of first-visible latency at 50 tok/s. B has to cover the detector's own delay, not
  just k: DeBERTa usually needs a good part of the injection before it flags, so at k = 8 zero-leak catches are
  12, 30, 42, 48 and 51 of the 51 catches for B = 0, 8, 16, 32 and 64.
- **k trades compute for delay.** Checks per 1,000 tokens follow 1,000/k when the classifier keeps up (W = 64:
  124.0, 62.3, 31.3, 15.9 for k = 8, 16, 32, 64); the full prefix at k = 8 gets 107.0 instead of 125 because its
  checks coalesce (busy 0.80). Detection delay p50 at full prefix: 32 tokens at k = 8 (coalescing again), 27 at
  k = 16, 76 at k = 64. Catches change little with k (48 to 51 of 124).
- **The injection starts on a check boundary, the best case for small B.** Token 128 is a multiple of every k
  swept, so the last check before the injection ends exactly where it starts and releases none of it. That is why
  k = 64 with B = 0 leaks nothing (48/48 catches leak-free, at 1.35 s to the first visible token: the cadence acts
  as a buffer). With an injection starting inside a check window, a check that passes it would release up to k - 1
  more injection tokens unless B covers them; this sweep does not measure that position.
- **Streaming multiplies false stops.** A benign 256-token stream that completes is checked 4 to 32 times, and any
  flag stops it.
  Checked once at the end, 5/124 [2, 9] benign streams are flagged; checked every 16 tokens on the full prefix,
  18/124 [9, 22]; with W = 64, 32/124 [19, 34]. The per-check false-positive rate is what matters for streaming,
  and a threshold chosen for one decision per document (D23's 0.5) is too low for 16 decisions per response.

### 5.2 Dilution: full prefix vs window

The `newsletter` pool has no benign false stop in any configuration (0/62 [0, 6] every time; `by_pool`), so its
catches are clean detections. At B = k = 16 (50 tok/s): full prefix 27/62 [32, 56], W = 128 30/62 [36, 61], W = 64
39/62 [50, 74]; checked once at the end, 23/62 [26, 50]. That is the dilution curve (`training/DILUTION.md`) in
streaming form: the full prefix around a short injection is mostly benign text, a 64-token window is mostly the
injection. Streaming the full prefix still catches more than checking the finished response once (27 vs 23 of 62),
because the prefix that ends just after the injection has less benign text after it than the whole response.
The `val_benign` pool shows the price: its benign streams are stopped 18/62 [19, 41] at full prefix and 32/62
[39, 64] at W = 64 (5/62 [3, 18] checked once), because val benign items are instruction-like prompts; its
"caught" counts include benign flags inside the window and are not clean detections. A smaller window is not
turned on here for the same reason the gateway does not turn it on for input (`serving/README.md`, Stage 1
anti-dilution): it trades dilution for false positives, and the threshold was never calibrated per window.

### 5.3 Token rate (`rates`, B = k = 16, stall)

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

### 5.4 Backpressure (`backpressure`, 100 tok/s, B = k = 16, lag budget 64 tokens)

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

### 5.5 Limits of this measurement

- The detector is an input classifier used as an output stand-in (§1); the streams are synthetic splices, one
  injection position (token 128) and one length (256 benign / 192 + injection tokens).
- Latency is from one stream at a time on an idle host, charged per distinct text from the cache; concurrent streams
  share the CPU (2 cores) and would be slower, which `slowdown` approximates without measuring.
- One check in flight per stream; no batching across streams or across a stream's windows.
- The injection always starts at a multiple of every k swept (§5.1), the most favourable alignment for small B.
- The 124 injection streams are 62 injections in two contexts, so they are not 124 independent draws; intervals
  treat them as independent and are narrower than they should be.

## 6. What is implemented

| Item | Where |
| --- | --- |
| Streaming guard, `StreamGuardConfig`, `CheckResult`, events, `service_check` adapter | `serving/streaming.py` |
| Virtual-time loop for tests and the harness | `serving/bench/vclock.py` |
| Tests: invariant grid, flag/retract, B = 0 leak, failure flag never overridden, timeout, lag, FAIL_OPEN release, exceptions, k = 1, B longer than the stream, tail check, empty stream, window coverage, window split, validation, real loop, early consumer close, adapter | `tests/serving/test_streaming.py` |
| Harness bookkeeping tests | `tests/serving/test_streaming_bench.py` |
| Harness and committed outputs | `serving/bench/streaming.py`, `serving/bench_results/streaming/` |

Not implemented: an SSE endpoint (§3), `FALLBACK` / `ASYNC_ONLY` stream modes, retracting already-released text,
batching checks across streams, a per-check threshold.

## 7. Recommendation (for discussion, not a decision)

If a stream guard were served with this detector: W = 64, k = 16, B = 16 to 32, stall mode with a generous lag
budget only as a safety valve, on a recalibrated threshold. The numbers say the window and the threshold matter more
than k or B: B buys zero leak at about 0.33 s per 16 tokens at 50 tok/s, while the false-stop rate (17 to 34 of 124 benign streams)
is what would make users notice.

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
