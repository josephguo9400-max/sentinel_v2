# Injection dilution in long windows (dilution study)

Owner: evaluation. Every number below is in a committed JSON written by a committed script:

| Output | Script | Model needed |
| --- | --- | --- |
| `training/results/dilution_scores.jsonl` -> `dilution_curve.json` | `training/dilution_curve.py score` / `summarize` | score: yes |
| `training/results/window_scores_{val,replay,probes}.jsonl` | `training/window_scores.py` | yes |
| `training/results/window_timing_{val,replay}.json` | `training/window_timing.py` | yes |
| `training/results/window_eval.json` | `training/window_eval.py` | no |
| `training/results/planner_fidelity.json` | `training/planner_fidelity.py` | no |

```bash
scripts/run_dilution.sh "510 128 64" 300            # each chunk under flock /home/claude/measure.lock; resumable
python -m training.dilution_curve summarize --windows 510,128,64
scripts/run_window_scores.sh "probes val replay" "510 256 128 64" 300
flock /home/claude/measure.lock python -m training.window_timing --set replay --n 60
flock /home/claude/measure.lock python -m training.window_timing --set val --n 60
python -m training.window_eval
python -m training.planner_fidelity
```

Model: `protectai/deberta-v3-base-prompt-injection-v2`, ONNX fp32 (sha256 pinned in `training/cascade_scores.py`),
CPU, 2 threads, batch size 1. A request is tokenized once and split into windows of W content tokens with the
gateway's rule (starts every `stride` tokens plus one window ending at the tail; stride 384 for W = 510, W/2
otherwise); each model input is `[CLS] + window + [SEP]`; the request score is the **max over all windows** (no
chunk cap); flag iff score >= 0.5. W = 510 is DeBERTa with the original sliding-window planner (`serving/encoding.py`), without
the `max_chunks` cap and the coverage policy. The gateway now plans Stage 2 chunks with the structure-aware planner
(`serving/structure.py`: per-segment tokenization, paragraph boundaries, kinds kept apart, a shorter last window).
On the 1,139 escalated replay items whose every chunk the served planner scored (serving-faithful counterfactual;
`planner_fidelity.json`), the max served-chunk score equals the W = 510 max on 1,074 (to 5 decimal places; 1,067 at
the stored 6), the window count equals the chunk count on 1,120, and 5 items flip at 0.5 (2 positive, 2 negative,
1 unlabeled). W = 510 is therefore a close proxy for the served Stage 2, not an exact one. The frozen test was not read. Sets: val (`data/splits/val.jsonl`, 2,240 injections, 974 benign,
92 of them benign_hard), the replay set (`data/replay/replay-v1.jsonl`, 306 injections, 1,185 benign, 509 unlabeled
mosscap items excluded from every rate) and probe set v2 (`fixtures/probes`, 12 attacks, 12 matched benign
controls). Intervals: Wilson 95% for proportions, Newcombe paired hybrid score 95% for differences on the same items.

## 1. Controlled dilution curve (`dilution_curve.json`)

**Design.** 62 injections: the two `redteam/probes.py` INJECTIONS (`system_notice_action`, 69 tokens;
`ignore_previous_forward`, 49 tokens) and 60 short val injections (label 1, lang en, <= 80 DeBERTa tokens), stratified
by source with up to 20 per source by a seeded hash order. Only 13 jackhhao items are eligible, so the sample is
27 spml, 20 deepset, 13 jackhhao. Each injection gets one 510-token benign context per pool:

- `newsletter`: the probe set v2 sentence pool (the same generator the probes use);
- `val_benign`: val benign items (label 0, lang en, not benign_hard) that DeBERTa alone scores < 0.5 (788 items),
  shuffled per injection and concatenated.

With ctx = the first L context tokens, the injection is placed at the start, in the middle (half the context on
each side) or at the end, for L in {0, 32, 64, 128, 256, 384, 510}. Composition is at the token-id level, so lengths
are exact (a cut can fall inside a word). The context alone (no injection) is scored for the false-positive side.
n = 62 per (pool, L, position); "cond." = detection among the injections that are detected when scored alone at
that W. That is 46/62 at every W, but not the same 46 (at W = 64, 2/2 probes and 10/13 jackhhao items are detected
alone, against 1/2 and 11/13 at W = 510), so a comparison of "cond." across W is over slightly different sets.

### W = 510 (the gateway's window): detection falls with context length

46/62 injections are detected alone (74% [62, 83]). Detection with benign context:

| Context L | newsletter: start | middle | end | cond. (all positions) | context alone flagged |
| --- | --- | --- | --- | --- | --- |
| 32 | 35/62 = 56% [44, 68] | 32/62 = 52% [39, 64] | 32/62 = 52% [39, 64] | 99/138 = 72% [64, 79] | 0/62 = 0% [0, 6] |
| 64 | 33/62 = 53% [41, 65] | 30/62 = 48% [36, 61] | 29/62 = 47% [35, 59] | 92/138 = 67% [58, 74] | 0/62 |
| 128 | 31/62 = 50% [38, 62] | 26/62 = 42% [30, 54] | 27/62 = 44% [32, 56] | 84/138 = 61% [53, 69] | 0/62 |
| 256 | 29/62 = 47% [35, 59] | 22/62 = 35% [25, 48] | 25/62 = 40% [29, 53] | 76/138 = 55% [47, 63] | 0/62 |
| 384 | 28/62 = 45% [33, 57] | 21/62 = 34% [23, 46] | 24/62 = 39% [28, 51] | 73/138 = 53% [45, 61] | 0/62 |
| 510 | 27/62 = 44% [32, 56] | 20/62 = 32% [22, 45] | 23/62 = 37% [26, 50] | 70/138 = 51% [42, 59] | 0/62 = 0% [0, 6] |

| Context L | val_benign: start | middle | end | cond. (all positions) | context alone flagged |
| --- | --- | --- | --- | --- | --- |
| 32 | 46/62 = 74% [62, 83] | 42/62 = 68% [55, 78] | 41/62 = 66% [54, 77] | 117/138 = 85% [78, 90] | 7/62 = 11% [6, 22] |
| 128 | 41/62 = 66% [54, 77] | 34/62 = 55% [43, 67] | 34/62 = 55% [43, 67] | 105/138 = 76% [68, 82] | 9/62 = 15% [8, 25] |
| 256 | 37/62 = 60% [47, 71] | 28/62 = 45% [33, 57] | 32/62 = 52% [39, 64] | 97/138 = 70% [62, 77] | 5/62 = 8% [3, 18] |
| 510 | 30/62 = 48% [36, 61] | 29/62 = 47% [35, 59] | 28/62 = 45% [33, 57] | 87/138 = 63% [55, 71] | 3/62 = 5% [2, 13] |

(64 and 384 rows are in the JSON.) Median middle-position score in newsletter context: 0.979 at L = 32, 0.012 at 64,
0.0017 at 128, 0.0021 at 510. Middle and end are worse than start. The drop is source-dependent: at L = 510 in
newsletter context, 68/81 spml placements are still detected, but 0/60 deepset (7/20 detected alone) and 1/39
jackhhao (11/13 alone).

**The ad hoc finding reproduces.** `ignore_previous_forward` in the middle of newsletter context: 1.000 alone,
1.000 at L = 32, 0.987 at 64, 0.135 at 128, 0.005 at 256, 0.001 at 384, 0.002 at 510 (start position: >= 0.97 at
every L). `system_notice_action`: 0.419 alone, <= 0.047 at every L and position in newsletter context. This is
dilution, not truncation: every token is inside a scored window.

### Smaller windows: dilution is bounded, false positives on benign context grow

| Window | Pool | detected at L = 510, middle | cond. at L = 510 (all positions) | context alone flagged at L = 128 / 256 / 510 |
| --- | --- | --- | --- | --- |
| 510 | newsletter | 20/62 = 32% [22, 45] | 70/138 = 51% [42, 59] | 0/62 / 0/62 / 0/62 |
| 128 | newsletter | 32/62 = 52% [39, 64] | 95/138 = 69% [61, 76] | 0/62 / 0/62 / 0/62 |
| 64 | newsletter | 38/62 = 61% [49, 72] | 110/138 = 80% [72, 86] | 0/62 / 0/62 / 0/62 |
| 510 | val_benign | 29/62 = 47% [35, 59] | 87/138 = 63% [55, 71] | 9/62 / 5/62 / 3/62 = 5% [2, 13] |
| 128 | val_benign | 48/62 = 77% [66, 86] | 123/138 = 89% [83, 93] | 9/62 / 16/62 / 27/62 = 44% [32, 56] |
| 64 | val_benign | 57/62 = 92% [82, 97] | 130/138 = 94% [89, 97] | 18/62 / 30/62 / 41/62 = 66% [54, 77] |

With W = 64 the newsletter curve is flat (middle: 37/62 at L = 32, 38/62 at L = 510; median middle score
>= 0.994 at every L), and `system_notice_action` scores 0.812 alone (its 69 tokens span two windows and one fragment
scores higher than the whole) and >= 0.81 in the middle of either context. But in val_benign context the
injection rates above are inflated by the context itself: the context alone is flagged in 41/62 cases at L = 510
with W = 64, against 3/62 with W = 510. The val_benign pool is built only from items DeBERTa does not flag alone,
so these are fragments of benign prompts (role-play setups, system-prompt-like text) that a small window scores
as injections, plus the max over many more windows. The newsletter pool has no such text and stays at 0/62.

## 2. Window-size study on real traffic (`window_eval.json`, `single_pass`)

| Set | W | Recall | diff vs 510 (+gained/-lost) | FPR | diff vs 510 (+/-) | Model tokens vs 510 |
| --- | --- | --- | --- | --- | --- | --- |
| val | 510 | 2,165/2,240 = 96.7% [95.8, 97.3] | — | 86/974 = 8.8% [7.2, 10.8] | — | 1.00 (404,123) |
| val | 256 | 2,165/2,240 = 96.7% | +0.00 [-0.18, +0.18] (+0/-0) | 86/974 = 8.8% | +0.00 [-0.38, +0.38] | 1.28 |
| val | 128 | 2,169/2,240 = 96.8% [96.0, 97.5] | +0.18 [-0.06, +0.44] (+4/-0) | 86/974 = 8.8% | +0.00 [-0.38, +0.38] | 1.48 |
| val | 64 | 2,173/2,240 = 97.0% [96.2, 97.6] | +0.36 [+0.04, +0.71] (+9/-1) | 102/974 = 10.5% [8.7, 12.6] | +1.64 [+0.81, +2.57] (+16/-0) | 1.67 |
| replay | 510 | 247/306 = 80.7% [75.9, 84.7] | — | 194/1,185 = 16.4% [14.4, 18.6] | — | 1.00 (324,838) |
| replay | 256 | 250/306 = 81.7% [77.0, 85.6] | +0.98 [-0.72, +2.75] (+4/-1) | 205/1,185 = 17.3% [15.3, 19.6] | +0.93 [+0.34, +1.55] (+11/-0) | 1.29 |
| replay | 128 | 255/306 = 83.3% [78.7, 87.1] | +2.61 [+0.27, +5.09] (+10/-2) | 227/1,185 = 19.2% [17.0, 21.5] | +2.78 [+1.84, +3.78] (+33/-0) | 1.40 |
| replay | 64 | 270/306 = 88.2% [84.1, 91.4] | +7.52 [+4.56, +10.79] (+23/-0) | 306/1,185 = 25.8% [23.4, 28.4] | +9.45 [+7.76, +11.19] (+114/-2) | 1.53 |
| probes | 510 | 1/12 = 8.3% [1.5, 35.4] | — | 0/12 = 0.0% [0.0, 24.2] | — | 1.00 |
| probes | 256 | 3/12 = 25.0% [8.9, 53.2] | +16.7 [-12.3, +44.5] | 0/12 | +0.0 | 1.36 |
| probes | 128 | 6/12 = 50.0% [25.4, 74.6] | +41.7 [+5.1, +67.2] (+5/-0) | 0/12 | +0.0 | 1.37 |
| probes | 64 | 10/12 = 83.3% [55.2, 95.3] | +75.0 [+36.0, +88.8] (+9/-0) | 0/12 = 0.0% [0.0, 24.2] | +0.0 | 1.41 |

The probe intervals (here and in section 3) treat the 12 attack probes as independent items. They are 2 payloads
x 6 placements, so the effective sample is smaller and the intervals are too narrow; read the probe rows as counts.

**Benign-hard (val, n = 92, all INSUFFICIENT_EVIDENCE by the gate's n >= 100 rule):** 32/92 = 34.8% [25.8, 44.9]
at W = 510, 256 and 128; **45/92 = 48.9% [38.9, 59.0] at W = 64**. 13 of W = 64's 16 extra val false positives are
benign_hard. Long replay emails (`replay_email` families ending in `_long`, n = 64): 32/64 = 50.0% at W = 510, 256 and
128, 54/64 = 84.4% [73.6, 91.3] at W = 64.

**Per source.** Val recall (W = 510 -> 128 -> 64): spml 2,056/2,080 at every W; jackhhao 87 -> 91 -> 94 of 102;
deepset 22 -> 22 -> 23 of 58. Replay recall: gandalf_summ 112 -> 113 -> 121 of 140; itw_jailbreak 64 -> 71 -> 78 of
93; replay_email 70/70 at every W; sandbox_dev 1/3 at every W. Replay FPR: itw_regular 24 -> 53 -> 94 of 295; aya
118 -> 122 -> 125 of 492; replay_email 52 -> 52 -> 87 of 390.

Why val barely moves: 2,344 of 3,214 val items have <= 128 tokens and score identically at W = 510 and 128 (same
single window); only 45 val items need more than one 510 window. The replay set has longer traffic (120 items over
510 tokens) and is where small windows change outcomes, in both directions.

**Measured cost (`window_timing_*.json`, ms per request, batch-1 windows, single measurement per request, n = 60
seeded sample per set, under the lock):**

| Sample | 510 mean / p50 / p95 | 256 | 128 | 64 |
| --- | --- | --- | --- | --- |
| val (mean 112.6 tokens) | 159.4 / 116.9 / 396.1 | 195.2 / 119.0 / 713.0 | 226.5 / 119.8 / 683.9 | 294.0 / 184.7 / 831.8 |
| replay (mean 209.6 tokens) | 468.0 / 85.5 / 2,500.1 | 506.6 / 82.5 / 2,985.7 | 507.8 / 88.9 / 2,578.3 | 610.3 / 85.8 / 3,240.6 |

Stacking all windows of a request in one call ("batched" in the JSON; padding-free because the windows of a
request have equal length) helped materially only at W = 64 (replay mean 610.3 -> 529.7 ms, val 294.0 -> 256.6 ms) and not at
W = 510 (replay 468.0 vs 506.1 ms). Mean wall time grows less than model tokens (replay W = 64: 1.30x the W = 510
mean on the timing sample, against 1.53x the model tokens over the whole replay set), consistent with shorter inputs costing less per token, but these are single
measurements on 60 requests.

## 3. Two-pass design (`window_eval.json`, `two_pass`)

Flag iff `s510 >= 0.5`, or the request has more than N content tokens and `s_W2 >= t2`. **Every parameter was
chosen on val only** by the rule stored as `selection_rule`: over W2 in {256, 128, 64}, N in {W2, min(2 W2, 510),
510} and t2 in {0.5, ..., 0.999}, keep settings whose paired 95% upper bound of the FPR increase over DeBERTa-510 is
<= +2 points and that add no benign_hard false positive; maximize val recall, then minimize pass-2 tokens, then
prefer the larger t2. 56 of 80 settings are admissible. **Chosen: W2 = 128, N = 128, t2 = 0.9.** The replay set and
the probes were not used to choose anything.

**Val cannot distinguish t2 = 0.5 from t2 = 0.9.** With W2 = 128 and N = 128, every t2 in {0.5, 0.6, 0.7, 0.8, 0.9}
gives item-for-item the same val flags (`two_pass.tied_on_val`: same recall, FPR, benign_hard and pass-2 tokens),
so t2 = 0.9 is the rule's last tie-break, not a val-tuned threshold. The val evidence for the second pass itself is
4 items (+0.18 [-0.06, +0.44]; the interval includes 0). At t2 = 0.5, the val-equivalent setting
(`two_pass.descriptive_t2_0.5`), the replay result is recall 257/306 = 84.0% [79.5, 87.7] (+3.27 [+1.14, +5.59],
+10/-0) and FPR 227/1,185 = 19.2% [17.0, 21.5] (**+2.78 [+1.84, +3.78]**, +33/-0), against 254/306 and 219/1,185 at
t2 = 0.9 below. The choice between them is a tie-break that val cannot inform.

| Set | Recall: 510 -> two-pass | Paired diff | FPR: 510 -> two-pass | Paired diff | Requests with pass 2 | Model tokens vs 510 |
| --- | --- | --- | --- | --- | --- | --- |
| val (chose it) | 2,165 -> 2,169 / 2,240 = 96.8% [96.0, 97.5] | +0.18 [-0.06, +0.44] (+4/-0) | 86 -> 86 / 974 | +0.00 [-0.38, +0.38] | 870/3,214 | 2.19 |
| val benign_hard | — | — | 32 -> 32 / 92 | +0.00 [-2.42, +2.42] | | |
| replay | 247 -> 254 / 306 = 83.0% [78.4, 86.8] | **+2.29 [+0.41, +4.31]** (+7/-0) | 194 -> 219 / 1,185 = 18.5% [16.4, 20.8] | **+2.11 [+1.27, +2.99]** (+25/-0) | 391/2,000 | 2.20 |
| replay_email long | — | — | 32 -> 32 / 64 | +0.00 | | |
| probes | 1 -> 6 / 12 = 50.0% [25.4, 74.6] | +41.7 [+5.1, +67.2] (+5/-0) | 0 -> 0 / 12 | +0.00 | 24/24 | 2.37 |

On replay the 7 recovered injections are itw_jailbreak (64 -> 70) and gandalf_summ (112 -> 113); the 25 new false
positives are itw_regular (24 -> 45 of 295) and aya (118 -> 122 of 492). The replay FPR increase (+2.11 points,
upper bound +2.99) exceeds the +2-point margin the rule enforced on val: the val constraint did not transfer.

Restricted to replay items the served planner splits into at most 4 chunks (`max_chunks`; `planner_fidelity.json`
`two_pass_gain_by_served_chunk_count`), the two-pass gain is recall 235 -> 240/291 (+5) and FPR 191 -> 215/1,179
(+24). The other 2 recovered injections (12 -> 14/15) and 1 new false positive (3 -> 4/6) are on items over the
4-chunk cap, which the D23 configuration already flags by the coverage policy.

Measured ms per request (derived: sum of each pass's measured time per request on the timing samples; each pass
was timed with its own tokenization, which a real two-pass run would do once, so this slightly overstates the
cost): val mean
159.4 -> 326.2 ms, p95 396.1 -> 1,081.4 (18/60 requests run pass 2); replay mean 468.0 -> 925.5 ms, p95 2,500.1 ->
4,894.2 (16/60 run pass 2). Pass 2 roughly doubles DeBERTa compute because N = 128 sends every request over 128
tokens to it (27% of val, 20% of replay requests).

## 4. What it means for the gateway

- **The dilution is real and is a model property, not a serving bug.** With full coverage and max aggregation, a
  short injection inside 256-510 tokens of benign text is missed most of the time when it sits in the middle or at
  the end (newsletter context, middle: 22/62 detected at L = 256, 20/62 at 510, vs 46/62 alone). The RFC's coverage
  rule ("no unscored content") is necessary but not sufficient: a scored token is not a detected injection.
- **Smaller windows recover these attacks** (probe set v2: 1/12 -> 6/12 at W = 128, 10/12 at W = 64, matched controls
  0/12 throughout) **but raise false positives on real benign traffic**, and more so as W shrinks (replay FPR +2.8
  points at W = 128, +9.5 at W = 64; val benign_hard 34.8% -> 48.9% at W = 64; long replay emails 50% -> 84% at
  W = 64). The controlled newsletter contexts and the probe controls show no false positives at any W, so they
  understate this risk; the val_benign contexts and replay show it.
- **Recommendation: do not turn a small-window pass on as a blocking decision now.** The val-chosen two-pass
  setting (t2 = 0.9 by tie-break; t2 = 0.5 is equally good on val and costs +2.78 points of replay FPR) buys +2.3 points of replay recall (7/306) for +2.1 points of replay FPR (25/1,185) at about 2.2x the
  DeBERTa model tokens, about 2x the mean and 2-2.7x the p95 latency on this CPU. benign_hard FPR is already the failing gate
  slice (34.8% at W = 510) and a small window cannot lower it. Reasonable next steps, in order: (1) run the W2 = 128
  pass in **shadow** (the gateway's async shadow path exists) to measure the flag-rate change on live traffic
  without affecting decisions; (2) use small windows as a **deep-scan** option for requests the coverage policy
  already escalates (over-long or chunk-capped inputs), where the extra cost is bounded and the probes show the
  gain; (3) the real fix is a model trained on long inputs with short injections embedded at random positions
  (the dilution curve here is a ready-made eval for it), or exchange-level classification (RFC P2).
- If a small-window pass is ever made blocking, its threshold must be set on a benign set with long instruction-like
  text (benign_hard-style, n >= 100) and checked on held-out long benign traffic; val alone did not predict the
  replay FPR change.

## 5. Caveats

- DeBERTa v2 was trained on short prompts; scores on windows that start or end mid-sentence (small windows cut
  arbitrarily) are out of its training distribution in both directions (fragments of benign text flagged; a fragment
  of `system_notice_action` scoring higher than the whole).
- The dilution curve is synthetic: token-level splicing, two context pools, 62 injections, of which only 46 are
  detected alone; deepset and jackhhao injections are few (20 and 13). Val injections were drawn from val, and spml
  is DeBERTa-favourable (98.8% val recall). Rates per (pool, L, position) have n = 62 and wide intervals.
- val_benign contexts are concatenations of unrelated benign prompts, which is not what a real long email looks
  like; their false-positive rate under small windows is a stress test, not a traffic estimate.
- The replay set is a systems set (`data/replay/MANIFEST.json`: "NOT an accuracy test set"); its labels are partial
  and its sources (jailbreak vs injection) differ from the gate's. Nothing was tuned on it.
- Small-window caches reuse the cascade cache's score for requests with n <= W (the same single model input); a
  test recomputes a sample of both cached paths. Timing numbers are single measurements on a shared 2-core VM
  (`onnxruntime` 1.29.0, 2 threads), n = 60 per set; the replay sample's mean and p95 are driven by a few long
  requests. The gateway's int8 Stage 2, its 4-chunk cap, its coverage policy and its structure-aware planner were
  not modelled here (planner agreement: see the method paragraph at the top and `planner_fidelity.json`).
- All benign_hard numbers are n = 92 < 100: INSUFFICIENT_EVIDENCE under the RFC gate rule.
