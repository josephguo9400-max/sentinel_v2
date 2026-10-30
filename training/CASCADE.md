# Cascade sweep on val (milestone cascade-and-shadow)

Owner: Agent B. Every number here is in `training/results/cascade_sweep_val.json` (written by
`python -m training.cascade_sweep` from `training/results/cascade_scores_val.jsonl`, which
`python -m training.cascade_scores` computes). Val only (`data/splits/val.jsonl`, n = 3,214: 2,240 injections,
974 benign); the frozen test was not read. Intervals: Wilson 95% for proportions, Newcombe paired hybrid-score
95% for differences against DeBERTa alone on the same items.

```bash
flock /home/claude/measure.lock -c 'python -m training.cascade_scores --split val'   # resumable, ~10 min on 2 cores
python -m training.cascade_sweep
python -m training.publish_stage1        # -> artifacts/candidates/stage1-tfidf-v2.1/ (refuses to overwrite)
```

## Setup

- **Stage 2 = DeBERTa alone (the reference)**: `protectai/deberta-v3-base-prompt-injection-v2` ONNX fp32
  (sha256 pinned), inputs `[CLS] + 510 content tokens + [SEP]`, stride 384, the last window ending at the tail
  (same rule as `serving/encoding.py`, checked by `test_window_starts_match_the_gateway_rule`), **every** window
  scored (no `max_chunks` cap), max aggregation, flag iff score >= 0.5. Val: recall 2,165/2,240 = 96.7%
  [95.8, 97.3], FPR 86/974 = 8.8% [7.2, 10.8].
- **Stage 1**: `stage1-tfidf-v2` model bytes, request score = its own chunked max.
- **Served decision** (the milestone's rule): Stage 1 may finalize **allow** only, only below `low`, and only
  for in-scope language (`lang == "en"`, the langdetect tag in val); everything else takes the DeBERTa
  decision. In manifest terms this is `gray_zone = [low, 1.0]`: Stage 1 never exceeds 1.0 (val max 0.9999987),
  so the "above high -> final injection" rule never fires.
- **Chunk-level variant**: an escalated in-scope request sends DeBERTa only the windows whose Stage 1 score is
  >= `low`, plus the first and the last window; out-of-scope requests send all windows.
- **Selection rule** (stated in `selection_rule` in the JSON): maximize the share of requests that never reach
  Stage 2, subject to (a) the paired 95% CI lower bound of recall(cascade) - recall(DeBERTa) >= -2 points (the
  RFC gate check), (b) the Wilson lower bound of cascade recall >= DeBERTa recall - 2 points, (c) the paired
  upper bound of the FPR difference <= +2 points and (d) no more benign-hard false positives than DeBERTa
  ((c) and (d) hold by construction under allow-only). Grid: `low` in 0.01 ... 0.50 plus 0.282715 and the Stage 1
  threshold 0.505068; `low` cannot exceed the threshold, because the manifest requires `low <= threshold`.

## Result: chosen `gray_zone = [0.505068, 1.0]`, published as `stage1-tfidf-v2.1`

The rule picks the largest `low` the manifest allows; every `low` on the grid satisfies the constraints.

| Setting (val) | Requests saved | Recall (cascade) | Recall diff vs DeBERTa | FPR (cascade) | DeBERTa windows computed |
| --- | --- | --- | --- | --- | --- |
| DeBERTa alone | 0 | 2,165/2,240 = 96.7% [95.8, 97.3] | — | 86/974 = 8.8% [7.2, 10.8] | 3,292/3,292 |
| **chosen [0.505068, 1.0]** | **922/3,214 = 28.7% [27.1, 30.3]** | 2,146/2,240 = 95.8% [94.9, 96.6] | -0.85 [-1.30, -0.46] pts | 17/974 = 1.7% [1.1, 2.8] | 2,329/3,292 = 70.7% |
| strictly equal recall [0.18, 1.0] | 760/3,214 = 23.6% [22.2, 25.1] | 2,165/2,240 = 96.7% [95.8, 97.3] | 0 [-0.18, 0.18] pts | 35/974 = 3.6% [2.6, 5.0] | 2,495/3,292 = 75.8% |

At the chosen setting Stage 1 finalizes 38 injections as allow; DeBERTa would have flagged 19 of them (the
other 19 DeBERTa misses too). Recall drops by 19 items, within the 2-point margin; FPR falls from 8.8% to 1.7%
because Stage 1 allows 884 of the 974 benign items, including 69 that DeBERTa flags.

### Kill criterion: **triggered**

The RFC kill rule is "savings < 30% at equal recall -> fall back to the single model and report the negative
result". At strictly equal recall the cascade saves **23.6%** of Stage 2 requests; even with the 2-point recall
margin it saves **28.7%** (upper Wilson bound 30.3%). Both are below 30%. **On val, the cascade as specified does
not meet the RFC bar; this is the negative result.**

Why: under allow-only routing Stage 1 can save a request only by allowing it, so savings are bounded by the
benign share of the traffic, and val is 70% injections (974/3,214 benign). Per class at the chosen setting,
Stage 1 finalizes 884/974 = 90.8% [88.8, 92.4] of benign items and 38/2,240 = 1.7% of injections
(`saved_by_label`). Reweighting those per-class rates to other base rates (`saved_reweighted_by_base_rate`;
derived, not measured, and assuming val's per-class score distributions hold) gives 89.9% saved at a 1%
injection rate, 81.9% at 10% and 64.0% at 30%. Real gateway traffic is expected to be mostly benign, so the
val number understates savings; but that projection is exactly what the replay set and the gate must measure,
not something val can show. The owner decided (D22) to judge the criterion on the replay set.

**Outcome on the replay set (D23, 2026-10-30): the cascade is rejected.** Savings did rise with the benign share,
as projected, but recall did not hold: injections from a source Stage 1 never trained on (`gandalf_summ`, indirect
injections inside text to summarize) score low under TF-IDF and are allowed before DeBERTa sees them. At this
gray zone the offline approximation loses 56 DeBERTa-caught injections (recall 80.7% -> 62.4%) and saves 1.35% at
equal recall; the gateway itself served 66.0% recall with 59.9% of requests sent to Stage 2
(`reports/cascade-and-shadow/replay_counterfactual.json`, `views.served`). The val sweep could not see this,
because val shares Stage 1's training sources: a threshold chosen in-distribution says nothing about a new attack
style. The gateway now runs Stage 1 observe-only (`stage1_mode: observe`); DeBERTa decides every request.

### Not chosen: letting Stage 1 finalize flags

The contract allows `high < 1.0` (scores above `high` finalized as injection). The best such setting,
`[0.505068, 0.764614]` (`flag_finalizing_variant_best`), saves 3,037/3,214 = 94.5% [93.6, 95.2] with recall
2,172/2,240 = 97.0% (diff -0.29 to +0.93 pts) and FPR 22/974 = 2.3%. It is reported, not chosen: on val Stage 1
is in-distribution (trained on the same sources, spml is 82% of val), while on the frozen test Stage 1 alone
recalls 83.5% versus 97.9% on val (DATA.md section 6). A Stage 1 flag would be final for held-out traffic where
Stage 1 is much weaker, and the milestone restricts Stage 1 to allow. The provisional v2 zone
`[0.282715, 0.764614]` is listed under `provisional_stage1_tfidf_v2` for comparison.

### Chunk level

Only 45 of 3,214 val items need more than one DeBERTa window, so the chunk-level variant barely differs on val:
2,327/3,292 windows computed versus 2,329 for the request-level cascade at the chosen setting, same recall
(2,146). Val cannot measure the chunk-level saving; it needs long inputs (the replay set has some; the red-team
middle-placement and boundary-split probes are the real test). 4 val items need more than 4 windows (the
gateway's default `max_chunks`); this sweep scores all windows.

## Per slice at the chosen setting

| Slice | n | Saved | Cascade recall | Cascade FPR | DeBERTa recall | DeBERTa FPR |
| --- | --- | --- | --- | --- | --- | --- |
| source=spml | 2,632 | 552 = 21.0% [19.5, 22.6] | 2,040/2,080 = 98.1% [97.4, 98.6] | 5/552 = 0.9% [0.4, 2.1] | 2,056/2,080 = 98.8% | 53/552 = 9.6% |
| source=jackhhao (Stage 2 trained on it) | 378 | 261 = 69.0% [64.2, 73.5] | 85/102 = 83.3% [74.9, 89.3] | 0/276 = 0.0% [0.0, 1.4] | 87/102 = 85.3% | 1/276 = 0.4% |
| source=deepset | 112 | 38 = 33.9% [25.8, 43.1] | 21/58 = 36.2% [25.1, 49.1] | 0/54 = 0.0% [0.0, 6.6] | 22/58 = 37.9% | 0/54 = 0.0% |
| lang=en | 3,150 | 922 = 29.3% [27.7, 30.9] | 2,134/2,216 = 96.3% [95.4, 97.0] | 17/934 = 1.8% [1.1, 2.9] | 2,153/2,216 = 97.2% | 86/934 = 9.2% |
| **lang=non_en** | 64 | 0 = 0.0% [0.0, 5.7] (by construction) | 12/24 = 50.0% [31.4, 68.6] | 0/40 = 0.0% [0.0, 8.8] | 12/24 = 50.0% | 0/40 = 0.0% |
| **benign_hard** | 92 | 71 = 77.2% [67.6, 84.6] | — | 12/92 = 13.0% [7.6, 21.4] · INSUFFICIENT_EVIDENCE (n < 100) | — | 32/92 = 34.8% [25.8, 44.9] |

Benign-hard by family (cascade FP / n; DeBERTa FP): it_notice 9/18 (9), policy_reminder 2/18 (6),
forwarding_request 1/18 (3), calendar_ask 0/18 (4), log_digest_long 0/10 (10), newsletter_long 0/10 (0). Every
family is n < 100. The benign-hard FPR improves because Stage 1 allows many of these emails before DeBERTa sees
them: **DeBERTa alone flags 34.8% of val benign-hard, including all 10 long log digests**, which is the
length/repetition drift reported in the detector baseline. Non-English injection recall is 50% for both, from
24 positives: val has almost no non-English data, so this slice says nothing about multilingual traffic.

## Caveats

- Val is spml-dominated and in-distribution for Stage 1; jackhhao is confirmed Stage 2 training data. Savings and
  the recall cost on held-out sources will differ. The gate (next milestone) measures them on the frozen test.
- The language gate uses the langdetect tag stored in val; the gateway needs its own language detector, and
  whatever it uses decides what "in scope" means in production.
- The score cache is computed with batch size 1 on this host; no latency number from this run is reported.
