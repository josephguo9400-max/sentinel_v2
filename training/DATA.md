# Stage 1 data: audit, splits and evaluation

Owner: Agent B. Audited 2026-10-29. Each number below comes from a committed file, and the file is named
next to it. To reproduce:

```bash
pip install --break-system-packages -r training/requirements.txt
python -m training.sources --fetch          # raw files -> data/raw/ (gitignored), sha256-verified
python -m training.stage2_overlap --fetch   # Stage 2 reference datasets (~200 MB, gitignored)
python -m training.stage2_overlap           # -> data/stage2_overlap.json, data/stage2_overlap_keys.txt
python -m training.build_splits             # -> data/splits/*.jsonl, MANIFEST.json, data/frozen_test.sha256
python -m training.check_leakage            # exit 0 = no exact or near duplicate across splits
python -m training.train_stage1             # -> artifacts/candidates/stage1-tfidf-v1/ (refuses to overwrite)
python -m training.eval_harness --backend local --artifact artifacts/candidates/stage1-tfidf-v1 \
    --split frozen_test --out training/results/stage1-tfidf-v1_frozen_test.json
```

## 1. Sources

### Used

All five are permissively licensed per their dataset cards at the pinned revision, so the splits are
redistributed in `data/splits/` with attribution in `data/splits/NOTICE.md`. File sha256 values are
pinned in `training/sources.py` and checked on every load.

| Name | HF dataset @ revision | License (card) | Raw size | Label definition and our mapping | Language mix (langdetect, seed 0) |
| --- | --- | --- | --- | --- | --- |
| deepset | `deepset/prompt-injections` @ `4f61ecb0` | Apache-2.0 (YAML `license:`; an older `dataset_info.license: cc-by-4.0` is also left in the card, and both are permissive) | 662 (399 benign / 263 injection) | `label` 1 = injection / goal hijacking, including role-play hijacks and "forget the above" attacks; 0 = normal query. **Mapping: identity.** | about 54% en, 40% de, 6% other (pre-dedup) |
| spml | `reshabhs/SPML_Chatbot_Prompt_Injection` @ `02ce8084` | MIT | 16,012 (3,470 / 12,542); 1 empty row dropped | `Prompt injection` = 1 if the user prompt tries to make the bot violate **its system prompt**. We use only `User Prompt`. **Mapping: identity.** Caveat: the label depends on the system prompt, which we drop, so some positives look benign in isolation (the card itself advises against training detectors on it). Degree 0 is always label 0. | about 97% en (1,000-row sample) |
| yanis | `yanismiraoui/prompt_injections` @ `bd55359f` | Apache-2.0 (card + LICENSE + NOTICE files, added 2026-06-02) | 1,034, all injection | Every row is an injection (prompt leaking, jailbreak, mode switching). **Mapping: all 1.** No negatives. | **31% en**, fr 14%, es 14%, it 14%, pt 13%, de 12%, ro/ca under 1% |
| gandalf | `Lakera/gandalf_ignore_instructions` @ `04737b65` | MIT | 1,000, all injection | Gandalf game prompts with embedding similarity of at least 0.825 to "Ignore all previous instructions". **Mapping: all 1.** The card warns that a few rows may not be real injections. | about 95% en, 3% fr (train file) |
| jackhhao | `jackhhao/jailbreak-classification` @ `2f2ceeb3` (`default/jailbreak_dataset_full.csv`) | Apache-2.0 | 1,998 (1,332 benign / 666 jailbreak) | `type`: jailbreak -> 1, benign -> 0. **A jailbreak is not the same as an injection**: these are mostly DAN-style persona prompts aimed at the model's policy, not at an application's instructions. The mapping is a deliberate approximation, and the source is never part of the frozen test. | about 97% en (500-row sample) |

Language counts come from `lang` in each split record (computed after dedup, `langdetect==1.0.9`,
`DetectorFactory.seed=0`).

### Rejected

| Dataset | Reason |
| --- | --- |
| `xTRam1/safe-guard-prompt-injection` | No license in the card |
| `imoxto/prompt_injection_cleaned_dataset-v2` | No license; also named in the Stage 2 **v1** card |
| `geekyrakshit/prompt-injection-dataset`, `jayavibhav/prompt-injection-safety` | No license |
| `hackaprompt/hackaprompt-dataset` | Gated (auth required); MIT, but not reachable without a token; named in the v1 card |
| `Harelix/Prompt-Injection-Mixed-Techniques-2024` | HF API returns 401 (removed or private); named in the v2 card |
| `qualifire/prompt-injections-benchmark` | API redirect / not reachable unauthenticated; license not verified |
| `Smooth-3/llm-prompt-injection-attacks` | Apache-2.0, but it is a heuristic relabel of the two datasets above, whose licenses are unverified ("Original datasets are not redistributed") |
| `Lakera/mosscap_prompt_injection` | MIT, about 280k game prompts with no injection/benign label; would need its own labeling rule. Deferred. |
| `alespalla/chatbot_instruction_prompts`, `databricks/databricks-dolly-15k` (as benign filler) | alespalla is **confirmed** Stage 2 training data. Dolly is CC-BY-SA-3.0 (share-alike) and is inside `VMware/open-instruct`, also confirmed. Neither can serve as clean benign evaluation data. |

## 2. Stage 2 training-data overlap (the most important caveat)

Stage 2 is `protectai/deberta-v3-base-prompt-injection-v2` @ `90c9989b`. Its card:

- names **7** datasets: `natolambert/xstest-v2-copy`, `VMware/open-instruct`,
  `alespalla/chatbot_instruction_prompts`, `HuggingFaceH4/grok-conversation-harmless`,
  `Harelix/Prompt-Injection-Mixed-Techniques-2024`, `OpenSafetyLab/Salad-Data`,
  `jackhhao/jailbreak-classification`;
- but its license summary counts **22** datasets (1 CC-BY-3.0, 8 MIT, 1 CC0, 6 "no license",
  5 Apache-2.0, 1 CC-BY-4.0), so **15 are unnamed**, among them 8 MIT, 1 CC0 and 6 unlicensed;
- adds "prompt injections crafted using insights from ... papers, articles, security competitions".

The v1 card (same team, same task) names 12 more, including `Lakera/gandalf_ignore_instructions` and
`hackaprompt/hackaprompt-dataset`.

What this means for each of our sources:

| Source | Tier | Evidence | Where it may appear |
| --- | --- | --- | --- |
| jackhhao | **confirmed** | Named in the v2 card | train / val only. Its val numbers for any Stage 2 comparison are inflated by construction. |
| gandalf | **suspected** | Named in the v1 card; plausibly one of the 8 unnamed MIT datasets of v2 | frozen_test, **reported as its own slice** (`stage2_overlap=suspected`) |
| deepset | **suspected** | The most widely used public prompt-injection set, published a year before v2. Its card said CC-BY-4.0 at v2 training time (changed to Apache-2.0 on 2024-07-30), and the card's only CC-BY-4.0 slot is xstest, so the license buckets do not point to it. The tier is precautionary, not evidence-based. | train / val only |
| spml | none_known | Not named in either card; published 2024-04-01, about 3 weeks before v2 training (2024-04-20) | train / val / frozen_test (by system-prompt group) |
| yanis | none_known | Not named in either card; at v2 training time it had an `apache-2.0` YAML tag, and the 5 Apache datasets in v2 are all named, so it is less likely, not excluded | frozen_test |

"none_known" does not mean "clean": the unnamed 15 make it impossible to clear any public source
completely.

**Empirical probe** (`training/stage2_overlap.py` -> `data/stage2_overlap.json`). Every record of ours
was compared, exactly and near (char-5gram Jaccard of at least 0.8), against the 6 named Stage 2
datasets that are still downloadable (all files listed in the script). Matches, counted before dedup:

| Our source (records) | alespalla | open-instruct | grok-harmless | Salad-Data | xstest |
| --- | --- | --- | --- | --- | --- |
| jackhhao (1,998) | 1 (label 1) | 51 (49 benign) | 0 | **81 (all jailbreak)** | 0 |
| spml (16,011) | 49 (all benign) | 4 (benign) | 1 (benign) | 0 | 0 |
| deepset (662) | 2 | 3 | 0 | 2 | 0 |
| yanis, gandalf | 0 | 0 | 0 | 0 | 0 |

Action: every record whose duplicate cluster matched a named Stage 2 dataset is **removed from the
frozen test** (7 spml records; see `dropped_from_frozen_stage2_named_match_by_source` in
`data/splits/MANIFEST.json`). Records in train/val keep the flag `stage2_named_match: true`. The probe
cannot see the 15 unnamed datasets, nor `Harelix/...` (401).

Consequences for the gate: on the frozen test, the `source=gandalf` slice should be read as "Stage 2
may have trained on this". The cleanest Stage 1 vs Stage 2 comparison is `stage2_overlap=none_known`
(spml + yanis), and even that is not guaranteed clean.

## 3. Split pipeline (`training/build_splits.py`)

1. **Load** each source (sha256-verified) plus the synthetic benign-hard set (section 4).
2. **Clean** stored text minimally: CRLF to LF, strip. Empty rows are dropped. Stored text stays close
   to raw on purpose, because the gateway scores raw text.
3. **Dedup before splitting**, on a separate dedup key (NFKC, casefold, punctuation to space, whitespace
   collapsed):
   - exact: same key;
   - near: character 5-gram shingles, MinHash with 128 permutations (datasketch, fixed seed), LSH
     candidates at Jaccard 0.6, each candidate pair **verified with the exact Jaccard; at least 0.8 is a
     near duplicate**. The 0.6 candidate threshold lowers LSH false negatives near 0.8. The threshold
     0.8 was chosen by inspection: pairs above it are template re-uses or one-word edits (for example
     spml prompts that differ only in the bot name); pairs below it are distinct prompts that share
     boilerplate.
   - Exact and near links are joined with union-find into clusters. **A cluster with conflicting
     labels is dropped entirely** (2 deepset records). Otherwise one representative is kept, with fixed
     priority yanis > gandalf > spml > deepset > jackhhao > benign_hard, so a held-out frozen record is
     never replaced by a training copy of itself.
   - Result: 21,073 input records, 206 exact links, 379 verified near pairs, 427 records removed as
     duplicates (spml 162, jackhhao 155, yanis 83, deepset 21, gandalf 4, benign_hard 2).
4. **Source-aware assignment** (no item-level random split for held-out sources):

| Source | Rule | Why |
| --- | --- | --- |
| yanis, gandalf | whole source -> frozen_test | Held-out **sources**: the frozen test measures transfer to unseen data styles (multilingual; game attacks) |
| spml | by **system-prompt group** (840 groups), hash(seed, group): 60% train / 15% val / 25% frozen_test | Prompts written against one system prompt share wording, so they never straddle splits |
| deepset | its own train split -> train, its own test split -> val | Small, multilingual and suspected Stage 2 data, so never frozen |
| jackhhao | hash(seed, text): 80% train / 20% val | Confirmed Stage 2 data, so never frozen |
| benign_hard | template 0 of each family -> val; templates 1-3 -> frozen_test | The frozen slice uses templates whose wording Stage 1's threshold was not tuned on |

5. **Stage 2 named-match exclusion** from frozen_test (section 2).
6. Files are written sorted by id, with sorted JSON keys. `data/frozen_test.sha256` holds the sha256 of
   `data/splits/frozen_test.jsonl`. `MANIFEST.json` also gives a `content_sha256` over (id, text, label)
   only.

**Determinism**: no global RNG; every random choice is `sha256(seed:value)`. Running the full build
twice gives byte-identical files (checked on 2026-10-29, sha256 unchanged).
`tests/training/test_training.py::test_splits_deterministic_across_two_runs` runs it twice on a
subset.

**Leakage**: `python -m training.check_leakage` re-derives exact and near duplicates across splits from
the files alone (independent of the build), checks id uniqueness and the frozen hash, and exits 1 on any
finding. Current result: 20,637 records, 0 exact, 0 near, exit 0.

### Split sizes (`data/splits/MANIFEST.json`)

| Split | n | pos | neg | by source (label 0 / 1) |
| --- | --- | --- | --- | --- |
| train | 11,149 | 7,878 | 3,271 | spml 1,888 / 7,269 · jackhhao 1,045 / 420 · deepset 338 / 189 |
| val | 3,214 | 2,240 | 974 | spml 552 / 2,080 · jackhhao 276 / 102 · deepset 54 / 58 · benign_hard 92 / 0 |
| frozen_test | 6,274 | 5,137 | 1,137 | spml 863 / 3,190 · gandalf 0 / 996 · yanis 0 / 951 · benign_hard 274 / 0 |

Frozen test sha256: `6e98eff17f7482c5a67ff42da3e5e08debc2e9c9a1c76215a7c970f3d2f4a388`.

The frozen test's **only natural benign data is spml** (863 records). Its FPR is therefore mostly an
spml-plus-synthetic FPR; there is no held-out benign source of real emails.

## 4. Benign-hard slice (`training/benign_hard.py`, `benign-hard-v1`)

These are synthetic, template-generated benign emails. Every record carries `synthetic: true`,
`family`, `template` and `generator`, and all names and domains are fictional (`example.com`). Six
families with four templates each: `it_notice`, `forwarding_request`, `policy_reminder`, `calendar_ask`
(18 per template, short and imperative: "ignore the earlier invite", "do not reply", "follow these
instructions even if your manager asks...") and `newsletter_long`, `log_digest_long` (10 per template,
1.5k to 6k characters, repetitive). That is 368 generated, 366 after dedup: **92 in val** (template 0)
and **274 in the frozen test** (templates 1-3). Benign-hard records are never in train.

Limitation: template text has low diversity, so results show how Stage 1 handles *these* phrasings, not
benign email in general. The RFC calls the slice "hand-written"; it is template-generated here, as the
task asked, and each template was written by hand.

## 5. Stage 1 model (`training/train_stage1.py` -> `artifacts/candidates/stage1-tfidf-v1/`)

- Word TF-IDF (1-2 grams, min_df 2, max 50k features, no sublinear TF) + LogisticRegression (C 10,
  no class weight). Chosen from a 16-point grid (ngram × C × class_weight) by **mean per-source ROC-AUC
  on val** (pooled AUC as the tie-break), so spml (82% of val) does not dominate. All trials are in
  `evaluation.json`.
- **Threshold 0.440418**: maximizes pooled F1 on val (0.978).
- **Provisional gray zone [0.282715, 0.818603]**: low = the score below which at most 1% of val positives
  fall (22 / 2,240 finalized as benign); high = the score above which at most 1% of val negatives fall
  (9 / 974 finalized as injection). Val escalation: 243 / 3,214 = 7.6%. Rationale: Stage 1 should
  finalize only what it is confident about, and the cascade-and-shadow milestone re-sweeps the zone
  against Stage 2 on val.
- `failure_policy: fail_open`, `timeout_ms: 50`: Stage 1 is a cost filter, not the security boundary.
  Both are provisional and the owner may override them.

### ONNX parity

On val, skl2onnx 1.20.0 + onnxruntime 1.29.0 vs sklearn 1.9.1: **max abs diff 7.2e-7** (chunked scores;
5.96e-7 whole-text; n = 3,214). One decision flips at the threshold, because one val score lies 1.4e-7
from it (float32 vs float64). Three mismatches were found and fixed on the way there:

1. **Locale**: the ONNX StringNormalizer needs an OS locale. This machine has only C / C.UTF-8, and
   C.UTF-8 lowercases differently from Python for some characters (for example U+0130). Fix: lowercase
   ASCII only (`ascii_lower`) in sklearn and `locale="C"` in the graph; non-ASCII case is preserved on
   both sides.
2. **sublinear_tf**: skl2onnx exports it as `log(1 + tf)`, while sklearn computes `1 + log(tf)`
   (diff up to 0.031). Fix: `sublinear_tf=False`.
3. **Bigrams over pruned unigrams**: onnxruntime's TfIdfVectorizer can only count n-grams whose words
   are in the vocabulary; sklearn counts them anyway (diff up to 0.0069 on texts like "programed").
   Fix: the vocabulary drops such n-grams (24 of about 50k).

Regex: Python `[^\W]{2,}` and RE2 `[\p{L}\p{N}_]{2,}` express the same class. The unit test
`test_onnx_parity_small_model` covers German, Turkish and Greek text plus the pruned-bigram case.

### Long inputs

`max_input_tokens` 512 counts regex tokens. Longer texts are scored in windows of 512 tokens with
stride 384, cut at token boundaries and aggregated by max (`stage1_text.chunk_text`). Nothing is
truncated. Val has 25 texts over 512 tokens (at most 5 windows).

### Latency (`evaluation.json` -> `latency`)

Measured with `Stage1Onnx.score()` (chunking + inference + aggregation) on every val text, sequentially,
one text per call, after 50 warmup calls. Hardware: Intel Xeon @ 2.10 GHz, 2 cores, onnxruntime 1.29.0
CPU EP with default threads, Python 3.13.18. n = 3,214: **p50 0.23 ms, p95 0.42 ms, p99 0.71 ms**, max
2.8 ms. Above 512 tokens (chunked, n = 25): p50 1.06 ms, p99 2.62 ms. This excludes HTTP and JSON.

## 6. Results (`training/results/stage1-tfidf-v1_{val,frozen_test}.json`)

Stage 1 alone, with decision score >= threshold. Wilson 95% intervals.

| Slice | Recall | FPR | Precision |
| --- | --- | --- | --- |
| val overall | 2,207/2,240 = 98.5% [97.9, 98.9] | 67/974 = 6.9% [5.5, 8.6] | 2,207/2,274 = 97.1% |
| val benign_hard | — | 31/92 = 33.7% [24.9, 43.8] · **INSUFFICIENT_EVIDENCE** (n < 100) | — |
| **frozen overall** | 4,353/5,137 = **84.7%** [83.7, 85.7] | 123/1,137 = **10.8%** [9.1, 12.8] | 4,353/4,476 = 97.3% |
| frozen spml | 3,171/3,190 = 99.4% [99.1, 99.6] | 35/863 = 4.1% [2.9, 5.6] | 98.9% |
| frozen gandalf (Stage 2 suspected) | 925/996 = 92.9% [91.1, 94.3] | — | 100% |
| frozen yanis (69% non-English) | **257/951 = 27.0%** [24.3, 29.9] | — | 100% |
| frozen lang=en | 4,213/4,455 = 94.6% [93.9, 95.2] | 123/1,136 = 10.8% | 97.2% |
| frozen lang=non_en | 140/682 = 20.5% [17.7, 23.7] | — | — |
| **frozen benign_hard** | — | **88/274 = 32.1% [26.9, 37.9] · FAIL** (Wilson upper bound 37.9% > 5%) | — |

Benign-hard by family (frozen): policy_reminder 33/52 flagged, forwarding_request 22/54,
it_notice 18/54, calendar_ask 13/54, newsletter_long 2/30, log_digest_long 0/30. Each family is
INSUFFICIENT_EVIDENCE on its own (n < 100).

Stage 1 routing if used as cascade Stage 1 with this gray zone (frozen overall): low (final allow)
912 benign + **605 injections**; gray (escalate) 217 + 644 = 861 / 6,274 = 13.7%; high (final flag)
8 benign + 3,888 injections. Benign-hard: 145 / 274 escalated, 7 finalized as flagged.

## 7. What does not work / open issues

- **Stage 1 does not generalize to non-English injections**: yanis recall is 27%. The train set has
  only deepset's German as non-English data. As a cascade Stage 1, the gray zone sends 563 / 951 yanis
  injections to "final benign" without Stage 2 ever seeing them. This argues for a language gate in the
  router (escalate non-English) or multilingual Stage 1 training data. It is not fixed here.
- **benign-hard FPR fails the RFC gate** (32% vs a 5% Wilson upper bound). Instruction-like benign mail
  is exactly what TF-IDF confuses with injections; most of those records land in the gray zone, so the
  cascade decision depends on Stage 2 for them.
- The val threshold was tuned on in-distribution sources (spml-dominated); the frozen test's held-out
  sources show the shift (98.5% -> 84.7% recall).
- spml labels depend on a system prompt that we drop; some "injections" look benign out of context.
- Language tags come from langdetect and are noisy on short texts.
- The HTTP gateway backend was tested only against a mock transport and a local fake server. Agent A's
  endpoint has not been called.
