# Stage 1 data: audit, splits and evaluation

Owner: Agent B. Audited 2026-10-29. Each number below comes from a committed file, and the file is named
next to it; every such file is written by committed code. To reproduce:

```bash
pip install --break-system-packages -r training/requirements.txt
python -m training.sources --fetch          # raw files -> data/raw/ (gitignored), sha256-verified
python -m training.stage2_overlap --fetch   # Stage 2 reference datasets (~200 MB, gitignored)
python -m training.stage2_overlap           # -> data/stage2_overlap.json, data/stage2_overlap_keys.txt
python -m training.build_splits             # -> data/splits/*.jsonl, MANIFEST.json, data/frozen_test.sha256
python -m training.check_leakage            # exit 0 = no exact/near duplicate and no shared benign-hard sentence
python -m training.language_mix             # -> data/language_mix.json (language column of section 1)
python -m training.train_stage1             # -> artifacts/candidates/stage1-tfidf-v2/ (refuses to overwrite)
python -m training.eval_harness --backend local --artifact artifacts/candidates/stage1-tfidf-v2 \
    --split frozen_test --out training/results/stage1-tfidf-v2_frozen_test.json
```

## 1. Sources

### Used

All five are permissively licensed per their dataset cards at the pinned revision, so the splits are
redistributed in `data/splits/` with attribution in `data/splits/NOTICE.md`. File sha256 values are
pinned in `training/sources.py` and checked on every load.

| Name | HF dataset @ revision | License (card) | Raw size | Label definition and our mapping | Language mix: raw, pre-dedup (`data/language_mix.json` -> `raw`) |
| --- | --- | --- | --- | --- | --- |
| deepset | `deepset/prompt-injections` @ `4f61ecb0` | Apache-2.0 (YAML `license:`; an older `dataset_info.license: cc-by-4.0` is also left in the card, and both are permissive) | 662 (399 benign / 263 injection) | `label` 1 = injection / goal hijacking, including role-play hijacks and "forget the above" attacks; 0 = normal query. **Mapping: identity.** | 355/662 = 53.6% en, de 40.0%, es 1.1%, other 5.3% |
| spml | `reshabhs/SPML_Chatbot_Prompt_Injection` @ `02ce8084` | MIT | 16,012 (3,470 / 12,542); 1 empty row dropped | `Prompt injection` = 1 if the user prompt tries to make the bot violate **its system prompt**. We use only `User Prompt`. **Mapping: identity.** Caveat: the label depends on the system prompt, which we drop, so some positives look benign in isolation (the card itself advises against training detectors on it). Degree 0 is always label 0. | 15,992/16,011 = 99.9% en |
| yanis | `yanismiraoui/prompt_injections` @ `bd55359f` | Apache-2.0 (card + LICENSE + NOTICE files, added 2026-06-02) | 1,034, all injection | Every row is an injection (prompt leaking, jailbreak, mode switching). **Mapping: all 1.** No negatives. | **322/1,034 = 31.1% en**, fr 14.3%, es 13.9%, it 13.9%, pt 13.4%, de 12.2%, other 1.1% |
| gandalf | `Lakera/gandalf_ignore_instructions` @ `04737b65` | MIT | 1,000, all injection | Gandalf game prompts with embedding similarity of at least 0.825 to "Ignore all previous instructions". **Mapping: all 1.** The card warns that a few rows may not be real injections. | 951/1,000 = 95.1% en, fr 2.9%, other 2.0% |
| jackhhao | `jackhhao/jailbreak-classification` @ `2f2ceeb3` (`default/jailbreak_dataset_full.csv`) | Apache-2.0 | 1,998 (1,332 benign / 666 jailbreak) | `type`: jailbreak -> 1, benign -> 0. **A jailbreak is not the same as an injection**: these are mostly DAN-style persona prompts aimed at the model's policy, not at an application's instructions. The mapping is a deliberate approximation, and the source is never part of the frozen test. | 1,966/1,998 = 98.4% en, other 1.6% |

Language tags: `langdetect==1.0.9`, `DetectorFactory.seed=0`, first 2,000 characters of each text, computed by
`python -m training.language_mix` over **every** raw record (no sampling). "other" sums tags under 1% of a
source; full counts are in the JSON. After dedup, in the committed splits (`data/language_mix.json` ->
`splits`, from each record's `lang` field): deepset 343/639 = 53.7% en, spml 15,823/15,842 = 99.9% en, **yanis
319/951 = 33.5% en (632 = 66.5% non-English)**, gandalf 947/996 = 95.1% en, jackhhao 1,811/1,843 = 98.3% en,
benign_hard 368/368 en. `tests/training/test_training.py::test_committed_language_mix_matches_committed_splits`
fails if the JSON is stale.

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
   - Result: 21,073 input records, 206 exact links, 377 verified near pairs, 425 records removed as
     duplicates (spml 162, jackhhao 155, yanis 83, deepset 21, gandalf 4, benign_hard 0).
4. **Source-aware assignment** (no item-level random split for held-out sources):

| Source | Rule | Why |
| --- | --- | --- |
| yanis, gandalf | whole source -> frozen_test | Held-out **sources**: the frozen test measures transfer to unseen data styles (multilingual; game attacks) |
| spml | by **system-prompt group** (840 groups), hash(seed, group): 60% train / 15% val / 25% frozen_test | Prompts written against one system prompt share wording, so they never straddle splits |
| deepset | its own train split -> train, its own test split -> val | Small, multilingual and suspected Stage 2 data, so never frozen |
| jackhhao | hash(seed, text): 80% train / 20% val | Confirmed Stage 2 data, so never frozen |
| benign_hard | template 0 of each family -> val; templates 1-3 -> frozen_test | Val and frozen records come from different templates **and** from disjoint sentence pools (section 4), so no benign-hard sentence Stage 1's threshold was tuned on reappears in the frozen slice |

5. **Stage 2 named-match exclusion** from frozen_test (section 2).
6. Files are written sorted by id, with sorted JSON keys. `data/frozen_test.sha256` holds the sha256 of
   `data/splits/frozen_test.jsonl`. `MANIFEST.json` also gives a `content_sha256` over (id, text, label)
   only.

**Determinism**: no global RNG; every random choice is `sha256(seed:value)` or a `random.Random` seeded with a
fixed string. Running the full build twice gives byte-identical files (checked on 2026-10-29 for splits-v2:
all three sha256 values identical across two builds).
`tests/training/test_training.py::test_splits_deterministic_across_two_runs` runs it twice on a
subset.

**Leakage**: `python -m training.check_leakage` re-derives exact and near duplicates across splits from
the files alone (independent of the build), checks id uniqueness, the frozen hash and benign-hard sentence
separation (section 4), and exits 1 on any finding. Current result: 20,639 records, 0 exact, 0 near, 0
benign-hard sentences shared by val and frozen_test, exit 0.

### Split sizes (`data/splits/MANIFEST.json`)

| Split | n | pos | neg | by source (label 0 / 1) |
| --- | --- | --- | --- | --- |
| train | 11,149 | 7,878 | 3,271 | spml 1,888 / 7,269 · jackhhao 1,045 / 420 · deepset 338 / 189 |
| val | 3,214 | 2,240 | 974 | spml 552 / 2,080 · jackhhao 276 / 102 · deepset 54 / 58 · benign_hard 92 / 0 |
| frozen_test | 6,276 | 5,137 | 1,139 | spml 863 / 3,190 · gandalf 0 / 996 · yanis 0 / 951 · benign_hard 276 / 0 |

Pipeline version `splits-v2`. Frozen test sha256:
`2a4e011f13250ca65d335bb12939a63f682254a2c6d1566ee78fc7686bb845ba`. Train is byte-identical to splits-v1
(benign_hard is never in train); val and frozen_test changed only in their benign-hard records.

The frozen test's **only natural benign data is spml** (863 records). Its FPR is therefore mostly an
spml-plus-synthetic FPR; there is no held-out benign source of real emails.

## 4. Benign-hard slice (`training/benign_hard.py`, `benign-hard-v2`)

These are synthetic, template-generated benign emails. Every record carries `synthetic: true`,
`family`, `template` and `generator`, and all names and domains are fictional (`example.com`). Six
families with four templates each: `it_notice`, `forwarding_request`, `policy_reminder`, `calendar_ask`
(18 per template, short and imperative: "ignore the earlier invite", "do not reply", "follow these
instructions even if your manager asks...") and `newsletter_long`, `log_digest_long` (10 per template,
1.5k to 6k characters, repetitive). That is 368 generated, 368 after dedup: **92 in val** (template 0)
and **276 in the frozen test** (templates 1-3). Benign-hard records are never in train.

**Separation between val and frozen.** Each family's sentence pools (the extra sentences, policy rules,
newsletter sections and footer, log services, messages and footer) exist twice: a dev pool used only by
template 0 and a frozen pool used only by templates 1-3. The two pools share no sentence; they express similar
instructions in different words (for example "Ignore the earlier invite; the time has changed." in frozen vs
"Disregard the first invitation I sent; it had the wrong day." in val). Only slot values (names, teams,
systems, dates, rooms, documents, topics, numbers) and fragments of at most 30 characters, such as greetings
and sign-off lines, are shared. Each (family, template) has its own RNG stream. `check_leakage` and two tests
(`test_benign_hard_generator_val_and_frozen_share_no_sentence`,
`test_committed_benign_hard_val_and_frozen_share_no_sentence`) fail if any sentence longer than 30 characters
occurs in both val and frozen benign-hard records, either exactly or after masking slot values and digits
(`benign_hard.sentence_skeletons`). The near-duplicate check alone cannot catch this, because records that
share a few sentences stay below Jaccard 0.8.

What the separation does **not** give: the frozen slice is still the same six families, written by the same
author, with the same kinds of instruction-like phrasing. It measures transfer to unseen wording of the same
families, not to unseen kinds of benign email.

Limitation: template text has low diversity, so results show how Stage 1 handles *these* phrasings, not
benign email in general. The RFC calls the slice "hand-written"; it is template-generated here, as the
task asked, and each template was written by hand.

## 5. Stage 1 model (`training/train_stage1.py` -> `artifacts/candidates/stage1-tfidf-v2/`)

Every number in this section is in `artifacts/candidates/stage1-tfidf-v2/evaluation.json`; the JSON key is
named in parentheses.

- Word TF-IDF (1-2 grams, min_df 2, max 50k features, no sublinear TF) + LogisticRegression (C 10,
  no class weight). Chosen from a 16-point grid (ngram × C × class_weight) by **mean per-source ROC-AUC
  on val** (pooled AUC as the tie-break), so spml (82% of val) does not dominate (`selection`). The
  procedure (grid, selection rule, threshold and gray-zone rules) is unchanged from v1 and was fixed before
  any frozen-test number existed; training reads only train and val.
- **Threshold 0.505068**: maximizes pooled F1 on val (0.978) (`threshold`).
- **Provisional gray zone [0.282715, 0.764614]**: low = the score below which at most 1% of val positives
  fall (22 / 2,240 finalized as benign); high = the score above which at most 1% of val negatives fall
  (10 / 974 finalized as injection). Val escalation: 209 / 3,214 = 6.5% (`gray_zone`). Rationale: Stage 1
  should finalize only what it is confident about, and the cascade-and-shadow milestone re-sweeps the zone
  against Stage 2 on val.
- `failure_policy: fail_open`, `timeout_ms: 50`: Stage 1 is a cost filter, not the security boundary.
  Both are provisional and the owner may override them.

### ONNX parity (`onnx_parity_val`, `onnx_parity_ablations`)

On val (n = 3,214), skl2onnx 1.20.0 + onnxruntime 1.29.0 vs sklearn 1.9.1: **max abs diff 7.2e-7** on
chunked scores (6.0e-7 whole-text). **0 decisions flip** at the threshold; the val score closest to the
threshold is 4.2e-7 away from it (`min_abs_distance_to_threshold_sklearn`), so float32 vs float64 could flip
an item that close. Three mismatches were found and fixed on the way there:

1. **Locale**: the ONNX StringNormalizer needs an OS locale. This machine has only C / C.UTF-8, and
   C.UTF-8 lowercases differently from Python for some characters (for example U+0130). Fix: lowercase
   ASCII only (`ascii_lower`) in sklearn and `locale="C"` in the graph; non-ASCII case is preserved on
   both sides. Covered by `test_onnx_parity_small_model` (German, Turkish and Greek text).
2. **sublinear_tf**: skl2onnx exports it as `log(1 + tf)`, while sklearn computes `1 + log(tf)`. Measured
   by retraining the selected configuration with `sublinear_tf=True` and exporting it: max abs diff
   **0.031** on val, 1,850 of 3,214 texts differ by more than 1e-4. Fix: `sublinear_tf=False`.
3. **Bigrams over pruned unigrams**: onnxruntime's TfIdfVectorizer can only count n-grams whose words
   are in the vocabulary; sklearn counts them anyway. Measured the same way without the filter: max abs
   diff **0.0035** on val, 4 texts over 1e-4. Fix: the vocabulary drops such n-grams (**24 of 50,000**,
   `ngrams_dropped_by_onnx_safe_filter`). Covered by `test_onnx_parity_small_model` ("rareword").

Regex: Python `[^\W]{2,}` and RE2 `[\p{L}\p{N}_]{2,}` express the same class.

### Long inputs

`max_input_tokens` 512 counts regex tokens. Longer texts are scored in windows of 512 tokens with
stride 384, cut at token boundaries and aggregated by max (`stage1_text.chunk_text`). Nothing is
truncated. Val has 25 texts over 512 tokens, at most 5 windows (`latency.by_length`, `latency.max_chunks`).

### Latency (`latency`)

`Stage1Onnx.score()` (chunking + inference + aggregation) on every val text, sequentially, one text per
call, after 50 warmup calls; Intel Xeon @ 2.10 GHz, 2 cores, onnxruntime 1.29.0 CPU EP with default threads,
Python 3.13.18. **Conditions: the host was shared with other agents' CPU-heavy load tests that may have
been running at the same time, so these are not benchmark numbers** (`latency.conditions.host_load`).
n = 3,214: p50 0.24 ms, p95 0.44 ms, p99 0.74 ms, max 2.8 ms; above 512 tokens (n = 25): p50 0.95 ms,
p99 2.56 ms. This excludes HTTP and JSON.

## 6. Results (`training/results/stage1-tfidf-v2_{val,frozen_test}.json`)

Stage 1 alone (local ONNX backend), with decision score >= threshold. Wilson 95% intervals. Frozen test =
splits-v2 (`2a4e011f...`); 0 errors, 0 policy decisions (local backend).

| Slice | Recall | FPR | Precision |
| --- | --- | --- | --- |
| val overall | 2,194/2,240 = 97.9% [97.3, 98.5] | 51/974 = 5.2% [4.0, 6.8] | 2,194/2,245 = 97.7% [97.0, 98.3] |
| val benign_hard | — | 21/92 = 22.8% [15.4, 32.4] · **INSUFFICIENT_EVIDENCE** (n < 100) | — |
| **frozen overall** | 4,289/5,137 = **83.5%** [82.5, 84.5] | 103/1,139 = **9.0%** [7.5, 10.8] | 4,289/4,392 = 97.7% [97.2, 98.1] |
| frozen spml | 3,164/3,190 = 99.2% [98.8, 99.4] | 25/863 = 2.9% [2.0, 4.2] | 3,164/3,189 = 99.2% [98.8, 99.5] |
| frozen gandalf (Stage 2 suspected) | 911/996 = 91.5% [89.6, 93.0] | — | 100% |
| frozen yanis (66.5% non-English) | **214/951 = 22.5%** [20.0, 25.3] | — | 100% |
| frozen lang=en | 4,169/4,455 = 93.6% [92.8, 94.3] | 103/1,138 = 9.1% [7.5, 10.9] | 97.6% |
| frozen lang=non_en | 120/682 = 17.6% [14.9, 20.6] | — | — |
| **frozen benign_hard** | — | **78/276 = 28.3% [23.3, 33.8] · FAIL** (Wilson upper bound 33.8% > 5%) | — |

Benign-hard by family (frozen): policy_reminder 30/54 flagged, it_notice 18/54, forwarding_request 17/54,
calendar_ask 11/54, newsletter_long 2/30, log_digest_long 0/30. Each family is INSUFFICIENT_EVIDENCE on its
own (n < 100).

Stage 1 routing if used as cascade Stage 1 with this gray zone (frozen overall, `stage1_zones`): low (final
allow) 894 benign + **605 injections**; gray (escalate) 221 + 552 = 773 / 6,276 = 12.3%; high (final flag)
24 benign + 3,980 injections. Benign-hard: 154 / 276 escalated, 18 finalized as flagged.

These numbers are not comparable one-to-one with the deleted v1 numbers: the frozen benign-hard slice is new
(benign-hard-v2), and the model was retrained because val changed.

## 7. What does not work / open issues

- **Stage 1 does not generalize to non-English injections**: yanis recall is 22.5%. The train set has
  only deepset's German as non-English data. As a cascade Stage 1, the gray zone sends 563 / 951 yanis
  injections to "final benign" without Stage 2 ever seeing them. This argues for a language gate in the
  router (escalate non-English) or multilingual Stage 1 training data. It is not fixed here.
- **benign-hard FPR fails the RFC gate** (28.3% vs a 5% Wilson upper bound). Instruction-like benign mail
  is exactly what TF-IDF confuses with injections; most of those records land in the gray zone, so the
  cascade decision depends on Stage 2 for them.
- **RFC deviation (needs owner sign-off)**: the RFC lists benign-hard as "Gate script only", but template 0
  (92 records) is in val and is used, with the rest of val, for threshold and gray-zone selection. With
  benign-hard-v2 the frozen slice no longer shares any sentence with it (section 4).
- The val threshold was tuned on in-distribution sources (spml-dominated); the frozen test's held-out
  sources show the shift (97.9% -> 83.5% recall).
- spml labels depend on a system prompt that we drop; some "injections" look benign out of context.
- Language tags come from langdetect and are noisy on short texts.
- No Stage 2 (gateway) result on the current frozen test is committed; see the changelog. The HTTP backend
  is tested against a mock transport, a local fake server, and the real serving app with a fake detector
  (`test_http_backend_against_real_gateway_app_contract`), including the timeout -> fail_closed path.

### How the HTTP (gateway) backend counts

A response is a **model prediction** only if every active `DetectorResult` has `status == "ok"` and
`policy_applied` is null; then `predicted = decision == "flag"`. A response decided by a failure policy
(timeout, error or skipped -> fail_open / fail_closed) is excluded from precision, recall and FPR and
reported per slice under `policy_decisions` (by status, by policy_applied, by decision and label).
Transport errors (after 2 retries by default), non-200 responses and malformed responses go to `errors`, and
the run continues. Each slice reports `truncated` (responses with any unscored content); truncated model
predictions stay in the metrics, since they are the decision the gateway made. With `--out`, partial results
are written every 500 items and at the end, with `complete` and `aborted` fields.

## Changelog

- **2026-10-29, splits-v2 / benign-hard-v2 / stage1-tfidf-v2** (review round 1):
  - Benign-hard sentence pools split by split. Under splits-v1, 56 distinct sentences (828 of 3,495
    sentence occurrences longer than 30 characters in frozen benign-hard records, counted by
    `benign_hard.sentences`) also occurred in val benign-hard records, and 271 skeletons were shared once
    slot values were masked. The review counted 711 / 3,293 with a slightly different sentence splitter.
    Now 0 and 0, and `check_leakage` enforces it. The splits were rebuilt: frozen test 6,274 -> 6,276
    records, sha256 `6e98eff1...` -> `2a4e011f...`; train unchanged; val changed in its 92 benign-hard records.
  - `stage1-tfidf-v1` (never released) and its results `training/results/stage1-tfidf-v1_{val,frozen_test}.json`
    were **deleted**: they were trained and evaluated on splits-v1. `stage1-tfidf-v2` was trained on the
    rebuilt train/val with the same procedure and published as a new immutable directory.
  - `training/results/stage2-deberta-gateway_frozen_test.json` was **deleted**: it refers to the
    splits-v1 frozen test, and it was produced by the HTTP backend before the fix that stops counting
    failure-policy decisions (gateway timeouts under fail_closed) as model predictions. The owner will
    re-run the Stage 2 gateway evaluation on splits-v2.
  - Language figures now come from `training.language_mix` (no sampling); the previous "1,000-row sample",
    "500-row sample" and "31% en" (raw yanis, but described as post-dedup) figures were not reproducible.
  - The parity numbers (0.031, 0.0035, 24 n-grams, distance to threshold) are now computed and stored by
    `train_stage1` and stored in `evaluation.json`. The v1 figures "1.4e-7" (distance of the flipped val
    score from the threshold), "0.031" and "0.0069" were not stored by any committed code; they are retired
    and replaced by the v2 values. The re-measured sublinear_tf value is again 0.031; the pruned-bigram value
    is 0.0035 on v2 (0.0069 was reported for v1).
  - `data/splits/NOTICE.md` reproduces the upstream yanis NOTICE verbatim.
