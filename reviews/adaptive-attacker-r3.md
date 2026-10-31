# Review r3: feat/adaptive-attacker, defensive cleanup (362ee18..c540287)

Reviewer: an independent verifier who did not write these commits. The scope is `git diff 362ee18 c540287`,
four commits:

- f459b91: ledger
- 7fe7c0f: do_not_train tests
- 6eac2fe: mutation backfill and gate label
- c540287: docs

I read the diff against r2 (`verify/adaptive-attacker-r2` a10160f, `reviews/adaptive-attacker-r2.md`).

How the review was run:
- Worktree `/home/claude/wt-verify-att3`, branch `verify/adaptive-attacker-r3`.
- No code was changed.
- **No API call was made**, and no gateway or model scoring was run.
- pytest ran under `flock /home/claude/measure.lock`, one directory at a time.

**Disclosure.** One repo-wide `git grep -i concentrat`, run to find support for a doc claim, printed three
matching lines from `data/splits/frozen_test.jsonl`. I did not otherwise open, use or modify that file. I
made no use of its content, and none of it appears here.

## Verdict

**Mergeable as a measurement-only cleanup. There are no Major findings.**

What holds:
- **Ledger.** The ledger arithmetic reproduces from the committed token counts.
- **Billable paths.** Every billable path in `redteam/adaptive/` is preflighted per call and recorded at
  split input/output rates.
- **Ledger writes.** The ledger is written before the first call and after every call.
- **do_not_train tests.** They fail when either binding is removed, so they are not vacuous.
- **Backfilled mutations.** They match the raw logs row for row. No body or address text was added.
- **Gate.** The gate block is still `REPORT_ONLY` and carries the requested label.

What remains:
- Doc contradictions: three Minor findings and some Nits.
- A pilot gate input whose hard-coded scope caveat is wrong.

## Findings

### Major

None.

### Minor

**r3-m1. The pilot gate input carries a scope caveat that is false for the pilot.**
`redteam/adaptive/report.py:185-188`; `reports/adaptive-attacker/pilot/adaptive_asr_gate_input.json`.
- **What it says.** `interval_scope` is a fixed string: "conditional on these 2 eval goals, 8 carriers per
  scenario and one seed".
- **Why that is wrong for the pilot.** The pilot gate input has `n_goals: 1`, a dev goal, with budget 20
  and a per-core budget of 3.
- **Other caveats.** `metric_definition` hard-codes "7 of the 21 cores". That is correct for the cores of
  these runs, but it would go stale if the library changed.
- **Fix.** Derive both caveats from `report["n_goals"]` and the goal split.
- **Impact.** Report-only, so none on any gate.

**r3-m2. §8 and §14.3 disagree about whether a real per-core budget is "attacker strength" or
"accounting".** `docs/design/adaptive-attacker.md:286-289` vs `:308` and `:430-432`.
- **§8 (`:286-289`).** It lists "give each core a real budget, report within-unit queries" among the fixes
  it rejects because they "would each make the attacker better".
- **§8 (`:308`) and §14.3 (`:430-432`).** They call the same items "query accounting… This is accounting,
  not a stronger attacker", and put them up as an owner question.
- **Which reading is right.** Reporting within-unit queries is pure reporting. Raising the per-core budget
  does strengthen the search.
- **Fix.** Split the two: report within-unit queries now; treat the budget change as out of scope.

**r3-m3. Two pre-r2 sentences in §6 contradict the corrected paragraph under them.**
`docs/design/adaptive-attacker.md:142,149` vs `:154-155`.
- **What is left.** `:142` still calls GullibleDefender "the worst-case victim". `:149` says "The
  gullible victim bounds harm from above".
- **The contradiction.** Five lines later (`:154-155`), the doc says it "does not bound harm from above
  across the library".
- **README.** `redteam/README.md:218-224` was corrected ("not a worst-case victim").
- **Fix.** Delete the two stale sentences or reword them. A reader who stops at `:149` gets the claim r2
  withdrew.

### Nit

- **§2 table row (`docs/design/adaptive-attacker.md:43`).** It still calls score-feedback "an *upper
  bound* on a realistic attacker". With 4 candidates per core it bounds only the *observation*, not
  attack strength; `:181` says as much.
- **§3 (`:59-60`).** The sentence "never against the served gateway's responses" is unchanged. It is
  caveated only in §7.1 (`:278-281`). The c540287 subject "correct every claim review r2 found
  unsupported" overstates this one; fix it in place.
- **"Does not detect this forwarding-injection shape" (`:264-268`).** This is stronger than the evidence:
  - **Selected bodies.** The 19 swapped bodies were chosen *because* they were allowed.
  - **What the swap shows.** Recipient-insensitivity.
  - **Better support.** The 78% candidate-level allow rate supports the claim better.
  - **Missing citation.** The new "same failure mode as the frozen-test findings… where D23's false
    positives concentrate" cites nothing. It is supported by `README.md:32` (val benign-hard FPR 33/92;
    gateway-mvp frozen 60/276), so cite that.
- **The `noise` flag means "operator applied", not "zero-width space inserted".** `_apply_noise` inserts
  with p = 0.04 per space (`mutators.py:75-82`).
  - Of the 106 committed candidates with `noise: true` (pilot 6, decision 45, score 55), 43 raw bodies
    contain no U+200B (1 / 17 / 25).
  - Say this in the `mutation_profile.note` or in doc §5 (`:94,111`), so nobody reads `noise: true` as
    "has ZWSP".
- **Ledger rounding.** The ledger's `accounting_notes[1]` gives the upper end as **$0.057126**
  (token-derived 0.0571260). The doc (`:365`) and `total_usd` give **$0.057127**, the sum of the rounded
  per-entry figures. Both are fine; state one.
- **§13 (`:409-411`).** It says the tests cover "dedup-key variants in train and in val". Variants are
  tested in train only, and val is tested on the exact body only. `:418` says "A test records this
  limit", but `test_real_do_not_train_fixtures_are_discovered` records it only in a comment. It asserts
  that 9 probes are found, not that the cores are invisible.
- **Stale docstring (`training/common.py:150`, not in range).** It still lists "the adaptive core library"
  as a do_not_train artifact, which §13 now withdraws.
- **Preflight estimate (`live_victim.py:71`).** The estimate omits the API's own tool-use system-prompt
  overhead, and a call that raises after the server has billed is not recorded. Both are cents-level, and
  the doc (`:373-376`) already avoids calling the estimate strict.
- **Out of milestone scope (`redteam/llm_run.py:48`).** It makes un-ledgered `AnthropicLLMClient` calls.
  This is the pre-milestone defender runner, not the adaptive path, so it was not required here. Noting
  it so the ledger is not mistaken for repo-wide accounting.

## Checks

### (1) Ledger and spend

**Call paths.** The only billable call sites in `redteam/adaptive/` are:
- `build_carriers.py:59` and `refusal_probe.py:50`, both through `AnthropicAttackerClient.complete`;
- `live_victim.py`, through `MeteredLLMClient.create`.

`search.py` makes no model call.

**Per-path behaviour.**

| Path | Preflight | Record | Ledger write | Cap |
| --- | --- | --- | --- | --- |
| `AnthropicAttackerClient` | before `messages.create` (`llm.py:89`); cache hits are free | split in/out (`llm.py:98`) | `on_record` → `LedgerRun.update` → atomic flush | — |
| `MeteredLLMClient` | every call, from the request's own size (system + tool schemas + messages) plus `max_tokens` (`live_victim.py:70-73`) | `resp.usage.input_tokens` / `output_tokens` separately (`:74`) | `on_record` → `LedgerRun.update` → atomic flush | — |
| `begin` / `update` / `finish` | — | — | entry written before the first call and after each call (`ledger.py:159-215`) | — |
| all three entry points | — | — | — | `prior_usd=ledger.prior_total_usd`, so the cap is read across runs |

**Tests (`tests/redteam/test_adaptive.py:73-155`).** They cover:
- output priced at the output rate;
- an in-progress entry on disk after each record;
- the reload total counting toward the cap;
- per-call preflight blocking before the inner client is called;
- the committed ledger re-priced.

**Recomputed at $0.10/M in and $0.50/M out.** All requests are far below the 100K tier.

| Entry | Priced |
| --- | --- |
| pilot (1,485 / 3,455) | 0.001876 |
| first-version eval_decision (2,970 / 5,295) | 0.0029445 → 0.002945 |
| carriers (2,217 / 3,938) | 0.002191 |
| refusal probe (6,528 / 7,449) | 0.004377 |
| **Sum of the four** | **0.011389** |
| live victim, 91,475 combined: all at input rate | 0.009148 |
| live victim, 91,475 combined: all at output rate | 0.045738 |
| **Range** | **$0.020536–$0.057126/7** |

How the inputs were verified:
- **First-version entries.** Their token counts equal `meta.spend` in `git show
  dc0aa1e:reports/adaptive-attacker/{pilot,eval_decision}/summary.json`. eval_score at dc0aa1e shows
  0 calls and 6 cache hits.
- **Carriers + refusal probe.** Together they total 8,745 in / 11,387 out. That equals the sum over the
  24 local response-cache entries (`/home/claude/wt-attacker/redteam/adaptive/cache/`). I read token
  fields only; each entry has the keys `text`, `input_tokens` and `output_tokens`.
- **The ~$0.097 figure.** It is $0.057 plus "roughly $0.04" (dc0aa1e doc: ≈$0.045 total, of which
  $0.0048 was compose runs). It is correctly labelled unverified, and kept out of `total_usd`.

### (2) do_not_train tests

`tests/training/test_do_not_train.py` exercises the real functions:
- **Loader.** Row-level refusal, refusal of a whole-object marked artifact, and refusal of the committed
  `fixtures/probes/adaptive/adaptive_evasions.json`.
- **Leakage check.** It is run through `check()` and `main()` on synthetic splits under `tmp_path`. It
  fails on an exact body and on case, whitespace and punctuation dedup variants. Unmarked, blank and
  unparseable fixtures are ignored.

**Mutation check (scratch pytest plugin, no repo edit).**
- With `read_training_jsonl` replaced by a non-refusing loader, **3 tests fail**.
- With `check_do_not_train` stubbed to `([], 0)`, **4 tests fail**.

So the tests bind. `test_trainer_loads_splits_through_the_refusing_loader` is a source-text check. That
is weak but acceptable.

### (3) Backfilled mutation fields

I compared each committed `queries_meta[i]` with raw `queries.jsonl[i]` (implementer's
`/home/claude/wt-attacker/reports/adaptive-attacker/<run>/`). My script read bodies only to test for
U+200B and printed counts only.

| Run | Rows (summary / raw) | Mismatches on shared fields | Mismatches on `mutation` | Old fields changed vs 362ee18 |
| --- | --- | --- | --- | --- |
| pilot | 22 / 22 | 0 | 0 | 0 |
| eval_decision | 119 / 119 | 0 | 0 | 0 |
| eval_score | 109 / 109 | 0 | 0 | 0 |

- **`mutation_profile`.** I recomputed it from the committed rows, for all candidates, successes and
  non-successes, and it matches field for field.
- **Top-level changes.** The only top-level changes to the summaries are `goals[*].queries_meta` and the
  new `mutation_profile`.
- **Gate inputs.** They gained only the caveat and label keys, and no measured value changed.
- **No new body text.**
  - The added report lines contain no addresses and no body text.
  - The `mutation` values are table indices and flags.
  - The doc quotes short core phrases ("Kindly cc X", etc.) that were already in `cores.py` and r2.
- **Operator attribution.** "The operator is not recoverable" is true: the raw log has no operator field.

### (4) Docs

**Traced to committed output or r2 evidence:**
- per-core budget 4 (r2: 50/59/48/54 queries used);
- within-core attempts 13/5/1 and 18/2/1;
- seed spread 0.452/0.452/0.429;
- medians 26/23 → 36/35;
- 7 never-functional cores;
- 74/74, 65/74, 56/74, 18/74 and 0;
- functional subset 0.68 [0.43, 0.89] and 0.75 [0.54, 0.93], correctly attributed to the reviewer and
  not recomputed;
- 93/119 allowed;
- controls 0/24 with 0/8 Wilson 0.32;
- swap check 19/19;
- live 2/6 successes, 0/2 live harm, mode C uninformative;
- refusals 12/12;
- 18/33 `after_512` fallbacks, which the new profile corroborates (33 / 18 reached-false).

**Limitations stated:**

| Limitation | Where |
| --- | --- |
| lower bound | `:177-181` |
| one-shot search | `:102-109` |
| seed spread | `:241-247` |
| unmatched controls, plus the reviewer's matched check | `:249-268` |
| live victim 2/6 | `:315-335` |
| intervals over 2 goals / 8 carriers / one seed | `:270-281` |
| mode-B row conditional | `:197-200` |
| weak feedback test | `:116-123` |
| ZWSP latent | `:125-132` |

**Gate.**
- `adaptive_asr` verdict is `REPORT_ONLY` in all three gate inputs.
- `training/gate.py:275-299` is unchanged in range and still `NOT_EVALUATED`.
- `metric_label` = `"fixed-budget evasion rate (lower bound)"`.

`redteam/README.md:218-224,394-397` is accurate.

Exceptions: r3-m2, r3-m3 and the Nits.

### (5) Key hygiene

- `git log -p 362ee18..c540287 | grep sk-ant-`: **0**.
- `git log -p c540287` (whole branch history): **0**.
- `git log --all -p`: 2 matches, both prose sentences in the r1 and r2 reviews.
- `git grep sk-ant- c540287` and the working tree (excluding `models`/`data`): 0.
- The local response cache: 0.

`git check-ignore` covers:
- `.env` and `redteam/.env`;
- `.sentinel.env`;
- `redteam/adaptive/cache/`;
- `reports/adaptive-attacker/**/queries.jsonl`;
- `/models`;
- `artifacts/baseline/deberta-pi-v2/onnx/`.

The ledger's `_safe` now also recurses into nested dicts (`ledger.py:59-60`).

## Tests

| Directory | Result |
| --- | --- |
| `tests/redteam` | 446 passed, 4 skipped (r2: 441; +5 new ledger tests) |
| `tests/training` | 182 passed, 2 skipped (r2: 170; +12 do_not_train tests) |

Setup: `models` symlink → `/home/claude/sentinel-day0/models`; DeBERTa ONNX hardlinked from
`/home/claude/wt-pr-fix2/artifacts/baseline/deberta-pi-v2/onnx/`. Both are gitignored and not
committed.

## Merge recommendation

**Merge.** No attack capability was expanded, and this range changes no measured value. r3-m1 to r3-m3
are doc and caveat fixes that can land as a small follow-up commit before or after merge.
