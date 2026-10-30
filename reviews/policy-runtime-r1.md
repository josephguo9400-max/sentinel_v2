# Review: milestone/policy-runtime (r1)

2026-10-30 · Reviewer: independent review agent · Head `bc49454` vs base `36ce683` (milestone/cascade-and-shadow) ·
Branch `verify/policy-runtime` · Design: `docs/design/capability-policy.md` (D25)

Probes written for this review: `reviews/policy-runtime-r1/test_reviewer_probes.py` (fakes only, 5 tests, all pass,
i.e. every finding they cover is reproduced) and `reviews/policy-runtime-r1/t3_subset_nocache.py` (a no-cache re-run
of T3 on a subset with real models; output `reviews/policy-runtime-r1/t3_subset_nocache.json`). Nothing was run on
`data/splits/frozen_test.jsonl`. No product code was changed.

## Verdict

**Mergeable after the two Major findings are fixed.** The refactor is careful and the equivalence evidence is real:
I found no decision difference between the policy engine and the frozen legacy service on the served D23 path, and my
own uncached re-run of T3 on 193 items (`T3 re-run` below) reproduced the committed report. But T3 as committed
cannot be re-run at HEAD (M1), and one engine path (parallel stage with a FAIL_OPEN member) allows content
that would have been flagged had the failing member succeeded (M2; not served, but it is in a contract-level engine
that the design calls tested).

**T3 is a valid merge condition for what it claims, and narrower than the docs say.** It proves the served D23 bundle
(no gate, single DeBERTa, coverage flag, observe Stage 1) decides identically to the legacy service with timeouts
disabled. It cannot speak to failure semantics, the gate/band/language logic, FALLBACK, deep scan or non-flag
coverage modes: two of its five compared decision fields are constant by construction (`stages_run` is `["stage2"]`
and `policy_applied` is `null` on all 2,000 items), `decided_by`/`coverage_policy_applied` vary on only 6 items, and
DeBERTa scores are memoized across both engines, so "identical scores" is true by construction. Those other paths are
covered, by T4 (gate, band, language, chunk selection, real Stage 1) and by fakes (T2, T6, T7), not by T3.

## Test results (HEAD `bc49454`, each directory under `flock /home/claude/measure.lock`)

| Directory | Result |
| --- | --- |
| `tests/contract` | 170 passed |
| `tests/serving` | 492 passed, 1 skipped (`test_quantize_script.py:57`, int8 weights not generated) |
| `tests/training` | 148 passed, 2 skipped |
| `tests/redteam` | 416 passed, 4 skipped (`test_firewall.py:122`, by design) |
| `tests/benchmarks` | 27 passed |
| `reviews/policy-runtime-r1` (reviewer probes) | 5 passed (each reproduces a finding) |

Worktree setup: `models` symlink as in the main checkout and hard links of
`artifacts/baseline/deberta-pi-v2/onnx/{model.onnx,tokenizer.json,config.json}` (not committed).

## Findings

### Major

**M1. T3, the merge condition, cannot be reproduced at HEAD.**
`training/replay_policy_equivalence.py:65-66` (`registry_state()` reads `artifacts/registry.json`), `:283-286`,
`:320` (report hard-codes `"artifacts/registry.json (sentinel-registry-v1)"`), `reports/policy-runtime/run_equivalence.sh:12-16`.
Scenario: commit `764e160` switched `artifacts/registry.json` to `sentinel-registry-v2` after T3 was run (`9ac0e92`).
`load_specs()` filters the v2 dict down to `{}` and `build_engines()` raises `KeyError: 'stage2'` before loading any
model. So `bash reports/policy-runtime/run_equivalence.sh` fails at its first step, and the committed report names an
input file whose content is no longer what it describes. The project rule is that every number comes from a
committed script; at HEAD the script that produced the merge-condition number does not run. The same report also
records `cache_path` as another session's scratch directory and its per-item progress (from several resumed
invocations) is gitignored, so nothing records which code version produced which items (see M1 note under T3).
Verified: ran `python -m training.replay_policy_equivalence t3 --limit 1 --no-cache` at HEAD (KeyError), and probe
`test_r4_t3_reads_artifacts_registry_which_is_now_v2`. Fix: read `reports/policy-runtime/registry_v1_d23.json`
(the file `serving/convert_registry.py` already uses since `764e160`), record the git commit and
`serving/engine_version.json` hashes in the report, and drop the scratch path.

**M2. Parallel decide: a FAIL_OPEN member failure skips the coverage rule.**
`serving/service.py:978-988` (`if not model_backed: ... return` before the coverage block at `:994-1001`).
Scenario: shape B, `any_flag`, coverage `flag`, a text with more chunks than `max_chunks`, both members scoring
below threshold. Both members healthy: unscored content exists, coverage flags (`decision=flag`,
`coverage_policy_applied=flag`). Member A raises (or times out) with `on_failure: FAIL_OPEN`: A's vote is the same
"allow", member B still scores exactly the same chunks, the unscored content is identical, yet the engine returns
`allow` with `policy_applied=fail_open` and `unscored_by_any_step=True`. A failing detector produces a more permissive
decision than the same detector succeeding. This contradicts the property the design keeps ("never allow with
unscored content unless the policy says allow", §4) and §11a's own claim that coverage in a parallel stage is about
content scored by no step. In the single-member case FAIL_OPEN allowing is the legacy semantics (no member scored
anything), but here a member did score and the coverage question is unchanged by A's failure. T6/T7 miss it because
every failure cell uses a one-chunk text (`BENIGN = "w1 w2 w3"`). Shape B is not served (D25.5), which is why this
is not a release blocker for D23, but it is a semantic bug in an engine path the milestone presents as specified and
tested. Verified: probe `test_r1_parallel_fail_open_member_turns_a_coverage_flag_into_an_allow`.
Fix direction (owner decision): apply coverage whenever the aggregated vote is allow and content is unscored,
regardless of which member's vote was a policy vote.

### Minor

**m1. A gate band without `scope_languages` silently drops language gating that the capability declares.**
`contracts/policy.py:399-407` (`_check_band`), `serving/service.py:353`, `contracts/policy.py:410-420` (`lint_policy`).
Scenario: the TF-IDF capability declares `validated_languages: ["en"]`; a hand-written route policy omits
`band.scope_languages` (and `language_detector`). `validate_policy` accepts it (the subset rule applies only "when
both are set"), `lint_policy` says nothing, and the engine clears a French chunk and finalizes `allow` at the gate.
The legacy gateway always gated on the manifest's `input_languages`, and D23's evidence is precisely that Stage 1
fails on non-English and indirect text. The converter writes the scope, so no committed policy is affected. Verified:
probe `test_r2_band_without_scope_clears_any_language_although_the_capability_declares_en`. Suggest: a validator
or at least a lint warning when a finalizing band has no scope but the capability declares languages.

**m2. `gate.py promote` of the bundle that is already active overwrites `previous` with itself.**
`training/gate.py:350-363`. Scenario: registry `active=A, previous=B`; the owner promotes the committed report
`gate_val_d23.json` (bound to A, the active bundle) with `--allow-not-pass`. Result: `active=A, previous=A`,
`status=approved`, B is lost, the audit entry has `bundle_sha256 == previous_bundle_sha256`, and a later rollback
is a no-op flip. Verified: probe `test_r3_promoting_the_already_active_bundle_overwrites_previous_with_itself`.
Suggest refusing a promotion whose candidate is already active (or keeping `previous` unchanged).

**m3. `promote` does not check that the report is complete, nor which split it used.**
`training/gate.py:353-362`. A report with `run.complete: false` (a `--max-seconds` run that stopped early) and a PASS
verdict is promoted without `--allow-not-pass`; a val report is accepted like a frozen-test one. Today PASS is
unreachable (p99 and ASR are always `NOT_EVALUATED`), so this is latent. Verified: probe
`test_r3b_promote_accepts_an_incomplete_report`.

**m4. A frozen-test run that crashes is not audited.**
`training/gate.py:418-435`: the `gate_on_frozen_test` audit entry is appended after `run_bundle` returns. An
exception mid-run (model load failure, OOM on this 2-core VM, Ctrl-C) leaves items of the frozen test read and
partially recorded in the progress file with no audit line, contradicting "each run of the gate on the frozen test
is recorded in the audit log" (`training/gate.py:37-38`, RFC gate rules). Code reading (not run: no frozen-test
access). Suggest writing an `intent` audit entry before reading the split.

**m5. Docs overstate T3 and understate the val benign-hard result.**
- `README.md:31` (and the `9ac0e92` commit message, "DeBERTa rows and active scores") say T3 found identical
  "DeBERTa rows and scores". The scores are identical
  by construction (one memoized score per row content, shared by both engines;
  `training/replay_policy_equivalence.py:30-33, 131-141`); the README does not say so. The 5 compared fields are
  listed as if all were exercised; see the T3 section for which are constant.
- `serving/README.md:765` calls the val verdict `INSUFFICIENT_EVIDENCE` "for two structural reasons rather than a
  defect". Benign-hard FPR on val is 33/92 = 35.9%, Wilson 95% [26.8%, 46.1%]; even if 8 more benign-hard samples
  were all allowed, 33/100 has a Wilson lower bound of 24.6%, so the 5% check is certain to FAIL once n reaches
  100. The gate rule is followed correctly (n < 100 → insufficient), but describing the outcome as "not a defect"
  hides that the served bundle fails this check on every plausible completion of the evidence. Numbers recomputed
  from `reports/policy-runtime/gate_val_d23.json` `slices.benign_hard`.

**m6. Milestone numbering is inconsistent.** `README.md:5,28` call policy-runtime "Milestone 4"; `docs/RFC.md`
lists it as row 3.5 and keeps 4 = adaptive-eval.

**m7. Contract ownership and snapshot.** `docs/contract_requests.md:3` ("Agents never edit `contracts/` directly")
and design §7 (`docs/design/capability-policy.md:414`: "schema_snapshot.json is regenerated by the owner"). The
implementer wrote `contracts/capability.py`, `contracts/policy.py`, edited `classify.py`/`detector.py` and regenerated
`tests/contract/schema_snapshot.json` itself. D25 approved the design, so the content is sanctioned, but the owner
should explicitly sign off the snapshot diff (ClassifyResponse and DetectorResult changed; CapabilityManifest and
ExecutionPolicy new; ArtifactManifest, ClassifyRequest, TraceEvent unchanged, checked by diffing the snapshot).

**m8. v1.3 is not readable by a v1.2 reader.** `contracts/classify.py` and `contracts/detector.py` keep
`extra="forbid"`; the gateway now writes `schema_version: "1.3"` on every `DetectorResult` and adds five response
fields. Any client pinned to the 1.2 contracts rejects every response, including D23 responses that contain no
observe result. Every in-repo consumer moves with the shared `contracts/` package, so nothing here breaks; the
RFC/README should say plainly that 1.3 is a breaking change for out-of-tree 1.2 readers (there is no response-level
version to negotiate on).

**m9. T4 does not bind its run to the committed cascade policy.** `training/replay_policy_equivalence.py:398-404,
513-515`. Unlike T3, T4 builds its policy in memory (timeouts 600 s, default rationale) and records the committed
path without checking they differ only in budgets. Its bundle hash `5bbccdf0…` is not the committed
`cascade-tfidf-allow-only` bundle `3385ef14…` (computed with `serving.policy.bundle_sha256`). Low risk (the converter
golden test T1 pins the committed file), but the report should carry the same `timeout_only_diff` T3 has.

**m10. Two FALLBACK budgets with different clocks, undocumented.** `serving/service.py:764` (gate FALLBACK: decide
deadline `t0 + fallback.timeout_ms`, i.e. from request start, as legacy) vs `:820` (decide-member FALLBACK: a fresh
`fallback.timeout_ms` from when the member failed). Design §4 says "run d@v within its own timeout_ms" for both.
The gate variant is what keeps T3/T4 faithful, so it is the right behaviour, but §11a should list it. Also in a
parallel stage after a gate failure (`:954-955`) every failed member votes the gate's terminal and its own
`on_failure`/FALLBACK is ignored; and after a decide-member FALLBACK, `coverage: deep_scan` jumps straight to
`deep_scan.fallback` (`:855-856`). Both are code comments only.

### Nit

- **n1.** `serving/legacy_service.py:1-2` says "byte-identical to serving/service.py at 36ce683"; it is identical
  except for 11 prepended docstring lines (`diff <(git show 36ce683:serving/service.py) serving/legacy_service.py`).
  Its imports (`structure`, `detector`, `batcher`, `language`, `shadow`, `metrics`, `config`) are unchanged in this
  milestone, so behaviour is frozen as claimed.
- **n2.** `docs/contract_requests.md:25`: the `escalation_reason` row's decision text was pasted into the "Proposed
  change" cell (cutting `detected_language: str \| None`), and the Decision column is empty.
- **n3.** `contracts/policy.py:13` says the differences are "recorded in the design doc §12"; they are in §11a.
- **n4.** The committed D23 policy's `rationale` (hashed into `bundle_sha256`) says "Converted from
  artifacts/registry.json (sentinel-registry-v1)", which is no longer true of that file. Changing it would change
  the bundle hash, so leave it, but `serving/convert_registry.py:68` still emits the same text for future
  conversions.
- **n5.** `training/gate.py:227`: the benign-hard `why` string always says "<= 5%" even when the verdict is FAIL.
- **n6.** `tests/training/test_gate.py:163-167`: the "gate is the only new reader of the frozen test" guard skips
  silently when `origin/milestone/cascade-and-shadow` is absent (e.g. a fresh CI clone), and only checks added
  files, not modified ones (I checked modified files by hand: only the pre-existing `training/eval_harness.py`
  `--split` choice mentions it).
- **n7.** `redteam/probe_runner.py:80`: `stage_scores` is keyed by stage, so two active `stage2` results (shape B,
  or a decide member plus its fallback) overwrite each other. Not reachable with any committed policy.
- **n8.** `training/replay_policy_equivalence.py:23-25` docstring says the two policies differ in "three timeout
  fields"; the report's expected set and the actual diff have two.
- **n9.** `band.scope_languages` is an ordered list in the hashed policy: `["en","de"]` and `["de","en"]` behave
  the same and hash differently (`finalize` is sorted on serialization, scope is not).

## Focus 1: is the T3 equivalence claim real?

What is compared (`item_view`, `differences`, `replay_policy_equivalence.py:203-241`): `decision`, `decided_by`,
`policy_applied`, `coverage_policy_applied`, `stages_run`, `path`, `n_chunks`, the observe outcome, every active and
observe result's status/score/n_chunks/truncated/failure_reason, and the exact DeBERTa row token ids in order.
New 1.3 fields (`unscored_by_any_step`, `escalation_reason`) are not compared (the legacy service has none).

Distribution in the committed report (2,000 items): `stages_run` `["stage2"]` 2,000/2,000; `policy_applied` `null`
2,000/2,000; `decided_by` `deberta-pi-v2` 1,994 / `coverage_policy:flag` 6; `coverage_policy_applied` `flag` 6;
`decision` allow 1,247 / flag 753; rows per item 1 (1,852), 2 (53), 3 (24), 4 (71). So:

- **Vacuous by construction:** `stages_run` and `policy_applied`. With no gate step and timeouts at 600 s, nothing
  else is possible. Not wrong, but "identical on 5 decision fields" is really "identical decision and chunk
  selection on the D23 path, plus 6 coverage cases".
- **Memoization does not hide an engine difference.** The cache key is the full row token-id list, and the row lists
  are compared exactly before scores matter, so a different chunk selection or row construction shows up as
  `stage2 rows differ`. What memoization hides is ONNX nondeterminism across batch composition and thread
  scheduling (the Recorder also scores one row per call, so batching is bypassed in both engines). That is a
  property of the runtime, not of the engine, so it is acceptable for a refactor test; but "scores identical" is a
  tautology and should not be reported as evidence. My uncached re-run (below) addresses it.
- **Raised timeouts are benign for equivalence.** Both sides use the same no-timeout `GatewayConfig`; the policy
  side is converted from that config, and the report proves the committed bundle differs only in
  `decide.members[0].timeout_ms` (5,000 → 600,000) and `observe[0].timeout_ms` (50 → 600,000). The served budgets
  therefore never decide anything in T3, which is what design §8 asks; T5 covers the live budgets (4 Stage 2
  timeouts listed, not compared).
- **The policy side is the converted policy, not the committed file loaded through registry v2.** The capability
  pins in the diffed policy match the committed ones, and the live smoke + gate dry run exercised the v2 loader, so
  this gap is closed elsewhere.
- **Provenance gap.** The run was resumed in chunks (the final invocation did 232 of 2,000 items); per-item records
  carry no code version. `serving/` did not change between `debbe1a` and `9ac0e92`, but the progress file could
  have been started before `debbe1a` was committed. Addressed by the re-run below.

**T3 re-run (reviewer, no cache, code at HEAD).** `reviews/policy-runtime-r1/t3_subset_nocache.py` re-points
`registry_state()` at `reports/policy-runtime/registry_v1_d23.json` (working around M1), builds both engines with
the committed `build_engines`, scores every row separately in each engine (no shared cache), and compares with the
committed `differences()` plus the committed per-item record. Subset: the first 100 items by id, the 6
coverage-decided items, and every item with ≥ 3 DeBERTa rows. Result: **193 items, 459 DeBERTa rows per engine, 0 items with an engine difference, 0 items differing from the committed report** (decision, decided_by, stages_run, policy_applied, coverage_policy_applied, path, observe outcome, row count), maximum active-score gap between the engines 0.0 (each engine scored its own rows, one row per call, so the runtime was deterministic here), 343 s wall. The committed T3 result holds for the code at HEAD on the items where chunk selection and coverage matter most; I did not re-run the other ~1,800 single-row items.

## Focus 2: engine vs legacy semantics

Read `serving/service.py` against `serving/legacy_service.py` line by line for the decision path. Equivalent for every
legacy-expressible configuration: prepare/deadline from `t0`, gate wait `min(gate budget, remaining)`, anytime soft
deadline with margin capped at half the budget, cleared = scored ∧ `< clear_below` ∧ language in scope, allow before
flag, `stage2_order`, chunk budget, deep scan, coverage tags and `decided_by` strings, admission shedding (legacy
"stage1 failure → Stage 2" = gate FALLBACK to the decider with terminal = the decider's failure policy; overloaded
never falls back; observe skipped when the decider shed load). New-only paths: M2, m1, m10. The language rule change
for undeclared languages is disclosed and is a refusal at load, not a silent behaviour change.

## Focus 3: bundle hash, registry v2, env guard, engine version

- `bundle_sha256` = canonical JSON (sorted keys, compact, `allow_nan=False`) of `{policy, sorted unique
  capability_sha256, engine_version}`; `finalize` (a set) is serialized sorted, so the hash is stable across
  processes. Recomputed the committed bundle: `17fac609…` matches the T3 report, `/health` smoke and gate report.
- v2 load is all-or-nothing for the active policy (any `RegistryError`/`PolicyError` → no service), path checks via
  `resolve_inside`, `capability_sha256` pin checked on load, every file pinned by sha256. `previous` is loaded and
  checked; a broken previous degrades `/health` but does not stop serving. Rollback is a file edit + reload, not an
  in-memory flip, which is fine.
- Env guard: `DECISION_FIELDS` covers every `GatewayConfig` field the design names; the four "injected only" fields
  (`threshold`, `timeout_ms`, `failure_policy`, `max_len`) are not read on the v2 path. The guard runs in
  `create_app` and again in `build_from_registry`, so a reload from v1 to v2 under a decision env var is refused.
  Smoke evidence `reports/policy-runtime/smoke_env_guard.txt`.
- Engine version: `serving/engine_version.json` hashes 12 decision-path files and the test fails on drift; it
  covers what matters (`shadow.py`, `config.py`, `convert_registry.py` are outside it; each either does not decide
  or changes the policy content, and thus the hash, anyway).

## Focus 4: gate.py

Evidence rules match `training/eval_harness.py` (≥ 200 positives for recall, benign-hard n ≥ 100 and Wilson upper ≤
5%, Wilson CIs from `summarize`); paired recall uses Newcombe with the correct argument order (candidate first).
Verdict logic: any FAIL → FAIL; any INSUFFICIENT or NOT_EVALUATED → INSUFFICIENT_EVIDENCE; PASS is therefore
unreachable in this milestone (disclosed in the RFC). The report is bound to `bundle_sha256` and promote refuses a
report bound to another bundle; `--approved-by` is required and an audit line is appended; promote never runs from
evaluate. Frozen test: refused without `--frozen-test-approved-by`; no committed report, progress file or audit log
used it (checked `reports/policy-runtime/*`, no `artifacts/audit_log.jsonl` in any commit). Issues: m2, m3, m4, n5.

## Focus 5: contracts and consumers

Snapshot diff: ClassifyResponse and DetectorResult changed, two new contracts, the rest unchanged. 1.3 readers accept
1.1-1.2 documents; 1.2 readers do not accept 1.3 (m8). All six consumers count only `role == "active"` and treat
`observe` and `shadow` alike (`redteam/classifier.py:60,66`, `redteam/judge.py:240`, `redteam/sandbox.py:204`,
`redteam/probe_runner.py:80`, `training/eval_harness.py:133-136`, `training/replay_runner.py:94-97`), tested by
`tests/redteam/test_observe_role_v13.py`.

## Focus 6: numbers in docs

Spot-checked against committed outputs: T3 2,000/0 differences; T4 201/306 and 191/1,185, 1,198 escalated; T5
1,970/1,974 compared identical, 4 failed-active + 26 transport errors not compared; gate val recall 2,164/2,240 =
96.6% [95.8, 97.3], FPR 88/974 = 9.0% [7.4, 11.0], benign-hard 33/92, 1 coverage decision, 0 failures; bundle
`17fac609…`; T6 "36 cells" (the MATRIX has 36 entries); T2 "76 cases" (76 collected). All match. Wording issues: m5,
m6.

## The implementer's deviations (design §11a and commit messages)

| # | Deviation | Justified? | Disclosed? |
| --- | --- | --- | --- |
| 1 | `ObserveStep` gains `max_chunks`, `deadline_margin_ms`, what-if `band` | Yes: needed for a faithful D23 observe step (T3 compares observe outcomes) | Yes (§11a, `contracts/policy.py`) |
| 2 | `subwindow_stride` None iff off; `Calibration.calibrated_with`/`notes`; `tokenizer_file` | Yes (one behaviour, one hash; D25.6) | Yes |
| 3 | Extra validators (band allow/clear_below 0, equal parallel `max_chunks`, member FALLBACK outside decide, deep_scan single, consistent pins) | Yes | Yes |
| 4 | `pack_overlap_tokens` recorded, not read | Yes (it was dead config before) | Yes |
| 5 | Parallel coverage: a chunk one member scored is covered | Yes, but incomplete: M2 | Yes |
| 6 | Undeclared-language gate cannot clear without a scope | Yes (design's own rule) | Yes; the converse gap is m1 |
| 7 | `compute_saved` NOT_EVALUATED for a policy without a gate; p99/ASR NOT_EVALUATED | Reasonable (nothing to measure); together they make PASS unreachable this milestone | Yes (RFC gate rules) |
| 8 | Registry v2 `status` field, `previous: null`, legacy constructor kept via `legacy_to_v2` | Yes | Yes (registry comment, service docstring) |
| - | Gate vs member FALLBACK clocks; deep scan skipped after a member fallback; gate terminal overrides parallel members' `on_failure` | Defensible | **No** (m10) |
