# Design: Capability Manifests and Execution Policies

2026-10-30 · Status: proposal for the next milestone (adaptive-eval), not implemented · Base: `milestone/cascade-and-shadow` @ `36ce683`

## 1. Problem

Today one file mixes two different questions:

- **What a detector can do**: model bytes, runtime, tokenizer, calibrated threshold, input limits.
- **How the gateway uses detectors**: who decides, which scores escalate, what happens on failure, what runs
  observe-only.

The coupling, with the files it lives in:

| Today | Where | What it actually is |
| --- | --- | --- |
| `gray_zone` | Stage 1 manifest (`contracts/manifest.py`, `artifacts/candidates/stage1-tfidf-v2.1/manifest.json`) | routing band (policy) |
| `cascade_role` | every manifest | position in a pipeline (policy) |
| `failure_policy` | every manifest | failure semantics (policy). The Stage 1 manifest's `fail_open` is never applied while a Stage 2 exists (`serving/service.py` step 3, `serving/README.md` "Request flow") |
| `stage1_mode: route\|observe` | `artifacts/registry.json` | who decides (policy) |
| `COVERAGE_POLICY`, `MAX_CHUNKS`, `STAGE1_FINALIZE_FLAG`, `STAGE1_SUBWINDOW_*`, `LANG_MIN_PROB`, `DEEP_SCAN_*` | environment (`serving/config.py`) | decision-affecting settings that are versioned nowhere |

Evidence that the coupling already hurts:

1. **A policy change was shipped as a model version.** `stage1-tfidf-v2` and `stage1-tfidf-v2.1` have the same
   model sha256 (`2a5df39a…`) and the same threshold (0.505068). Only `gray_zone` differs
   (`[0.282715, 0.764614]` → `[0.505068, 1.0]`); `training/publish_stage1.py` records "model.onnx is
   byte-identical".
2. **The coupling constrained the search.** The val sweep could not try `low` above the threshold, "because the
   manifest requires `low <= threshold`" (`training/CASCADE.md`, Setup). That rule is a manifest validator
   (`stage1 threshold must lie inside gray_zone`), not a property of the model.
3. **"Allow only" is encoded as `high = 1.0`** and depends on the router's comparison operator
   (`docs/contract_requests.md`, row 2026-10-30 / `stage1_finalizes`).
4. **Decision-affecting settings live outside the registry.** In the D23 replay, 5 of the 15 decisions that differ
   from offline DeBERTa-only came from the coverage policy and 3 from the failure policy
   (`reports/cascade-and-shadow/d23/replay_counterfactual.json`, `views.served.same_items.flips_vs_deberta_only_offline`);
   `coverage_flag` decided 18 of 1,998 requests (`reports/cascade-and-shadow/d23/metrics.txt`). `COVERAGE_POLICY` is an
   environment variable.
5. **Scoring rule and calibration have drifted apart.** Stage 1's threshold was chosen by max F1 on val
   (`training/train_stage1.py` `choose_threshold`) on "its own chunked max" (`training/CASCADE.md`); serving adds
   64/32-token sub-windows by default (`serving/config.py` `stage1_subwindow_tokens`). `training/` contains no
   sub-window code. The serving-faithful replay found 98 items where the serving Stage 1 scored at or above `low`
   while the offline score did not (`replay_counterfactual_served_faithful.json`, `offline_low_vs_gateway`). How a
   detector is applied to a chunk is part of what its threshold means.

## 2. Principle: a policy change is a deployment

Same model bytes, different policy: on the replay set, letting TF-IDF Stage 1 finalize allows changed recall from
247/306 to 201/306, a paired **−15.0 pt [−19.4, −10.7]**, with no FPR benefit (−0.25 pt [−0.91, +0.40])
(`reports/cascade-and-shadow/replay_counterfactual_served_faithful.json`, `paired.chunk_level_minus_deberta_recall`
/ `_fpr`). No model changed. So:

- The **release gate's unit is the bundle**: one execution policy plus the exact capability versions it pins, plus
  the engine version that interprets it. The gate evaluates it **as served** (`training/eval_harness.py`
  `as_served`, `gate_view: "as_served"`), never a detector alone.
- A policy change goes through the same lifecycle as a model change (RFC: CANDIDATE → SHADOW → GATED → ACTIVE,
  plus ROLLBACK): observe/shadow evidence, gate verdict (PASS / FAIL / `INSUFFICIENT_EVIDENCE`), owner approval,
  audit record, rollback by pointer flip. No auto-promotion.
- A gate verdict is bound to a `bundle_sha256`. Any change to the policy, a pinned capability, or the engine
  version produces a new hash, and the verdict does not carry over.
- Decision-affecting settings must be in the policy. Under the new registry an environment variable that would
  change a decision is a startup error (§7).

The existing gate code is `training/eval_harness.py` (`evidence()`: recall needs ≥ 200 positives, benign-hard FPR
needs ≥ 100 samples and a Wilson upper bound ≤ 5%, otherwise `INSUFFICIENT_EVIDENCE`) plus
`training/cascade_sweep.py` `newcombe_paired` for paired differences. The full gate script is still to be built
(RFC milestone adaptive-eval); this design fixes its input type.

## 3. Capability Manifest (what a detector can do)

Immutable, keyed by `(detector_id, detector_version)`. A new threshold is a new version. Nothing in it names
another detector or a pipeline position.

```python
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

class InputContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text", "token"]                 # text: ONNX takes strings; token: ids
    preprocessing: Literal["tfidf-v1", "deberta-v3-window-v1"]  # closed set the gateway implements (row 16)
    tokenizer: str = Field(pattern=r"^(hf|regex):.+$")         # units of max_input_tokens / input_tokens
    max_input_tokens: int = Field(gt=0)
    validated_languages: list[str] | None          # ISO 639; "what it was validated on", not a routing rule

class ScoringRule(BaseModel):
    """How the model is applied to one chunk. Part of the calibration: the threshold is only valid with it."""
    model_config = ConfigDict(extra="forbid")
    aggregation: Literal["max"] = "max"            # over windows of a chunk and over chunks
    subwindow_tokens: int = Field(ge=0)            # 0 = off (TF-IDF anti-dilution; today a GatewayConfig env var)
    subwindow_stride: int = Field(ge=1)
    score_segment_pieces: bool                     # TF-IDF scores each packed segment piece too (service.py)

class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(ge=0.0, le=1.0)       # flag iff score >= threshold (DetectorResult rule)
    chosen_on: str                                 # dataset version, e.g. "splits-v2:val:d84853ecc84cc87e"
    rule: str                                      # e.g. "maximize F1 on val (pooled)"
    evidence_path: str | None                      # e.g. evaluation.json

class OperationalLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommended_timeout_ms: int = Field(gt=0)      # the researcher's budget; the policy sets the real deadline
    max_batch: int = Field(ge=1)
    latency_evidence_path: str | None              # e.g. serving/bench_results/stage1_latency.json

class CapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["2.0"] = "2.0"
    detector_id: str
    detector_version: str
    artifact_format: Literal["onnx"]
    files: dict[str, Sha256]                       # every file the detector loads, tokenizer included (row 13)
    model_file: str                                # a key of `files`
    runtime: Runtime                               # unchanged from contracts/manifest.py
    label_mapping: dict[Literal["0", "1"], Literal["benign", "injection"]]
    input: InputContract
    scoring: ScoringRule
    calibration: Calibration
    limits: OperationalLimits
    training_data_version: str | None
    created_at: str                                # RFC 3339 UTC
```

Removed relative to `ArtifactManifest` v1.2: `cascade_role`, `gray_zone`, `failure_policy`, `timeout_ms` (becomes
`limits.recommended_timeout_ms`). `serving/registry.py` `KNOWN_TOKENIZERS` is replaced by `files` (the tokenizer's
sha256 is pinned by the manifest, as row 13 requests). All load checks (`path_escape`, `sha256_mismatch`,
`runtime_mismatch`, …) stay as they are and apply to every entry of `files`.

`capability_sha256` = sha256 of the canonical JSON (sorted keys, no whitespace) of the manifest. Policies pin it.

## 4. Execution Policy (when and how detectors run and combine)

A fixed structure, not a graph: an optional **gate** step, one **decide** stage (one detector, or two in
parallel), any number of **observe** steps, an optional **shadow**, a **coverage** rule, an optional **canary**.

```python
class DetectorRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    detector_id: str
    detector_version: str
    capability_sha256: Sha256                      # pins the manifest, so an edited manifest cannot slip in

class Failure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["FAIL_OPEN", "FAIL_CLOSED", "FALLBACK", "ASYNC_ONLY"]
    fallback: "FallbackStep | None" = None         # required iff mode == FALLBACK

class FallbackStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    terminal: Literal["FAIL_OPEN", "FAIL_CLOSED"]  # depth 1: a fallback cannot fall back again

class Band(BaseModel):
    """Per-chunk routing on the gate detector's score. Independent of the gate detector's threshold."""
    model_config = ConfigDict(extra="forbid")
    clear_below: float = Field(ge=0.0, le=1.0)     # chunk cleared iff score <  clear_below (strict)
    flag_at_or_above: float | None = None          # set only if finalize includes "flag"; score >= value
    finalize: set[Literal["allow", "flag"]]        # replaces gray_zone=[low, 1.0] (row 2026-10-30 stage1_finalizes)
    scope_languages: list[str] | None              # chunks outside it are never cleared; must be a subset of
                                                   # the capability's validated_languages

class GateStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef
    band: Band
    timeout_ms: int = Field(gt=0)
    deadline_margin_ms: int = Field(ge=0)          # anytime soft deadline (today STAGE1_DEADLINE_MARGIN_MS)
    max_chunks: int = Field(ge=1)                  # today STAGE1_MAX_CHUNKS
    on_failure: Failure                            # FALLBACK -> must target the decide detector (= "escalate")

class DecideMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    max_chunks: int = Field(ge=1)                  # today MAX_CHUNKS (first, last, then uncleared by gate score)
    on_failure: Failure                            # FAIL_OPEN | FAIL_CLOSED | FALLBACK

class DecideStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["single", "parallel"]
    members: list[DecideMember]                    # 1 for single, exactly 2 for parallel
    aggregate: Literal["any_flag", "all_flag"] | None  # parallel only; decision-level, never score averaging

class ObserveStep(BaseModel):                      # today's stage1_mode=observe, generalized
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    concurrency: int = Field(ge=1)                 # today STAGE1_OBSERVE_CONCURRENCY; busy -> dropped
    on_failure: Failure                            # must be ASYNC_ONLY

class ShadowStep(BaseModel):                       # today's async mirror (serving/shadow.py)
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef
    timeout_ms: int = Field(gt=0)
    queue: int = Field(ge=1)
    max_chunks: int = Field(ge=1)
    intra_op_threads: int = Field(ge=0)
    on_failure: Failure                            # must be ASYNC_ONLY

class DeepScan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detector: DetectorRef                          # must be a decide member
    max_chunks: int = Field(ge=1)
    timeout_ms: int = Field(gt=0)
    fallback: Literal["flag", "quarantine", "allow"]

class Coverage(BaseModel):                         # content scored by no step (unchanged semantics)
    model_config = ConfigDict(extra="forbid")
    mode: Literal["flag", "quarantine", "allow", "deep_scan"]
    deep_scan: DeepScan | None = None              # required iff mode == deep_scan

class Chunking(BaseModel):                         # one chunk plan shared by every step (today GatewayConfig)
    model_config = ConfigDict(extra="forbid")
    plan_tokenizer: DetectorRef                    # today: Stage 2's tokenizer defines chunk units
    stride: int; structure_aware: bool; pack_overlap_tokens: int

class LanguageDetector(BaseModel):                 # resolves row 2026-10-30 "name the language identifier"
    model_config = ConfigDict(extra="forbid")
    impl: Literal["langdetect==1.0.9"]
    seed: int; min_letters: int; min_prob: float; max_chars: int
    tag_map: dict[str, str]                        # e.g. {"zh-cn": "zh"} -> ISO 639

class Canary(BaseModel):                           # designed now, implemented later (§6)
    model_config = ConfigDict(extra="forbid")
    candidate: "PolicyRef"
    share_bp: int = Field(ge=0, le=10_000)         # basis points of traffic
    salt: str                                      # changing it reshuffles assignment

class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    policy_id: str
    policy_version: str
    entrypoint: Literal["classify"]                # the only API it serves (POST /classify)
    engine_version: str                            # semantic version of the policy interpreter it was validated on
    chunking: Chunking
    language_detector: LanguageDetector | None     # required iff a gate band declares scope_languages
    gate: GateStep | None
    decide: DecideStage
    observe: list[ObserveStep] = []
    shadow: ShadowStep | None = None
    coverage: Coverage
    canary: Canary | None = None
    rationale: str                                 # links the decision record (e.g. "D23")
```

Validators (all cheap, all unit-testable):

- Every `DetectorRef` resolves to a loaded capability whose `capability_sha256` matches.
- `decide.mode == "single"` ⇔ 1 member and `aggregate is None`; `parallel` ⇔ 2 members and `aggregate` set.
- `ASYNC_ONLY` is required on observe/shadow steps and forbidden on gate/decide steps.
- Gate `FALLBACK` must target a decide member (this is today's "Stage 1 failure clears nothing, escalate").
- `band.flag_at_or_above` is set iff `"flag" in band.finalize`, and then `clear_below < flag_at_or_above`.
- `band.scope_languages ⊆ capability.input.validated_languages` (when both are set). A detector with no declared
  languages may not finalize unless the policy states a scope; this replaces `STAGE1_UNDECLARED_LANGUAGES`.
- The gate detector must be `kind="text"` and decide members `kind="token"` while the engine only implements those
  (today's `role_mismatch` rules in `serving/registry.py`).
- **No** validator ties `band.clear_below` to the gate detector's `calibration.threshold` (the coupling in §1, item
  2). The gate report prints both.

### Threshold split

| Value | Owner | Changes when | Versioned in |
| --- | --- | --- | --- |
| `calibration.threshold` ("is injection") | detector author | a new calibration, model or scoring rule | capability (new `detector_version`) |
| `band.clear_below`, `band.flag_at_or_above` | policy owner | routing trade-off | policy (new `policy_version`), pinned to exact detector versions |

A band is meaningless without the score distribution it was chosen on, so a band always references a pinned
`DetectorRef`; a new detector version forces a new policy version, which forces a new gate run.

### Failure semantics

| Mode | Decision effect | Applies to | Today |
| --- | --- | --- | --- |
| `FAIL_OPEN` | the step votes allow | gate (not useful: see below), decide | manifest `failure_policy: fail_open` |
| `FAIL_CLOSED` | the step votes flag | decide | manifest `failure_policy: fail_closed` (Stage 2) |
| `FALLBACK(d@v)` | run `d@v` within its own `timeout_ms`; if it also fails, `terminal` decides | gate (target = decide member), decide | gate: Stage 1 failure escalates (`service.py` step 3); decide: none |
| `ASYNC_ONLY` | never read by the decision | observe, shadow | `stage1_mode: observe`, shadow mirror |

Rules: failure means `status != ok` with `failure_reason ∈ {timeout, exception, overloaded}`
(`contracts/detector.py`). **`FALLBACK` is not attempted on `overloaded`**: running a second model under load
shedding adds load; the `terminal` mode applies directly. The gate keeps today's anytime behaviour: chunks it
scored before the soft deadline count, the rest are uncleared (not a failure). Coverage stays a separate,
policy-level rule because it is about unscored content after every step succeeded; it keeps its tested property
"never allow with unscored content unless the policy says allow"
(`test_property_never_allow_with_unscored_content_unless_policy_allow`).

Values considered and dropped: `ESCALATE` (identical to gate `FALLBACK` to the decide member, so the validator
expresses it instead of a new value); `RETRY` (an ONNX call cannot be cancelled and keeps its worker, see
`serving/README.md` "Timeouts"; a retry doubles the damage); chained fallbacks (unbounded latency, untestable
combinatorics). `FAIL_OPEN` on a gate is allowed by the schema but flagged by the policy linter: it turns a 50 ms
Stage 1 hiccup into an unscored allow, which is why the gateway ignores it today.

## 5. The two supported shapes, and today's deployments as policies

**Shape A, sequential:** `gate` (cheap) → band → `decide.single` (expensive).
**Shape B, gate + parallel:** `gate` (optional) → `decide.parallel` (two detectors on the same chunk selection,
concurrently) → `aggregate` (`any_flag` or `all_flag` over per-member votes, where a failed member votes by its
`on_failure`). Only decision-level aggregation: two detectors' scores are calibrated differently, so averaging them
has no meaning without a joint calibration, which is out of scope.

D23, the served deployment (shape A without a gate; DeBERTa decides, Stage 1 observe-only), as converted from
`artifacts/registry.json` and the committed manifests with the default `GatewayConfig`:

```yaml
policy_id: d23-deberta-decides
policy_version: "1"
entrypoint: classify
engine_version: "1.0.0"
rationale: "D23 (docs/RFC.md); registry change pending gate (serving/README.md 'Registry')"
chunking: {plan_tokenizer: deberta-pi-v2@2.0.0+90c9989b.onnx.fp32, stride: 384, structure_aware: true, pack_overlap_tokens: 126}
language_detector: null          # no gate band, so no language routing
gate: null
decide:
  mode: single
  members:
    - detector: deberta-pi-v2@2.0.0+90c9989b.onnx.fp32
      timeout_ms: 5000
      max_chunks: 4
      on_failure: {mode: FAIL_CLOSED}
observe:
  - detector: stage1-tfidf-lr@stage1-tfidf-v2.1
    timeout_ms: 50
    concurrency: 1
    on_failure: {mode: ASYNC_ONLY}
shadow: null
coverage: {mode: flag}
```

The rejected cascade (route mode, `reports/cascade-and-shadow/registry_integration_route.json`), shape A:

```yaml
policy_id: cascade-tfidf-allow-only
policy_version: "1"
rationale: "Rejected by D23: replay recall -15.0 pt [-19.4, -10.7] (replay_counterfactual_served_faithful.json)"
language_detector: {impl: "langdetect==1.0.9", seed: 0, min_letters: 20, min_prob: 0.80, max_chars: 2000, tag_map: {}}
gate:
  detector: stage1-tfidf-lr@stage1-tfidf-v2.1
  band: {clear_below: 0.505068, flag_at_or_above: null, finalize: [allow], scope_languages: [en]}
  timeout_ms: 50
  deadline_margin_ms: 15
  max_chunks: 512
  on_failure: {mode: FALLBACK, fallback: {detector: deberta-pi-v2@2.0.0+90c9989b.onnx.fp32, timeout_ms: 5000, terminal: FAIL_CLOSED}}
decide: {mode: single, members: [{detector: deberta-pi-v2@…fp32, timeout_ms: 5000, max_chunks: 4, on_failure: {mode: FAIL_CLOSED}}]}
coverage: {mode: flag}
# chunking as in D23
```

Note that the band no longer needs `high = 1.0`: `finalize: [allow]` says it, and the comparison operators are
fixed by the schema.

Shape B example (schema and engine tested with fakes; not deployable on the 2-core host, see §9):
`gate` as above, `decide: {mode: parallel, members: [deberta-pi-v2@…fp32, <second token model>], aggregate: any_flag}`.

**Out of scope:** a general DSL or workflow engine; arbitrary DAGs, loops or more than two parallel members;
nested or chained fallbacks; routing on caller metadata, tenant or platform; score-level ensembling; online or
automatic band tuning; more than one entrypoint (exchange-level classification is RFC P2); per-step hardware
placement; auto-promotion.

## 6. Canary (designed now, implemented later)

- Field: `ExecutionPolicy.canary = {candidate: PolicyRef, share_bp, salt}`. The candidate is a complete policy and
  may not itself have a canary (depth 1).
- Assignment: `int.from_bytes(sha256(salt + "\x00" + request_id)[:8], "big") % 10_000 < share_bp` → candidate.
  Deterministic across processes and restarts; the same `request_id` always lands in the same arm, so caller
  retries (the harness reuses `request_id`, `training/eval_harness.py`) stay in one arm. A request without
  `request_id` gets a generated uuid today (`service.py`), so it is assigned at random; callers that need
  stickiness must send one.
- The response says which arm decided (`policy_id`, `policy_version`, `bundle_sha256`, §7), metrics carry a
  `bundle` label, and the gate never pools arms: each arm is its own bundle with its own minimum-n checks, so a
  small canary returns `INSUFFICIENT_EVIDENCE` until it has enough labeled traffic.
- Automatic rollback (RFC adaptive-eval: error rate, p99, flag-rate deviation) sets `share_bp` to 0 by writing a new
  registry state, audit-logged; it never promotes. Thresholds are an owner decision (§10).
- Not implemented in this milestone: the engine rejects a policy with `canary` set until the feature lands, so the
  field cannot be half-supported.

## 7. Registry, bundle, contracts and migration

**Registry v2** (`artifacts/registry.json`, schema `sentinel-registry-v2`):

```json
{"schema": "sentinel-registry-v2",
 "active": "policies/d23-deberta-decides/1/policy.json",
 "previous": "policies/d23-deberta-decides/0/policy.json",
 "capabilities_root": "capabilities/"}
```

`previous` stays loaded so rollback is a pointer flip (RFC adaptive-eval). Capabilities live in immutable
directories `artifacts/capabilities/<detector_id>/<detector_version>/` (model files may stay where they are; `files`
paths are resolved inside `artifacts/` with the existing `resolve_inside` checks).

**Bundle hash:** `bundle_sha256 = sha256(canonical(policy) ‖ sorted(capability_sha256 …) ‖ engine_version)`. Model
and tokenizer bytes are covered through `files`. Promotion record: `{actor, time, bundle_sha256, gate_report_path,
previous_bundle_sha256}`, appended to an audit log.

**Decision-affecting environment variables** (`COVERAGE_POLICY`, `DEEP_SCAN_*`, `MAX_CHUNKS`, `STAGE1_*`, `LANG_*`,
`STRIDE`, `STRUCTURE_AWARE`, `PACK_OVERLAP_TOKENS`, `SHADOW_*` except log path): with a v2 registry, setting any of
them is a startup error naming the policy field to change. Capacity settings (`MAX_BATCH`, `WORKERS`, `MAX_QUEUE`,
`MAX_INFLIGHT*`, `TOKENIZE_WORKERS`, `INTRA_OP_THREADS`, `MAX_TEXT_CHARS`) stay in `GatewayConfig`; they change
decisions only through overload and timeouts, which the gate's error-rate and p99 checks measure under a fixed load
profile, and the gate report records them.

**Contracts:**

| Contract | Change | Version |
| --- | --- | --- |
| `CapabilityManifest` | new (§3) | 2.0 |
| `ExecutionPolicy` | new (§4) | 1.0 |
| `DetectorResult` | `role` gains `"observe"` (observe steps; `"shadow"` keeps meaning the async mirror only). `stage` is unchanged: the gate reports `stage1`, decide members `stage2` (parallel members are told apart by `detector_id`), to avoid churn in every consumer | 1.2 → 1.3 |
| `ClassifyResponse` | optional `policy_id`, `policy_version`, `bundle_sha256`; `unscored_by_any_step: bool`; `escalation_reason: Literal["band", "language", "coverage", "gate_failed"] \| None`; `coverage_policy_applied` unchanged | 1.2 → 1.3 |
| `ArtifactManifest` | read-only: still accepted (1.1, 1.2) through the converter, never written | frozen |

Readers accept 1.1–1.3; writers emit 1.3; `tests/contract/schema_snapshot.json` is regenerated by the owner.

**`docs/contract_requests.md` rows this resolves:** `aux_files` (→ `files`); documented `preprocessing_version`
values (→ closed `Literal`); `stage1_finalizes` (→ `band.finalize`, strict/inclusive operators fixed);
`escalation_reason` (resolved; `detected_language` stays a diagnostic, not on the wire); language identifier and tag
mapping (→ `LanguageDetector`); observe-only on the wire (→ `role="observe"`); `unscored_by_any_stage` (→
`unscored_by_any_step`). **Not resolved:** `decision: "quarantine"` (wire-visible decision change for every caller;
owner question, §10).

**Migration, in order (each step keeps the suite green):**

1. Add the contracts and a pure converter `legacy_to_v2(registry_v1, manifests, GatewayConfig) -> (policy,
   capabilities)`. The two committed Stage 1 manifests convert to two capabilities with identical content except
   version; their gray zones become the policies `cascade-tfidf-provisional` (v2's) and `cascade-tfidf-allow-only`
   (v2.1's).
2. Refactor, do not rewrite: `ClassifierService` takes an `ExecutionPolicy` plus loaded capabilities instead of
   `stage1/stage2/stage1_mode/config`. The decision path (`_classify_active`, `stage2_order`, `_stage1_pass`,
   coverage, deep scan) keeps its code; parameters come from the policy. Observe steps reuse `_observe_guarded`.
3. The gateway accepts registry v1 (converted in memory, `/health` shows `policy_source: "legacy_registry"` and the
   derived `bundle_sha256`) and v2. Reload stays all-or-nothing.
4. Commit `artifacts/policies/d23-deberta-decides/1/policy.json` and switch the registry to v2 only after the
   equivalence tests (§8) pass; this is not a behaviour change, so the D23 "pending gate" status carries over to
   that bundle unchanged: **the D23 bundle is the first thing the gate evaluates**.
5. Update consumers that interpret `role`: `redteam/classifier.py`, `redteam/judge.py`, `redteam/sandbox.py`,
   `redteam/probe_runner.py`, `training/eval_harness.py`, `training/replay_runner.py` (they skip `role="shadow"`
   today and must also skip `role="observe"`), and record `bundle_sha256` per response.
6. Remove the v1 path one milestone later.

## 8. Test plan (what proves equivalence with today's served behaviour)

| # | Test | Pass condition |
| --- | --- | --- |
| T1 | Converter golden files: committed registry + manifests → the two YAML policies in §5 | byte-identical canonical JSON |
| T2 | Differential unit tests: every input in `tests/serving/test_cascade.py`, `test_stage1_observe.py`, `test_gateway.py`, `test_shadow.py`, run through legacy service and policy engine, all four coverage modes, with fakes that hang, raise, hit the soft deadline | responses equal field for field except `latency_ms` and new 1.3 fields; Stage 2 scores the same rows in the same order |
| T3 | **D23 replay equivalence**, in-process, timeouts disabled (as `training/replay_counterfactual_served.py` does): all 2,000 items of `data/replay/replay-v1.jsonl` (sha256 `47ac4c84…`) through the legacy D23 config and the converted policy, real models | identical `decision`, `decided_by`, `policy_applied`, `coverage_policy_applied`, `stages_run` and Stage 2 row set on **every** item (score differences ≤ `THRESHOLD_TOLERANCE`) |
| T4 | Cascade replay equivalence: the route policy reproduces `replay_counterfactual_served_faithful.json` `items` (Stage 2 chunks, Stage 1 scores) | identical per item; headline 201/306 reproduced |
| T5 | Against the live D23 run (`reports/cascade-and-shadow/d23/replay/requests.jsonl`, 1,974 responses) | identical decisions on every item whose served response had no failed active result; the others (4 Stage 2 timeouts, `d23/replay/summary.json`) are listed, not compared, because timing decided them |
| T6 | Failure matrix with fakes: {gate, decide single, decide parallel member} × {timeout, exception, overloaded} × {FAIL_OPEN, FAIL_CLOSED, FALLBACK ok, FALLBACK fails} | documented outcome per cell; FALLBACK never started on `overloaded`; ASYNC_ONLY steps never change any decision field |
| T7 | Shape B with fakes: `any_flag`/`all_flag` truth tables, member failure votes, both members score identical rows | as specified |
| T8 | Bundle: changing only the band, only a capability's threshold, or only `engine_version` changes `bundle_sha256`; a gate verdict for one hash is refused for another; a decision-affecting env var with registry v2 refuses startup | as specified |
| T9 | Validators: one negative test per rule in §4 | each rejected with a named error |
| T10 | Canary (when implemented): assignment stable across processes; share within a binomial 99% interval over 100,000 synthetic ids; salt change reshuffles | as specified |

T3 is the acceptance test for the refactor. Live replay (T5) cannot be exact because timeouts depend on load; that
is why T3 disables them.

## 9. Health in three layers

| Layer | Question | Existing signals (file) | Gaps |
| --- | --- | --- | --- |
| **Model health** | Is each detector still good on what it sees? | Recall/FPR with Wilson CIs per slice (`training/eval_harness.py`, `training/results/*_frozen_test.json`, `*_val.json`); shadow agreement (`sentinel_shadow_comparisons_total`, `shadow_overhead.json`); observe outcomes (`sentinel_stage1_observed_total{outcome}`: 806 would-allow, 1,186 would-escalate, 6 not scored in `d23/metrics.txt`); int8 vs fp32 agreement 484/500 (`serving/bench_results/int8_compare.json`); dilution curve (`training/DILUTION.md`); probes (`reports/cascade-and-shadow/d23/probes.json`) | No live score histogram per detector; no calibration check against delayed labels; language mix is not measured by the gateway (only replay metadata `non_en_share`); scoring rule vs calibration mismatch is not checked (§1 item 5) |
| **Serving health** | Does the gateway run the bundle within its budgets? | `sentinel_detector_latency_seconds`, `sentinel_detector_failures_total{reason,role}`, `inflight`, `queue_depth`, `busy_workers`, batch size, real/computed tokens, `sentinel_stage1_deadline_hits_total`, `sentinel_shadow_dropped_total`, registry loads/rejections, `/health` (`d23/health.json`); `serving/loadtest_results/`, `serving/bench_results/observe_overhead.json` | No `bundle` label on any series; no per-step deadline-vs-budget metric; stuck tokenization invisible (RFC serving-hardening); load generator shares the 2 cores (RFC) |
| **End-to-end safety health** | Does the deployed bundle catch injections and spare benign traffic, as served? | As-served recall/FPR (`eval_harness` `gate_view: "as_served"`; `d23/replay_counterfactual.json` `views.served`: 248/303, 199/1,169); how requests were decided (`sentinel_cascade_path_total`, `sentinel_coverage_policy_applied_total`, `sentinel_classify_requests_total{decision,policy_applied}` incl. fail-open allows); flag and escalation rate per window with delayed labels (`training/replay_runner.py`, `d23/replay/windows.jsonl`); sandbox modes B/C (`redteam/README.md`) | Delayed labels exist only in replay, not live; no flag-rate deviation alarm (needed for canary rollback); no counter for "allow with any unscored content" across steps (`unscored_by_any_step` adds it); no adaptive-attacker ASR yet (RFC adaptive-eval) |

The gate reads layer 3 for its verdict and layers 1–2 for its error-rate and p99 checks. A detector that is
healthy in layer 1 can still fail layer 3 under a bad policy, which is exactly D23.

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| The refactor subtly changes routing (chunk order, anytime Stage 1, coverage) | Refactor in place; T2–T4 differential tests are the merge condition |
| Evaluating many policy versions on the same replay set overfits it | RFC rule: a future Stage 1 decision uses a fresh replay draw; the replay manifest itself says it is "NOT an accuracy test set" (`data/replay/MANIFEST.json`); gate accuracy checks use the gate's splits |
| Shape B on 2 cores: parallel DeBERTa-class members compete for CPU and turn into timeouts, so failure semantics decide | A DeBERTa shadow already roughly doubled active p50 (`serving/README.md`, `shadow_overhead.json`); observe-only TF-IDF cost +78 ms p50 / +311 ms p99 at concurrency 4 (`observe_overhead.json`). Shape B is tested with fakes only until separate cores or a host exist |
| `role="observe"` breaks consumers that only know active/shadow | Migration step 5 lists them; contract tests accept 1.2 documents |
| Bundle hash includes `engine_version`, a manual bump that can be forgotten | Also record the git commit of `serving/` in the gate report; a CI check fails when decision-path files change without a bump |
| Canary arms are too small for the gate's minimum n | Report `INSUFFICIENT_EVIDENCE` per arm; never pool arms |
| onnxruntime results differ slightly across thread counts | T3 compares decisions with `THRESHOLD_TOLERANCE`, and runs with the same thread settings as the served run |

## 11. Effort (estimate, not a measurement)

| Work | Estimate |
| --- | --- |
| Contracts, validators, snapshot, contract tests (T1, T9) | 1 day |
| Converter, registry v2 loader, bundle hash, env-var guard (T8) | 1 day |
| Engine refactor for gate/decide single/observe/shadow/coverage + failure enum (T2, T6) | 2 days |
| Shape B and FALLBACK with fakes (T7) | 1 day |
| Replay equivalence runs (T3–T5) and consumer updates | 1 day |
| Canary (later milestone) | 1 day plus the gate's arm handling |

About 6 engineer-days before canary. Compute on the 2-core VM, from committed measurements: the serving-faithful
route-mode replay took 8.6 s for phase A and 331.5 s for phase B on 1,439 DeBERTa rows
(`replay_counterfactual_served_faithful.json` `inputs.wall_s`). T3 for D23 sends every request to Stage 2, so it
scores more rows; its wall time is not measured and should be run under `/home/claude/measure.lock`. Unit and
differential tests use fakes and need no model.

## 12. Open questions for the owner

1. Is `decision: "quarantine"` added in v1.3, or kept as `flag` plus `coverage_policy_applied`?
2. `engine_version`: manual semantic version plus CI check (proposed), or the git commit of `serving/` directly?
3. Should decision-affecting environment variables be a hard startup error under registry v2 (proposed) or a
   warning for one milestone?
4. Canary: minimum `share_bp`, which arm must reach the gate's minimum n, and the automatic-rollback thresholds
   (error rate, p99, flag-rate deviation).
5. Shape B's second decider: no candidate exists today (MiniLM is planned as a Stage 1 v2, i.e. a gate). Is shape B
   required to be served in adaptive-eval, or is a fake-tested engine enough?
6. Stage 1 scoring rule: re-calibrate `stage1-tfidf-v2.1`'s threshold with the served sub-windows (a new capability
   version), or declare `subwindow_tokens: 0` in the capability and accept that serving changes? Either is a new
   bundle.
7. Collapse `stage1-tfidf-v2` and `v2.1` into one capability version (identical bytes and threshold), or keep both
   for history?
