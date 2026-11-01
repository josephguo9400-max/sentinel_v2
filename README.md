# Sentinel v2

A safety-classifier inference gateway prototype: DeBERTa prompt-injection detection behind a gateway with chunked scoring and an explicit coverage policy for unscored content, manifest-driven deployment with shadow and observe-only candidates, a release gate with confidence intervals, and an adaptive red-team harness used as acceptance testing.

**Status:** milestones 1 (detector-baseline) and 2 (gateway-mvp) done and independently reviewed. Milestone 3 (cascade-and-shadow) is in review: its cost-saving cascade (cheap TF-IDF Stage 1 finalizing allows) **failed its kill criterion on the traffic replay set** and was turned off (D23); DeBERTa now decides every request and Stage 1 runs observe-only. Plan and rationale: [`docs/RFC.md`](docs/RFC.md).

## Layout

| Path | Owner | Purpose |
| --- | --- | --- |
| `contracts/` | owner | Frozen shared schemas: `DetectorResult`, `ArtifactManifest`, `TraceEvent`, `ClassifyRequest`/`ClassifyResponse` (v1.2; readers accept 1.1) |
| `tests/contract/` | owner | Contract tests every agent must keep green |
| `serving/`, `web/` | Agent A | Gateway, registry, Stage 1 routing/observe modes, coverage policy, shadow, static UI |
| `training/`, `artifacts/candidates/` | Agent B | Data splits, Stage 1, candidate manifests |
| `redteam/`, `fixtures/` | Agent C | Email sandbox, deterministic judge, adaptive attacker |
| `reports/verify/` | Verifier | Independent checkpoint reports |
| `benchmarks/` | owner | Measured results with hardware noted |

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_models.py           # downloads the ONNX detector into artifacts/baseline/deberta-pi-v2/ (not committed)
pytest -q
python -m benchmarks.detector_baseline.bench_deberta  # latency + truncation test on this machine
```

## Cascade-and-shadow findings (replay set, 2,000 requests; Wilson 95% intervals, Newcombe paired 95% for differences)

- **The cascade does not pay for itself on drifted traffic.** DeBERTa alone recalls 247/306 = 80.7% [75.9, 84.7] of replay injections. With TF-IDF Stage 1 allowed to finalize allows, the gateway served 202/306 = 66.0% [60.5, 71.1] while still sending 1,196/1,995 = 59.9% [57.8, 62.1] of requests to DeBERTa (`reports/cascade-and-shadow/replay_counterfactual.json`, `views.served`). Simulated the way the gateway routes (`reports/cascade-and-shadow/replay_counterfactual_served_faithful.json`), the cascade recalls 201/306 = 65.7% [60.2, 70.8], a paired -15.0 pt [-19.4, -10.7] against DeBERTa alone, with **no FPR benefit either** (191/1,185 vs 194/1,185, paired -0.25 pt [-0.91, +0.40]); it saves at most 2.5% of Stage 2 calls at identical recall (bar: 30%). The loss is concentrated in `gandalf_summ`, indirect injections inside text to summarize, a source Stage 1 never trained on. Val did not show it because val shares Stage 1's training sources. Details: `training/CASCADE.md`.
- **The D23 deployment (DeBERTa decides, Stage 1 observe-only), replayed through the gateway:** recall 248/303 = 81.8% [77.1, 85.8], FPR 199/1,169 = 17.0% [15.0, 19.3], against 245/303 and 191/1,169 for offline DeBERTa-only on the same items. The 15 decisions that differ come from the coverage policy (5), the failure policy (3) and other causes (7) (`reports/cascade-and-shadow/d23/replay_counterfactual.json`, `views.served.same_items`).
- **Coverage is necessary but not sufficient.** With every token scored, a short injection in the middle of 256-510 tokens of benign text is usually missed by DeBERTa (newsletter context, middle placement: 20/62 detected at 510 tokens of context, against 46/62 alone; probe set v2: 1/12 at the gateway's 510-token window). Smaller windows recover some of these but raise false positives on real benign traffic, so no small-window pass is turned on. Details: [`training/DILUTION.md`](training/DILUTION.md).
- **Probe set v1 measured nothing for long emails:** its filler was largely random words, which DeBERTa flags with or without an injection. Probe set v2 uses distinct sentences.

## Gateway MVP findings (frozen test sha256 `2a4e011f…`, 6,276 items, Wilson 95% intervals, as served)

| Metric | Stage 2: DeBERTa via gateway | Stage 1: TF-IDF v2 |
| --- | --- | --- |
| Recall | 5083/5137 = 98.9% [98.6, 99.2] | 4289/5137 = 83.5% [82.5, 84.5] |
| Recall, multilingual held-out source (yanis) | 933/951 = 98.1% | 214/951 = 22.5% |
| FPR | 148/1139 = 13.0% [11.2, 15.1] | 103/1139 = 9.0% [7.5, 10.8] |
| FPR, instruction-like benign emails | 60/276 = 21.7% [17.3, 27.0], gate FAIL | 78/276 = 28.3% [23.3, 33.8], gate FAIL |

- Neither stage is good enough alone: Stage 1 cannot finalize non-English inputs, and both fail the benign-hard FPR gate. Long log digests are the worst slice for DeBERTa (30/30 flagged).
- In the sandbox, attacks the classifier missed were all stopped by the deterministic authorization firewall in mode C.
- Results: `training/results/`, data audit: `training/DATA.md`, gateway design and load tests: `serving/README.md`, sandbox: `redteam/README.md`.

## Detector baseline findings (2-core Xeon 2.1 GHz, no GPU, fp32 ONNX)

- **Silent truncation misses injections:** an injection appended to a 705-token synthetic email scores 0.0185 when truncated to 512 tokens (same as the benign email) and 0.9966 with chunking.
- **Long benign text drifts upward:** benign scores rise with length and repetition (0.0096 at 120 tokens, 0.3163 at 510), so long benign emails need their own false-positive slice.
- **CPU latency:** p95 118 ms at 64 tokens, 335 ms at 256, 774 ms at 510; batching does not help on 2 cores → int8 quantization and the cascade are next.

Details: [`benchmarks/detector_baseline/README.md`](benchmarks/detector_baseline/README.md).

## Non-goals

Not a production safeguard, not a replica of any company's architecture. The authorization firewall, not the classifier, is the security boundary.
