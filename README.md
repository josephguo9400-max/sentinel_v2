# Sentinel v2

A safety-classifier inference gateway prototype: a two-stage cascade (cheap Stage 1 → DeBERTa Stage 2 only for gray-zone scores), manifest-driven self-service deployment into shadow, a release gate with confidence intervals, and an adaptive red-team harness used as acceptance testing.

**Status:** milestones 1 (detector-baseline) and 2 (gateway-mvp) done and independently reviewed; next is cascade-and-shadow. Plan and rationale: [`docs/RFC.md`](docs/RFC.md).

## Layout

| Path | Owner | Purpose |
| --- | --- | --- |
| `contracts/` | owner | Frozen shared schemas: `DetectorResult`, `ArtifactManifest`, `TraceEvent` (v1.1) |
| `tests/contract/` | owner | Contract tests every agent must keep green |
| `serving/`, `web/` | Agent A | Gateway, cascade routing, shadow, gate, static UI |
| `training/`, `artifacts/candidates/` | Agent B | Data splits, Stage 1, candidate manifests |
| `redteam/`, `fixtures/` | Agent C | Email sandbox, deterministic judge, adaptive attacker |
| `reports/verify/` | Verifier | Independent checkpoint reports |
| `benchmarks/` | owner | Measured results with hardware noted |

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_models.py           # downloads the ONNX detector into models/ (not committed)
pytest -q
python -m benchmarks.detector_baseline.bench_deberta  # latency + truncation test on this machine
```

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
