# Sentinel v2

A safety-classifier inference gateway prototype: a two-stage cascade (cheap Stage 1 → DeBERTa Stage 2 only for gray-zone scores), manifest-driven self-service deployment into shadow, a release gate with confidence intervals, and an adaptive red-team harness used as acceptance testing.

**Status:** Day 0 — contracts v1.1 and detector feasibility measurements. Plan and rationale: [`docs/RFC.md`](docs/RFC.md).

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
pytest tests/contract
python -m benchmarks.day0.bench_deberta  # latency + truncation test on this machine
```

## Day 0 findings (2-core Xeon 2.1 GHz, no GPU, fp32 ONNX)

- **Silent truncation misses injections:** an injection appended to a 705-token synthetic email scores 0.0185 when truncated to 512 tokens (same as the benign email) and 0.9966 with chunking.
- **Long benign text drifts upward:** benign scores rise with length and repetition (0.0096 at 120 tokens, 0.3163 at 510), so long benign emails need their own false-positive slice.
- **CPU latency:** p95 118 ms at 64 tokens, 335 ms at 256, 774 ms at 510; batching does not help on 2 cores → int8 quantization and the cascade are next.

Details: [`benchmarks/day0/README.md`](benchmarks/day0/README.md).

## Non-goals

Not a production safeguard, not a replica of any company's architecture. The authorization firewall, not the classifier, is the security boundary.
