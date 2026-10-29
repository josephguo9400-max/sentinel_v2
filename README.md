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
python benchmarks/day0/bench_deberta.py  # latency + truncation test on this machine
```

## Day 0 findings (2-core Xeon 2.1 GHz, no GPU, fp32 ONNX)

- **Truncation bypass reproduced:** an injection at the end of a 709-token email scores 0.002 when truncated to 512 tokens and 0.999 with chunking.
- **Long repetitive text false-positives:** repeated benign text scores 0.94 at 512 tokens (0.009 at 120 tokens).
- **CPU latency:** p95 95 ms at 64 tokens, 326 ms at 256, 769 ms at 512 → int8 quantization and the cascade are next.

Details: [`benchmarks/day0/README.md`](benchmarks/day0/README.md).

## Non-goals

Not a production safeguard, not a replica of any company's architecture. The authorization firewall, not the classifier, is the security boundary.
