# Day 0: Stage 2 detector feasibility

**Model:** `protectai/deberta-v3-base-prompt-injection-v2`, official ONNX fp32 export, Apache-2.0. Labels `0=SAFE, 1=INJECTION`.
**Hardware:** Claude cloud workspace, 2-core Intel Xeon @ 2.10 GHz, 7 GB RAM, no GPU. onnxruntime 1.29.0, CPUExecutionProvider, sequential requests, 3 warmup runs.
**Run:** `python benchmarks/day0/bench_deberta.py` (raw output in `day0_results.json`).

## Latency

| Config | p50 | p95 | p99 | per-item p50 |
| --- | --- | --- | --- | --- |
| batch 1, 64 tokens (n=60) | 78 ms | 95 ms | 101 ms | 78 ms |
| batch 1, 256 tokens (n=40) | 297 ms | 326 ms | 345 ms | 297 ms |
| batch 1, 512 tokens (n=30) | 723 ms | 769 ms | 771 ms | 723 ms |
| batch 8, 256 tokens (n=15) | 2,492 ms | 2,572 ms | 2,608 ms | 312 ms |
| batch 32, 128 tokens (n=8) | 4,623 ms | 4,941 ms | 4,941 ms | 145 ms |

Kill criterion (p95 > 150 ms at batch 1) is met for realistic email lengths (200–700 tokens). Next: int8 quantization comparison. Batching does not reduce per-item latency on 2 cores. GPU numbers will be measured separately on an RTX 3080.

## Findings

1. **Truncation bypass.** A realistic 709-token email with an injection appended at the end: 0.002 truncated at 512 tokens, 0.999 with overlapping chunks (window 448, stride 384, max aggregation). The same injection at the start scores 0.996. Silent truncation is a serving-layer bug; contracts v1.1 require `truncated=True` whenever content is not scored.
2. **Repetitive-text false positive.** Benign text repeated to 512 tokens scores 0.94; the same text at 120 tokens scores 0.009. Long repetitive emails (newsletters, logs) go into the `benign-hard` evaluation slice.
3. **Sanity.** A short benign request, an instruction-like IT notice, and a direct injection score 0.000 / 0.000 / 1.000. Three examples are a smoke test, not an evaluation.
