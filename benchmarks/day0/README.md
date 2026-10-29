# Day 0: Stage 2 detector feasibility

**Model:** `protectai/deberta-v3-base-prompt-injection-v2`, official ONNX fp32 export, Apache-2.0, revision `90c9989b`, model sha256 `f0ea7f23…7b228c`. Labels `0=SAFE, 1=INJECTION`.
**Hardware:** Claude cloud workspace, 2-core Intel Xeon @ 2.10 GHz, no GPU. Python 3.13.18, onnxruntime 1.29.0 (CPUExecutionProvider, default threads), tokenizers 0.23.3.
**Run:** `python scripts/fetch_models.py && python -m benchmarks.day0.bench_deberta`. Every number below is in `day0_results.json`, produced by that command.

Inputs are always `[CLS] + content + [SEP]`, in both the truncated and the chunked path (`encoding.py`, tested in `tests/benchmarks/test_encoding.py`). Chunks are 510-content-token windows with stride 384, built on token ids, covering the tail.

## Finding 1: silent truncation misses an injection at the end of a long email

A synthetic multi-paragraph meeting-recap email (`fixtures.py`, 705 content tokens) with one injection sentence appended (724 tokens):

| Input | Truncated to 512 | Chunked (max over 2 windows) |
| --- | --- | --- |
| Benign email | 0.0185 | 0.3718 |
| Injection at the end | **0.0185** (miss at threshold 0.5) | **0.9966** |
| Injection at the start | 0.9990 | — |

The truncated score for the injected email equals the benign one because the injection is never seen. Silent truncation is a serving-layer bug; contracts v1.1 require `truncated=True` whenever content is not scored, and the gateway must chunk.

**Side effect to track:** max aggregation raised the benign email's score from 0.0185 to 0.3718 (still below 0.5). Chunking can raise false positives on long benign emails; the `benign-hard` slice must include long emails, and aggregation is a measured choice, not a default.

## Finding 2: scores of benign repeated text drift upward with length

The same benign sentence repeated: 0.0096 at 120 content tokens, 0.0467 at 256, 0.1117 at 400, 0.3163 at 510. No false positive at threshold 0.5, but the trend matters for any threshold below ~0.35. (An earlier version of this benchmark reported 0.94 here; that was a bug that dropped `[SEP]` during truncation, caught in review.)

## Latency (sequential requests, timing includes tokenization)

| Config | n | p50 | p95 | p99 | max | per-item p50 |
| --- | --- | --- | --- | --- | --- | --- |
| batch 1, 64 tokens | 100 | 80 ms | 118 ms | 193 ms | 196 ms | 80 ms |
| batch 1, 128 tokens | 100 | 138 ms | 162 ms | 177 ms | 184 ms | 138 ms |
| batch 1, 256 tokens | 100 | 286 ms | 335 ms | 350 ms | 379 ms | 286 ms |
| batch 1, 510 tokens | 50 | 715 ms | 774 ms | 802 ms* | 802 ms | 715 ms |
| batch 8, 256 tokens | 30 | 2,485 ms | 2,624 ms | 2,650 ms* | 2,650 ms | 311 ms |
| batch 32, 128 tokens | 20 | 4,588 ms | 4,768 ms | 4,770 ms* | 4,770 ms | 143 ms |

\* p99 equals the maximum at this sample size. Batch rows are distinct texts of identical length (no padding cost).

- Batching does not reduce per-item latency on 2 cores: 311 ms (batch 8) vs 286 ms (batch 1) at 256 tokens; 143 ms (batch 32) vs 138 ms (batch 1) at 128 tokens.
- The kill criterion (p95 > 150 ms at batch 1) is met from 128 tokens up. The synthetic email above is 705 tokens. Next: int8 quantization, and the cascade so most requests never reach this model. GPU numbers will be measured separately on an RTX 3080.

## Sanity

Short benign request 0.0000, instruction-like IT notice 0.0005, direct injection 1.0000. Three examples are a smoke test, not an evaluation.
