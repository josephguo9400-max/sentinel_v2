# deberta-pi-v2-int8 (Stage 2 candidate, int8 dynamic quantization)

Generated from the verified fp32 baseline (`artifacts/baseline/deberta-pi-v2`) by

```bash
python scripts/fetch_models.py
flock /home/claude/measure.lock -c 'python scripts/quantize_deberta.py'   # verifies the output against manifest.json
```

Settings (pinned in the script): `onnxruntime.quantization.quantize_dynamic`, QInt8 weights, MatMul/Gemm only,
`MatMulConstBOnly`, per-channel weight scales (v2; v1 was per-tensor), no reduce_range; onnxruntime 1.29.0, onnx 1.23.2. Two runs produced the same
sha256 (pinned in `manifest.json`). The output is 539 MB, not ~180 MB: the 128,100 x 768 word embedding
(98.4 M of the model's parameters) is a Gather, not a MatMul, and stays fp32.

Only the manifest is committed; `onnx/` is gitignored. This is a candidate: it is not in the registry's
active pointers, and promoting it is the release gate's decision. Measurements: `serving/README.md`.
