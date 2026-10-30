# deberta-pi-v2 (Stage 2 baseline, fp32)

`protectai/deberta-v3-base-prompt-injection-v2` at revision `90c9989b1a342275dd0d1a95aad283c04e075671`
(Apache-2.0), ONNX fp32 build from the upstream repository. Only `manifest.json` is committed; the weights land
in `onnx/` (gitignored) via:

```bash
python scripts/fetch_models.py            # downloads, then verifies sha256 of model.onnx and tokenizer.json
```

`max_input_tokens` (512) is the model input length in `hf:deberta-v3` tokens including `[CLS]` and `[SEP]`,
so a window holds 510 content tokens. `preprocessing_version` `deberta-v3-window-v1` names that window format
(serving/structure.py). `input_languages` is null: the upstream card does not state a validated language set.
