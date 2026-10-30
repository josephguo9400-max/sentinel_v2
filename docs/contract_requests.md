# Contract change requests

Agents never edit `contracts/` directly. Append a request here; the owner decides and bumps the schema version.

| Date | Requester | Contract | Proposed change | Reason | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-10-29 | Agent B | `ArtifactManifest` | Add optional `input_languages: list[str] \| None` (e.g. `["en"]`) and let the router escalate (or apply `failure_policy`) for inputs outside it | Stage 1 `stage1-tfidf-v1` recalls 27.0% (257/951) of the multilingual held-out source `yanis`, and 20.5% of non-English frozen-test injections; as Stage 1 it would finalize 563 of them as benign without Stage 2 seeing them (`training/results/stage1-tfidf-v1_frozen_test.json`) | |
| 2026-10-29 | Agent B | `DetectorResult` / `ArtifactManifest` | Document that `input_tokens` and `max_input_tokens` are in the detector's own tokenizer units, and add `tokenizer: str` to the manifest (e.g. `regex:[\p{L}\p{N}_]{2,}` for Stage 1, `hf:deberta-v3` for Stage 2) | Stage 1 counts regex word tokens and chunks at 512 of them; DeBERTa counts subword tokens. The same text can be over the limit for one and under it for the other, so the router cannot compare the two numbers | |
