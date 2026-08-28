# EXP-012 inference-only publication plan

Status: **validated local publication candidate; public release deferred**.
The package is `artifacts/exp012-publication/`; EXP-012 is the current best
evaluated research checkpoint, not a permanently frozen or final GIBC
submission model. Its local version is `exp012-evaluated-v1`. Public model
publication is deferred until the final GIBC model is selected; EXP-012 may
later be retained as a historical version if useful.

## Publication decision

No public model repository, release, tag, account connection, or upload is
authorized for EXP-012 now. The local candidate is retained only for future
review after final GIBC model selection.

## Deterministic package layout

```text
gibc-v2-track01-exp012-evaluated-checkpoint/
  model.safetensors       # strict terminal model tensors only; no pickle
  tokenizer.json          # frozen tokenizer byte copy
  exp012.yaml             # byte copy of configs/exp012.yaml
  SHA256SUMS              # standard SHA-256 checksum list for payload files
  manifest.json           # SHA-256 and byte count for every payload file
  provenance.json         # hashes, parameter count, source commit, credits
  equivalence.json        # fixed-logit comparison evidence
  tensor-inventory.json   # exact tensor names, shapes, dtypes, finiteness
  LICENSE                 # Apache License 2.0 for original code/weights
  LICENSE-NOTICE.md       # dataset-rights and attribution boundary
```

`model.safetensors` must exclude optimizer, scheduler, RNG, data cursor,
training metrics, and caches. Its provenance must link to original terminal checkpoint
SHA-256 `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c`.
The manifest is generated after every file is finalized, in sorted filename
order, using lowercase SHA-256 digests and exact byte counts.

## Required local candidate verification

1. Rehash the original checkpoint, tokenizer, and config; fail on mismatch.
2. Extract only `payload['model']`; require strict keys and 49,860,480
   trainable parameters after reload.
3. Reconstruct CPU FP32 models from `exp012.yaml` using original and exported
   state; score fixed token IDs `[[1, 2, 3, 4]]` and require maximum absolute
   logit difference `<= 1e-6` before writing `equivalence.json`.
4. Verify every manifest entry and the final parameter-count command. Do not
   run benchmark examples.

The package works with the explicit README command:
`scripts/generate.py --config ... --checkpoint model.safetensors --tokenizer ...`.

## Completed local gate

The local package contains a 199,450,256-byte `model.safetensors` with SHA-256
`1f56454bf7098a9b8a33f0346c42fcf8c2dfbc86cc7403bd5d7eb01829530102` and
`SHA256SUMS` SHA-256
`da2641348715fa020268efbae369dcd84cb0be606c35cadef9e07d75eee79564`.
`SHA256SUMS` uses standard `sha256sum -c` syntax; `manifest.json` records the
same payload entries with exact byte sizes.
The original full checkpoint and exported safetensors state were strict-loaded
into CPU FP32 models and compared on fixed non-benchmark token IDs; maximum
absolute logit difference was `0.0` at `atol=1e-6`, `rtol=0`. All 83 tensor
names, shapes, FP32 dtypes, and finite values were independently validated;
the reloaded export recounted exactly 49,860,480 trainable parameters.

## Publication boundary

Do not upload EXP-012, create a Hugging Face repository, create a release/tag,
or connect an account. Revisit public distribution only after final GIBC model
selection.
