# EXP-012 inference-only publication plan

Status: **designed; not built or uploaded**. This plan does not authorize an
external account action, a release, or a model upload.

## Recommended publication target

Publish to a Hugging Face model repository after explicit approval. The local
terminal training checkpoint is 598,446,963 bytes because it includes training
state. The inference-only FP32 model tensors are approximately 190 MiB before
container overhead (`49,860,480 × 4` bytes), plus a 546,551-byte tokenizer.
That size is technically possible in a GitHub Release but is better suited to
a Hugging Face model repository, which provides model-oriented discovery,
versioning, and file checksums. A GitHub Release may later mirror immutable
release metadata, not replace the primary model repository.

## Deterministic package layout

```text
gibc-v2-track01-exp012-inference/
  model_state.pt          # {'model': strict terminal model state only}
  tokenizer.json          # frozen tokenizer byte copy
  exp012.yaml             # byte copy of configs/exp012.yaml
  manifest.sha256         # SHA-256 and byte count for every shipped file
  provenance.json         # hashes, parameter count, source commit, credits
  equivalence.json        # fixed-logit comparison evidence
```

`model_state.pt` must exclude optimizer, scheduler, RNG, data cursor, training
metrics, and caches. Its provenance must link to original terminal checkpoint
SHA-256 `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c`.
The manifest is generated after every file is finalized, in sorted filename
order, using lowercase SHA-256 digests and exact byte counts.

## Required local verification before upload

1. Rehash the original checkpoint, tokenizer, and config; fail on mismatch.
2. Extract only `payload['model']`; require strict keys and 49,860,480
   trainable parameters after reload.
3. Reconstruct CPU FP32 models from `exp012.yaml` using original and exported
   state; score fixed token IDs `[[1, 2, 3, 4]]` and require maximum absolute
   logit difference `<= 1e-6` before writing `equivalence.json`.
4. Verify every manifest entry and the final parameter-count command. Do not
   run benchmark examples.

The package works with the explicit README command:
`scripts/generate.py --config ... --checkpoint model_state.pt --tokenizer ...`.

## Approval boundary

After local package creation and verification, stop and request approval before
creating a Hugging Face repository, connecting an account, uploading files, or
creating a GitHub Release. Do not make either artifact public silently.
