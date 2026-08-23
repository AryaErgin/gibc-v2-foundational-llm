# EXP-001C — Full-Run Entrypoint and Dry Run

Status: PASS. This launch-readiness stage materialized the exact hardened full artifact and exercised only the authorized 5-update full-path dry run plus a separate-process one-update resume. It did not start the 3,052-update baseline.

Implementation was introduced by `14bb5dc2df79a5ee5240ae39da97411f319ca4a8` (`feat: add EXP-001 full-run entrypoint`), followed by `027900cf30117a3cd9a767ba3b364a755ade1e26` to use the already-approved immutable FineWeb SHA directly rather than re-querying dataset metadata.

## Full artifact

- FineWeb: `HuggingFaceFW/fineweb`, `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`.
- Tokenizer: byte-level BPE, 8,192 entries, sole special token `<|endoftext|>`; SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.
- Train stream: 100,007,937 stored uint16 IDs / 100,007,936 prediction tokens / 195,328 non-cycled 512-token examples; 200,015,874 bytes; SHA-256 `86b84dc30f88ac1ba8daee4f7b160f581d3e9a5987fbf86fff5dbab967647d04`.
- Held-out deterministic validation: 131,072 prediction tokens (256 examples, 148 validation documents); inputs SHA-256 `f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7`; targets SHA-256 `2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e`.
- Full-artifact manifest SHA-256: `f1b2bd8c6dbd73b2e31c795103e2fc1088f123c14021f3c966df2e6333ebd730`.
- Hardened all-public-split screening: 124,383 FineWeb documents scanned; 124,218 accepted; 165 rejected (0.13265%); 121,721 train documents contributed. Preparation wall time: 1,703.105 seconds.

The manifest locks the public HellaSwag, ARC-Easy, PIQA, and WinoGrande train/validation/test sources and the WikiText-103 held-out validation/test sources at the revisions recorded in `provenance/exp001-benchmark-revisions.json`. It stores hashes/counts/provenance only, not benchmark contents.

## Entrypoint and dry run

`scripts/train_exp001_full.py` validates the full artifact before use, asserts the 8,392,960-parameter model, uses the fixed 3,052-step schedule horizon, sequential mmap batches, BF16 autocast with FP32 parameters, and explicit checkpoints/resume provenance. Its default is 3,052 updates; `--max-steps` is explicit and logged as `DRY RUN / INCOMPLETE TRAINING`.

The authorized first invocation was:

```powershell
python scripts/train_exp001_full.py --artifact-dir artifacts/exp001c-full --run-dir artifacts/exp001c-full/dry-run --max-steps 5
```

It used microbatch 32, accumulation 2, and effective batch 32,768. It completed step 5, 163,840 prediction tokens, and `next_sequence_index=320`; first/final train loss was 9.074440/8.981425; validation loss before/after was 9.073967/8.940254 (PPL 8,725.17/7,633.14); mean/final throughput was 294,516.84/346,519.94 tokens/s; peak allocated/reserved VRAM was 3,218,745,856/4,068,474,880 bytes. It wrote `checkpoint-step-0005.pt`.

The required independent resume invocation was:

```powershell
python scripts/train_exp001_full.py --artifact-dir artifacts/exp001c-full --run-dir artifacts/exp001c-full/dry-run --resume artifacts/exp001c-full/dry-run/checkpoints/checkpoint-step-0005.pt --max-steps 1
```

It completed step 6, 196,608 prediction tokens, and `next_sequence_index=384`, with finite train loss 8.950151 and validation loss 8.900388 (PPL 7,334.82). The full-horizon schedule remained at 3,052 steps. This is dry-run evidence only, not a partial baseline result.

## Verification

- Complete pytest suite: 33 passed.
- `pip check`: no broken requirements.
- Exact uint16 contiguous-batch path equals stacked individual 513-token reference views.
- Full manifest rejects missing/mismatched tokenizer, stream, validation data, token counts, context, dtype, hash, and non-cycling invariants.
- CPU fixture checkpoint/resume and mmap full-stream checkpoint/resume retain exact cursor/model equality; the actual CUDA full-run entrypoint resumed from cursor 320 to 384 as required.

No official benchmarks were evaluated during training. All generated artifacts, caches, checkpoints, and logs remain ignored and local-only.
