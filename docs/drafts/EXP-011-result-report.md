# EXP-011 Result Report — Terminal Evidence

> **Status:** Populated from verified terminal artifacts. The compact source-of-truth record is [results/EXP-011-summary.md](../../results/EXP-011-summary.md).

## Attribution and frozen controls

- Training-source commit: `cfe78547eecb37079a3da102a3f5e6b02f725017` (0→900M) and `59db721ad1ed35446a36db1a3807a9ab0ffbdd5a` (verified continuation)
- Evidence/result commit: `<this documentation commit>`
- Config hash: not recorded; existing tooling does not reproducibly generate one
- Seed: `42`
- Architecture: Near-Cap Recipe v3 — SwiGLU with SiLU gate; vocab 8,192; d_model 640; 9 layers; 20 heads x 32; d_ff 1,728; tied embeddings/output; **49,860,480 trainable parameters**
- Context: `512`
- Physical batch / accumulation / effective prediction tokens per update: `32 / 2 / 32,768`
- Precision: BF16 autocast with FP32 model and optimizer state
- Optimizer: AdamW beta1/beta2/eps `0.9/0.95/1e-8`; matrix weight decay `0.1`; clipping `1.0`
- LR schedule: warmup `100`; peak/minimum `6e-4/6e-5`; cosine horizon `45,777` steps from step zero
- System controls: native Windows; OMEN Performance mode; AC power

## Data-integrity evidence

### Artifact cardinality and frozen prefixes

- Required full artifact: **1,500,020,737 stored uint16 token IDs**, yielding **1,500,020,736 prediction tokens**.
- EXP-004 prefix: first `300,023,809` stored IDs / `300,023,808` prediction tokens; record expected and observed raw-byte SHA-256 plus the exact byte count.
- EXP-006 prefix: first `900,071,425` stored IDs / `900,071,424` prediction tokens; record expected and observed raw-byte SHA-256 plus the exact byte count.
- Full stream / manifest / tokenizer SHA-256: `092fc4a02f991b15fd8fcd2c209754e014485c74bea642c1a57270462141b671` / `b2ed5e461d753beb581c0d88668371c16abc63c6c9a67673f453a46f27d9feeb` / `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.

### Data Recipe v1 evidence

- Target source mixture: `2:1` FineWeb / FineWeb-Edu, with target prediction-token contributions `1,000,013,824 / 500,006,912`.
- Realized prediction-token contributions: FineWeb `1,000,014,372`; FineWeb-Edu `500,006,364`.
- Deduplication method/scope: globally content-hash-deduplicated, whole-document deterministic selection. Record the manifest method, unique-document count, within-source and cross-source duplicate skips, and the reused-state provenance.
- Global-state requirement: document evidence that deduplication remained global across the inherited 900M prefix and the new extension; it must not have been restarted solely for the extension.
- Contamination screen: record the NFKC/casefold/tokenized normalized 13-gram SHA-256-overlap method, benchmark-index SHA-256, index provenance/reuse from EXP-006, benchmark-source scope, and per-source accepted/rejected counts.
- Contamination limitation: no-overlap under this finite normalized 13-gram index is evidence against detected overlap, not proof of zero benchmark contamination or semantic contamination.

## Checkpoints, scheduler, and resume

- Promotable checkpoint path / SHA-256: `<compute from selected checkpoint after promotion>`.
- Step-27,468 source checkpoint SHA-256: `<record from actual checkpoint file>`.
- Fresh-process resume evidence: `<record checkpoint, restored optimizer/RNG/scheduler/cursor state, and first continuation update>`.
- Each LR value below must be copied from the actual logged or checkpoint scheduler state; do not reconstruct it manually for documentation.

## Validated training milestones

| Step | Stored IDs consumed | Prediction tokens | General | Edu | Combined | LR from actual state | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 9,156 | 300,023,809 | 300,023,808 | 3.6302440166 | 3.3181973100 | 3.4742206633 | 0.0005492980000078377 | n/a |
| 18,312 | 600,047,617 | 600,047,616 | 3.4739109874 | 3.1538517475 | 3.3138813674 | 0.000414472387127598 | n/a |
| 27,468 | 900,071,425 | 900,071,424 | 3.3710331023 | 3.0461599529 | 3.2085965276 | 0.00024724035994133734 | `e3e3aa36…00670896` |
| 36,624 | 1,200,095,233 | 1,200,095,232 | 3.2905968130 | 2.9606119394 | 3.1256043762 | 0.00011175768037925403 | `1ca9f931…256d87c` |
| 45,777 | 1,500,020,737 | 1,500,020,736 | 3.2471743524 | 2.9129971564 | 3.0800857544 | 0.00006000000000000000 | `c1e65c7d…9a37928c0` |

## Runtime accounting

- 900M training wall time: `8,971.3878 s`.
- Continuation training wall time: `5,875.3530 s`.
- Total model-training wall time: `14,846.7408 s`.
- Data-build/preparation wall time: `23,989.5951 s`.
- Mean / final continuation throughput: `102,624.21 / 105,443.83 tok/s`.
- Peak allocated / reserved VRAM across phases: `7,687,993,856 / 8,491,368,448` bytes.
- Final step / prediction tokens / final LR: `45,777 / 1,500,020,736 / 6e-5`.

## Completion boundary

- Finite loss/gradient status: verified across 18,320 machine-readable continuation events; final model tensors finite.
- Parameter, tokenizer, data, prefix, checkpoint, cursor, and resume invariants: verified from the strict loader and final checkpoint load.
- Deviations or aborts: reboot lost no completed updates; continuation resumed from the preserved step-27,468 checkpoint after the provenance-gate key correction.
- Official benchmarks: **not run**.
- Next action: stop after 1.5B and await research review.
