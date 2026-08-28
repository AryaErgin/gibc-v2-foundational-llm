# EXP-013 — Warmup-Stable-Decay Scheduler Ablation

**Status:** predeclared; implementation and source commit required before training.

## Question

Can Warmup-Stable-Decay (WSD) improve capability and/or horizon-scaling
efficiency relative to the frozen 300M-token Recipe-v3 cosine control?

## Hypothesis

A long stable `6e-4` learning-rate phase followed by a 10%-scale cooldown
will match or improve frozen 300M-token validation and provide a reusable
stable-stage checkpoint for later horizon experiments.

## Fixed control conditions

Both arms use Recipe v3 without any architectural, tokenizer, data, objective,
optimizer, batch, seed, or validation-set change:

- decoder-only LM: vocabulary 8,192; `d_model=640`; 9 layers; 20 heads;
  `d_ff=1728`; SwiGLU; tied input/output embeddings;
  **49,860,480 trainable parameters**;
- context length 512 and causal next-token objective;
- Data Recipe v1: contamination-screened FineWeb/FineWeb-Edu 2:1 mixture,
  exact non-cycled 300,023,808 prediction-token prefix;
- frozen tokenizer SHA-256
  `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`;
- seed 42; AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, 0.1 decay on matrices,
  gradient clip 1.0;
- physical batch 32 sequences × accumulation 2 (32,768 prediction
  tokens/update); BF16 autocast with FP32 parameters and optimizer state;
- frozen General and Edu validation tensors, each 131,072 prediction tokens;
- peak/minimum LR `6e-4` / `6e-5`; 9,156 updates.

The source data artifact is `artifacts/exp004-full`.  Its exact stream
SHA-256 is
`8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`.

## Arms

| Arm | ID | Schedule |
| --- | --- | --- |
| C | `EXP-013-C` | Existing `CosineWithWarmup`: 100-update linear warmup then cosine decay to `6e-5` at update 9,156. |
| W | `EXP-013-W` | WSD: updates 1–100 linearly warm to `6e-4`; updates 101–8,240 hold exactly `6e-4`; updates 8,241–9,156 are the 916-update cosine cooldown to exactly `6e-5`. |

Scheduler steps are numbered by completed optimizer update. `schedule.step()`
sets the LR used by that update, before `optimizer.step()`. The WSD
stable-stage checkpoint is written after update 8,240, immediately before
cooldown update 8,241, and includes model, optimizer, scheduler, RNG, and
sequential data-cursor state.

Both arms begin from a fresh seed-42 initialization. Neither arm runs an
official benchmark or benchmark example.

## Primary metric and decision rule

The predeclared primary metric is combined frozen validation loss at exactly
300,023,808 prediction tokens. Define:

`delta = WSD combined loss − cosine combined loss`.

- `delta <= -0.020`: **CAPABILITY WIN**; WSD is eligible for confirmation.
- `-0.020 < delta <= +0.010`: **PERFORMANCE TIE**; no capability claim.
  WSD may only be considered an engineering/horizon-flexibility candidate.
- `delta > +0.010`: **REJECT WSD**.

A capability promotion is also rejected if either General or Edu validation
loss regresses by more than 0.020 nat. If a candidate meets the capability
threshold, stop after these two arms: do not automatically launch another seed
or a longer run.

## Evidence to record

For each arm, record final General, Edu, and combined validation loss;
wall-clock duration; mean and final token/s; peak allocated/reserved VRAM;
data/tokenizer/manifest hashes; configuration and source commit; and the full
checkpoint list. The WSD arm additionally records the stable-stage checkpoint
SHA-256 and deterministic save/resume gate.

No benchmark evaluation is part of this experiment.
