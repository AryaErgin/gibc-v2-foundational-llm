# EXP-016 — Momentum-Aligned Gradient Masking (Magma)

Status: preregistered and preflighted on 2026-08-30. No EXP-016 training arm
has been launched. This document freezes the design before either endpoint is
observed.

## Question

Does Momentum-Aligned Gradient Masking (Magma) improve the promoted Recipe-v3
+ WSD baseline at exactly 300,023,808 prediction tokens under the
scientifically qualified WSL runtime?

The algorithmic source is Taejong Joo et al., *On Surprising Effectiveness of
Masking Updates in Adaptive Optimizers*, arXiv:2602.15322v1 (2026). The
implementation is independent; no third-party Magma package or code is used.
See `provenance/exp016-magma-paper.json`.

## Frozen shared conditions

- Runtime: WSL/Linux, Python 3.11, official `torch==2.13.0+cu132`, RTX 5090
  Laptop GPU; OMEN Performance mode during actual training.
- Model: Recipe-v3, exactly 49,860,480 trainable parameters.
- Training: fresh seed 42, BF16, AdamW beta1=0.9 beta2=0.95 eps=1e-8, matrix
  weight decay 0.1, gradient clip 1.0, WSD, context 512, physical batch 32,
  gradient accumulation 2, 64 windows per optimizer update, 9,156 updates,
  and exactly 300,023,808 prediction tokens.
- Data: immutable EXP-015 Schedule A
  `39c509f59489d125904be61e7e3094e0e87af5ee7ead46afe6742cac35185eb2`,
  frozen stream
  `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`,
  and unchanged frozen General/Edu validation arrays.
- WSD: warmup through 100; stable through 8,240; cooldown 8,241–9,156; final
  learning rate exactly 6e-5.
- CPU thread limits: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
  OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1.

## Arms and execution order

The future experiment has two fresh seed-42 arms on the same integration
commit and schedule. The control must run before the treatment regardless of
control validation performance.

1. Control: `configs/exp016-control.yaml`; Recipe-v3 + WSD + dense AdamW.
2. Treatment: `configs/exp016-magma.yaml`; Recipe-v3 + WSD + AdamW + Magma.

SYS-002 is infrastructure qualification only and is not the formal EXP-016
control. Neither arm may run an official benchmark.

## Magma treatment

Magma is applied only to one complete matrix per block: each transformer
layer's attention q/k/v/o projection weights and SwiGLU value/gate/out
projection weights. Recipe-v3 therefore has 9 × 7 = 63 masked blocks covering
44,605,440 parameters. The tied token embedding/output matrix, RMSNorm
parameters, and every other non-attention/non-MLP parameter are excluded and
receive dense AdamW. No trainable parameter is added.

For each target matrix at every optimizer update, sample one blockwise
Bernoulli mask m_t with survival probability p=0.5. Compute the alignment
score after AdamW has incorporated the current gradient in its dense first
moment:

`score_t = sigmoid(cos(mu_t, g_t) / tau)`, with tau=2.0,

and update the per-block scalar:

`s_t = 0.9 * s_(t-1) + 0.1 * score_t`.

This AdamW operationalization is explicit: the base AdamW step first updates
all dense optimizer moments and produces its full parameter delta, including
decoupled weight decay. Magma then replaces the selected block with:

`theta_final = theta_old + s_t * m_t * (theta_adamw - theta_old)`.

Thus a masked block does not move for that update, while its AdamW first and
second moments still update densely. Excluded parameters retain their ordinary
dense AdamW delta. The paper's main table uses Adam rather than this exact
AdamW+WSD system; this is a faithful, declared adaptation rather than a claim
of identical paper conditions.

Masks use a dedicated torch.Generator seeded 42. It does not advance global,
model, or data RNG state. Each checkpoint stores the exact block name/order,
Magma settings, alignment EMA scalars, Magma generator state, optimizer state,
WSD state, model state, global RNG state, and data cursor. Resume must reject
a different Magma configuration or block mapping.

## Preregistered decision rule

Magma capability passes only if all conditions hold:

- Magma combined NLL <= contemporaneous control combined NLL - 0.010.
- Magma General NLL <= control General NLL + 0.020.
- Magma Edu NLL <= control Edu NLL + 0.020.

Operational efficiency requires mean training tokens/s degradation versus the
contemporaneous control to be <=10%. A capability pass with >10% degradation
is QUALITY_PASS / EFFICIENCY_FAIL and cannot be promoted until an
implementation-only optimization preserves numerical semantics. A capability
failure rejects Magma with no p/tau/smoothing/granularity/module/LR tuning.
A capability pass does not authorize official benchmarks; any seed-43 paired
confirmation is a separate decision.

## Integrity and safety

Before launch, verify the parameter count, Schedule-A and stream hashes,
frozen validation hashes, fresh initialization, exact WSD endpoints,
checkpoint/resume semantics, and no benchmark invocation. Abort any GPU smoke
or future arm on GPU temperature >=85 C, CUDA/runtime error, or nonfinite
loss. The performance smoke is infrastructure-only and must not contribute
model-selection evidence.
