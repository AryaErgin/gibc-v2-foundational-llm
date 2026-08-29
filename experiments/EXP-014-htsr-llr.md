# EXP-014 — HT-SR Layerwise Learning Rate

**Status:** preregistered implementation preflight; no training started.

## Question and control

At fixed Recipe-v3 architecture, Data Recipe v1 stream, token budget, seed,
AdamW and promoted WSD schedule, can HT-SR LLR improve held-out quality or
convergence efficiency? The frozen seed-42 WSD control is General
`3.5252154767513275`, Edu `3.2107785046100616`, combined
`3.3679969906806946` at 9,156 updates / 300,023,808 prediction tokens.

## Upstream provenance and adaptation

Method reference: He et al., *One LR Doesn't Fit All*, arXiv:2605.22297v3
(27 May 2026), with `hed-ucas/Layer-wise-Learning-Rate`, main inspected on
29 August 2026. The upstream repository is an OLMo/AlphaDecay pre-release;
EXP-014 does not import it or inherit its OLMo scheduler, logging, distributed,
or optimizer assumptions. Its source revision must be resolved to a full Git
commit before a training launch; network access was unavailable during this
preflight, so this remains a launch gate.

The adaptation uses the paper/source's weight-spectrum PL_Alpha_Hill metric,
positive `tb_linear_map`, upper-bound embedding treatment, and linear soft
switch. For each 2-D named matrix in Recipe-v3 there is one AdamW group:
each block's `attention.{q,k,v,o}_proj`, `mlp.{value,gate,out}_proj`; and the
tied `token_embedding.weight`/output matrix as exactly one upper-bound group.
All 1-D RMSNorm scales remain in one no-decay, multiplier-1 group because the
author code's layernorm adaptation is opt-in and is not enabled in its LLM
command. No parameter is added.

## Candidate schedule

Global WSD is unchanged: updates 1–100 linear warmup to `6e-4`, 101–8240
stable, 8241–9156 cosine cooldown, final `6e-5`. At every update,
`actual_group_lr = global_wsd_lr * multiplier`.

PL_Alpha_Hill uses squared singular values of each 2-D matrix, sorted
descending, with Hill `k=floor(n/2)` and threshold `lambda[k]`. Alpha is mapped
linearly and positively over `[1,5]`. Recompute cadence is updates 100, 200,
..., 1800 (the last complete 100-update cadence inside 20% of 9,156). The
last 50-step soft transition runs 1800–1850, then multipliers freeze. The
frozen multipliers continue to scale the instantaneous WSD LR through
cooldown. Embedding/output is always multiplier 5; RMSNorm remains 1.

## Fixed protocol and evidence

Fresh seed 42, 49,860,480 trainable parameters, tokenizer SHA-256
`c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, stream
SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`,
frozen General/Edu validation, BF16 autocast/FP32 params and optimizer,
32×2 physical batch, and 300,023,808 non-cycled prediction tokens remain
unchanged. Checkpoints include model, AdamW, WSD, LLR state, RNG and cursor.
At recomputes telemetry records alpha, multiplier, actual LR, parameter norm,
and spectral wall seconds; run telemetry retains throughput and peak VRAM.
No official benchmark or benchmark example is part of EXP-014.

## Decision

Discovery capability pass requires candidate combined <= `3.3511570057272912`
(at least 0.5% relative NLL reduction) and neither domain worse by >0.020 nat.
Otherwise stop after seed 42 without tuning any LLR knob or starting another
optimizer experiment. Tokens-to-equal-loss analysis may use only observed
logged trajectories; sparse cadence must be reported as a limitation.
