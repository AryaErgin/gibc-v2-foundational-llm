# Architecture — frozen EXP-020

EXP-020 uses Recipe v3 with **49,860,480 trainable parameters**, 139,520 below the 50,000,000 cap. Its checkpoint SHA-256 is `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`. The [submission evidence](results/exp020-submission-evidence.json) independently counts stored FP32 tensors with tied storage counted once; historical [Recipe-v3 accounting](results/exp012-parameter-count.json) agrees.

| Component | Frozen setting |
|---|---|
| Decoder blocks / width | 9 / 640 |
| Vocabulary / context | 8,192 / 512 |
| Attention | 20 heads × 32 dimensions; causal native PyTorch SDPA |
| Position | RoPE, theta 10,000; adjacent-pair rotation over all 32 head dimensions |
| MLP | Bias-free SwiGLU: SiLU gate; intermediate width 1,728 |
| Normalization | Pre-RMSNorm, epsilon 1e-5; final RMSNorm |
| Embeddings / output | Shared weight matrix |
| Bias / dropout | None / 0 |
| Initialization | Fresh seed 42; existing Normal(0, 0.02) linear/embedding initialization; norm scales 1 |
| Excluded methods | QK-Norm off; CWD off; no WSD, LLR, curriculum or Magma in final run |

Each block applies pre-normalized causal attention plus a residual connection, then pre-normalized SwiGLU plus a residual connection. The final norm precedes the tied output projection. RoPE is parameter-free. SDPA retains its standard head-dimension scale.

## Exact budget

| Tensor family | Calculation | Parameters |
|---|---|---:|
| Token embedding | 8,192 × 640 | 5,242,880 |
| Q/K/V/output attention projections | 9 × 4 × 640 × 640 | 14,745,600 |
| SwiGLU projections | 9 × 3 × 640 × 1,728 | 29,859,840 |
| RMSNorm scales | (9 × 2 + 1) × 640 | 12,160 |
| Tied output additional | Shares embedding | 0 |
| **Total** | | **49,860,480** |

Implementation: [model.py](src/gibc_llm/model.py). Configuration: [frozen EXP-020 YAML](configs/exp020-final-7p2b-cosine.yaml). Independent command: `python scripts/count_parameters.py --config configs/exp020-final-7p2b-cosine.yaml --expected-total 49860480 --json`.

## Optimization

Ordinary AdamW: betas 0.9/0.95, epsilon 1e-8; weight decay 0.1 on matrix parameters including tied embedding, no decay on norm scales; gradient clipping 1.0. Peak/minimum LR 6e-4/6e-5, warmup 100 updates, cosine over the entire 219,726-update horizon. BF16 forward/autocast; FP32 parameters and optimizer state. Microbatch 32, accumulation 2, 32,768 prediction tokens per update. Operational pacing adds wall time, not scheduler steps.

## Evidence for the allocation

EXP-001 began at 8,392,960 parameters (256 width, 8 blocks, GELU). EXP-007B selected the 49,491,840-parameter GELU Recipe v2 by a preregistered engineering tiebreak. EXP-008's SwiGLU allocation improved Combined internal NLL by 0.0300662965 and became Recipe v3. EXP-009 retained the original LR under its tie rule; EXP-010's alternative depth/width allocation lost the engineering tiebreak. EXP-020 kept Recipe v3 unchanged and increased the fresh cosine training horizon.

These components are established methods, not inventions of this project. Their sources are credited in [SOURCE_LEDGER.md](SOURCE_LEDGER.md) and [CODE_ATTRIBUTION.md](CODE_ATTRIBUTION.md).
