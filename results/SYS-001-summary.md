# SYS-001 Near-Cap Systems Optimization Ladder

Status: `COMPLETE`. This is a bounded systems study only: no official benchmark, checkpoint promotion, or long-horizon training was run.

## Fixed systems and scientific controls

All measurements used the Windows native environment, OMEN Performance mode, and original AC power connected (manually established before the run). Software/hardware observed: Python 3.11.9, PyTorch 2.13.0+cu132, CUDA 13.2, NVIDIA GeForce RTX 5090 Laptop GPU, BF16 supported. Every phase retained EXP-007B exactly: 49,491,840 trainable parameters; vocabulary 8,192; d_model 640; 9 layers; 20 heads of 32; d_ff 2,560; context 512; frozen EXP-004 tokenizer/data/order; seed 42; 32,768 effective prediction tokens/update; AdamW hyperparameters; BF16 autocast; objective; clipping; and uncompressed 100-warmup / 9,156-step LR schedule.

Each timing result is a freshly initialized, sequential 200-update bounded run: 100 warmup updates followed by 100 timed updates. The two floating-point-order-changing phases also received independent fresh 500-update dual-validation comparisons.

## Hardware/software baseline

The unchanged original EXP-007B 32x2 runner recorded 110,434.48 tok/s over post-warmup updates 101–200 (3,276,800 tokens in 29.6721 s). Its complete 200-update bounded invocation, including the initial/final dual validations and final checkpoint, took 60.8522 s. Peak allocated/reserved memory was 6,955,832,320 / 7,763,656,704 bytes (6.956 / 7.764 GB).

One-hertz `nvidia-smi` samples during steady training showed 96–99% GPU utilization, approximately 153–166 W draw, P0, 73 C peak temperature, and about 9.0 GB driver-reported memory use. This is materially faster than the prior Balanced-mode EXP-007B full-run mean of 85,060.71 tok/s: +25,373.77 tok/s, or +29.83%. The mode/power-state comparison is operational rather than a controlled scientific intervention because only the current Performance/AC state is directly observed here.

## Sequential ladder results

| Phase | Systems change relative to predecessor | 100 timed updates tok/s | Peak allocated / reserved GB | Result |
| --- | --- | ---: | ---: | --- |
| Baseline | Existing 32x2 runner with per-update synchronization | 110,434.48 | 6.956 / 7.764 | Reference |
| 64x1 | 64 sequences x 1 accumulation; same 32,768 effective tokens | 100,532.07 | 12.749 / 14.099 | Slower and higher memory |
| Production timing | 64x1 with terminal-window CUDA synchronization only | 101,056.77 | 12.749 / 14.099 | +0.52% over synchronized 64x1; still below baseline |
| RoPE cache/reuse | Reuse non-persistent RoPE sine/cosine tables | 101,154.90 | 12.749 / 14.099 | +0.10% over prior phase; still below baseline |
| SDPA audit | Profile actual SDPA operation at the same 512-token workload | 101,061.40 | 12.749 / 14.141 | Actual backend: memory-efficient SDPA |
| Fused AdamW | CUDA fused AdamW, after live support probe | 102,324.89 | 12.749 / 14.099 | Supported; still below baseline |

The profiler recorded `aten::_scaled_dot_product_efficient_attention` and its backward operator. Flash, memory-efficient, and math SDPA flags were enabled, but the audit reports the operator actually executed, not merely eligible backends.

## Floating-order stability comparisons

| Comparison | Reference / candidate final combined validation loss after 500 updates | Candidate minus reference |
| --- | ---: | ---: |
| 32x2 vs 64x1 | 5.2078539133 / 5.2066164613 | -0.0012374520 |
| RoPE-cache 64x1 vs fused-AdamW 64x1 | 5.2071888447 / 5.2063994408 | -0.0007894039 |

These are bounded stability/validation controls, not architecture-selection results or a new loss threshold. They provide no evidence for a long-horizon change. Since every tested 64x1 combination was slower and required substantially more peak memory than the Performance-mode 32x2 reference, SYS-001 retains the original EXP-007B 32x2 training path. The harness keeps all phases available for reproducible future systems retests; it does not enable a training run by itself.
