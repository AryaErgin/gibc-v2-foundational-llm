# EXP-010A SwiGLU Depth/Width Preflight

Status: `PASS` for the authorized full-horizon EXP-010A candidate. This is bounded stability/readiness evidence only; its losses do not decide the architecture comparison.

The candidate was independently instantiated at **49,985,504** trainable parameters. Strict artifact loading verified the frozen EXP-004 manifest SHA-256 `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`, stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, and frozen general/Edu validation tensors. The invocation retained native Windows, OMEN Performance mode, AC power, BF16 autocast with FP32 model/optimizer state, seed 42, context 512, AdamW controls, peak/min LR 6e-4/6e-5, and physical batch 32 x 2.

The initial bounded run completed 60 updates / 1,966,080 prediction tokens at source commit `430d3757ff9f34d43e4bc28e347e42b60fc46853`: step-60 train/general/Edu loss `7.2849016190 / 7.2598409057 / 7.2911592722`; mean/final throughput `96,890.7149 / 97,842.9347` tok/s; peak allocated/reserved memory `8,061,879,808 / 8,835,301,376` bytes; wall time `21.8705` s; and exact cursor `3,840`.

A fresh process restored the step-60 checkpoint and completed exactly one update to step 61, 1,998,848 prediction tokens, and cursor `3,904`. Its train/general/Edu losses were finite (`7.2706634998 / 7.2545218468 / 7.2846311927`) and peak allocated/reserved memory was `8,079,410,176 / 8,719,958,016` bytes. The checkpoint, optimizer, RNG, schedule, and sequential data cursor path therefore pass the preflight requirement.
