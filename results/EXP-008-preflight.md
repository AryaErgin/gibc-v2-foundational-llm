# EXP-008A Near-Cap SwiGLU Preflight

Status: `PASS` for the authorized controlled EXP-008A run. No official benchmark was run.

Implementation commit: `c6638df2864cde16e91f3c00b705c35af92be69a`.

## Independent count and frozen controls

The real instantiated candidate has **49,860,480** trainable parameters, independently matching `8192*640 + 9*4*640*640 + 9*3*640*1728 + (9*2*640 + 640)`. This is 139,520 below the 50,000,000 cap.

The runner independently loaded only the exact frozen EXP-004 stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, and frozen general/Edu validations. It retained context 512; seed 42; BF16 autocast with FP32 parameters/optimizer state; AdamW 0.9/0.95/1e-8 with 0.1 matrix decay; clip 1.0; peak/min LR 6e-4/6e-5; 100-warmup / 9,156-step cosine schedule; and 32 sequences x 2 accumulation / 32,768 prediction tokens per update.

## Bounded run and resume

The initial dry run completed 60 updates / 1,966,080 prediction tokens with finite losses and dual validations. Mean throughput was 97,690.05 tok/s; final-update throughput was 100,377.73 tok/s. Peak allocated/reserved memory was 7,686,099,968 / 8,491,368,448 bytes (7.686 / 8.491 GB). The bounded runner wall time was 21.9338 s, including its required final validation/checkpoint work.

The step-60 checkpoint was loaded in a fresh process for exactly one update. It reached step 61, 1,998,848 prediction tokens, and `next_sequence_index=3904`, with finite train, general-validation, and Edu-validation losses. No physical-batch fallback was used.

This is engineering readiness evidence only; its short-run losses do not decide the GELU-versus-SwiGLU question. The one full equal-token candidate run is authorized. Official benchmarks and all training beyond 300,023,808 tokens remain unauthorized.
