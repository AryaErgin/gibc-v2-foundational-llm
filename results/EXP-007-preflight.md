# EXP-007 Near-Cap Architecture Preflight

Status: `PASS` for bounded engineering readiness only. Neither 9,156-update / 300,023,808-token EXP-007A nor EXP-007B training is authorized. No official EXP-007 benchmark was run.

Implementation commit: `08274c324c94e39e7698706323d43e84e1106b19`.

## Frozen controls

Both candidates read the exact EXP-004 train stream directly: SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`; 300,023,809 stored uint16 IDs; 300,023,808 prediction tokens; 585,984 sequential examples. No stream was copied or rematerialized. Data manifest SHA-256: `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`.

Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`. Frozen general validation input/target SHA-256: `f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7` / `2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e`. Frozen educational validation input/target SHA-256: `cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a` / `300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4`.

Both runs retained seed 42; BF16 autocast with FP32 parameters/optimizer state; AdamW (.9, .95, eps 1e-8); 0.1 matrix decay; clip 1.0; peak/min LR 6e-4/6e-5; 100 warmup updates; uncompressed 9,156-step cosine horizon; 32 sequences x 2 accumulation; and 32,768 prediction tokens/update.

## Equal 60-update preflights

| Candidate | Allocation / exact parameters | First/final train loss | Final general / educational validation loss | Mean/final throughput (tok/s) | Peak allocated / reserved bytes | Update-only 300M estimate |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-007A | d_model 608; 10 layers; 19 heads; d_ff 2432; 49,353,184 | 9.145676612854004 / 7.058586835861206 | 7.033175349235535 / 7.068580210208893 | 81686.12550787536 / 82891.46727177713 | 7,250,241,024 / 8,036,286,464 | 3672.885770192084 s (61.21476283653473 min) |
| EXP-007B | d_model 640; 9 layers; 20 heads; d_ff 2560; 49,491,840 | 9.163904190063477 / 7.032688617706299 | 7.006724953651428 / 7.035873651504517 | 84444.63102132562 / 85915.50034398763 | 6,955,832,320 / 7,763,656,704 | 3552.905665775626 s (59.21509442959377 min) |

The estimates are `300,023,808 / mean tok/s`, exclude validation/checkpoint overhead, and are not end-to-end wall-time promises.

## Resume and stability

Each initial dry run completed 60 updates / 1,966,080 prediction tokens and wrote a step-60 checkpoint. A fresh process resumed each checkpoint for exactly one update. EXP-007A and EXP-007B both reached step `61`, `1,998,848` prediction tokens, and `next_sequence_index=3904`, with finite train loss and dual validation losses. Both candidates fit the required initial 32x2 physical batch; no fallback batch was used.

These short-run losses are not architecture-selection evidence. The eventual predeclared decision uses final full-horizon combined validation loss, with 0.02 nats as the engineering-tie threshold. Full training and official evaluation remain unauthorized.
