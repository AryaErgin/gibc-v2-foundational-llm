# EXP-005 Architecture Preflight

Status: `PASS` for bounded engineering readiness only. Neither full 9,156-update / 300,023,808-token EXP-005A nor EXP-005B training is authorized by this record. No official benchmark evaluation was run.

Implementation commit: `10f67608569cb66d2765853b30b859998c8fd533`.

## Frozen controls

Both candidates read `artifacts/exp004-full/train-token-stream.uint16` directly; no stream was copied or rematerialized. The verified SHA-256 is `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, with 300,023,809 stored uint16 IDs / 300,023,808 prediction tokens / 585,984 sequential examples. The referenced EXP-004 manifest SHA-256 is `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`.

Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`. Frozen `general_validation` input/target SHA-256s: `f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7` / `2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e`. Frozen `edu_validation` input/target SHA-256s: `cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a` / `300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4`.

The fixed schedule remains 100 linear-warmup steps and cosine decay through step 9,156. Both bounded runs used seed 42, BF16 autocast with FP32 parameters, AdamW (.9, .95, eps 1e-8), 0.1 matrix decay, clip norm 1.0, 32 sequences x 2 accumulation, and 32,768 prediction tokens/update. The actual run logs show LR `5.999999999999999e-06` after update 1 and `0.00035999999999999997` after update 60 for both candidates, preserving the uncompressed 9,156-step horizon.

## Equal bounded preflights

Both candidates ran exactly 60 updates / 1,966,080 prediction tokens and saved a step-60 checkpoint. Each was then loaded by a fresh process for exactly one further update. The resumed checkpoint states were step 61, 1,998,848 prediction tokens, `next_sequence_index` 3,904, and scheduler step 61 for both candidates.

| Candidate | Config | Exact parameters | First/final train loss | Final general / edu validation loss | Mean/final throughput (tok/s) | Peak allocated / reserved bytes | Estimated update-only 300M runtime |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| EXP-005A deep/thin | d_model 256, 24 layers, 8 heads, head_dim 32, d_ff 1024 | 20,984,064 | 9.063342094421387 / 7.2807817459106445 | 7.245031237602234 / 7.278079390525818 | 120346.56168289184 / 117759.56393125361 | 6,677,613,056 / 7,879,000,064 | 2492.9985851241036 s (41.549976418735056 min) |
| EXP-005B wide/shallow | d_model 384, 10 layers, 12 heads, head_dim 32, d_ff 1536 | 20,848,512 | 9.093114852905273 / 7.120301008224487 | 7.09125143289566 / 7.121268689632416 | 154866.68586621148 / 157925.53093905473 | 4,892,792,320 / 5,555,355,648 | 1937.3037288289943 s (32.28839548048324 min) |

The full-runtime estimates are `300,023,808 / measured mean tok/s`; they exclude validation/checkpoint overhead and are not promises of end-to-end wall time.

## Stability and reproducibility observations

EXP-005A logged finite loss and gradient norm on all 60 updates. Gradient norm first/final/min/max/mean: 1.7644319534301758 / 0.5961177349090576 / 0.5961177349090576 / 1.8383326530456543 / 1.0882230579853058. EXP-005B likewise logged finite loss and gradient norm on all 60 updates: 2.2477123737335205 / 0.6381507515907288 / 0.3763597309589386 / 2.387416362762451 / 1.1035496667027473. No depth-related instability was observed in this bounded preflight.

For each candidate, resetting the global seed to 42 before construction produced bitwise-identical CPU model state dictionaries. Both candidates read the same first contiguous 64-sequence stream batch (shape `[64, 512]` for inputs and targets). This validates deterministic start and common sequential data ordering, not eventual architecture quality.

Short-run loss differences are not interpreted as EXP-005 results. The predeclared eventual primary metric remains `combined_validation_loss = (general_validation_loss + edu_validation_loss) / 2` after a separately authorized full horizon. If final combined losses differ by under 0.02 nats, loss alone does not establish architecture superiority.
