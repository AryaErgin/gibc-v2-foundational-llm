# EXP-007 Near-Cap Architecture Comparison

Status: `COMPLETE`. Both 9,156-update runs finished under the frozen EXP-004 Data Recipe v1 stream. No official benchmark was run.

Implementation/training commit: `9523e744082fb4dedbcd9856964cb625bde08b0c`.

## Frozen shared controls

Both candidates retained the exact EXP-004 non-cycled training stream (SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`), tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, dual held-out validation tensors, source order, seed 42, context 512, BF16 autocast with FP32 parameters/optimizer state, AdamW (betas 0.9/0.95, eps 1e-8, matrix decay 0.1), clip 1.0, and the 100-warmup / 9,156-step cosine schedule at peak/min LR 6e-4/6e-5. Each update used the fixed 32 sequences x 2 accumulation, or 32,768 effective prediction tokens. Each completed 300,023,808 prediction tokens.

## Final equal-token comparison

| Candidate | Exact parameters | Final general / educational validation loss | Final combined validation loss | Mean throughput | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| EXP-007A | 49,353,184 | 3.5797968507 / 3.2717535496 | 3.4257752001 | 82,068.63 tok/s | 7.250 GB |
| EXP-007B | 49,491,840 | 3.5819328129 / 3.2808714211 | 3.4314021170 | 85,060.71 tok/s | 6.956 GB |

The primary metric is `(general_validation_loss + educational_validation_loss) / 2`. B-A is `+0.0056269169` nats. Although EXP-007A was numerically lower, this difference is within the predeclared `0.02`-nat engineering-tie threshold.

## Decision

This is an engineering tie. Apply the predeclared efficiency tiebreak and select EXP-007B: it delivered higher mean throughput and lower peak allocated memory while retaining the same frozen data, training, and evaluation controls.

Near-Cap Architecture Recipe v2 is now frozen as: `vocab=8192`, `d_model=640`, `n_layers=9`, `n_heads=20`, `head_dim=32`, `d_ff=2560`, `params=49,491,840`.

EXP-007A is not selected; its numerically lower loss is recorded above and is not hidden. This result does not authorize official benchmarks, architecture/data changes, FP8, reduced context/model/tokens, 1B+ training, or any change to Windows security settings.
