# EXP-003 FineWeb-Edu Data-Quality Ablation

Training completed at source commit `95b782156220c6747f02a526dd7fb64d182e8eb3` with the fixed 8,392,960-parameter model, frozen tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, 9,156 optimizer updates, and 300,023,808 prediction tokens. The local final checkpoint is `artifacts/exp003-full/run/checkpoints/checkpoint-step-9156.pt`; it must not be resumed or modified. Data manifest SHA-256: `b209eca3fd4b7a1e5ddffae5e2fe71962dbef3341150b9faf42ec2f5898e2226`.

Final train loss was 3.6129820346832275. Mean throughput was 315124.69830428046 tokens/s and wall time was 958.6512514999922 s. The run used seed 42; context 512; physical microbatch 32 with accumulation 2; BF16 autocast with FP32 parameters; AdamW (.9, .95, eps 1e-8); matrix-only weight decay 0.1; LR 6e-4 to 6e-5; 100-step warmup; and a 9,156-step cosine schedule.

| Step | Prediction tokens | general_validation loss / PPL | edu_validation loss / PPL |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 9.073967218399048 / 8725.16988899444 | 9.069507122039795 / 8686.341444207374 |
| 3052 | 100,007,936 | 4.409337937831879 / 82.21501403128084 | 3.956002175807953 / 52.24802942647881 |
| 6104 | 200,015,872 | 4.137368530035019 / 62.637774963517515 | 3.6481595933437347 / 38.40392213628989 |
| 9156 | 300,023,808 | 4.043534308671951 / 57.027539978663555 | 3.5508443117141724 / 34.84272319140925 |

The frozen EXP-002 final checkpoint measured 3.8954159319400787 on `general_validation` and 3.738632768392563 on `edu_validation`. EXP-003 therefore changed general validation by +0.1481183767318724 nats and educational validation by -0.1877884566783906 nats. FineWeb-Edu strongly improved educational modeling by the predeclared >0.10-nat criterion, but exceeded the predeclared +0.05-nat general-loss regression limit.

Scientific decision: reject pure FineWeb-Edu as the sole training corpus. The result supports a separately controlled mixture-preparation stage; it does not justify post-hoc tuning, benchmark evaluation, or modification of this checkpoint.
