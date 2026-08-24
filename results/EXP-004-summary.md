# EXP-004 Mixed FineWeb / FineWeb-Edu Data Ablation

The training source commit was `7edfcb03e44d3f6dd3dcfcae12644ba41ada44f5`. Status: `FULL HORIZON COMPLETE`. The fixed 8,392,960-parameter model completed 9,156 optimizer updates and 300,023,808 prediction tokens. Final train loss: 3.730197787284851. Data manifest SHA-256: `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`.

Final general validation loss/PPL: 3.9005910456180573 / 49.431656837295954. Final educational validation loss/PPL: 3.6293802559375763 / 37.68945156393033. Mean/final throughput: 305300.5078776953 / 308598.7237217538 tokens/s. Training wall time: 991.879451300003 s.

| Step | Prediction tokens | general_validation loss | edu_validation loss |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 9.073967218399048 | 9.069507122039795 |
| 3052 | 100,007,936 | 4.241046905517578 | 4.004450470209122 |
| 6104 | 200,015,872 | 3.9868789315223694 | 3.7299690544605255 |
| 9156 | 300,023,808 | 3.9005910456180573 | 3.6293802559375763 |

The actual globally deduplicated mixture contributed 200,017,577 FineWeb and 100,006,231 FineWeb-Edu prediction tokens. It selected 322,643 unique documents. Cross-source duplicates skipped: FineWeb 6, FineWeb-Edu 59. Intra-source duplicates skipped: FineWeb 14, FineWeb-Edu 1.

Relative to EXP-002, general validation changed by +0.005175113677978516 nats and educational validation changed by -0.10925251245498657 nats. EXP-004 met both predeclared thresholds: general <= 3.9454159319400787 and educational <= 3.708632768392563.

Scientific decision: accept the approximately 2:1 globally deduplicated FineWeb:FineWeb-Edu mixture as Data Recipe v1. It captured approximately 58.2% of pure FineWeb-Edu's educational-loss improvement while incurring approximately 3.5% of its general-loss regression relative to EXP-002. This establishes internal language-modeling behavior only; it does not claim benchmark or reasoning improvement.
