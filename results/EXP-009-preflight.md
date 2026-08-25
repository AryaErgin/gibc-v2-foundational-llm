# EXP-009 Learning-Rate Calibration Preflight

Status: `PASS` for the two authorized controlled full-horizon candidates. These are bounded engineering/stability checks, not a learning-rate selection result. No official benchmark was run.

Both configurations were loaded through the strict EXP-009 guards at source commit `89297b19d25894a27c270616a7799d3785c62a47`. Each instantiated exactly **49,860,480** trainable parameters and independently validated the immutable EXP-004 stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, frozen general-validation inputs SHA-256 `f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7`, and frozen Edu-validation inputs SHA-256 `cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a`.

| Candidate | Peak / min LR | Step-60 train / general / Edu loss | Mean / final tok/s | Peak allocated / reserved bytes | 60-step wall time |
| --- | --- | --- | --- | --- | --- |
| EXP-009A | `4e-4` / `4e-5` | 7.2943713665 / 7.2737907767 / 7.3082421422 | 107302.1105 / 107017.1548 | 7,686,099,968 / 8,491,368,448 | 19.6234 s |
| EXP-009B | `8e-4` / `8e-5` | 7.2602643967 / 7.2307779789 / 7.2620210052 | 106561.6908 / 107186.6545 | 7,686,099,968 / 8,491,368,448 | 19.7555 s |

Each invocation retained native Windows, OMEN Performance mode, AC power, BF16 autocast, FP32 model/optimizer state, 32 sequences x 2 accumulation, 32,768 prediction tokens/update, and the fixed schedule shape. Finite preflight loss is intentionally not used to select a candidate.

Fresh-process resume was also exercised from each step-60 checkpoint for exactly one update. Both reached step 61, 1,998,848 prediction tokens, and `next_sequence_index=3904` with finite losses:

| Candidate | Step-61 train / general / Edu loss | Resume peak allocated / reserved bytes | Resume wall time |
| --- | --- | --- | --- |
| EXP-009A | 7.2880954742 / 7.2713914514 / 7.3046608567 | 7,687,993,856 / 8,363,442,176 | 1.7148 s |
| EXP-009B | 7.2346787453 / 7.2219830751 / 7.2491734624 | 7,687,993,856 / 8,363,442,176 | 1.7398 s |

The checkpoint/optimizer/RNG/cursor path is therefore ready for the authorized, independently sequential 9,156-update runs. No early-loss decision or official evaluation is authorized.
