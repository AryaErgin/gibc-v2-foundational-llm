# EXP-005B Promoted Final-Checkpoint Evaluation

Status: complete. Only the final EXP-005B checkpoint was evaluated: `artifacts/exp005b-full/run/checkpoints/checkpoint-step-9156.pt`, SHA-256 `c1f4718fdfaf34ea43b300bd19f51cb1167552a5a0e0096df9c6c9b1eb0302d5`. EXP-005A was not evaluated.

## Protocol and provenance

- lm-evaluation-harness: `0.4.9.1`
- Task definitions: same pinned zero-shot/default definitions as EXP-004A
- `num_fewshot`: `0`
- Batch size: `16` for every task
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`
- Data manifest SHA-256: `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`

Each task was evaluated separately over its full available examples. Ignored local raw-result files persist the task, checkpoint, tokenizer through the adapter configuration, batch size, lm-eval version, few-shot count, wall seconds, and unmodified lm-eval result object.

## Raw metrics

| Task | Raw metric | Value | Delta versus EXP-004A |
| --- | --- | ---: | ---: |
| HellaSwag | `acc,none` | 0.26638119896434975 | |
| HellaSwag | `acc_stderr,none` | 0.004411624374176717 | |
| HellaSwag | `acc_norm,none` | 0.2681736705835491 | 0.0 |
| HellaSwag | `acc_norm_stderr,none` | 0.004421031403685238 | |
| ARC-Easy | `acc,none` | 0.31186868686868685 | |
| ARC-Easy | `acc_stderr,none` | 0.009505823345817656 | |
| ARC-Easy | `acc_norm,none` | 0.32491582491582494 | +0.01641414141414144 |
| ARC-Easy | `acc_norm_stderr,none` | 0.009610203604504817 | |
| PIQA | `acc,none` | 0.5652883569096845 | |
| PIQA | `acc_stderr,none` | 0.011565943814308855 | |
| PIQA | `acc_norm,none` | 0.5495103373231773 | +0.0043525571273123 |
| PIQA | `acc_norm_stderr,none` | 0.011608491028638188 | |
| WinoGrande | `acc,none` | 0.49171270718232046 | -0.01104972375690614 |
| WinoGrande | `acc_stderr,none` | 0.014050555322824192 | |
| WikiText | `word_perplexity,none` | 158.55633061459332 | |
| WikiText | `word_perplexity_stderr,none` | N/A | |
| WikiText | `byte_perplexity,none` | 2.57896429099718 | |
| WikiText | `byte_perplexity_stderr,none` | N/A | |
| WikiText | `bits_per_byte,none` | 1.3667917973388595 | -0.0833302536218889 |
| WikiText | `bits_per_byte_stderr,none` | N/A | |

Per-task wall seconds: HellaSwag `898.8432191999746`; ARC-Easy `44.15621879999526`; PIQA `66.59871570000541`; WinoGrande `22.32190409998293`; WikiText `492.92955260002054`. Total sequential task wall time: `1524.84961039997874` seconds.

## Interpretation

This promoted capacity-transfer evaluation shows lower WikiText BPB, higher ARC-Easy and PIQA normalized accuracy, unchanged HellaSwag normalized accuracy, and lower WinoGrande accuracy relative to EXP-004A. These are descriptive final-checkpoint results. They do not authorize architecture, mixture, learning-rate, tokenizer, or prompt-template tuning from small benchmark movements. EXP-005B remains the accepted Architecture Recipe v1 because of the predeclared internal-validation and efficiency result; EXP-006 remains unauthorized.
