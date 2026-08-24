# EXP-004A Promoted Final-Checkpoint Evaluation

Status: complete. This promoted evaluation used only the final EXP-004 checkpoint, `artifacts/exp004-full/run/checkpoints/checkpoint-step-9156.pt`, produced by training source commit `7edfcb03e44d3f6dd3dcfcae12644ba41ada44f5`. The checkpoint and raw evaluation outputs remain local-only artifacts.

## Protocol and provenance

- lm-evaluation-harness: `0.4.9.1`
- Task definitions: the same pinned zero-shot/default task definitions used for EXP-002A
- `num_fewshot`: `0`
- Batch size: `16` for every task
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`
- EXP-004 data manifest SHA-256: `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`
- Checkpoint: `artifacts/exp004-full/run/checkpoints/checkpoint-step-9156.pt`

Each task was invoked separately over its full available examples. Persisted local raw-result files contain the task, checkpoint, batch size, lm-eval version, few-shot count, wall-clock seconds, and unmodified lm-eval result object.

## Raw task metrics

| Task | Raw metric | Value | EXP-002 comparison |
| --- | --- | ---: | ---: |
| HellaSwag | `acc,none` | 0.2638916550487951 | |
| HellaSwag | `acc_stderr,none` | 0.004398404992933866 | |
| HellaSwag | `acc_norm,none` | 0.2681736705835491 | +0.00438159729137624 |
| HellaSwag | `acc_norm_stderr,none` | 0.004421031403685238 | |
| ARC-Easy | `acc,none` | 0.30176767676767674 | |
| ARC-Easy | `acc_stderr,none` | 0.009418994158522532 | |
| ARC-Easy | `acc_norm,none` | 0.3085016835016835 | +0.01725589225589225 |
| ARC-Easy | `acc_norm_stderr,none` | 0.009477472342978112 | |
| PIQA | `acc,none` | 0.5544069640914037 | |
| PIQA | `acc_stderr,none` | 0.01159655408098765 | |
| PIQA | `acc_norm,none` | 0.545157780195865 | +0.0043525571273123 |
| PIQA | `acc_norm_stderr,none` | 0.011618148261187403 | |
| WinoGrande | `acc,none` | 0.5027624309392266 | +0.0023677979479086 |
| WinoGrande | `acc_stderr,none` | 0.014052271211616441 | |
| WikiText | `word_perplexity,none` | 215.93541845703254 | |
| WikiText | `word_perplexity_stderr,none` | N/A | |
| WikiText | `byte_perplexity,none` | 2.732311655328376 | |
| WikiText | `byte_perplexity_stderr,none` | N/A | |
| WikiText | `bits_per_byte,none` | 1.4501220509607484 | -0.0232674028573234 |
| WikiText | `bits_per_byte_stderr,none` | N/A | |

Per-task wall time, in seconds: HellaSwag `755.0133916999912`; ARC-Easy `50.41123290001997`; PIQA `74.78981489999569`; WinoGrande `27.38254359998973`; WikiText `272.10187410001527`. Total sequential wall time: `1179.69885720001186` seconds.

## Interpretation

This is a promoted milestone transfer check for frozen Data Recipe v1, not a mixture-ratio optimization. The mixture improved the reported normalized HellaSwag, ARC-Easy, and PIQA metrics and WinoGrande accuracy relative to EXP-002, while WikiText bits per byte decreased. These benchmark movements are descriptive final-checkpoint evidence; they do not establish reasoning capability or justify data-recipe tuning from small fluctuations. Do not start EXP-005 from this result alone.
