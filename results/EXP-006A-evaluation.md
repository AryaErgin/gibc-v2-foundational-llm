# EXP-006A Promoted Final-Checkpoint Evaluation

Status: complete. Only final checkpoint `artifacts/exp006-full/run/checkpoints/checkpoint-step-27468.pt` was evaluated. Checkpoint SHA-256: `a3566034223a0cd64b0212f0a9aed84192fbb93cd2def9636b6e69612fa1c896`.

## Protocol

- lm-evaluation-harness `0.4.9.1`; zero-shot (`num_fewshot=0`); batch size `16`
- Full available examples, each task separately, with unchanged task definitions and corrected CustomCausalLM semantics
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`
- EXP-006 data manifest SHA-256: `be8dd1674b4d993483a6986710829eb8a35aff9d62d7e0e307d61c1145153b17`
- Local ignored raw files preserve unmodified harness output, stderr fields, task, checkpoint/path, tokenizer, lm-eval version, batch, few-shot count, and measured wall seconds.

## Raw aggregate metrics

| Task | Raw metric | Value | Delta versus EXP-005B | Wall seconds |
| --- | --- | ---: | ---: | ---: |
| HellaSwag | `acc,none` | 0.2658832901812388 | | 896.02466309999 |
| HellaSwag | `acc_stderr,none` | 0.0044089948686501035 | | |
| HellaSwag | `acc_norm,none` | 0.2741485759808803 | +0.0059749053973312 | |
| HellaSwag | `acc_norm_stderr,none` | 0.004451725530626298 | | |
| ARC-Easy | `acc,none` | 0.3287037037037037 | | 46.74086700001499 |
| ARC-Easy | `acc_stderr,none` | 0.009638903167022166 | | |
| ARC-Easy | `acc_norm,none` | 0.3181818181818182 | -0.00673400673400674 | |
| ARC-Easy | `acc_norm_stderr,none` | 0.009557408782506372 | | |
| PIQA | `acc,none` | 0.5750816104461371 | | 65.49790159999975 |
| PIQA | `acc_stderr,none` | 0.01153354794665477 | | |
| PIQA | `acc_norm,none` | 0.5685527747551686 | +0.0190424374319913 | |
| PIQA | `acc_norm_stderr,none` | 0.01155565729886461 | | |
| WinoGrande | `acc,none` | 0.5074980268350434 | +0.01578531965272294 | 22.69744509999873 |
| WinoGrande | `acc_stderr,none` | 0.014050905521228584 | | |
| WikiText | `word_perplexity,none` | 113.3989299730652 | | 493.7604642000224 |
| WikiText | `word_perplexity_stderr,none` | N/A | | |
| WikiText | `byte_perplexity,none` | 2.4222677404790756 | | |
| WikiText | `byte_perplexity_stderr,none` | N/A | | |
| WikiText | `bits_per_byte,none` | 1.2763583392324693 | -0.0904334581063902 | |
| WikiText | `bits_per_byte_stderr,none` | N/A | | |

Total sequential evaluation wall time: `1524.7213410000259` seconds.

## Interpretation

This promoted horizon-transfer evaluation records higher normalized HellaSwag and PIQA accuracy, lower ARC-Easy normalized accuracy, higher WinoGrande accuracy, and lower WikiText BPB than EXP-005B. These small benchmark movements are descriptive final-checkpoint evidence only. They do not authorize tuning the horizon, architecture, data recipe, tokenizer, prompts, or learning rate.

The WikiText task emitted lm-eval's documented default-aggregation warnings for word/byte perplexity and bits-per-byte. The fields above are reported unchanged.
