# EXP-002A Final-Checkpoint Evaluation

Status: complete. Protocol: lm-eval 0.4.9.1, zero-shot (`num_fewshot=0`), batch size 16, final checkpoint `artifacts/exp002-full/run/checkpoints/checkpoint-step-9156.pt`, frozen tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, and data manifest SHA-256 `ab36ea1ffae92e3e93d3a57c4d369240c33456da93d7672919874ee157033779`. Raw JSON remains local-only in `artifacts/exp002-full/eval/`.

- HellaSwag: `acc,none` 0.26269667396932883; `acc_stderr,none` 0.0043919956375421225; `acc_norm,none` 0.26379207329217286; `acc_norm_stderr,none` 0.004397872471854957. Delta versus EXP-001 `acc_norm`: +0.00926110336586336.
- ARC-Easy: `acc,none` 0.2845117845117845; `acc_stderr,none` 0.00925805092561882; `acc_norm,none` 0.29124579124579125; `acc_norm_stderr,none` 0.009322788837938854. Delta versus EXP-001 `acc_norm`: +0.01178451178451178.
- PIQA: `acc,none` 0.5544069640914037; `acc_stderr,none` 0.01159655408098765; `acc_norm,none` 0.5408052230685527; `acc_norm_stderr,none` 0.011626910523588569. Delta versus EXP-001 `acc_norm`: +0.014689880304679.
- WinoGrande: `acc,none` 0.500394632991318; `acc_stderr,none` 0.014052481306049516. Delta versus EXP-001 `acc`: +0.011838989739542123.
- WikiText-103: `word_perplexity,none` 235.38480417301673; `word_perplexity_stderr,none` N/A; `byte_perplexity,none` 2.7767349131686996; `byte_perplexity_stderr,none` N/A; `bits_per_byte,none` 1.4733894538180718; `bits_per_byte_stderr,none` N/A. BPB delta versus EXP-001: -0.2554314472198112.

The task wrapper was instrumented to print task wall-clock time. The execution environment detached each task after initial output, so these printed values were not retained in raw JSON; completed-task timing cannot be reconstructed exactly and is deliberately not fabricated. Future wrapper output should persist `wall_seconds` beside raw results.

Scientific interpretation: the 300M run substantially improved the frozen FineWeb language-modeling metric and WikiText compression/perplexity metrics. Multiple-choice benchmarks improved modestly across all reported comparison metrics, with PIQA remaining the strongest of these tasks. This single controlled horizon comparison does not justify a new experiment or post-hoc tuning.
