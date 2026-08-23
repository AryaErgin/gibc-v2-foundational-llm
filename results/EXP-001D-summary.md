# EXP-001D Final-Checkpoint Baseline Evaluation

Status: PASS. Final checkpoint only: `artifacts/exp001-full/checkpoints/checkpoint-step-3052.pt`. Raw machine-readable lm-eval outputs are local-only in `artifacts/exp001-full/eval/`.

Protocol: lm-eval 0.4.9.1; task names `hellaswag`, `arc_easy`, `piqa`, `winogrande`, and `wikitext`; zero-shot (`num_fewshot=0`); batch size 16 for every task; requested limit 1000000 (all available task samples). Tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`; data manifest SHA-256 `f1b2bd8c6dbd73b2e31c795103e2fc1088f123c14021f3c966df2e6333ebd730`.

- HellaSwag: `acc,none` 0.2621987651862179; `acc_stderr,none` 0.00438931274801214; `acc_norm,none` 0.2545309699263095; `acc_norm_stderr,none` 0.004347070019527483.
- ARC-Easy: `acc,none` 0.2777777777777778; `acc_stderr,none` 0.00919077990964993; `acc_norm,none` 0.27946127946127947; `acc_norm_stderr,none` 0.00920783814259724.
- PIQA: `acc,none` 0.5489662676822633; `acc_stderr,none` 0.01160974720073308; `acc_norm,none` 0.5261153427638737; `acc_norm_stderr,none` 0.011649900854263425.
- WinoGrande: `acc,none` 0.48855564325177586; `acc_stderr,none` 0.014048804199859325.
- WikiText-103 (`wikitext` held-out rolling likelihood): `word_perplexity,none` 606.6768421816571; `word_perplexity_stderr,none` N/A; `byte_perplexity,none` 3.314568115491924; `byte_perplexity_stderr,none` N/A; `bits_per_byte,none` 1.728820901037883; `bits_per_byte_stderr,none` N/A.

Evaluation engineering: `09fd935` batches exact causal scoring events and `f068c81` streams rolling events in fixed-size batches. Regression tests compare serial/reference and batched likelihoods and greedy flags across BPE boundaries, whitespace, empty context, long contexts/continuations, and rolling text beyond 512 tokens. Semantic deviations: none. No historical serial wall-clock baseline exists, so speedup is not reported.

Warning: the pre-existing evaluation CLI did not record task start timestamps; per-task and total wall runtime cannot be reconstructed from completed raw JSON without inventing measurements. HellaSwag/ARC-Easy/PIQA/WinoGrande/WikiText raw result files and completion timestamps are preserved locally. WikiText emitted its harness default-aggregation warnings for word/byte perplexity; these fields are reported unchanged. Internal FineWeb PPL is distinct from WikiText metrics.

Scientific interpretation: the baseline demonstrates above-chance PIQA accuracy, near-chance multiple-choice behavior on the other reported tasks, and high WikiText perplexity. These measurements are a single frozen-baseline capability profile, not grounds for post-hoc model changes or an EXP-002 proposal.
