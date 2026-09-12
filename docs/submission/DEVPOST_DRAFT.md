# Devpost draft — GIBC V2 Track 01

## Title

49.86M Parameters, 7.2B Tokens: A Laptop-Trained Language Model with an Auditable Experiment Trail

## One-line summary

A from-scratch base language model trained on one laptop GPU, with frozen held-out results and a transparent record of what worked—and what did not.

## Inspiration / problem

Under a 50M-parameter cap, the challenge was not to assemble the largest collection of modern methods. It was to find a defensible recipe, test interventions under matched controls, and invest the remaining budget in unique-data training. Our own short-horizon rankings repeatedly weakened or reversed at longer horizons.

## What it does

EXP-020 is a decoder-only base language model with 49,860,480 trainable parameters, trained from scratch on 7,199,981,568 prediction tokens. It is not instruction-tuned and is not presented as a general-purpose assistant. Required official scores are final reporting, not another optimization loop.

## Evidence mapped to the five judging criteria

| Criterion | Claim supported here | Evidence to show | Limit |
|---|---|---|---|
| Perplexity & Accuracy | WikiText-103 token PPL 31.783; HellaSwag acc_norm 30.163%; ARC-Easy acc_norm 39.141%; PIQA acc_norm 60.446%; WinoGrande acc 48.777% | [RESULTS](../../RESULTS.md), task hashes and metric definitions in [digest](../../results/exp020-submission-evidence.json) | Single final seed; no SOTA or statistical-significance claim |
| Reasoning Performance | HellaSwag/ARC-Easy improve versus EXP-012; PIQA is roughly stable; WinoGrande regresses | [Progression figure](../assets/exp020-devpost/03-benchmark-development.png), explicit +1.404/+2.694/+0.218/−1.579 percentage-point changes | These task scores are limited proxies, not evidence of universal reasoning gains |
| Training Efficiency | Final 7.2B run completes in 37.94h on one RTX 5090 Laptop GPU; 8.680GB peak reserved VRAM | [Final summary](../../results/EXP-020-summary.md), wall-time/active/paced distinctions | Final run only, not all experiments/data/evaluation; no measured energy claim |
| Innovation | Controlled horizon-dependent intervention evaluation, preregistered gates, retained negative results, benchmark-blind final checkpoint selection, exact contamination/provenance controls | [Decision figure](../assets/exp020-devpost/02-horizon-dependent-findings.png), [experiment log](../../EXPERIMENT_LOG.md), preregistrations | Standard Transformer components and named methods are credited, not claimed as inventions |
| Documentation & Demo | Readable architecture, source counts, artifact hashes, reproducible figures and offline verification | [README](../../README.md), [pipeline](../assets/exp020-devpost/06-training-evaluation-pipeline.png), [storyboard](DEMO_STORYBOARD.md) | Public weights and hash verification complete; competition source is published at the frozen tag; video remains a human action |

## How we built it

Recipe-v3 uses an 8,192-token scratch byte-level BPE vocabulary, nine width-640 blocks, twenty heads, SwiGLU, RoPE, pre-RMSNorm and tied embeddings. Ordinary AdamW and a cosine schedule span all 219,726 updates. The deterministic 2:1 FineWeb/FineWeb-Edu corpus has global exact-document deduplication and normalized 13-gram SHA-256 exclusion screening. Its EXP-011/012 prefixes reproduce the historical byte identities exactly.

The final model was selected using frozen General/Edu validation before its official benchmark scores. Evaluation was CPU FP32, zero-shot, batch 16 and context 512, with a separately pinned held-out WikiText-103 rolling evaluator.

## Challenges / what we learned

Thermal aborts and a pathologically slow initial data build are retained as systems incidents—not hidden and not called scientific failures. Combined operational thermal controls and fixed pacing supported the completed run; no individual control receives causal credit.

WSD's replicated proxy improvement missed its longer-horizon gate. QK-Norm improved 1.5B validation by about 0.0122 nats but missed its preregistered 0.015 gate. CWD's intermediate improvement reversed to an endpoint regression. All three are excluded from the final recipe. This is project-specific evidence about horizon sensitivity, not a universal rejection of those methods.

## Honest limitations / what is next

The final benchmark profile is mixed, especially WinoGrande. Exact n-gram filtering is not proof of absence of all semantic overlap. The full evaluator source commit is resolved; PIQA evaluation/exclusion snapshot equivalence remains unproved. There is no fresh training, tuning or checkpoint change planned from these scores. The [frozen model is public](https://huggingface.co/AryaErgin/gibc-v2-exp020); the frozen source is published; next actions are human review and video recording—not another experiment.

## Built with / assistance

Python, PyTorch, CUDA, NumPy, Hugging Face datasets/tokenizers, EleutherAI lm-eval, SQLite, WSL2, Git and Pillow for submission plots. Credit original datasets, benchmarks and component papers via [SOURCE_LEDGER](../../SOURCE_LEDGER.md) and [CODE_ATTRIBUTION](../../CODE_ATTRIBUTION.md).

ChatGPT, Deep Research and Codex materially supported research, explanation/learning, planning, coding/review/debugging, analysis, writing and submission packaging. The submitted neural model itself was trained from scratch; it is not an API wrapper around an AI assistant. [Full AI disclosure](../../AI_ASSISTANCE.md).

**Before submission:** replace no numbers; use [the permanent model URL](https://huggingface.co/AryaErgin/gibc-v2-exp020), publish the reviewed source revision and attach the human-recorded video. No fabricated demo output or external leaderboard comparison.
