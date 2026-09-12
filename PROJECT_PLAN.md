# Project status — model development and evaluation frozen

**Current competition model: EXP-020.** Full training completed at step 219,726, 7,199,981,568 prediction tokens and 49,860,480 parameters. Official evaluation completed on 2026-09-08. Model development and method hunting are closed.

## Completed scientific sequence

| Stage | Outcome |
|---|---|
| Baseline and data controls | EXP-001 measurement system; EXP-004 promoted 2:1 FineWeb/FineWeb-Edu |
| Parameter allocation | EXP-007 near-cap allocation; EXP-008 Recipe v3 SwiGLU; EXP-009/010 retain recipe |
| Horizon calibration | Fresh EXP-011 1.5B and EXP-012 2.4B cosine runs |
| Controlled methods | WSD initially passed at 300M, failed 2.4B promotion; LLR, curriculum, Magma, QK-Norm and CWD rejected for the final recipe |
| Final corpus | Deterministic 7.2B rebuild from zero; historical prefix hashes passed |
| Final model | One fresh EXP-020 ordinary-AdamW/cosine run; fixed operational pacing |
| Selection and reporting | Internal-validation selection, then five required official benchmarks |
| Submission packaging | Evidence digest, documentation, seven native 3:2 judge figures, preserved scientific figures, Devpost draft and timed storyboard |

The definitive current results are [RESULTS.md](RESULTS.md). Earlier authorizations in the historical [experiment log](EXPERIMENT_LOG.md) are records of their time, not current instructions.

## Final controls

Recipe v3; ordinary AdamW; full-horizon cosine; 2:1 FineWeb/FineWeb-Edu; frozen 8K tokenizer; seed 42; context 512; batch 32 × accumulation 2. QK-Norm and CWD are off.

The final run used WSL2 on one RTX 5090 Laptop GPU, BF16 forward with FP32 model/optimizer state, and 0.300-second inter-update sleep. The operational envelope records OMEN Balanced, Turbo/Boost off, AC maximum processor state 60%, AC power, AUTO fans and one-thread environment limits. Human settings are not equivalent to continuous sensor verification. The combined thermally paced production configuration completed the full run; no individual control is assigned causal credit.

## Remaining human release work

- Review this working-tree documentation and the [skeptical scorecard](docs/submission/JUDGE_REVIEW.md).
- Model delivery is complete: [public EXP-020 package](https://huggingface.co/AryaErgin/gibc-v2-exp020); [anonymous remote hash verification](docs/submission/PUBLIC_RELEASE.md) passed.
- Record and edit the [3:30 demo](docs/submission/DEMO_STORYBOARD.md); upload the existing figures.
- Confirm team information, rights/notices and submission fields against current organizer requirements.
- Competition source is published at `gibc-v2-track01-exp020-v1.0.0` / `69f56d7b1f4f377f94a30216a9945df8a6662cce`. Subsequent docs/media polish does not move this frozen tag.

No training, resume, method search, benchmark rerun or result-driven checkpoint selection is a remaining project task.

## Final compliance handoff

Exact inference-only weights are publicly available without changing the frozen checkpoint; see [release guide](docs/submission/EXP020_RELEASE.md). Frozen source publication is complete; demo recording and team eligibility/Devpost entry remain human actions. [Six-component checklist](docs/submission/COMPLIANCE.md) and [license limits](docs/submission/LICENSE_AUDIT.md) are current. The [public-user audit](docs/submission/PUBLIC_USABILITY.md) records isolated verification and short authorized non-benchmark generation. No training or official benchmark rerun occurred.
