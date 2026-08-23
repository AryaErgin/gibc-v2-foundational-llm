# Experiment Log

## EXP-001

Status: training complete; final-checkpoint baseline evaluation pending EXP-001D.

The approved run at `fd99d1035c6c3d1e677620875426eb8a4cfcda93` completed 3,052 updates and 100,007,936 prediction tokens. Source-of-truth training record: `results/EXP-001-summary.md`. The final checkpoint is local-only at `artifacts/exp001-full/checkpoints/checkpoint-step-3052.pt`; do not resume or modify it.

## EXP-003

Status: completed and accepted. The pure FineWeb-Edu run completed 9,156 updates / 300,023,808 prediction tokens at source commit `95b782156220c6747f02a526dd7fb64d182e8eb3`. It improved educational validation but exceeded the predeclared general-validation regression limit. See `results/EXP-003-summary.md`; do not resume, modify, or benchmark its checkpoint.

## EXP-004

Status: preparation only. This is a globally content-hash-deduplicated roughly 2:1 FineWeb:FineWeb-Edu unique-document mixture at the unchanged EXP-002/003 training controls. See `experiments/EXP-004.md`; do not materialize, train, or benchmark without later authorization.
