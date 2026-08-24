# Experiment Log

## EXP-001

Status: training complete; final-checkpoint baseline evaluation pending EXP-001D.

The approved run at `fd99d1035c6c3d1e677620875426eb8a4cfcda93` completed 3,052 updates and 100,007,936 prediction tokens. Source-of-truth training record: `results/EXP-001-summary.md`. The final checkpoint is local-only at `artifacts/exp001-full/checkpoints/checkpoint-step-3052.pt`; do not resume or modify it.

## EXP-003

Status: completed and accepted. The pure FineWeb-Edu run completed 9,156 updates / 300,023,808 prediction tokens at source commit `95b782156220c6747f02a526dd7fb64d182e8eb3`. It improved educational validation but exceeded the predeclared general-validation regression limit. See `results/EXP-003-summary.md`; do not resume, modify, or benchmark its checkpoint.

## EXP-004

Status: completed and accepted as Data Recipe v1. The 2:1 globally content-hash-deduplicated mixture completed 9,156 updates / 300,023,808 prediction tokens and met both predeclared internal-validation thresholds. See `results/EXP-004-summary.md`. Its separately authorized final-checkpoint EXP-004A evaluation completed under lm-eval `0.4.9.1`, zero shot, batch size 16; see `results/EXP-004A-summary.md`. Do not resume or modify the checkpoint.
