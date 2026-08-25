# Experiment Log

## EXP-001

Status: training complete; final-checkpoint baseline evaluation pending EXP-001D.

The approved run at `fd99d1035c6c3d1e677620875426eb8a4cfcda93` completed 3,052 updates and 100,007,936 prediction tokens. Source-of-truth training record: `results/EXP-001-summary.md`. The final checkpoint is local-only at `artifacts/exp001-full/checkpoints/checkpoint-step-3052.pt`; do not resume or modify it.

## EXP-003

Status: completed and accepted. The pure FineWeb-Edu run completed 9,156 updates / 300,023,808 prediction tokens at source commit `95b782156220c6747f02a526dd7fb64d182e8eb3`. It improved educational validation but exceeded the predeclared general-validation regression limit. See `results/EXP-003-summary.md`; do not resume, modify, or benchmark its checkpoint.

## EXP-004

Status: completed and accepted as Data Recipe v1. The 2:1 globally content-hash-deduplicated mixture completed 9,156 updates / 300,023,808 prediction tokens and met both predeclared internal-validation thresholds. See `results/EXP-004-summary.md`. Its separately authorized final-checkpoint EXP-004A evaluation completed under lm-eval `0.4.9.1`, zero shot, batch size 16; see `results/EXP-004A-summary.md`. Do not resume or modify the checkpoint.

## EXP-005

Status: completed. Under the exact frozen EXP-004 Data Recipe v1 stream, EXP-005B wide/shallow beat EXP-005A deep/thin by 0.08317044377326965 nats on final combined validation, above the predeclared 0.02-nat tie threshold, and also used less wall time and peak VRAM. EXP-005B is accepted as Architecture Recipe v1; see `results/EXP-005-summary.md`. Its separately authorized final-checkpoint evaluation completed under lm-eval `0.4.9.1`, zero shot, batch size 16; see `results/EXP-005B-evaluation.md`. EXP-005A was not benchmarked.

## EXP-006

Status: completed and accepted. EXP-006 held Architecture Recipe v1 and Data Recipe v1 fixed through 27,468 updates / 900,071,424 prediction tokens. Its final 600M-to-900M combined-validation improvement was `0.0848376601934433`, above the predeclared strongly-data-limited threshold. See `results/EXP-006-summary.md`. Do not resume or modify its checkpoint; only final-checkpoint EXP-006A evaluation is authorized.

## EXP-007

Status: completed and accepted. Under the exact frozen EXP-004 stream, EXP-007A had the numerically lower final combined validation loss (3.4257752001 versus 3.4314021170), but the 0.0056269169-nat difference is inside the predeclared 0.02-nat engineering-tie region. EXP-007B is selected by the predeclared throughput/memory efficiency tiebreak and freezes Near-Cap Architecture Recipe v2. See `results/EXP-007-summary.md`. Neither candidate was officially benchmarked.

## EXP-008

Status: completed and accepted. EXP-008A improved final combined frozen validation by 0.0300662965 nats over EXP-007B, exceeding the predeclared 0.02-nat threshold; SwiGLU is the capability winner and freezes Near-Cap Architecture Recipe v3. See `results/EXP-008-summary.md`. No official benchmark was run.

## EXP-009

Status: completed. EXP-009A 4e-4/4e-5 was worse than the existing 6e-4/6e-5 control by 0.0408950746 nats and is rejected. EXP-009B 8e-4/8e-5 improved by 0.0075877458 nats, inside the predeclared 0.01-nat engineering/statistical tie band; retain 6e-4 to avoid proxy-horizon over-tuning. See `results/EXP-009-summary.md`. No official benchmark was run and no >300M follow-up is authorized.

## EXP-010

Status: specification and non-training guard preparation only. EXP-010A proposes the single depth/width allocation change from Recipe v3 SwiGLU 640 x 9 x 20, d_ff 1,728 to 608 x 10 x 19, d_ff 1,656 (49,985,504 parameters). See `experiments/EXP-010.md`. Neither the 60-update preflight nor a 300M run is authorized by existing governance.

Status update: completed under subsequent research authorization. EXP-010A's combined loss was 0.0056370944 nats worse than Recipe v3, inside the predeclared 0.02-nat tie region. It was also slower and used more allocated memory, so the committed engineering tiebreak retains Recipe v3. See `results/EXP-010-summary.md`.
