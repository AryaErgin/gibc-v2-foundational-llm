# Project Plan

The project follows a baseline-first experimental sequence: infrastructure validation, baseline measurement, controlled interventions, final training, evaluation, and submission documentation.

## Completed current stage

Recipe v3 (49,860,480 parameters, SwiGLU) was selected by EXP-008. EXP-009 retained peak/minimum LR `6e-4/6e-5` because the 8e-4 result was inside its predeclared 0.01-nat tie region. EXP-010 retained Recipe v3 under its predeclared engineering tiebreak.

EXP-011 completed the authorized fresh seed-42 1.5B calibration. EXP-012 then completed the single authorized fresh 2.4B-class calibration: 73,242 updates / 2,399,993,856 prediction tokens, with its cosine horizon fixed from step zero. The rebuilt stream retained exact EXP-004, EXP-006, and EXP-011 raw-byte prefixes and the 2:1 globally deduplicated, contamination-screened Data Recipe v1 mixture. The final approximately-300M EXP-012 tranche is approaching diminishing returns. Details: [results/EXP-012-summary.md](results/EXP-012-summary.md).

## Frozen controls

Native Windows, OMEN Performance mode, AC power, context 512, physical batch 32 x accumulation 2, effective batch 32,768 prediction tokens, BF16 autocast with FP32 model/optimizer state, and Recipe v3 + WSD remain the recorded production controls. EXP-013 verified the WSD 100-update warmup, stable `6e-4` phase through update 8,240, and 916-update cooldown to exactly `6e-5` at update 9,156 across seeds 42 and 43.

## Next action

The selected EXP-012 terminal checkpoint completed the frozen official CPU FP32
evaluation successfully on 2026-08-28; see
`experiments/EXP-012-official-evaluation.md` and `RESULTS.md`. No additional
training, LR/architecture/data change, alternate checkpoint selection, or
benchmark rerun is authorized. The remaining submission task is approval-gated
publication of the inference-only artifact described in
`docs/EXP-012-INFERENCE-PUBLICATION.md`. EXP-014 then tested the
pre-registered HT-SR LLR treatment at fixed total tokens and failed its seed-42
capability gate; Recipe v3 + WSD remains the promoted baseline. No follow-up
LLR tuning or confirmation is authorized by that negative result.
