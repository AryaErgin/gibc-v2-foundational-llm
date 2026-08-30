## EXP-017A 2.4B WSD horizon branch

EXP-016 is finalized negative: Magma worsened combined frozen NLL by
`0.09226645529270172` nat and failed both domain guards. It is rejected with
no tuning or seed-43; Recipe v3 + WSD remains promoted.

EXP-017A is preregistered only, not launched: a fresh seed-42 2.4B-token
Recipe-v3 + WSD lineage on the immutable EXP-012 stream. Scheduler-derived
WSD is stable through 65,918 and cools down across updates 65,919-73,242
(7,324 updates), ending at `6e-5`. The pre-cooldown checkpoint at
2,160,001,024 prediction tokens is the required future continuation trunk;
the cooled endpoint does not continue. Future 4.8B/7.2B/9.6B branches require
separate authorization. No benchmark or training is authorized by this plan.

# Project Plan

The project follows a baseline-first experimental sequence: infrastructure validation, baseline measurement, controlled interventions, final training, evaluation, and submission documentation.

## Completed current stage

Recipe v3 (49,860,480 parameters, SwiGLU) was selected by EXP-008. EXP-009 retained peak/minimum LR `6e-4/6e-5` because the 8e-4 result was inside its predeclared 0.01-nat tie region. EXP-010 retained Recipe v3 under its predeclared engineering tiebreak.

EXP-011 completed the authorized fresh seed-42 1.5B calibration. EXP-012 then completed the single authorized fresh 2.4B-class calibration: 73,242 updates / 2,399,993,856 prediction tokens, with its cosine horizon fixed from step zero. The rebuilt stream retained exact EXP-004, EXP-006, and EXP-011 raw-byte prefixes and the 2:1 globally deduplicated, contamination-screened Data Recipe v1 mixture. The final approximately-300M EXP-012 tranche is approaching diminishing retuns. Details: [results/EXP-012-summary.md](results/EXP-012-summary.md).

## Frozen controls

Native Windows, OMEN Performance mode, AC power, context 512, physical batch 32 x accumulation 2, effective batch 32,768 prediction tokens, BF16 autocast with FP32 model/optimizer state, and Recipe v3 + WSD remain the recorded production controls. EXP-013 verified the WSD 100-update warmup, stable `6e-4` phase through update 8,240, and 916-update cooldown to exactly `6e-5` at update 9,156 across seeds 42 and 43.

## Next action

The selected EXP-012 terminal checkpoint completed the frozen official CPU FP32
evaluation successfully on 2026-08-28; see
`experiments/EXP-012-official-evaluation.md` and `RESULTS.md`. No additional
training, LR/architecture/data change, altenate checkpoint selection, o
benchmark rerun is authorized. The remaining submission task is approval-gated
publication of the inference-only artifact described in
`docs/EXP-012-INFERENCE-PUBLICATION.md`. EXP-014 then tested the
pre-registered HT-SR LLR treatment at fixed total tokens and failed its seed-42
capability gate; Recipe v3 + WSD remains the promoted baseline. No follow-up
LLR tuning or confirmation is authorized by that negative result.

EXP-015 subsequently tested FineWeb-Edu-enriched fixed-example temporal
placement at the fixed 300,023,808-token seed-42 Recipe-v3 + WSD protocol.
Neither ordering met its preregistered capability gate, and C failed the
preregistered phase-interaction gate against B. This negative result rejects
the curriculum from the promoted recipe and does not authorize seed-43,
EXP-016, a benchmark, or a final-scale run. Recipe v3 + WSD remains the
promoted training baseline.

## SYS-002 WSL runtime qualification

On 2026-08-30, the secure WSL Python 3.11 / torch 2.13.0+cu132 runtime on the
RTX 5090 Laptop reproduced the frozen seed-42 Recipe-v3 + WSD 300M endpoint
within all predeclared Windows equivalence gates. This is runtime-system
qualification only, not model-selection evidence: it does not change Recipe
v3 + WSD, promote a new model, or authorize EXP-016, seed43, a benchmark, or
additional training. The extenal run record is

## EXP-016 Magma preflight

The research chat has authorized preflight only for a future paired
fresh-seed-42 EXP-016 ordinary-AdamW control followed by Magma treatment.
The WSL runtime is scientifically qualified, but neither EXP-016 arm is
authorized to start by this document. Recipe v3 + WSD remains the promoted
baseline. The frozen specification, gates, and independent paper provenance
are in `experiments/EXP-016-magma.md` and
`provenance/exp016-magma-paper.json`; no benchmarks, tuning, or seed-43
confirmation are authorized at preflight.

## EXP-016 result

EXP-016 completed fresh seed-42 Control then Magma on the scientifically
qualified WSL runtime. Magma worsened combined frozen NLL by 0.09226645529270172
nat and failed both domain guards. Its <=10% throughput guard passed but does
not rescue capability failure. Magma is rejected without tuning; no seed-43,
benchmark, or additional experiment is authorized. Recipe v3 + WSD remains the
promoted training recipe.
