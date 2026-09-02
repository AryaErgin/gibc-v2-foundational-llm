## EXP-017A closure, EXP-018 closure, and EXP-019 CWD preregistration (2026-09-02)

EXP-017A Attempt 3 completed the full 2.4B WSD horizon but failed its frozen scientific promotion gate (Combined 3.0358807146549225 versus required <= 3.010453929901123). EXP-012 cosine remains promoted; WSD is rejected for the final recipe without retuning. The combined thermally paced production configuration was stable for the full run; no individual thermal control is given causal credit.

EXP-018 completed its fresh seed-42, 1.5B-token, EXP-011-matched cosine QK-Norm ablation. Its terminal Combined NLL was 3.0679232478141785 versus EXP-011's 3.0800857544: a 0.0121625066-nat improvement, below the frozen 0.015-nat promotion threshold. Domain guards passed, but scientific promotion fails. The shrinking advantage from about -0.02233 at 300M to -0.01216 at 1.5B is recorded as a proxy/horizon-decay warning. QK-Norm remains implemented but disabled by default and rejected from the final recipe; no retuning is authorized.

EXP-019 is the one remaining, preregistered source-faithful Cautious Weight Decay (CWD) ablation. It holds the EXP-011 1.5B Recipe-v3 cosine protocol fixed and changes only ordinary decoupled weight decay to the source Algorithm 1 entrywise mask. It must use the independently verified EXP-011-identical prefix of EXP-012, start fresh at seed 42, run 45,777 updates, and use frozen General/Edu validation only. The pass, strong-pass, domain, horizon, and systems gates are recorded in provenance/exp019-cwd-preregistration.json. Broad method hunting is closed; EXP-020 final-scale recipe selection waits for this result.

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


## Final recipe freeze and EXP-020 preparation (2026-09-02)

EXP-019 CWD completed but regressed the exact EXP-011 1.5B endpoint by 0.0062686652 Combined NLL, so CWD is rejected without retuning. Method hunting is closed. The sole final training plan is EXP-020: fresh 49,860,480-parameter Recipe-v3, ordinary AdamW, 2:1 FineWeb/FineWeb-Edu, seed 42, and a 219,726-step / 7,199,981,568-token cosine schedule. QK-Norm and CWD are explicitly disabled. This is one final model run, not a paired 7.2B comparison.

Because the EXP-012 materialization did not retain serializable global-dedup/cursor/mixture continuation state, a 7.2B append is not defensible. EXP-020 first rebuilds a unique token stream deterministically from zero and must reverify both EXP-012 and EXP-011 byte-prefix SHA-256 gates before a trainer may start. Required competition benchmarks remain embargoed until post-training checkpoint selection is frozen using the preregistered General/Edu-only rule.
