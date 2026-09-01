## EXP-017A  final closure (2026-09-01)

Status: **execution PASS; scientific promotion FAIL**. Attempt 3 completed the
full fresh seed-42 horizon: 73,242 updates, 2,399,993,856 prediction tokens,
49,860,480 parameters, and a final checkpoint at step 73,242. stderr was
empty. Wall time was 45,052.716201469 seconds; mean active-compute throughput
was 104,962.2439 tokens/s and mean paced throughput was 53,494.0248 tokens/s,
with operational inter-update sleep of 0.300 seconds.

Frozen final validation was General 3.2023668587207794, Edu
2.8693945705890656, Combined 3.0358807146549225. This misses the
predeclared gate of Combined <= 3.010453929901123; EXP-012 cosine remains
the promoted scientific baseline (General 3.1909595430, Edu
2.8499483168, Combined 3.020453929901123). WSD is rejected for the final
recipe. No post-hoc WSD retuning is authorized.

Thermal history is preserved accurately: Attempt 1 aborted for thermal
concern; Test B passed its 15-minute qualification but was insufficient to
expose longer heat soak; Attempt 2 was a genuine thermal abort after the user
directly observed repeated 100 C CPU temperatures; Attempt 3 completed under
combined operational thermal controls. **The combined thermally paced
production configuration was stable for the full run.** No individual control
is assigned causal credit.

The closure record is
provenance/exp017a-attempt-3-closure.json. The EXP-017A terminal checkpoint
is preserved but must not be used to claim scientific promotion.

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

Status: completed. EXP-010A tested the single depth/width allocation change from Recipe v3 SwiGLU 640 x 9 x 20, d_ff 1,728 to 608 x 10 x 19, d_ff 1,656 (49,985,504 parameters). Its combined loss was 0.0056370944 nats worse than Recipe v3, inside the predeclared 0.02-nat tie region; it was slower and used more allocated memory, so the committed engineering tiebreak retains Recipe v3. See `results/EXP-010-summary.md`.

## EXP-011

Status: completed long-horizon calibration. The selected 49,860,480-parameter Recipe v3 completed 45,777 updates / 1,500,020,736 prediction tokens from fresh seed-42 initialization, using the exact 45,777-step 6e-4-to-6e-5 cosine from step zero. The verified full stream preserved the EXP-004 300M and EXP-006 900M raw-byte prefixes. The final 1.2B-to-1.5B combined-validation gain was 0.0455186218 nats, therefore meaningfully data-limited under the predeclared diagnostic. This result does not authorize more training or official evaluation. See `results/EXP-011-summary.md`; no official benchmark was run.

## EXP-012

Status: completed fresh 2.4B-token calibration and finalized official evaluation. Recipe v3 completed 73,242 updates / 2,399,993,856 prediction tokens from fresh seed-42 initialization, using the exact 73,242-step 6e-4-to-6e-5 cosine from step zero. The deterministically rebuilt Data Recipe v1 stream passed exact EXP-004 300M, EXP-006 900M, and EXP-011 1.5B raw-byte prefix gates. The final 299,827,200-token tranche improved combined frozen validation by 0.0212369710 nats, therefore approaching diminishing returns under the predeclared diagnostic. The frozen selected terminal checkpoint then completed HellaSwag, ARC-Easy, PIQA, WinoGrande, and competition-correct WikiText-103 on 2026-08-28; see `experiments/EXP-012-official-evaluation.md` and `RESULTS.md`. These results do not authorize training, retuning, alternate checkpoint selection, or benchmark reruns.

## EXP-013

Status: completed and promoted. The preregistered 300M-token paired WSD scheduler ablation retained every Recipe v3, Data Recipe v1, optimizer, batch, seed-specific initialization, and validation control. At seed 42, WSD improved combined frozen validation by `0.03376075625419617` nat versus its contemporaneous cosine control, meeting the capability-win threshold with General and Edu both improving. At seed 43, the paired confirmation improved by `0.027358993887901306` nat, exceeding its `0.010`-nat confirmation threshold with no domain regression. Recipe v3 + WSD is therefore the frozen promoted training baseline for a future EXP-014; no EXP-014 run, benchmark, or retrospective tuning is authorized by this decision. See `results/EXP-013-summary.md`.
# EXP-014 — HT-SR LLR pre-registration (2026-08-29)

Status: completed seed-42 negative result. EXP-014 tested
the paper's HT-SR PL_Alpha_Hill positive linear layerwise LR mapping on the
promoted Recipe-v3 + WSD baseline, with all model/data/tokenizer/seed/AdamW/
batch/validation conditions fixed. The source specification is
`experiments/EXP-014-htsr-llr.md`; candidate config is
`configs/exp014-llr.yaml`. The algorithmic source is arXiv:2605.22297v3; the
unlicensed author repository was consulted, not copied, at immutable commit
`bbd0dcf86af80b8843866a9a041086a37de35897`. See
`provenance/exp014-upstream-provenance.json`.

The LLR candidate completed 9,156 updates / 300,023,808 tokens with General
`3.58173605799675`, Edu `3.265502631664276`, and combined
`3.423619344830513`, versus the frozen WSD control combined
`3.3679969906806946` (`+0.05562235414981842` nat). It fails the preregistered
capability gate and both domain-safety conditions. No seed-43 run, LLR tuning,
benchmark, or new experiment is authorized. Recipe v3 + WSD remains the
promoted baseline; see `results/EXP-014-summary.md`.

## EXP-015

Status: completed seed-42 negative result. EXP-015 tested FineWeb-Edu-enriched
fixed-example temporal placement with Recipe v3 + WSD. A/B/C each completed
9,156 updates and 300,023,808 prediction tokens from fresh seed-42 starts at
recovered integration commit `ac7078f3d0cf327d4f8dc7ab9f8ec7edfd263321`. B
improved Edu but slightly regressed General and improved combined NLL only
`0.005854308605194092` nat, below the preregistered 0.010-nat gate. C slightly
improved General but materially regressed Edu, worsening combined NLL by
`0.010336264967918396` nat. C was `0.016190573573112488` nat worse than B, so
phase interaction failed. Reject the curriculum; no seed-43 is authorized.
Recipe v3 + WSD remains promoted. See `results/EXP-015-summary.md`.

## EXP-016 — Magma pre-registration (2026-08-30)

Status: preflight complete; no EXP-016 training arm has been launched. EXP-016
will test independent Momentum-Aligned Gradient Masking (Magma) under the
scientifically qualified WSL runtime while holding Recipe v3 + WSD, seed,
batch, data stream, tokenizer, schedule, and frozen validations fixed.
The two future fresh seed-42 arms are ordinary AdamW control
`configs/exp016-control.yaml` followed by AdamW+Magma treatment
`configs/exp016-magma.yaml`, at 9,156 updates / 300,023,808 prediction
tokens each. The treatment uses p=0.5, tau=2.0, and EMA smoothing=0.9 over
63 attention/SwiGLU matrix blocks (44,605,440 parameters); embeddings/output,
RMSNorm, and all non-attention/non-MLP tensors remain dense AdamW.

The independent implementation applies Magma to the full post-step AdamW
delta, including decoupled weight decay, after dense first/second-moment
updates. It maintains an isolated seed-42 torch generator and checkpoints
block mapping, alignment scalars, generator state, and configuration. The
predeclared capability gate is combined NLL improvement of at least 0.010 nat
against the contemporaneous control with neither domain regressing by more
than 0.020 nat; mean training throughput degradation must be no more than
10%. No benchmark, hyperparameter retuning, or seed-43 confirmation is
authorized by this preflight. See `experiments/EXP-016-magma.md` and
`provenance/exp016-magma-paper.json`.

Focused single-threaded Magma tests passed (8 tests) and the read-only frozen
artifact preflight verified parameter, Schedule-A, stream, validation, and WSD
invariants. A bounded 60-update WSL GPU smoke completed with finite loss,
79,076.26 post-warmup tok/s, 8,665,432,064 reserved bytes, no data stalls, and
61 C sampled peak GPU temperature. Its estimated 7.13% throughput degradation
relative to the prior WSL production baseline is within the predeclared 10%
operational guard. This is implementation/infrastructure evidence only, not
model-selection evidence; see `results/EXP-016-preflight.md`.

## EXP-016 - Magma (2026-08-30)

Status: completed fresh seed-42 negative result. Control reached General
`3.526235818862915`, Edu `3.2133454084396362`, combined `3.3697906136512756`.
Magma reached General `3.616050601005554`, Edu `3.3080635368824005`, combined
`3.4620570689439774`: combined worsened by `0.09226645529270172` nat and both
domain guards failed. Its 7.43% throughput decline passed the efficiency guard
but cannot rescue capability failure. Reject Magma without tuning; no seed-43,
benchmark, or new experiment is authorized. Recipe v3 + WSD remains promoted.
See `results/EXP-016-summary.md`.


## EXP-018 — QK-Norm closure (2026-09-02)

Status: execution PASS; scientific promotion FAIL. Fresh seed-42 EXP-018 completed 45,777 updates / 1,500,020,736 prediction tokens with 49,860,489 parameters and terminal General 3.2337925136089325, Edu 2.9020539820194244, Combined 3.0679232478141785. Against EXP-011 Combined 3.0800857544, the improvement was 0.0121625066 nat, below the preregistered 0.015-nat promotion requirement; the domain guards passed. The matched combined advantages were approximately -0.02233, -0.01640, -0.01306, -0.01288, and -0.01216 at 300M through 1.5B, respectively, strengthening the project's proxy/horizon-decay warning. QK-Norm is rejected from the final recipe without retuning, but remains implemented behind its default-off configuration. No held-out benchmark was invoked. See provenance/exp018-closure.json.

## EXP-019 — Cautious Weight Decay preregistration (2026-09-02)

Status: pre-launch. EXP-019 is the sole remaining method ablation: a fresh seed-42 Recipe-v3, 1.5B-token, EXP-011-matched cosine run using only source-faithful CWD. All architecture, data/order, tokenizer, batch, Adam betas/epsilon, nominal 0.1 decay coefficient, LR/warmup/cosine, validation, and pacing controls remain fixed; QK-Norm is off. CWD replaces ordinary decoupled decay entrywise with Algorithm 1's I(u*x >= 0) mask using the pre-update parameter and Adam adaptive update. The frozen gate is Combined <=3.0700857544 for PASS, <=3.0650857544 for STRONG PASS, with General <=3.2571743524 and Edu <=2.9229971564. Frozen matched horizon measurements are at steps 9,156, 18,312, 27,468, 36,624, and 45,777. No benchmark or post-result retuning is authorized. Full provenance is in provenance/exp019-cwd-preregistration.json.
