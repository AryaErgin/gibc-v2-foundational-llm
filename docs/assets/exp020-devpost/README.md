# EXP-020 judge-facing figures — native 3:2

Seven approved presentation assets, copied byte-for-byte from the final 1800×1200 exports. Each PNG is under 5 MB. SVG counterparts retain editable text; Segoe UI fallback may change typography when re-rendered, so use the PNGs for submission. No font binaries are distributed. Contact sheets and temporary rendering environments are deliberately excluded.

These are presentation-only assets, not new scientific measurements. The earlier frozen figures remain in [../exp020/](../exp020/README.md).

## Ordered gallery and evidence

### 1. exp020 final results

![Final results: WikiText-103 31.783 at 49.86M parameters; official zero-shot accuracy and final-run context. The 11.56% PPL decrease is computed from unrounded EXP-012/020 values.](01-exp020-final-results.png)

Final results: WikiText-103 31.783 at 49.86M parameters; official zero-shot accuracy and final-run context. The 11.56% PPL decrease is computed from unrounded EXP-012/020 values. [Editable SVG](01-exp020-final-results.svg).

### 2. horizon dependent findings

![Horizon-dependent intervention findings. WSD, QK-Norm and CWD outcomes refer to this sub-50M regime, not universal method rankings.](02-horizon-dependent-findings.png)

Horizon-dependent intervention findings. WSD, QK-Norm and CWD outcomes refer to this sub-50M regime, not universal method rankings. [Editable SVG](02-horizon-dependent-findings.svg).

### 3. benchmark development

![Historical benchmark development. Each position/change panel spans 16 percentage points with its local range explicit. Architecture, budget and recipe changed: not a controlled scaling ablation. WinoGrande is retained.](03-benchmark-development.png)

Historical benchmark development. Each position/change panel spans 16 percentage points with its local range explicit. Architecture, budget and recipe changed: not a controlled scaling ablation. WinoGrande is retained. [Editable SVG](03-benchmark-development.svg).

### 4. validation trajectory

![Internal General/Edu validation and training-loss bins, not official benchmark scores. Main plot details 1.5B onward; initial validation and full training-loss context are stated.](04-validation-trajectory.png)

Internal General/Edu validation and training-loss bins, not official benchmark scores. Main plot details 1.5B onward; initial validation and full training-loss context are stated. [Editable SVG](04-validation-trajectory.svg).

### 5. training efficiency

![Final-run execution only. Active-update mean and tokens/full-wall throughput are distinct. The 6NT estimate is theoretical, not measured energy or executed hardware FLOPs.](05-training-efficiency.png)

Final-run execution only. Active-update mean and tokens/full-wall throughput are distinct. The 6NT estimate is theoretical, not measured energy or executed hardware FLOPs. [Editable SVG](05-training-efficiency.svg).

### 6. training evaluation pipeline

![Data screening, training, internal selection and official evaluation are separate stages. EXP-020 checkpoint selection preceded its official benchmark scores.](06-training-evaluation-pipeline.png)

Data screening, training, internal selection and official evaluation are separate stages. EXP-020 checkpoint selection preceded its official benchmark scores. [Editable SVG](06-training-evaluation-pipeline.svg).

### 7. public model verification

![Public exact model-state export, 83 tensors, clean-environment inference and round-trip hash match. The source training checkpoint and inference weights have different SHA identities.](07-public-model-verification.png)

Public exact model-state export, 83 tensors, clean-environment inference and round-trip hash match. The source training checkpoint and inference weights have different SHA identities. [Editable SVG](07-public-model-verification.svg).

## Evidence and integrity

Source facts: [frozen submission evidence](../../../results/exp020-submission-evidence.json), [results](../../../RESULTS.md), [experiment log](../../../EXPERIMENT_LOG.md), [public release receipt](../../../results/exp020-public-release.json) and [clean-environment smoke](../../../results/exp020-prepublication-smoke.json). Frozen competition source: `69f56d7b1f4f377f94a30216a9945df8a6662cce`.

All seven images were reviewed against their final Desktop references before copying. [Hash inventory](manifest.json) binds the public PNG/SVG files to those approved exports. No cropping, stretching, or re-editing was performed during publication polish.
