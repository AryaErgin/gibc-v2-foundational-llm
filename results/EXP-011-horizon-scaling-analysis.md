# EXP-011 Horizon Scaling Analysis

This is an extrapolation of the frozen EXP-011 combined-validation curve, not a training authorization or a replacement for an observed result. Token counts are expressed in billions in all fitted functions.

## Fits

The model `L(N) = L_inf + A * N^(-alpha)` was fit with non-negative `L_inf`, `A`, and `alpha` using bounded nonlinear least squares.

| Data | L_inf | A | alpha | RMSE |
|---|---:|---:|---:|---:|
| All 5 points | ~0.0000 | 3.1777 | 0.0757 | 0.00733 |
| Last 4 points | 1.8938 | 1.2834 | 0.1994 | 0.00381 |
| Last 3 points | 2.9501 | 0.2243 | 1.3462 | 0.00000 |

The exact three-point interpolation is structurally underconstrained and is used as a pessimistic tail, not evidence that the irreducible loss is 2.9501. Leave-one-out four-point fits predict combined loss at 3.0B from 2.9219 to 2.9306, whereas the last-three-point fit predicts 3.0013. This model-form spread is more important than optimizer fit residuals.

Observed 300M gains were 0.16034, 0.10528, 0.08299, and 0.04552 nats. A log-linear geometric fit to all four gains gives a per-tranche multiplier of 0.6693; fitting only the last three gives 0.6575. The all-gains geometric model is the central diminishing-gain trajectory below.

## Envelope

At the established 101,033.67 tok/s overall rate, each additional 300M tokens costs about 0.825 GPU-hours. Values are predicted combined validation loss; cumulative gains are relative to the observed 1.5B value 3.0800857544.

| Fresh horizon | Extra GPU-hours | Pessimistic loss / gain | Central loss / gain | Optimistic loss / gain |
|---:|---:|---:|---:|---:|
| 1.8B | 0.825 | 3.0518 / 0.0283 | 3.0473 / 0.0327 | 3.0353 / 0.0448 |
| 2.1B | 1.650 | 3.0328 / 0.0473 | 3.0254 / 0.0547 | 3.0007 / 0.0793 |
| 2.4B | 2.474 | 3.0192 / 0.0609 | 3.0108 / 0.0693 | 2.9717 / 0.1084 |
| 2.7B | 3.299 | 3.0090 / 0.0710 | 3.0009 / 0.0792 | 2.9467 / 0.1334 |
| 3.0B | 4.124 | 3.0013 / 0.0788 | 2.9944 / 0.0857 | 2.9248 / 0.1553 |

The predicted next-tranche gain per GPU-hour is approximately 0.0343–0.0543 nats/hour at 1.8B, 0.0231–0.0419 at 2.1B, 0.0165–0.0353 at 2.4B, 0.0120–0.0303 at 2.7B, and 0.0080–0.0265 at 3.0B (pessimistic to optimistic). The central geometric trajectory is reported separately in the table because simple models may cross at the tail.

## Interpretation

The three asymptotic fits agree on continuing improvement but diverge increasingly after 2.1B. The all-five and last-four fits are also leave-one-out stable, but their apparent low asymptotes are not credible estimates from five points. The last-three fit describes recent slowing but has zero degrees of freedom. The geometric-gain model lies between these shapes.

The recommended largest next fresh-run horizon supported by reasonably convergent extrapolations is **2.1B prediction tokens**. It is a bounded +600M extension with a 0.047–0.079 predicted cumulative gain and retains a nontrivial predicted second-tranche marginal gain. A 2.4B recommendation would rely on an envelope already about 0.047 nats wide; 2.7B and 3.0B are not defensibly precise with five observations.

Critical confound: EXP-011 used a cosine schedule whose horizon was 1.5B/45,777 updates from step zero. A fresh 2.1B run would require its own longer cosine horizon from step zero. Its 300M, 600M, and later losses would therefore not reproduce EXP-011 point-for-point, so the table is a horizon-planning envelope, not a continuation forecast.
