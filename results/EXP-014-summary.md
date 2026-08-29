# EXP-014 — HT-SR Layerwise Learning Rate, seed 42

**Status:** completed 2026-08-29; **FAIL** under the preregistered seed-42
capability gate. No official benchmark, seed-43 confirmation, retuning, or
follow-up optimizer experiment was started.

## Controlled comparison

EXP-014 changed only the pre-registered HT-SR LLR control relative to the
promoted Recipe-v3 + WSD seed-42 control. Architecture, 49,860,480 parameter
count, seed 42 initialization, 300,023,808-token non-cycled Data Recipe v1
prefix, tokenizer, AdamW, WSD, batch, precision, and frozen General/Edu
validation tensors were unchanged.

| Arm | General loss | Edu loss | Combined loss |
| --- | ---: | ---: | ---: |
| EXP-013 WSD control | 3.5252154767513275 | 3.2107785046100616 | 3.3679969906806946 |
| EXP-014 HT-SR LLR | 3.58173605799675 | 3.265502631664276 | 3.423619344830513 |
| LLR minus WSD | +0.05652058124542236 | +0.05472412705421448 | +0.05562235414981842 |

The combined change is a **-1.65149655132491% relative improvement** (that
is, a 1.6515% relative NLL worsening). Descriptive combined perplexity is
`30.680256741483365`. The candidate does not meet the required combined-loss
threshold `<= 3.3511570057272912`; both domains also worsen by more than
0.020 nat. EXP-014 seed 42 is therefore **FAIL**.

## LLR execution and diagnostics

The WSD scheduler reached 9,156 / 9,156 updates and exactly 300,023,808
prediction tokens. The final LR was `6e-5`; the LLR controller recomputed at
steps 100 through 1,800, completed its 50-step final transition at step 1,850,
and stored frozen multipliers thereafter. All 1,134 recorded alphas were
finite (range 1.6812543869018555–2.9989683628082275); no NaN, Inf, or invalid
alpha was observed.

Final frozen multipliers across 65 groups: minimum `1.0`, median
`1.792471142322491`, mean `2.1537904779772696`, maximum `5.0`. The tied
embedding/output group was exactly one group at `5.0`. Highest values were the
tied embedding and `blocks.6.mlp.out_proj.weight` (`5.0`), followed by MLP
output projections; the lowest were `blocks.7.attention.o_proj.weight` and
the uniform RMSNorm group (`1.0`). The last recorded stable-phase effective
LRs at global `6e-4` ranged from `6e-4` to `3e-3`; the frozen multipliers then
scaled proportionally through cooldown.

## Integrity, runtime, and limits

- terminal checkpoint SHA-256:
  `187c64f852423a089bb5638ca91c54808298eb901989ed553813d916d9795590`;
- final cursor: `585984` sequential examples; checkpoint contains model,
  optimizer, WSD schedule, LLR state, RNG, run state, and data cursor;
- fresh initialization: one run-start record at step 0; no resume record or
  crash anomaly;
- frozen manifest/tokenizer SHA-256:
  `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f` /
  `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`;
- runtime: `3837.3573023000063` s; mean `79446.38170209524` tok/s; peak
  allocated/reserved VRAM `7,686,099,968` / `8,491,368,448` bytes;
- spectral computation: `41.207680399980745` s, `1.0738557072931936%` of
  measured wall time.

Only step-0 and terminal validation are recorded for both arms. Therefore a
defensible tokens-to-control-final-loss comparison is **NOT MEASURABLE** from
the existing validation trajectory; no interpolation is used.

The independent implementation and consulted-method provenance are recorded
in `provenance/exp014-upstream-provenance.json`. Per the preregistration,
the negative result does not authorize changing `s=5`, cadence, transition,
or active phase. Recipe-v3 + WSD remains the promoted training baseline.
