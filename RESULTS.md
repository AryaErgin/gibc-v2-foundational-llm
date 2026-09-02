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

# Results Index

| Record | Accepted outcome | Detailed evidence |
|---|---|---|
| Data Recipe v1 | EXP-004 accepts the 2:1 globally deduplicated FineWeb/FineWeb-Edu stream. | [EXP-004](results/EXP-004-summary.md) |
| Near-cap Recipe v3 | EXP-008 selects 49,860,480-parameter SwiGLU Recipe v3. EXP-009 retains 6e-4/6e-5; EXP-010 retains Recipe v3. | [EXP-008](results/EXP-008-summary.md), [EXP-009](results/EXP-009-summary.md), [EXP-010](results/EXP-010-summary.md) |
| Long-horizon calibration | EXP-011 completed 1,500,020,736 prediction tokens; the final tranche remains meaningfully data-limited. | [EXP-011](results/EXP-011-summary.md) |
| Fresh 2.4B calibration and official evaluation | EXP-012 completed 2,399,993,856 prediction tokens, then completed its frozen CPU FP32 official evaluation. | [Training record](results/EXP-012-summary.md), [official protocol and provenance](experiments/EXP-012-official-evaluation.md) |
| WSD scheduler selection | EXP-013 seed-42 discovery and seed-43 paired confirmation both improved frozen combined validation without a domain regression; Recipe v3 + WSD is promoted. | [EXP-013](results/EXP-013-summary.md) |
| HT-SR LLR ablation | EXP-014 seed-42 LLR worsened combined frozen validation by `0.05562235414981842` nat and failed its preregistered gate; Recipe v3 + WSD remains baseline. | [EXP-014](results/EXP-014-summary.md) |

## EXP-014 HT-SR LLR ablation — finalized 2026-08-29

At the fixed 300,023,808-token seed-42 Recipe-v3 + WSD protocol, HT-SR LLR
finished with combined frozen validation loss `3.423619344830513` versus the
WSD control's `3.3679969906806946` (`+0.05562235414981842` nat). Both General
and Edu losses regressed by more than 0.020 nat. **EXP-014 is a negative
result; Recipe v3 + WSD remains the frozen promoted baseline.** No official
benchmark, LLR retuning, or seed-43 confirmation followed. Details are in
[results/EXP-014-summary.md](results/EXP-014-summary.md).

## EXP-013 WSD scheduler ablation â€” finalized 2026-08-29

The four completed 300,023,808-token arms used the frozen 49,860,480-paramete
Recipe v3, tokenizer, Data Recipe v1 stream, and validation tensors. WSD
improved combined loss by `-0.03376075625419617` nat at seed 42 and by
`-0.027358993887901306` nat in the paired seed-43 confirmation; General and
Edu validation both improved in each comparison. **Model Recipe v3 + WSD is
the frozen promoted training baseline for EXP-014.** This is an intenal
validation/scheduler conclusion only; it does not run or reinterpret any
official benchmark. Full provenance, runtime, scheduler, and checkpoint data
are in [results/EXP-013-summary.md](results/EXP-013-summary.md).

## EXP-012 official evaluation — finalized 2026-08-28

The frozen selected checkpoint is `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c` (49,860,480 parameters); the frozen tokenizer SHA-256 is `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`. The guarded WSL CPU FP32 suite completed successfully with CUDA unavailable, `CUDA_VISIBLE_DEVICES=""`, zero-shot, batch size 16, context 512, `lm-eval==0.4.9.1`, and CPU amendment commit `49bfe789fb8e6ebd23b00b5774f4f2e97ee1c464`.

| Task | Official metric(s) |
|---|---|
| HellaSwag | acc `0.273451503684525`; acc stderr `0.004448196648383006`; acc_norm `0.28759211312487554`; acc_norm stderr `0.004517148434180435` |
| ARC-Easy | acc `0.38552188552188554`; acc stderr `0.009987250004629016`; acc_norm `0.36447811447811446`; acc_norm stderr `0.00987572928248244` |
| PIQA | acc `0.6039173014145811`; acc stderr `0.011411089031912477`; acc_norm `0.6022850924918389`; acc_norm stderr `0.011419114133117227` |
| WinoGrande | acc `0.5035516969218626`; acc stderr `0.014052131146915853` |
| WikiText-103 held-out | perplexity `35.93897257521639`; BPB `1.4083853215598`; scored tokens `350948`; documents `2891`; mean NLL `3.581822293724097` |

### Directly comparable reasoning-task record: EXP-006A to EXP-012

| Task / metric | EXP-006A | EXP-012 | Descriptive change |
|---|---:|---:|---|
| HellaSwag acc_norm | 0.2741485759808803 | 0.28759211312487554 | Higher in EXP-012 |
| ARC-Easy acc_norm | 0.3181818181818182 | 0.36447811447811446 | Higher in EXP-012 |
| PIQA acc_norm | 0.5685527747551686 | 0.6022850924918389 | Higher in EXP-012 |
| WinoGrande acc | 0.5074980268350434 | 0.5035516969218626 | Lower in EXP-012 |

These are descriptive comparisons only. A single combined architecture-and-scale change cannot establish causal attribution.

The previous EXP-001 through EXP-006 “WikiText” BPB figures came from the earlier lm-eval `wikitext` task, not this competition-correct WikiText-103 evaluator. They are **not directly comparable**; this record does not present a `1.276 -> 1.408` regression.

The first display-corruption attempt and terminal-closure attempt remain preserved in EXP-012 history as non-results: neither produced a valid official result artifact or informed checkpoint/model selection. No training, checkpoint selection, retuning, or benchmark rerun is authorized by these results.
| Fixed-example temporal placement | EXP-015 seed-42 A/B/C found a General/Edu specialization tradeoff, no preregistered capability winner, and no phase-interaction support; Recipe v3 + WSD remains baseline. | [EXP-015](results/EXP-015-summary.md) |

## EXP-015 fixed-example temporal placement - finalized 2026-08-30

At the fixed 300,023,808-token seed-42 Recipe-v3 + WSD protocol, B
cooldown-Edu reached combined frozen validation NLL `3.3636962473392487`
versus A static `3.3695505559444427` (`-0.005854308605194092` nat), below the
preregistered `-0.010` capability threshold. C pre-cooldown-Edu reached
`3.379886820912361` (`+0.010336264967918396` nat versus A) and was
`+0.016190573573112488` nat versus B, failing both capability and
phase-interaction gates. **EXP-015 is a preserved negative result.** Identical
fixed examples with only temporal placement produced a domain tradeoff, not
broad capability improvement. No seed-43 follows; Recipe v3 + WSD remains
frozen. Details: [EXP-015](results/EXP-015-summary.md).

## SYS-002 WSL runtime qualification - finalized 2026-08-30

The fresh WSL runtime reproduction of Recipe-v3 + WSD completed 9,156 updates
and 300,023,808 prediction tokens under the frozen A schedule. General, Edu,
and combined endpoint differences versus Windows A were all within the

| Momentum-Aligned Gradient Masking | EXP-016 fresh seed-42 Magma worsened combined frozen validation by `0.09226645529270172` nat and failed both domain guards; its 7.43% throughput decline met the efficiency guard but cannot rescue capability failure. Recipe v3 + WSD remains baseline. | [EXP-016](results/EXP-016-summary.md) |

## EXP-016 Magma - finalized 2026-08-30

Magma combined NLL was `3.4620570689439774` versus Control
`3.3697906136512756` (`+0.09226645529270172` nat), with both domain guards
failing. **EXP-016 is a preserved negative result; Magma is rejected without
tuning or seed-43 confirmation.** The efficiency guard passed with a 7.43%
mean-throughput degradation, which does not alter capability failure. Details:
[results/EXP-016-summary.md](results/EXP-016-summary.md).


## EXP-018 QK-Norm — finalized 2026-09-02

EXP-018 executed its full fresh 45,777-update / 1,500,020,736-token horizon. Terminal frozen validation was General 3.2337925136089325, Edu 2.9020539820194244, Combined 3.0679232478141785; versus EXP-011 Combined 3.0800857544, this is a -0.0121625066-nat improvement. Although neither domain guard failed, this misses the preregistered >=0.015-nat promotion threshold. EXP-018 is an execution pass and scientific promotion fail: QK-Norm is rejected from the final recipe without retuning. No required benchmark was run.


## EXP-019 Cautious Weight Decay - finalized 2026-09-02

EXP-019 completed the exact 1.5B EXP-011-matched Recipe-v3 cosine horizon with source-faithful CWD, QK-Norm off, and no benchmark evaluation. Terminal General/Edu/Combined validation was 3.253117263317108 / 2.919591575860977 / 3.0863544195890425, compared with EXP-011s Combined 3.0800857544. The +0.0062686652 endpoint regression fails the preregistered PASS threshold (Combined <=3.0700857544). CWD is rejected from the final recipe without retuning. It appeared favorable through intermediate horizons but reversed by 1.5B; this is project-specific evidence that short-horizon method rankings were insufficiently reliable for further method hunting.

## EXP-020 final scaling run - preregistered, not launched

EXP-020 is the sole final fresh 7.2B-token Recipe-v3 ordinary-AdamW cosine run. QK-Norm and CWD are off. Its data must be a fresh deterministic 2:1 FineWeb/FineWeb-Edu 7.2B stream with exact inherited EXP-012 and EXP-011 prefix checks. No training or benchmark result exists yet.
