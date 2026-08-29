# Results Index

| Record | Accepted outcome | Detailed evidence |
|---|---|---|
| Data Recipe v1 | EXP-004 accepts the 2:1 globally deduplicated FineWeb/FineWeb-Edu stream. | [EXP-004](results/EXP-004-summary.md) |
| Near-cap Recipe v3 | EXP-008 selects 49,860,480-parameter SwiGLU Recipe v3. EXP-009 retains 6e-4/6e-5; EXP-010 retains Recipe v3. | [EXP-008](results/EXP-008-summary.md), [EXP-009](results/EXP-009-summary.md), [EXP-010](results/EXP-010-summary.md) |
| Long-horizon calibration | EXP-011 completed 1,500,020,736 prediction tokens; the final tranche remains meaningfully data-limited. | [EXP-011](results/EXP-011-summary.md) |
| Fresh 2.4B calibration and official evaluation | EXP-012 completed 2,399,993,856 prediction tokens, then completed its frozen CPU FP32 official evaluation. | [Training record](results/EXP-012-summary.md), [official protocol and provenance](experiments/EXP-012-official-evaluation.md) |
| WSD scheduler selection | EXP-013 seed-42 discovery and seed-43 paired confirmation both improved frozen combined validation without a domain regression; Recipe v3 + WSD is promoted. | [EXP-013](results/EXP-013-summary.md) |

## EXP-013 WSD scheduler ablation â€” finalized 2026-08-29

The four completed 300,023,808-token arms used the frozen 49,860,480-parameter
Recipe v3, tokenizer, Data Recipe v1 stream, and validation tensors. WSD
improved combined loss by `-0.03376075625419617` nat at seed 42 and by
`-0.027358993887901306` nat in the paired seed-43 confirmation; General and
Edu validation both improved in each comparison. **Model Recipe v3 + WSD is
the frozen promoted training baseline for EXP-014.** This is an internal
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
