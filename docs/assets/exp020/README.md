# EXP-020 submission figures

Four figures, each 1280×720 PNG plus editable SVG. All plotted values come from recorded results. No generated/model-scored example is used.

| Figure | Source and interpretation |
|---|---|
| [Trajectory PNG](trajectory.png) / [SVG](trajectory.svg) | [Evidence digest](../../../results/exp020-submission-evidence.json): 1,000-update mean training-loss bins (final bin 726 updates); General/Edu/Combined internal validation. Step-0 Combined 9.147 is omitted from the zoomed validation plot and remains in the table. Curves connect recorded points, not an interpolated performance claim. |
| [Progression PNG](progression.png) / [SVG](progression.svg) | [EXP-001D](../../../results/EXP-001D-summary.md), [EXP-012 provenance](../../../results/exp012-official-provenance.json), final digest. First three accuracy metrics are acc_norm, WinoGrande acc. Shared zero baseline; historical architecture/data/horizon differ. Old word/byte PPL is excluded from token-PPL comparison. Error bars are not plotted; recorded harness stderr is in [RESULTS](../../../RESULTS.md). |
| [Decisions PNG](decisions.png) / [SVG](decisions.svg) | [RESULTS](../../../RESULTS.md), EXP-004/008/013–016 summaries and EXP-017/018/019 closure records. Own-horizon/own-control decisions, not a common-budget method leaderboard. No generic component novelty claim. |
| [Pipeline PNG](pipeline.png) / [SVG](pipeline.svg) | Frozen config, corpus manifest, training summary, preregistration and official metadata in the digest. Final selection precedes its own scoring; historical project exposure is disclosed. |

## Reproduce without model/data execution

Use an environment where Pillow is already available (the packaging render used Pillow 12.3.0). No changes were made to the qualified training/evaluation environment. Fonts are locally installed Segoe UI, not redistributed. A different font can alter pixels/layout and must be visually rechecked.

```powershell
python scripts/render_submission_figures.py --repo . --output /path/to/NEW-figure-directory --font C:/Windows/Fonts/segoeui.ttf --bold-font C:/Windows/Fonts/segoeuib.ttf
```

The output directory must be fresh. The [render manifest](figure-provenance.json) records evidence/script/source/font SHA-256s, Pillow version and all output hashes. SVG uses a Segoe UI/DejaVu Sans/sans-serif fallback; PNG fixes the rendered appearance. This is plotting/layout software only, with no model, dataset loader, network or benchmark invocation.

The captions intentionally report limitations alongside the visuals so they remain interpretable when reused in Devpost or the video.
