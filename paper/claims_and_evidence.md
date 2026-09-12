# Final claims and evidence ledger

| Claim | Evidence | Scope |
|---|---|---|
| 49,860,480 parameters; scratch training; 7,199,981,568 tokens | [architecture](../ARCHITECTURE.md), [audit](../results/exp020-submission-evidence.json) | Exact checkpoint/config/run identity |
| 37.94h on one laptop GPU | [summary](../results/EXP-020-summary.md) | Final run including pacing, not campaign/energy |
| PPL 31.783; mixed task changes | [RESULTS](../RESULTS.md) | HellaSwag/ARC better, PIQA roughly stable, WinoGrande worse |
| WSD proxy gain did not meet 2.4B gate | [EXP-013](../results/EXP-013-summary.md), [EXP-017 closure](../provenance/exp017a-attempt-3-closure.json) | Local horizon sensitivity |
| QK-Norm gain below promotion gate | [closure](../provenance/exp018-closure.json) | −0.01216 versus required −0.015; not “no gain” |
| CWD intermediate win reversed | [closure](../provenance/exp019-closure.json) | +0.00627 terminal regression |
| Final selection precedes its official scores | [chronology](../docs/submission/EVIDENCE_AUDIT.md) | Not absence of all historical exposure |
| Exact dedup/screen/prefix controls | [data](../DATA_SOURCES.md) | Not proof of no semantic contamination |

No SOTA, universal reasoning improvement, universal method failure, energy advantage or causal single-control thermal claim is supported.
