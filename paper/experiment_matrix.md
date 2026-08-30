# Experiment matrix (working)

| Area | Frozen comparison | Outcome | Promotion status |
|---|---|---|---|
| WSD | Recipe-v3 schedule comparison, 300M, seeds 42/43 | Replicated positive | Promoted recipe |
| LLR | Recipe-v3 + WSD, 300M seed 42 | Negative transfer | Rejected; no tuning |
| Fixed-example placement | EXP-015 A/B/C, same fixed examples | Domain tradeoff, no broad gain | Rejected |
| Magma | EXP-016 contemporaneous control/treatment, 300M seed 42 | Strong negative transfer | Rejected; no tuning |
| WSD horizon | EXP-017A versus fixed 2.4B cosine reference | Pending | No decision |

The table is not a benchmark leaderboard and must not be used to infer
unmeasured official task performance.
