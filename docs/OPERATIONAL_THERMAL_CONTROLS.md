# Operational thermal controls

`inter_update_sleep_seconds` is an infrastructure control, not a scientific hyperparameter. It defaults to `0.0`; when nonzero, `train_smoke` sleeps only after a completed optimizer update and after all model, optimizer, scheduler, token, and cursor state has advanced. The sleep does not affect learning rate, batch order, RNG, gradient accumulation, validation cadence, checkpoints, or model mathematics.

The trainer records active-compute throughput separately from paced wall-clock throughput. Wall-clock run duration includes the intentional idle time. EXP-017A Attempt 3 used `0.300` seconds under combined external thermal controls. No individual control is assigned causal credit for the observed full-run stability.

## Frozen EXP-020 completion

EXP-020 completed with fixed 0.300-second pacing, 37.9415132885h total training wall time, mean active step throughput 104,140 tok/s and mean paced step throughput 53,276 tok/s. The combined thermally paced production configuration was stable for the full run. This is not causal evidence for any individual control and does not imply a continuous reliable CPU-temperature record.

Historical external controls were OMEN Balanced, Turbo/Boost OFF, AC maximum processor state 60%, AC power and AUTO fans; thread environment variables were fixed to one. These are recorded operational settings, not a new launch prescription. Model development and evaluation are frozen. [Final summary](../results/EXP-020-summary.md).
