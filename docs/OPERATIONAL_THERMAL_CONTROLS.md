# Operational thermal controls

`inter_update_sleep_seconds` is an infrastructure control, not a scientific hyperparameter. It defaults to `0.0`; when nonzero, `train_smoke` sleeps only after a completed optimizer update and after all model, optimizer, scheduler, token, and cursor state has advanced. The sleep does not affect learning rate, batch order, RNG, gradient accumulation, validation cadence, checkpoints, or model mathematics.

The trainer records active-compute throughput separately from paced wall-clock throughput. Wall-clock run duration includes the intentional idle time. EXP-017A Attempt 3 used `0.300` seconds under combined external thermal controls. No individual control is assigned causal credit for the observed full-run stability.
