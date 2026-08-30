# EXP-016 Magma preflight evidence

Status: completed before either formal EXP-016 arm. This is implementation and
infrastructure evidence only; it is not model-selection evidence and contains
no frozen validation result.

The focused single-threaded test module passed 8 tests. It covered the
49,860,480-parameter invariant; the 63-block / 44,605,440-parameter mapping;
dense AdamW moments under a zero mask; identity-mode AdamW+WSD trajectory;
known alignment and EMA values; Bernoulli-frequency diagnostics; dedicated RNG
isolation; CPU and CUDA checkpoint/resume continuity; and unchanged WSD
endpoints. The read-only preflight independently verified Schedule A
`39c509f59489d125904be61e7e3094e0e87af5ee7ead46afe6742cac35185eb2`, frozen
stream `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, and
all frozen General/Edu validation-array hashes. No benchmark was invoked.

The bounded WSL Magma smoke used the frozen treatment configuration, physical
batch 32, accumulation 2, BF16, seed 42, and Schedule A for 60 optimizer
updates / 1,966,080 prediction tokens. It deliberately performed no terminal
validation and saved no experiment checkpoint. Mean throughput after a
10-update warm-up was 79,076.26 tokens/s; final throughput was 79,149.01
tokens/s. Relative to the prior WSL production baseline of 85,149 tokens/s,
the estimated implementation overhead was 7.13%, within the preregistered
10% operational-efficiency guard. Peak allocated/reserved VRAM was
7,874,934,784 / 8,665,432,064 bytes; sampled GPU peak temperature/power was
61 C / 93.47 W; utilization peaked at 99%; loss stayed finite; and no data
stall was observed. This result does not establish capability.
