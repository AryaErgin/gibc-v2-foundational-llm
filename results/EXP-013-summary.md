# EXP-013 — Warmup-Stable-Decay scheduler ablation

**Status:** completed and promoted on 2026-08-29. No official benchmark was
run for this scheduler experiment.

## Protocol

The four arms used the exact 49,860,480-parameter Recipe v3 model, frozen
FineWeb/FineWeb-Edu 2:1 300,023,808-token prefix (stream SHA-256
`8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`),
frozen tokenizer SHA-256
`c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`,
and frozen General/Edu validation tensors. Every arm completed 9,156 updates,
300,023,808 prediction tokens, and deterministic cursor 585,984 from fresh
initialization; no arm resumed.

WSD used a 100-update linear warmup, `6e-4` stable phase through step 8,240,
and a 916-update cosine cooldown starting at step 8,241. Logged and pure
scheduler values agree: steps 100, 101, and 8,240 are `6e-4`; step 8,241 is
`0.000599998412030558`; step 9,156 is exactly `6e-5`.

## Results

| Seed | Schedule | General loss | Edu loss | Combined loss | Wall s | Mean tok/s | Peak allocated VRAM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Cosine | 3.5564022958278656 | 3.247113198041916 | 3.4017577469348907 | 3982.595513100001 | 76067.3597637185 | 7,686,099,968 B |
| 42 | WSD | 3.5252154767513275 | 3.2107785046100616 | 3.3679969906806946 | 3670.8824978999983 | 82203.7618492167 | 7,686,099,968 B |
| 43 | Cosine | 3.555551588535309 | 3.247914671897888 | 3.4017331302165985 | 3705.1132580999983 | 81411.93453244497 | 7,686,099,968 B |
| 43 | WSD | 3.5318978428840637 | 3.2168504297733307 | 3.374374136328697 | 3699.4708794000035 | 81722.50225145479 | 7,686,099,968 B |

Seed-42 discovery deltas (WSD minus cosine): combined
`-0.03376075625419617`, General `-0.031186819076538086`, Edu
`-0.03633469343185425`. It is a **CAPABILITY WIN** under the preregistered
`<= -0.020` threshold, with no domain regression.

Seed-43 paired-confirmation deltas: combined `-0.027358993887901306`, General
`-0.023653745651245117`, Edu `-0.031064242124557495`. It beats the paired
cosine control by at least 0.010 nat and has no domain regression; the
confirmation succeeds.

## Checkpoints and resume gates

Both WSD stable checkpoints contain model, optimizer, scheduler, RNG, and
sequential data-cursor state at completed update 8,240, immediately before
cooldown update 8,241. Their run state, data cursor, and schedule fields were
independently reloaded and verified.

| Seed | WSD stable checkpoint SHA-256 | WSD terminal checkpoint SHA-256 |
|---:|---|---|
| 42 | `c10198c10bfb94494be9552f59aaf59c2082d5a2c0b89266fb93ba3fdab21b78` | `217e7cdd0503837191596895465b8a87ed068806ca302f5421777fdcaed1bcfb` |
| 43 | `e45ecf89ee409855dc894592f3803582ebf65be07efcbd2a2453b75c2bdd9b99` | `f4bd7b5a7a29d2a70009dde1da2eea924eba6ec44b5440ed7e1949df302ff007` |

The seed-42 source/spec commit was
`bcc7e0df6208a77377c09273c9836ddf41267e3c`; seed-43 confirmation used
`838ef26`.

## Decision

**Model Recipe v3 + WSD becomes the frozen promoted training baseline for
EXP-014.** This decision changes the selected scheduler only; it does not
authorize an EXP-014 run, a benchmark, a model publication, or retrospective
tuning of these completed experiments.
