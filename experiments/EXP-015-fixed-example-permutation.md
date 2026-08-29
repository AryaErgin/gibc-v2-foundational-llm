# EXP-015 - Fixed-example FineWeb-Edu phase permutation

**Status:** preregistered reconstruction preflight; no training or benchmark evaluation is authorized.

The original EXP-015 source-private-pool proposal remains blocked and historical. This revision changes neither model nor tokenization: it uses the immutable EXP-004 300,023,808-prediction-token stream and permutes only IDs of its existing 585,984 fixed windows.

## Immutable examples and attribution gate

The loader's window `i` reads stored token IDs `[512i : 512i+513]`; its 512 prediction targets are stored positions `[512i+1 : 512i+513]`. A deterministic replay of the original `GlobalDeduplicatedTokenMixer` must emit a source byte for every stored token while matching the immutable stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1` exactly. Prediction-token source totals must be FineWeb `200,017,577` and FineWeb-Edu `100,006,231`, exactly as recorded by the original manifest. Any mismatch blocks EXP-015.

Each window's `edu_count` is the count of FineWeb-Edu labels over exactly its existing 512 target-token positions. EOD tokens retain the source of the document that emitted them. No retokenization, re-packing, flattening, re-windowing, EOD change, or cross-document-context change is permitted.

## WSD-aligned schedules

Phase 1 is updates 1-7,324 / window IDs 0-468,735 (468,736 windows); phase 2 is 7,325-8,240 (58,624); phase 3 is 8,241-9,156 (58,624). All schedules are permutations of every original ID exactly once:

- A Static: original IDs `0..585983` in original order.
- B Cooldown-Edu: original phase 1, then LOW, then HIGH.
- C PreCooldown-Edu: original phase 1, then HIGH, then LOW.

LOW/HIGH are constructed only from original tail IDs `468736..585983`: rank ascending by `(edu_count, original_id)`, split lower/upper 58,624 IDs, then restore original relative ID order inside each block. The HIGH Edu share must exceed LOW by at least 0.15 absolute. B and C therefore have exactly identical phase 1 and identical LOW/HIGH membership; only the two block positions differ.

Arm A can reuse EXP-013 seed-42 WSD only if its static schedule is proved identical to the original ordered sequence IDs and the fixed stream provenance remains exact. The frozen comparison values are General `3.5252154767513275`, Edu `3.2107785046100616`, combined `3.3679969906806946`.

## Gates and future execution requirements

The predeclared capability contrast remains candidate minus Static combined NLL <= -0.010 with neither domain worse by more than 0.020 nat. The scheduler-phase interaction requires C minus B <= -0.010 under the same domain guard. A reversed B-over-C effect is reported honestly. Any promotion requires later paired seed-43 confirmation.

Before any future B/C training, tests must prove stream SHA and source totals, every ID exactly once, window/tensor identity to the original indexed window, A original-order identity, B/C phase-1 and block-swap identity, parameter count 49,860,480, unchanged WSD, and checkpoint restoration of arm/schedule hash/cursor. This revision does not authorize training.
