# EXP-015 - Fixed-membership curriculum x WSD phase study

**Status:** preregistered preflight **BLOCKED**; no data artifact was built and no training was launched.

## Question

At fixed total FineWeb/FineWeb-Edu composition and exact fixed corpus membership, does placing an Edu-enriched block immediately before WSD cooldown improve held-out quality relative to placing the identical enriched exposure in cooldown?

Recipe-v3 + WSD remains the promoted baseline: 49,860,480 trainable parameters, seed 42, AdamW, BF16 autocast with FP32 parameters/optimizer, 32 sequences x 2 accumulation, context 512, frozen tokenizer and validations, and 300,023,808 non-cycled prediction tokens. No official benchmark or benchmark-derived example is part of this experiment.

The completed EXP-013 seed-42 WSD run is an historical scheduler control, not an automatically reusable EXP-015 static control. Its frozen stream is SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1` and its EXP-004 manifest is SHA-256 `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`.

## Fixed WSD phases

| Phase | Updates | 512-prediction sequences | Prediction tokens | Global LR |
| --- | ---: | ---: | ---: | --- |
| 1 | 1-7,324 | 468,736 | 239,992,832 | stable `6e-4` |
| 2, pre-cooldown | 7,325-8,240 | 58,624 | 30,015,488 | stable `6e-4` |
| 3, cooldown | 8,241-9,156 | 58,624 | 30,015,488 | cosine `6e-4` to `6e-5` |
| total | 1-9,156 | 585,984 | 300,023,808 | WSD unchanged |

The desired exact aggregate source allocation is 390,656 FineWeb sequences (200,015,872 prediction tokens) and 195,328 FineWeb-Edu sequences (100,007,936 prediction tokens): exactly 2:1 at the 512-prediction sequence unit. The following integer allocation is preregistered as the proposed future source-private-pool schedule; it is **not executable with the current artifact** and has not been materialized.

| Arm / phase | FineWeb sequences | FineWeb-Edu sequences | Total |
| --- | ---: | ---: | ---: |
| A phase 1, static | 312,491 | 156,245 | 468,736 |
| A phase 2, static | 39,082 | 19,542 | 58,624 |
| A phase 3, static | 39,083 | 19,541 | 58,624 |
| B phase 1 | 322,261 | 146,475 | 468,736 |
| B phase 2, ordinary | 39,083 | 19,541 | 58,624 |
| B phase 3, Edu-enriched | 29,312 | 29,312 | 58,624 |
| C phase 1 | 322,261 | 146,475 | 468,736 |
| C phase 2, Edu-enriched | 29,312 | 29,312 | 58,624 |
| C phase 3, ordinary | 39,083 | 19,541 | 58,624 |

Thus B and C have identical phase 1; B phase 2 equals C phase 3; B phase 3 equals C phase 2; and every candidate arm has the same aggregate source totals. The one-sequence rounding in the ordinary phase is unavoidable at a 58,624-sequence phase length; it is exactly compensated in phase 1. It does not affect aggregate 2:1 equality.

## Blocking data/packing finding

The current builder (`GlobalDeduplicatedTokenMixer`) selects whole documents from both sources and writes one concatenated uint16 token stream. The training dataset then creates each example by a fixed 513-token window at a global `next_sequence_index`; it records no source cursor, document ID list, source-to-token map, or source-to-example map. A window can therefore cross a document boundary and, because the mixer interleaves documents, a source boundary. The only durable cursor is a global sequential example index.

The existing EXP-004 manifest records aggregate actual contributions of 200,017,577 FineWeb and 100,006,231 FineWeb-Edu prediction tokens, selected document counts (244,245 / 78,398), and duplicate skips (cross-source 6 / 59; intra-source 14 / 1). It does not contain document/example IDs or their hashes. Those actual contributions are not the required exact 2:1 sequence allocation and cannot be attributed per fixed example from the retained stream.

Consequently, reordering the present global stream would change fixed-window contents, EOD/cross-document contexts, and source attribution - not only the time at which an otherwise identical example is consumed. Reconstructing source-private token/sequence pools would itself change packing and produce a new static stream, so it cannot establish byte-identical reuse of EXP-013. It would require a separately approved data-representation change and a fresh static control with newly materialized, source-level manifests. Under the hard packing-only-ordering requirement, EXP-015 must not launch from the current artifact.

No candidate-arm membership manifests, document-ID hashes, source-pool hashes, or no-duplicate/no-omission audits exist. Reporting invented hashes would be invalid. Existing data and checkpoint tests validate the current contiguous cursor only; they cannot validate the required source-private queues.

## Required conditions before any future launch

Only after an approved source-aware representation is designed and validated may a fresh static control and the B/C arms be considered. It must materialize and persist per-source document IDs, per-source token/sequence-pool hashes, exact source counts/tokens, duplicate/omission counts, a deterministic schedule state with source cursors, and resume data. Tests must prove:

- exact parameter count, step/token horizon, WSD boundaries, and frozen tokenizer/validation hashes;
- exact aggregate source totals and equality of source membership/pool hashes across all arms;
- byte-identical B/C phase 1 and exact phase-2/phase-3 swap;
- preserved example contents/EOD placement within each source pool;
- exact resume of both source cursors and schedule state with no duplicate or dropped example; and
- no official benchmark invocation.

## Preregistered decision rules

For curriculum capability, a candidate must beat the new static control by at least 0.010 nat combined held-out NLL, with neither General nor Edu regressing by more than 0.020 nat. For the scheduler-phase interaction, PreCooldown-Edu must beat Cooldown-Edu by at least 0.010 nat under the same domain-safety constraint. Negative or reversed results remain results. Any promotion requires a later paired seed-43 confirmation of the relevant comparison. No training, optimizer change, or official benchmark evaluation is authorized by this blocked preflight.
