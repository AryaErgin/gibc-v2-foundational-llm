# EXP-008A Near-Cap GELU versus SwiGLU Ablation

Status: `COMPLETE`. This record covers the one authorized 9,156-update / 300,023,808-token candidate run. No official benchmark or training beyond 300M was run.

Candidate source commit: `71f3ebf0babb9253437bd937e024260b82fd4cf8`.

## Frozen controls and candidate invariant

EXP-008A retained the exact frozen EXP-004 non-cycled first 300,023,808-token stream (SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`), tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, frozen general/Edu validation tensors, data order, context 512, seed 42, BF16 autocast with FP32 model/optimizer state, AdamW (0.9, 0.95, eps 1e-8, matrix decay 0.1), clip 1.0, peak/min LR 6e-4/6e-5, 100 warmup updates, cosine horizon 9,156, and physical/effective batch 32x2 / 32,768 prediction tokens.

The only principal scientific change was the MLP: EXP-007B's two-projection exact-GELU d_ff 2,560 versus EXP-008A's three-projection, SiLU-gated SwiGLU d_ff 1,728. Candidate trainable count was independently derived and instantiated as **49,860,480**, below the 50,000,000 cap. Native Windows, OMEN Performance mode, and original AC power were retained.

## Equal-token result

| Model | Final general loss | Final Edu loss | Final combined loss |
| --- | ---: | ---: | ---: |
| EXP-007B GELU control | 3.5819328129 | 3.2808714211 | 3.4314021170 |
| EXP-008A SwiGLU candidate | 3.5561715066 | 3.2465001345 | 3.4013358206 |

Candidate minus control combined loss: `-0.0300662965` nats. Candidate advantage: `0.0300662965` nats. The predeclared capability-winner threshold is `0.02` nats.

## Decision

SwiGLU is the **capability winner** for this near-cap equal-token ablation: the candidate advantage exceeds the predeclared threshold. This did not use post-hoc tuning, a control retraining run, or an official benchmark.

## Engineering record

EXP-008A completed all 9,156 updates and 300,023,808 prediction tokens with final cursor 585,984. Mean/final throughput was 98,395.41 / 90,263.71 tok/s. Peak allocated/reserved memory was 7,686,099,968 / 8,491,368,448 bytes (7.686 / 8.491 GB). Full runner wall time was 3,075.0358 s; this includes the required initial/final and milestone validation/checkpoint work.

No official benchmark was run. Do not start EXP-009, any final-style near-cap run, or any 900M/1.5B/other >300M run without research review and a new authorization.
