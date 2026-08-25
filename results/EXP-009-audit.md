# EXP-009 Post-Commit Audit

Audit target: committed EXP-009 evidence at `8a3fd50f39043d46f2a4d53796e9932d41040732`. Result: **PASS**.

The local machine-readable full-run summaries and terminal `run_end` records agree with `results/EXP-009-summary.md`:

| Run | General | Edu | Combined | Delta versus EXP-008A control |
| --- | ---: | ---: | ---: | ---: |
| Control EXP-008A, 6e-4/6e-5 | 3.5561715066432953 | 3.2465001344680786 | 3.4013358205556870 | 0.0000000000000000 |
| EXP-009A, 4e-4/4e-5 | 3.5922982096672060 | 3.2921635806560516 | 3.4422308951616287 | +0.0408950746059418 |
| EXP-009B, 8e-4/8e-5 | 3.5453133583068848 | 3.2421827912330627 | 3.3937480747699738 | -0.0075877457857132 |

Both candidate summaries report exactly 49,860,480 parameters, 9,156 updates, 300,023,808 prediction tokens, final cursor 585,984, fixed 32 x 2 batch, checkpoints at 3,052 / 6,104 / 9,156, and training source commit `fb2555af5cd648cfa9a8b5511aaf419da90a2b5b`.

The artifact loader independently validated the actual EXP-004 stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`, tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`, and manifest SHA-256 `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`. Fresh-process preflight resumes for both candidates advanced from step 60 / cursor 3,840 to step 61 / cursor 3,904 with finite losses.

The predeclared decision is correctly applied. EXP-009A is worse by more than 0.01 nats and is rejected. EXP-009B is better by 0.0075877458 nats, strictly inside the absolute 0.01-nat tie band; retaining peak/min LR 6e-4/6e-5 is therefore required. The split guard is not triggered because EXP-009B improves both individual validation splits. No training or benchmark was performed during this audit.
