# EXP-009 Near-Cap Learning-Rate Calibration

Status: `FULL HORIZON COMPLETE`. EXP-009 independently and sequentially calibrated the learning-rate amplitude for frozen Near-Cap Architecture Recipe v3. The existing EXP-008A 6e-4/6e-5 run is the control; it was not retrained. EXP-009A used 4e-4/4e-5 and EXP-009B used 8e-4/8e-5, retaining the 10:1 ratio, 100-update warmup, and exactly 9,156-step cosine horizon.

Both candidates completed at source commit `fb2555af5cd648cfa9a8b5511aaf419da90a2b5b`: 9,156 updates, 300,023,808 prediction tokens, final `next_sequence_index=585984`, and checkpoints/dual validations at 3,052 / 6,104 / 9,156. They used the exact 49,860,480-parameter Recipe v3 model; the immutable non-cycled EXP-004 stream SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`; tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`; frozen general/Edu validation tensors; seed 42; context 512; BF16 autocast with FP32 model/optimizer state; AdamW betas 0.9/0.95, eps 1e-8, matrix decay 0.1; clip 1.0; and native Windows, OMEN Performance mode, AC power, and 32 sequences x 2 accumulation / 32,768 prediction tokens per update.

| Schedule | General | Edu | Combined | Delta vs 6e-4 control | Mean / final tok/s | Peak allocated / reserved | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control 6e-4 / 6e-5 (EXP-008A) | 3.5561715066 | 3.2465001345 | 3.4013358206 | 0.0000000000 | 98,395.41 / 90,263.71 | 7.686 / 8.491 GB | 3,075.0358 s |
| EXP-009A 4e-4 / 4e-5 | 3.5922982097 | 3.2921635807 | 3.4422308952 | +0.0408950746 | 98,037.28 / 101,987.14 | 7.686 / 8.491 GB | 3,083.1388 s |
| EXP-009B 8e-4 / 8e-5 | 3.5453133583 | 3.2421827912 | 3.3937480748 | -0.0075877458 | 98,132.14 / 103,825.23 | 7.686 / 8.491 GB | 3,084.3461 s |

The primary metric is `(general + Edu) / 2`. Relative to the existing 6e-4 control, EXP-009A has a disadvantage greater than or equal to 0.01 nats and is **rejected**. EXP-009B improves by 0.0075877458 nats, strictly below the predeclared 0.01-nat promotion threshold. It is therefore an **engineering/statistical tie**; retain **peak/min LR 6e-4/6e-5** to avoid proxy-horizon over-tuning.

The split guard is `PASS / not triggered`: EXP-009B, the only candidate with lower combined loss, improved both general (-0.0108581483 nats) and Edu (-0.0043173432 nats) relative to control. Both 60-update preflights were finite and each fresh-process checkpoint resume reached step 61 at the exact cursor; see `results/EXP-009-preflight.md`.

No official benchmark was run. No long-horizon training beyond these two authorized 300M ablations, no architecture/data/batch change, and no EXP-010 follow-up is authorized.
