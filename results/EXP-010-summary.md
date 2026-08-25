# EXP-010A Near-Cap SwiGLU Depth/Width Allocation

Status: `FULL HORIZON COMPLETE`. The authorized EXP-010A candidate completed at source commit `0f33c0249139a3343f206245b0a64c0b12e2fa4e`: exact 49,985,504 trainable parameters; 9,156 updates; 300,023,808 prediction tokens; final cursor 585,984; and checkpoints/dual validations at 3,052 / 6,104 / 9,156. It retained the exact frozen EXP-004 stream, tokenizer, validations, 32 x 2 physical batch, seed, context, optimizer, BF16/FP32 precision, and 6e-4/6e-5 schedule.

| Model | General | Edu | Combined | Mean / final tok/s | Peak allocated / reserved | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recipe v3 control (EXP-008A) | 3.5561715066 | 3.2465001345 | 3.4013358206 | 98,395.41 / 90,263.71 | 7.686 / 8.491 GB | 3,075.0358 s |
| EXP-010A 608 x 10 SwiGLU | 3.5597349703 | 3.2542108595 | 3.4069729149 | 93,646.15 / 96,723.39 | 8.062 / 8.835 GB | 3,228.0944 s |

Candidate-minus-control combined loss is `+0.0056370944` nats, strictly below the predeclared 0.02-nat capability threshold. The split guard is not triggered because the candidate does not improve combined loss. This is an engineering tie, but EXP-010A fails the committed tiebreak: it is slower, uses more allocated memory, and has more transformer blocks. **Retain Near-Cap Architecture Recipe v3** (640 width, 9 layers, 20 x 32 heads, SwiGLU d_ff 1,728, 49,860,480 parameters).

No official benchmark was run. The full-horizon tie result is valid and unambiguous; the subsequent user authorization conditionally permits EXP-011 using Recipe v3 only.
