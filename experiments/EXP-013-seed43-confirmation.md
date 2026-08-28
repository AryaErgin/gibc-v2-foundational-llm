# EXP-013 seed-43 paired confirmation

**Authorization:** conditional authorization triggered by the completed valid
seed-42 EXP-013 capability win.

## Discovery result

At seed 42, Arm C combined frozen validation loss was `3.401757746934891`.
Arm W combined loss was `3.3679969906806945`, a WSD-minus-cosine delta of
`-0.0337607562541965`. General and Edu both improved, so the preregistered
seed-42 discovery threshold was met.

## Paired confirmation

Run `EXP-013-C43` first, then `EXP-013-W43`, both from fresh seed-43
initialization. Every condition other than the already-approved schedule arm
and training seed is identical to EXP-013: Recipe v3, 49,860,480 parameters,
the exact frozen 300,023,808-token stream, frozen tokenizer, validation
tensors, AdamW/batch/context/LR endpoints, and native system configuration.

The primary confirmation statistic is seed-43 `WSD combined loss − cosine
combined loss`. Confirmation succeeds only when that delta is at most
`-0.010` nat and neither General nor Edu loss regresses by more than `0.020`
nat. This is a paired seed-43 comparison only; seed-43 WSD must not be
compared to seed-42 Arm C.

No official benchmark, additional seed, EXP-014 run, or longer-horizon
continuation is authorized by this confirmation.
