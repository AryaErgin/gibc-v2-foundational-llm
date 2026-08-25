# EXP-011 Preflight Evidence

Source commit for the 60-update preflight: `30ca55470bd6543efa19e8f27f6bc53cf988fb63` (the subsequent preparation-evidence commit records the artifact-count regression fix found before training started).

The exact selected Recipe v3 configuration instantiated at 49,860,480 trainable parameters. The runner accepted only the existing immutable EXP-006 900,071,424-prediction-token artifact after rechecking its tokenizer, stream, 300M EXP-004 prefix, frozen general validation, frozen Edu validation, 32 x 2 physical batch, and 45,777-step schedule from step zero.

The bounded 60-update run completed at step 60 / 1,966,080 prediction tokens / cursor 3,840. It had finite losses throughout, mean throughput `102,002.8457` tok/s, final throughput `89,966.3175` tok/s, peak allocated/reserved memory `7,686,099,968 / 8,491,368,448` bytes, and `21.0963` seconds wall time. Final general/Edu preflight losses were `7.2547822595 / 7.2862058282`; they are readiness evidence only, not an EXP-011 result.

A fresh process loaded the step-60 checkpoint from the same exact artifact and reached step 61 / 1,998,848 prediction tokens / cursor 3,904 with finite loss. Its final general/Edu losses were `7.2460433245 / 7.2754790783`; it used `7,687,993,856 / 8,363,442,176` allocated/reserved bytes. This verifies checkpoint, optimizer, RNG, scheduler, and sequential-cursor restoration for the 900M phase.

No official benchmark and no long-horizon token was run before the preflight passed.
