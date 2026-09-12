# EXP-020 — frozen final competition model

**Execution: FULL HORIZON COMPLETE. Official evaluation: COMPLETE. Model selection: frozen terminal checkpoint.** No further training, checkpoint selection or benchmark rerun is authorized.

## Final training

Recipe-v3, ordinary AdamW, full-horizon cosine, seed 42, fresh initialization, QK-Norm OFF, CWD OFF. 49,860,480 trainable parameters. Context 512; microbatch 32; accumulation 2; 32,768 prediction tokens/update. Step 219,726; 7,199,981,568 prediction tokens; cursor 14,062,464 sequences.

One NVIDIA GeForce RTX 5090 Laptop GPU. WSL2; Python 3.11.9; torch 2.13.0+cu132; CUDA 13.2. BF16 forward, FP32 model/optimizer state.

| Measured quantity | Value |
|---|---:|
| Final training wall seconds | 136,589.44783848198 |
| Final training wall hours | 37.9415132885 |
| Mean active step throughput, tok/s | 104,140.02524072649 |
| Mean paced step throughput, tok/s | 53,275.50408849002 |
| Total tokens / full wall seconds, tok/s | 52,712.5754 |
| Fixed idle after update | 0.300 s |
| Peak allocated VRAM | 7,876,458,496 bytes |
| Peak reserved VRAM | 8,680,112,128 bytes |

Step-throughput means are not interchangeable with total tokens/full wall time. The latter also reflects validation/checkpoint overhead. The final-run wall time is not the full campaign cost. Approximate 6NT training compute is 2.154×10^18 FLOPs, a conventional analytical estimate, not measured kernel work or energy.

The combined thermally paced production configuration completed the full run. Do not attribute stability to any single control or claim continuous CPU-temperature coverage.

## Frozen internal validation (NLL, nats)

| Step | Prediction tokens | General | Edu | Combined |
|---:|---:|---:|---:|---:|
| 0 | 0 | 9.145263433 | 9.148761272 | 9.147012353 |
| 45,777 | 1,500,020,736 | 3.407622755 | 3.079545468 | 3.243584111 |
| 73,242 | 2,399,993,856 | 3.342972964 | 3.015445769 | 3.179209366 |
| 91,553 | 3,000,008,704 | 3.310950160 | 2.981790662 | 3.146370411 |
| 109,863 | 3,599,990,784 | 3.274529666 | 2.956444740 | 3.115487203 |
| 128,174 | 4,200,005,632 | 3.244010776 | 2.912871063 | 3.078440920 |
| 146,484 | 4,799,987,712 | 3.202632248 | 2.873861730 | 3.038246989 |
| 164,795 | 5,400,002,560 | 3.173533052 | 2.838792622 | 3.006162837 |
| 183,105 | 5,999,984,640 | 3.133437186 | 2.803406835 | 2.968422011 |
| 201,416 | 6,599,999,488 | 3.110759914 | 2.776944965 | 2.943852440 |
| 219,726 | 7,199,981,568 | 3.100797236 | 2.769774169 | 2.935285702 |

Combined = (General + Edu)/2. Step 0 is initialization. Official benchmark tasks are not these validation sets.

The preregistered candidate set begins at 4.8B. Terminal is default; choose lowest Combined among late candidates, preferring later within 0.003 NLL. Terminal has the lowest Combined, so checkpoint-step-219726.pt is selected. Required benchmark scores played no role in this selection.

EXP-012's terminal Combined is 3.020453929901123; endpoint change is −0.0851682275533676. Earlier EXP-020 positions follow a different cosine horizon from EXP-011/012 and are not matched-schedule ablations.

## Final official results

WikiText-103 token PPL **31.783151406614728**; mean NLL **3.4589363193959604**; BPB **1.3600661174573045**. HellaSwag acc_norm **30.163314%**; ARC-Easy acc_norm **39.141414%**; PIQA acc_norm **60.446137%**; WinoGrande acc **48.776638%**.

[RESULTS.md](../RESULTS.md) provides comparisons, uncertainty, partitions and interpretation. CPU FP32 zero-shot scoring completed 2026-09-08 after selection. No new model/selection decision follows.

## Provenance

Training source: `d88800733846c7a30e2044fa0a20f9e4a448f328`. Builder: `79a61239bfa216e812fb93c81b082812dfeeca89`. The four raw lm-eval artifacts record `raw_lm_eval_result.git_hash = 3733279`, resolving uniquely to `37332797909df963ca7c77a945cea8752b60d481`. The full SHA expands recorded execution provenance, not merely the presence of evaluator code in Git.

Checkpoint: 598,447,091 bytes; SHA-256 `95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89`.

Full hashes, paths, task definitions, runtime and source accounting are preserved in the [derived audit digest](exp020-submission-evidence.json). The audit independently rehashed the full corpus and both historical prefixes and read the checkpoint tensors on CPU without executing the model. All 219,726 training rows passed finite/sequential token/cursor checks.
