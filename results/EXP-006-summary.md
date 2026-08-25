# EXP-006 — 900M-Token Horizon Scaling

Status: `FULL HORIZON COMPLETE`. Training source commit: `3d3fd727adb17ebda52be7c06f51f0123b3dcff2`. EXP-006 holds Architecture Recipe v1 (the accepted EXP-005B 20.85M allocation) and Data Recipe v1 fixed while extending the training horizon and its cosine schedule to 27,468 updates. It is not a pure token-count intervention because the scheduler horizon changed.

## Exact run record

- Parameters: `20,848,512`
- Optimizer updates: `27,468`
- Prediction tokens: `900,071,424`
- Stored stream IDs: `900,071,425`
- Data manifest SHA-256: `be8dd1674b4d993483a6986710829eb8a35aff9d62d7e0e307d61c1145153b17`
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`
- Seed: `42`; context: `512`; physical batch: `32` sequences x `2` accumulation; effective batch: `32,768` prediction tokens
- Optimizer/schedule: AdamW beta1 `0.9`, beta2 `0.95`, eps `1e-8`, matrix weight decay `0.1`, clip `1.0`, peak/min LR `6e-4`/`6e-5`, 100-step warmup, cosine horizon `27,468`
- Precision: BF16 autocast with FP32 parameters and optimizer states
- Final train loss: `3.403317093849182`
- Mean throughput: `156553.99345383173` tok/s; final throughput: `156124.4097473811` tok/s
- Wall time: `5765.500875700003` s
- Peak allocated/reserved VRAM: `4892792320` / `5555355648` bytes
- Environment: Windows `10.0.26200`, Python `3.11.9`, PyTorch `2.13.0+cu132`, CUDA `13.2`, NVIDIA GeForce RTX 5090 Laptop GPU

## Dual frozen-validation trajectory

| Step | Prediction tokens | General loss | General PPL | Educational loss | Educational PPL | Combined loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 9.103128671646118 | 8983.354749419672 | 9.097827434539795 | 8935.857863244648 | |
| 9,156 | 300,023,808 | 3.750222384929657 | 42.53053909941152 | 3.4502077996730804 | 31.506938760123266 | 3.6002150923013687 |
| 18,312 | 600,047,616 | 3.5702926218509674 | 35.526987603604276 | 3.261716604232788 | 26.09429229068764 | 3.4160046130418777 |
| 27,468 | 900,071,424 | 3.487837463617325 | 32.71512350902944 | 3.174496442079544 | 23.91477436177875 | 3.3311669528484344 |

The within-run combined-loss improvements are `0.184210479259491` from 300M to 600M and `0.0848376601934433` from 600M to 900M. The final tranche exceeds the predeclared `0.05`-nat strongly-data-limited threshold.

## Comparison and decision

Against EXP-005B's 300M endpoint, EXP-006 changed general validation from `3.7012462317943573` to `3.487837463617325` (`-0.2134087681770323` nats), educational validation from `3.409231722354889` to `3.174496442079544` (`-0.23473528027534485` nats), and combined validation from `3.555238977074623` to `3.3311669528484344` (`-0.22407202422618866` nats).

Scientific decision: under the frozen 20.85M Architecture Recipe v1 and Data Recipe v1, the model remains strongly data-limited at 900M tokens according to the predeclared within-run final-tranche criterion. The EXP-006 step-9,156 checkpoint is not a controlled reproduction of EXP-005B's endpoint because their cosine schedule horizons differ. No claim of pure token-count causality follows. Benchmark capability is not recorded here; only the final checkpoint is authorized for the separately promoted EXP-006A evaluation.
