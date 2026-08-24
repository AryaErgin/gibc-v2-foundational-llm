# EXP-005 Architecture Allocation Experiment

Status: `FULL HORIZON COMPLETE`. Training source commit: `329f02ffe54c501da8815d8db8858b955cc1b9ec`. EXP-005 compares only the approved allocation of approximately 21M parameters under the frozen EXP-004 Data Recipe v1. It is not a claim about wide versus deep architectures generally.

## Frozen common controls

- Exact EXP-004 Data Recipe v1 uint16 stream SHA-256: `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`
- Referenced data manifest SHA-256: `7b96284987ab81a2c1704907689aded6623bdf58c5037d6ba76c9f1a87d9407f`
- Prediction tokens / updates: 300,023,808 / 9,156
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`
- Seed 42; context 512; BF16 autocast with FP32 parameters; AdamW beta1 .9, beta2 .95, eps 1e-8; 0.1 matrix weight decay; peak/min LR 6e-4/6e-5; 100-step warmup; cosine horizon 9,156; gradient clip 1.0; physical batch 32 x 2 = 32,768 prediction tokens/update.
- Hardware/software: NVIDIA GeForce RTX 5090 Laptop GPU; Windows-10-10.0.26200-SP0; Python 3.11.9; PyTorch 2.13.0+cu132; CUDA 13.2.

The frozen general and educational validation tensors were used at steps 0, 3052, 6104, and 9156. Their hashes remain those recorded by EXP-004.

## Results

| Candidate | Allocation | Parameters | Final train loss | Final general loss / PPL | Final educational loss / PPL | Final combined loss | Mean throughput (tok/s) | Wall seconds | Peak allocated / reserved bytes |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| EXP-005A deep/thin | d_model 256; 24 layers; 8 heads; head_dim 32; d_ff 1024 | 20,984,064 | 3.611304759979248 | 3.7769566774368286 / 43.68289808942507 | 3.499862164258957 / 33.11088778039228 | 3.6384094208478928 | 120889.67674597578 | 2490.6542591000034 | 6677613056 / 7879000064 |
| EXP-005B wide/shallow | d_model 384; 10 layers; 12 heads; head_dim 32; d_ff 1536 | 20,848,512 | 3.539416193962097 | 3.7012462317943573 / 40.49774249903524 | 3.409231722354889 / 30.242001078563153 | 3.555238977074623 | 156476.33467190273 | 1924.2128658000147 | 4892792320 / 5555355648 |

| Step | Prediction tokens | EXP-005A general | EXP-005A edu | EXP-005B general | EXP-005B edu |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 9.05635142326355 | 9.058774471282959 | 9.103128671646118 | 9.097827434539795 |
| 3052 | 100,007,936 | 4.190706849098206 | 3.9573460519313812 | 4.051126688718796 | 3.787585198879242 |
| 6104 | 200,015,872 | 3.8812103867530823 | 3.621362179517746 | 3.798127442598343 | 3.522021770477295 |
| 9156 | 300,023,808 | 3.7769566774368286 | 3.499862164258957 | 3.7012462317943573 | 3.409231722354889 |

## Predeclared comparison and decision

EXP-005B's combined-loss advantage over EXP-005A is 0.08317044377326965 nats; its general and educational advantages are 0.07571044564247131 and 0.090630441904068 nats. This exceeds the predeclared 0.02-nat engineering-tie threshold. EXP-005B also achieved approximately 29.44% higher mean throughput, 22.74% lower wall time, and 26.73% lower peak allocated VRAM.

Against EXP-004, EXP-005B changed general validation from 3.9005910456180573 to 3.7012462317943573 (-0.1993448138237 nats), educational validation from 3.6293802559375763 to 3.409231722354889 (-0.22014853358268738 nats), and combined validation from 3.7649856507778168 to 3.555238977074623 (-0.20974667370319366 nats).

Scientific decision: EXP-005B wide/shallow is accepted as Architecture Recipe v1 at the tested approximately 21M scale. Under this frozen data/training recipe, it clearly outperformed the deep/thin alternative while also being materially more efficient. The conclusion is conditional on these two configurations and controls; it does not claim universal superiority of wider architectures.

The EXP-005A checkpoint must not be resumed or benchmarked. Only the final EXP-005B checkpoint `artifacts/exp005b-full/run/checkpoints/checkpoint-step-9156.pt` is authorized for the separately approved EXP-005B promoted evaluation. EXP-006 is not authorized.
