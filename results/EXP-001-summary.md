# EXP-001 Baseline Training Record

Status: training complete. Transcribed exactly from `artifacts/exp001-full/summary.json`; this record did not modify, resume, or otherwise use the final checkpoint.

- Training commit: `fd99d1035c6c3d1e677620875426eb8a4cfcda93`; final checkpoint: `artifacts/exp001-full/checkpoints/checkpoint-step-3052.pt`.
- Parameters: 8,392,960. Seed 42; 3,052 updates; 100,007,936 prediction tokens; effective batch 32,768; microbatch 32; accumulation 2.
- FP32 parameters/AdamW state with CUDA BF16 autocast. AdamW beta1 0.9, beta2 0.95, eps 1e-8; matrix weight decay 0.1/no norm decay; clip 1.0. LR: 6e-4 peak, 6e-5 minimum, 100-step linear warmup, cosine decay through 3,052.
- Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`. Data manifest SHA-256: `f1b2bd8c6dbd73b2e31c795103e2fc1088f123c14021f3c966df2e6333ebd730`.
- Hardware/software: NVIDIA GeForce RTX 5090 Laptop GPU; CUDA 13.2; PyTorch 2.13.0+cu132; Python 3.11.9; Windows-10-10.0.26200-SP0.
- Final internal FineWeb validation loss/PPL: 4.4311341643333435 / 84.02666293188594.
- Mean/final throughput: 306750.2051883061 / 299815.7260517263 tokens/s. Peak allocated/reserved VRAM: 3218745856 / 4068474880 bytes. Wall time: 330.7768236000047 seconds.
- Estimated FLOPs: 5.035175155392512e+15, explicitly the approximate `6 x trainable parameters x training tokens` calculation.

Validation trajectory (step, loss, PPL): (0, 9.073967218399048, 8725.16988899444), (500, 5.358493864536285, 212.40479501991373), (1000, 4.910816252231598, 135.7501755815464), (1500, 4.697930335998535, 109.71985406472149), (2000, 4.581826567649841, 97.69267361503925), (2500, 4.4970256090164185, 89.74978295317315), (3000, 4.43733412027359, 84.54924285213718), (3052, 4.4311341643333435, 84.02666293188594).

Scientific interpretation: the run completed correctly; held-out FineWeb loss decreased monotonically through the horizon; 100M tokens did not demonstrate saturation; benchmark capability has not yet been measured.
