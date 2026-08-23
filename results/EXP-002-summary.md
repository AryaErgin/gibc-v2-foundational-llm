# EXP-002 Training Record

EXP-002 held architecture, tokenizer, data family/order, validation artifact, optimizer, effective batch, context, and seed fixed while extending the horizon/schedule to 9,156 updates / 300,023,808 tokens. Training commit: `ce635f0aa67ee9c8ad8097c33af01ddc9a3a7845`; final checkpoint: `artifacts/exp002-full/run/checkpoints/checkpoint-step-9156.pt`.

Parameters: 8,392,960. Final FineWeb validation loss/PPL: 3.8954159319400787 / 49.17650318789857. First/final train loss: 9.074440002441406 / 3.771830677986145. Mean throughput: 312351.5265864468 tok/s. Wall time: 968.2099003999974 s. Tokenizer SHA-256: `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`; manifest SHA-256: `ab36ea1ffae92e3e93d3a57c4d369240c33456da93d7672919874ee157033779`.

Validation loss: 9.073967218399048 (step 0), 4.2281957268714905 (3052), 3.976299613714218 (6104), 3.8954159319400787 (9156). Relative to EXP-001 final loss 4.4311341643333435, the predeclared final improvement is 0.5357182323932648 nats, exceeding the >=0.10 materially-valuable threshold. Within EXP-002, the 100M-to-200M change was -0.25189611315727234 and 200M-to-300M was -0.0808836817741394.

Caveat: EXP-002 step 3052 is not a controlled reproduction of EXP-001 step 3052 because its cosine horizon is 9,156 rather than 3,052. This does not establish pure token-count causality.
