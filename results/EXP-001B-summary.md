# EXP-001B Readiness Hardening Summary

Status: PASS. This research-chat-requested stage preserves EXP-001A's bounded validation evidence and hardens future full-run correctness without starting the 3,052-update run.

Implemented and locally verified: lm-eval 0.4.9.1 TemplateLM causal pair tokenization (including trailing-space and empty-context behavior), disjoint rolling likelihood windows, an exact 100,007,936-prediction-token non-cycled uint16 stream design, input/target view equivalence, checkpointed next-sequence cursor continuity, public-split benchmark provenance locking, and FineWeb revision pinning.

Fresh local verification on implementation commit `2b96697655abef98e9dae3a3d08e1e39eecd01ed`: 29 pytest tests passed; `pip check` reported no broken requirements; exact parameter count remains 8,392,960. Fresh hardened bounded artifact manifest SHA-256: `bb29d36592459a480d350f7658272fc70eeacc2842c41518535e25223aa14062`; 34,332 documents scanned, 34,287 accepted, 45 rejected (0.1311%); tokenizer SHA-256 `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14`.

Fresh GPU checks: the 500-step tiny overfit reduced 9.0621 to 0.001672. The 60-update BF16 smoke trained 1,966,080 prediction tokens with microbatch 32 and accumulation 2: train loss 9.0744 to 7.2980, validation loss 9.0740 to 7.2433, final validation PPL 1398.66, mean/final-10 throughput 312,655/316,953 tokens/s, peak allocated/reserved 3.00/6.48 GiB, and exact checkpoint round trip. The smoke script's own Git subprocess was denied by Windows ownership policy, but the active verified commit during invocation was the implementation commit above.

Limit-1 lm-evaluation-harness integration is complete for HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText rolling likelihood. Each isolated task exited 0 and serialized its own result JSON. The earlier combined invocation was terminated by the execution runner before later tasks completed; it was not a TemplateLM interface, result-path, or Windows serialization defect. These calls are integration-only and are not benchmark scores. The full EXP-001 run remains unauthorized.

EXP-001A measurements in `results/EXP-001A-summary.md` remain historical values from their original artifact and must not be attributed to EXP-001B.
