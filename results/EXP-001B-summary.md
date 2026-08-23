# EXP-001B Readiness Hardening Summary

Status: PARTIAL. This research-chat-requested stage preserves EXP-001A's bounded validation evidence and hardens future full-run correctness without starting the 3,052-update run.

Implemented and locally verified: lm-eval 0.4.9.1 TemplateLM causal pair tokenization (including trailing-space and empty-context behavior), disjoint rolling likelihood windows, an exact 100,007,936-prediction-token non-cycled uint16 stream design, input/target view equivalence, checkpointed next-sequence cursor continuity, public-split benchmark provenance locking, and FineWeb revision pinning.

Fresh local verification: 29 pytest tests passed; `pip check` reported no broken requirements; exact parameter count remains 8,392,960.

Unresolved preflight boundary: a fresh bounded artifact could not be completed in this session. Hugging Face returned HTTP 429 for the pinned HellaSwag revision during the all-public-split index build, and subsequent foreground attempts terminated before producing a manifest. No tokenizer hash, hardened decontamination counts, fresh tiny-overfit/BF16 losses, throughput/VRAM, or limit-1 harness integration result is recorded here. The full EXP-001 run remains unauthorized.

EXP-001A measurements in `results/EXP-001A-summary.md` remain historical values from their original artifact and must not be attributed to EXP-001B.
