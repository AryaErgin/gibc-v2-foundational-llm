# EXP-011 Long-Horizon Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely prepare and execute the conditionally authorized EXP-011 1.5B-token Recipe v3 run without changing its frozen scientific controls.

**Architecture:** Reuse the existing eager-PyTorch full-run path. Add a strictly EXP-011-only two-phase artifact gate: the validated EXP-006 900M stream is accepted only through step 27,468, while a separately built EXP-011 1.5B stream is accepted only for the resume phase after raw-byte 900M-prefix verification.

**Tech Stack:** Python 3.11, PyTorch 2.13 CUDA BF16, YAML, pytest, uint16 token streams.

**Spec:** `experiments/EXP-011.md`

## Global Constraints

- Use Recipe v3 only: 49,860,480 parameters; 640 width; 9 layers; 20 x 32 heads; SwiGLU d_ff 1,728.
- Preserve Data Recipe v1, the frozen tokenizer/validation tensors, seed 42, context 512, 32 x 2 physical batch, AdamW, BF16, and 6e-4/6e-5 LR.
- Schedule horizon is exactly 45,777 from step zero; total is exactly 1,500,020,736 prediction tokens.
- EXP-006 remains immutable; checkpoint/artifact files are local-only.
- Stop on every hash, prefix, parameter, finite-loss, environment, or resume failure. Do not run official benchmarks.

---

### Task 1: Create EXP-011 control specification and configuration

**Files:**
- Create: `experiments/EXP-011.md`
- Create: `configs/exp011.yaml`
- Test: `tests/test_exp011.py`

- [ ] Write tests that load the config, require exact Recipe v3 dimensions, exact 45,777-step/1,500,020,736-token arithmetic, 2:1 source targets, and milestones.
- [ ] Run the focused test and observe the missing configuration failure.
- [ ] Add the immutable specification and config.
- [ ] Re-run the focused test and require pass.

### Task 2: Add the deterministic 1.5B extension builder and prefix gate

**Files:**
- Create: `src/gibc_llm/exp011.py`
- Create: `scripts/prepare_exp011.py`
- Test: `tests/test_exp011.py`

- [ ] Write tests for exact 900M raw-byte prefix verification and the 3:1 scale preserving the EXP-006 fixture prefix.
- [ ] Run the tests and observe the missing module/function failure.
- [ ] Implement the builder by reusing the complete global-deduplication/data-screening path, copying only frozen tokenizer/validation controls, and rejecting any nonidentical EXP-006 prefix.
- [ ] Re-run the focused tests and require pass.

### Task 3: Gate two-phase use in the production runner

**Files:**
- Modify: `src/gibc_llm/utils.py`
- Modify: `src/gibc_llm/full_run.py`
- Modify: `scripts/train_exp001_full.py`
- Test: `tests/test_exp011.py`

- [ ] Write tests that reject a 900M artifact beyond step 27,468, reject the wrong artifact identity, and accept only phase-compatible manifest/token accounting.
- [ ] Run the tests and observe the missing EXP-011 behavior failure.
- [ ] Add minimal EXP-011-only accepted capacities, identities, mixture accounting, fixed milestones, and run-end capacity validation. Preserve all existing experiment checks unchanged.
- [ ] Re-run focused tests and require pass.

### Task 4: Verify and commit preparation

**Files:**
- Modify: `EXPERIMENT_LOG.md`
- Modify: `PROJECT_PLAN.md`

- [ ] Run full pytest, parameter/config/hash checks, existing 900M artifact verification, and 60-step preflight/fresh-process resume.
- [ ] Record only measured evidence, commit preparation without artifacts/checkpoints, and push before starting EXP-011.
- [ ] Start the 900M phase only when all conditions pass.
