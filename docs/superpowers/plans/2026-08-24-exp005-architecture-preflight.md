# EXP-005 Architecture Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly provenance-checked, shared-trainer preflight for the approved deep/thin and wide/shallow EXP-005 candidates without authorizing full training.

**Architecture:** `configs/exp005a.yaml` and `configs/exp005b.yaml` express the sole candidate architecture allocation difference. The current full-run loader consumes the existing EXP-004 artifact directly for EXP-005, validates its exact stream/tokenizer/dual-validation provenance, and the current runner retains its sequential cursor/checkpoint behavior.

**Tech Stack:** Python 3.11, PyTorch BF16 CUDA training, pytest, YAML configuration.

**Spec:** `experiments/EXP-005.md`

## Global Constraints

- Never alter or duplicate the EXP-004 Data Recipe v1 stream; require SHA-256 `8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1`.
- Preserve tokenizer SHA `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` and both frozen validation hashes.
- Preserve seed 42, BF16 autocast/FP32 parameters, AdamW .9/.95/1e-8, 0.1 matrix decay, 32x2 physical batch, 100 warmup, and 9,156-step cosine horizon.
- Full EXP-005A/B training and benchmark evaluation are not authorized.

---

### Task 1: Candidate configuration and invariant tests

**Files:**
- Create: `configs/exp005a.yaml`, `configs/exp005b.yaml`
- Modify: `src/gibc_llm/utils.py`
- Test: `tests/test_exp005.py`

- [ ] Write failing tests that load both configs, assert 20,984,064/20,848,512 real-model parameters, and reject changed physical-batch or schedule/token arithmetic.
- [ ] Run the focused test and verify it fails because EXP-005 configs/config validation do not exist.
- [ ] Add only candidate dimensions and EXP-005 validation support; preserve every approved shared control.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Frozen EXP-004 artifact reuse guard

**Files:**
- Modify: `src/gibc_llm/full_run.py`
- Test: `tests/test_exp005.py`

- [ ] Write failing fixture tests that require the EXP-004 manifest identity, exact stream SHA, frozen tokenizer, and exact general/educational validation hashes for either candidate.
- [ ] Run the focused test and verify it fails because the loader only accepts the config experiment ID as manifest identity.
- [ ] Generalize the existing loader minimally so EXP-005 may reference only a valid EXP-004 full-stream artifact and cannot accept a changed stream or validation control.
- [ ] Re-run the focused test and verify it passes.

### Task 3: Shared runner architecture preflight

**Files:**
- Modify: `scripts/train_exp001_full.py`
- Test: `tests/test_exp005.py`

- [ ] Write failing tests for candidate parameter-count enforcement, predeclared 3,052/6,104/9,156 milestone defaults, and a bounded dry-run/resume cursor plan.
- [ ] Run the focused test and verify it fails because the runner only enforces the 8,392,960 EXP-001 count and lacks EXP-005 milestone defaults.
- [ ] Replace the baseline-specific parameter check with candidate-specific approved counts and extend existing milestone selection; do not change training, scoring, token ordering, or optimizer behavior.
- [ ] Re-run focused and full tests, then run `python -m pip check`.

### Task 4: Equal-step CUDA preflight evidence

**Files:**
- Create: ignored `artifacts/exp005a-preflight-run/`, `artifacts/exp005b-preflight-run/`
- Create: `results/EXP-005-preflight.md`

- [ ] Run the existing shared full runner for the same explicit bounded step count per candidate, against `artifacts/exp004-full`.
- [ ] Resume each from its saved checkpoint in a fresh process for one bounded update and verify step/cursor/token arithmetic.
- [ ] Record actual measured loss, throughput, VRAM, finite-gradient behavior, and estimated full-horizon runtime without interpreting short-run loss scientifically.
- [ ] Commit only code/tests/concise evidence; never commit streams, checkpoints, caches, or benchmark data.
