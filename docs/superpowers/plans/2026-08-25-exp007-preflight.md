# EXP-007 Near-Cap Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two near-cap controlled architecture configurations that reuse the exact EXP-004 artifact and validate both with equal bounded engineering preflights.

**Architecture:** The existing full-runner remains the sole trainer. EXP-007A/B are registered as frozen-EXP-004 consumers, so the artifact loader enforces the train-stream SHA, tokenizer and dual-validation hashes before construction. No data preparation path is added.

**Tech Stack:** Python 3.11, PyTorch, NumPy memmap, pytest, YAML.

**Spec:** User-provided EXP-007 near-cap architecture-preflight request, 2026-08-25.

## Global Constraints

- Do not materialize a stream, start either 9,156-step EXP-007 run, or run official EXP-007 benchmarks.
- Preserve the exact EXP-004 300,023,808-token stream and all Data Recipe v1 controls.
- Freeze physical batch at 32 sequences x 2 accumulation for the first preflight attempt; report rather than silently change an OOM condition.
- Verify real model counts 49,353,184 and 49,491,840, both below the 50M cap.

---

### Task 1: Frozen configurations and runner guards

**Files:**
- Create: `configs/exp007a.yaml`
- Create: `configs/exp007b.yaml`
- Modify: `src/gibc_llm/utils.py`
- Modify: `src/gibc_llm/full_run.py`
- Modify: `scripts/train_exp001_full.py`
- Test: `tests/test_exp007.py`

- [ ] Write a failing configuration/control test for both exact model counts, 138,656 parameter difference, 32x2 batch, 9,156-step arithmetic, and 585,984 examples.
- [ ] Run `python -m pytest -q tests/test_exp007.py` and observe missing EXP-007 config failures.
- [ ] Add only exact configurations and shared-runner EXP-004 frozen-artifact support.
- [ ] Run the focused test and confirm it passes.

### Task 2: Artifact and cursor regression coverage

**Files:**
- Modify: `tests/test_exp007.py`

- [ ] Add a fixture manifest carrying the exact EXP-004 stream/tokenizer/validation provenance.
- [ ] Assert that `load_full_run_artifact` accepts the exact fixture and rejects a changed stream SHA.
- [ ] Assert 60-update dry-run plus one-step resume arithmetic and 0/3052/6104/9156 milestones.
- [ ] Run the focused test and confirm it passes.

### Task 3: Bounded preflight evidence

**Files:**
- Create: `experiments/EXP-007.md`
- Create: `results/EXP-007-preflight.md`

- [ ] Run equal 60-update full-path invocations on the exact EXP-004 stream using default 32x2 batch, then one explicitly resumed update each.
- [ ] Record finite loss/gradients, throughput, peak allocated/reserved VRAM, cursor, and an update-only 300M runtime estimate.
- [ ] Do not interpret short-run loss as a candidate selection result.

### Task 4: Verification and public evidence

**Files:**
- Modify: relevant files above

- [ ] Run `python -m pytest -q` and `python -m pip check`.
- [ ] Inspect staged paths to exclude artifacts, checkpoints, caches, and `AGENTS.md`.
- [ ] Commit/push implementation and preflight evidence separately.
