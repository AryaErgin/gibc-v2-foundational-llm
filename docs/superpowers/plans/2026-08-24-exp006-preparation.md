# EXP-006 Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a preparation-only, prefix-controlled 900M-token Data Recipe v1 experiment configuration and shared-runner validation without materializing or training it.

**Architecture:** EXP-006 reuses the EXP-004 deterministic global-deduplication mixer with targets scaled exactly threefold. Its materializer rebuilds the stream from the beginning and treats the EXP-004 byte prefix hash as a hard authorization gate. The existing full-runner remains shared and validates the EXP-006 manifest, milestone schedule, and cursor arithmetic.

**Tech Stack:** Python 3.11, PyTorch, NumPy memmaps, pytest, YAML.

**Spec:** User-provided EXP-006 preparation request, 2026-08-24.

## Global Constraints

- Do not materialize the 900M-token stream, train EXP-006, or run official benchmarks in this implementation task.
- Freeze EXP-005B architecture at 20,848,512 parameters and Data Recipe v1/tokenizer/validation hashes.
- Use 27,468 updates, 900,071,424 prediction tokens, physical batch 32x2, and a 27,468-step cosine horizon.
- The first 600,047,618 raw stream bytes must hash to the exact EXP-004 stream SHA before an EXP-006 manifest is written.
- Generated artifacts, caches, checkpoints, and local instructions remain untracked.

---

### Task 1: Test the EXP-006 controlled configuration and prefix primitives

**Files:**
- Create: `tests/test_exp006.py`
- Modify: `configs/exp006.yaml`
- Modify: `src/gibc_llm/utils.py`

- [ ] **Step 1: Write failing tests** for fixed model count, 900M arithmetic, 2:1 targets, physical batch, milestones, and prefix success/failure on a fixture stream.
- [ ] **Step 2: Run** `pytest -q tests/test_exp006.py` and confirm it fails because EXP-006 is not implemented.
- [ ] **Step 3: Implement** config validation and bounded prefix hashing.
- [ ] **Step 4: Run** `pytest -q tests/test_exp006.py` and confirm the primitive tests pass.

### Task 2: Test and implement deterministic prefix-controlled materialization

**Files:**
- Create: `src/gibc_llm/exp006.py`
- Create: `scripts/prepare_exp006.py`
- Modify: `src/gibc_llm/exp004.py` only if a small reusable mixer helper is necessary
- Test: `tests/test_exp006.py`

- [ ] **Step 1: Write failing tests** proving threefold targets preserve a fixture prefix and a mismatched EXP-004 prefix blocks artifact loading.
- [ ] **Step 2: Run** the focused tests and confirm expected missing-module/validation failures.
- [ ] **Step 3: Implement** a reusable prep path that copies only frozen EXP-004 controls, replays the two source streams from the beginning, globally deduplicates, verifies the raw prefix, and writes the manifest only on a match.
- [ ] **Step 4: Run** focused tests and confirm success.

### Task 3: Generalize runner validation and document the predeclared curve

**Files:**
- Modify: `src/gibc_llm/full_run.py`
- Modify: `scripts/train_exp001_full.py`
- Create: `experiments/EXP-006.md`
- Test: `tests/test_exp006.py`

- [ ] **Step 1: Write failing tests** for EXP-006 milestones and dry-run/resume cursor arithmetic.
- [ ] **Step 2: Run** focused tests and confirm the old 300M-only guard rejects EXP-006.
- [ ] **Step 3: Implement** 0/9156/18312/27468 milestones, parameter assertions, and an artifact validator that independently rechecks the manifest prefix record and raw stream prefix.
- [ ] **Step 4: Run** focused tests and confirm success.

### Task 4: Verify and publish source-only preparation work

**Files:**
- Modify: relevant files above

- [ ] **Step 1: Run** `pytest -q` and `pip check`.
- [ ] **Step 2: Inspect** `git diff --check`, status, and staged file list to exclude artifacts and `AGENTS.md`.
- [ ] **Step 3: Commit and push** source/config/test/documentation changes to public `main`.
