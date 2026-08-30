# EXP-017A Thermal Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable, separate long-run progress logging without changing Recipe-v3/WSD numerical semantics; keep EXP-017A stopped unless existing read-only CPU-package telemetry proves the required automatic safety guard.

**Architecture:** `train_smoke` emits an optional post-update record. `DurableProgressLogger` appends, flushes, and fsyncs it without supplying input to model, data, optimizer, scheduler, or RNG. The full runner owns `progress.jsonl` and only invokes this path at a positive configurable interval no greater than 100 updates.

**Tech Stack:** Python 3.11, PyTorch 2.13.0+cu132, JSONL, focused pytest, Linux fsync.

**Spec:** `experiments/EXP-017A-2.4b-wsd.md` and the approved overnight thermal-recovery request dated 2026-08-30.

## Global Constraints

- Do not change Recipe-v3, data, tokenizer, optimizer, WSD, batch, context, seed, or validation.
- No full pytest, official benchmark, scientific result, resumption, or relaunch from this work.
- Progress records contain only completed-step scalars: step, prediction tokens, data cursor, loss, LR, timestamp, throughput.
- Progress logging must not consume or restore model/global/data RNG and must not modify model, optimizer, scheduler, batches, or checkpoints.
- Without a trustworthy existing read-only CPU-package source, set `UNATTENDED_EXP017A_ALLOWED = NO`; do not shakedown or launch Attempt 2.

---

### Task 1: Test-first durable progress hook

**Files:**
- Modify: `src/gibc_llm/train.py`
- Modify: `scripts/train_exp001_full.py`
- Create: `tests/test_exp017a_durable_progress.py`

**Interfaces:**
- `DurableProgressLogger(path: Path).log(record: dict[str, Any]) -> None`
- optional `progress_logger` and `progress_interval_updates` keywords on `train_smoke`
- `--progress-interval-updates` on the full runner, default 100 and positive

- [ ] Write a failing test that expects exactly steps 2 and 4 in an interval-2 JSONL stream, including token count, cursor, loss, LR, timestamp, and throughput.
- [ ] Run the single test with CPU thread limits; confirm it fails because the progress interface is absent.
- [ ] Implement a minimal append/flush/fsync writer and post-update hook. Emit only when `state.step % interval == 0`; do not invoke RNG APIs.
- [ ] Write a failing deterministic equality test: the same tiny CPU run, with and without progress logging, has identical model parameters, optimizer state, scheduler state, run state, and checkpoint payload.
- [ ] Run the focused module with CPU thread limits; confirm both tests pass.

### Task 2: Focused checks and documentation

**Files:**
- Modify: `EXPERIMENT_LOG.md`
- Modify: `PROJECT_PLAN.md`
- Create: `paper/outline.md`
- Create: `paper/claims_and_evidence.md`
- Create: `paper/related_work.md`
- Create: `paper/experiment_matrix.md`
- Create: `paper/figures_plan.md`
- Create: `paper/reviewer_risks.md`
- Create: `docs/storage-manifest-2026-08-30.md`

- [ ] Run only `test_exp017a_durable_progress.py`, `test_exp017a_preflight.py`, and `test_checkpoint.py`, with all requested CPU thread limits. Stop if a verified package source reports 95 C sustained.
- [ ] Record Attempt 1 exactly as `TECHNICAL_ABORT_THERMAL` and `NO SCIENTIFIC RESULT`; preserve artifact hashes and describe the original chunk-buffering limitation. Do not edit scientific results tables.
- [ ] Create the provisional paper skeleton. It records WSD’s replicated 300M positive, LLR negative transfer, fixed-example curriculum specialization tradeoff, Magma negative transfer, and EXP-017A pending. It excludes universal and novelty claims.
- [ ] Inventory only directory sizes; classify artifacts `KEEP_CRITICAL`, `KEEP_UNTIL_PAPER`, `ARCHIVE`, or `DELETE_CANDIDATE`; perform no deletion, recursive hash, compression, or data rebuild.

### Task 3: Read-only thermal gate and reconciliation

**Files:**
- Modify: `PROJECT_PLAN.md`
- Modify: `experiments/EXP-017A-2.4b-wsd.md`

- [ ] Discover only existing Windows/HP/OMEN/WSL process, service, package, WMI, `/sys`, or documented local telemetry. Do not install a package/driver or control hardware.
- [ ] If no trustworthy package sensor exists, write `UNATTENDED_EXP017A_ALLOWED = NO`, skip shakedown and Attempt 2, and document the specific missing precondition. If it does exist, require an independently tested 95 C sustained 30–60 second abort, 100 C immediate abort, and GPU 85 C abort before a future shakedown.
- [ ] Before a scoped commit, run `git diff --check`, inspect status, and rerun focused tests. Exclude pre-existing `src/gibc_v2_foundational_llm.egg-info/`. Generate/verify a new bundle outside WSL and attempt one bounded push.
