# EXP-013 WSD ablation implementation plan

> **For the implementation agent:** execute in order and do not start model
> training until the specification, scheduler implementation, tests, and
> source/spec commit are complete.

**Goal:** Run the predeclared EXP-013 300M-token cosine-versus-WSD scheduler
ablation without changing any non-scheduler scientific variable.

## Steps

1. Add the EXP-013 scientific record and two explicit configurations.
2. Add an independently testable WSD scheduler while leaving
   `CosineWithWarmup` behavior unchanged.
3. Extend configuration and full-run artifact guards for only `EXP-013-C` and
   `EXP-013-W`, both requiring the frozen EXP-004 byte stream and validation
   tensors.
4. Ensure WSD checkpoints at completed update 8,240, before cooldown.
5. Add unit tests for LR endpoints and save/resume, configuration guards,
   stream/prefix guards, and parameter count.
6. Run non-benchmark tests and commit only EXP-013 source/spec/test files in
   the clean Git checkout. Attempt a push without rewriting history.
7. Run each arm once from fresh seed-42 state on the native Windows CUDA
   environment; never invoke evaluation/benchmark scripts.
8. Validate final counters, artifact hashes, result arithmetic, and WSD
   stable-checkpoint completeness; write evidence; commit and attempt push.

## Scheduler convention

At completed optimizer update `s`, `schedule.step()` sets LR immediately before
the optimizer update. WSD uses `warmup=100`, `stable_end=8240`, and
`cooldown=916`. It has LR zero at scheduler state 0, reaches peak at step 100,
holds peak through step 8240, then decays on steps 8241–9156. Step 9156 is
represented exactly as the configured minimum LR.

## Verification

Run the whole non-benchmark test suite plus targeted scheduler and static
runner/config tests before training. After training, inspect summaries,
metrics, checkpoint payloads, parameter counts, data-prefix hashes, and all
source-result references. Do not execute benchmark examples.

This plan ends before any follow-on confirmation or horizon-extension run.
