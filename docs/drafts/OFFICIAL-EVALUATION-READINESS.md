# Official Evaluation Readiness — Draft Checklist

> This is a readiness checklist only. It does not authorize evaluation, load an evaluation checkpoint, or run any benchmark.

## Promotion and provenance

- [ ] EXP-011 has a terminal, reviewed result; no integrity stop condition is open.
- [ ] Research review explicitly selects the checkpoint under the frozen validation-based promotion rule.
- [ ] The selected checkpoint SHA-256 is recorded and the checkpoint is retained unchanged.
- [ ] Exact parameter count is recomputed after loading the selected evaluation checkpoint; it remains at or below 50,000,000.
- [ ] The evaluation loader and evaluation configuration are verified to load that selected checkpoint, not a default, latest, or alternative checkpoint.
- [ ] Training source commit, evidence/result commit, config provenance, seed, architecture, tokenizer hash, stream/manifest hashes, source revisions, and system environment are recorded.

## Benchmark integrity

- [ ] Official benchmark evaluation is run only on the checkpoint selected by the frozen validation-based promotion rule, with no benchmark-driven checkpoint selection.
- [ ] No benchmark result is used to tune architecture, data, optimizer, LR, batch, context, or checkpoint choice.
- [ ] Data Recipe v1 deduplication and contamination-screen evidence, including limitations, is linked from the evaluation record.
- [ ] Task revisions, splits, templates, shot count, batch size, library versions, command lines, and result-artifact hashes are frozen before execution.

## Required benchmark plan

- [ ] HellaSwag protocol is frozen in writing.
- [ ] ARC-Easy protocol is frozen in writing.
- [ ] PIQA protocol is frozen in writing.
- [ ] WinoGrande protocol is frozen in writing.
- [ ] WikiText-103 perplexity protocol is frozen in writing, including tokenizer identity, context length and stride/windowing, BOS/EOS treatment, loss-token masking, and metric aggregation.

## Execution and reporting

- [ ] Evaluation environment and package versions are recorded; no pre-trained weights are loaded.
- [ ] Each task runs once under the frozen plan, with runtime and hardware metadata recorded.
- [ ] Raw machine-readable outputs, aggregate metrics, failures, and any deviations are preserved.
- [ ] Results are reported separately from internal frozen validation; no benchmark score is represented as a training-selection signal.
