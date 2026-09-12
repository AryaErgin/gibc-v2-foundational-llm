# Hostile judge review — submission self-audit

These are **internal editorial ratings**, not official judge scores, predicted placement or externally calibrated performance ranks. Five-point scale: 1 = weak evidence presentation, 5 = strong judge-accessible evidence with few material gaps. No weighting or total is invented.

| Criterion | Editorial score | Skeptical objection | What this package fixes / retains |
|---|---:|---|---|
| Perplexity & Accuracy | 3.5/5 | How do I know the rounded scores belong to this checkpoint? | Exact output/checkpoint/config hashes, metric definitions, scored sample counts and harness stderr; no SOTA claim |
| Reasoning Performance | 2.5/5 | WinoGrande is below the prior control; why call this a reasoning advance? | README/plot openly show regression; PIQA called roughly stable; no universal-reasoning or significance claim |
| Training Efficiency | 4/5 | Is 37.94h the entire project? Did sleep inflate the throughput claim? | Final-run-only scope, full wall time including pacing, separate active/paced/whole-run rates, exact VRAM; no energy estimate presented as measurement |
| Innovation | 4/5 | These are standard Transformer parts and borrowed optimizer methods. | Credit established components; narrow contribution to controlled horizon evidence, gates, negative-result retention and reproducibility |
| Documentation & Demo | 3.5/5 | Can I obtain the model? Is there an actual video? | Auditable docs/figures and 3:30 storyboard now exist; public final-weight delivery/hash verification complete; recorded video remains incomplete |

## Corrections made during adversarial review

- Replaced stale EXP-012/future-EXP-020 status with the frozen final state.
- Updated six paper notes, AI disclosure and evaluator status; marked twelve ambiguous snapshots archival.
- Kept 54 historical experiment documents unchanged instead of erasing old conclusions.
- Distinguished General/Edu internal validation from held-out official tasks.
- Stated actual benchmark partitions, normalized metrics, uncertainty and mixed changes.
- Excluded incompatible EXP-001 word/byte WikiText PPL from token-PPL comparison.
- Disclosed earlier project benchmark exposure and public-text exclusion screening.
- Did not attribute a thermal improvement to a single simultaneously changed control.
- Labeled the abbreviated builder command. Later compliance audit found recorded `3733279` in all four raw harness results and resolved its full evaluator commit.
- Put every judge-required document/figure/tool in the authoritative source checkout, not only scratch.

## Remaining limits and resolved delivery issue

1. **Model delivery resolved (2026-09-09):** [public EXP-020](https://huggingface.co/AryaErgin/gibc-v2-exp020); all 19 release files independently hash-matched via anonymous HTTPS. [Receipt](PUBLIC_RELEASE.md).
2. **Video:** storyboard exists; actual recording/editing is a human task. No fake generation demo.
3. **Dataset snapshot coverage:** official PIQA task uses `baber/piqa`; exclusion-index historical source pins do not establish byte-identical evaluated snapshot coverage.
4. **Evaluator identity resolved:** all four raw task files record `3733279`, resolving to `37332797909df963ca7c77a945cea8752b60d481`; no frozen output was rewritten.
5. **Statistics/generalization:** one final seed, historical reference, no uncertainty across training seeds, no controlled alternative at 7.2B.
6. **Costs:** no complete campaign-energy/compute ledger or continuous reliable CPU sensor record.
7. **Attribution:** [core primary-source audit](LICENSE_AUDIT.md) completed with explicit PIQA/WikiText license limitations; exploratory bibliography VERIFY entries remain historical and unverified.

No benchmark result may trigger a model/checkpoint/training change. Submission is review-ready, not a claim that all publication/delivery steps have been completed.
