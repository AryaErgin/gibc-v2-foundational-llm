# Repository-wide documentation freshness audit

Audited all **84 tracked Markdown/text documents** at starting HEAD `37332797909df963ca7c77a945cea8752b60d481`, before edits. Classification preceded editing: 18 current/canonical, 54 historical experiment records, 12 archived snapshots. 44 contain one or more requested status/numeric search terms. A match is not automatically an error: old experimental scores are valid historical evidence.

The search covered current best, final model, EXP-012/020, 2.4B/7.2B, WikiText/HellaSwag/ARC-Easy/PIQA/WinoGrande, parameter count, training time, current status, TODO, pending and not yet evaluated. Source-code/config/provenance matches were also reviewed as implementation/history; **none of those scientific artifacts is changed**. Historical code flags and old expected results must remain available for reproducibility.

## Stale current claims corrected

- README/ARCHITECTURE/RESULTS: EXP-012 or earlier final/current state superseded by immutable EXP-020, exact final scores and mixed benchmark comparison.
- PROJECT_PLAN: preparation/method-hunting stages superseded by completed model/evaluation and human release work only.
- AGENTS: EXP-001 launch guidance replaced with frozen-submission boundaries.
- EXPERIMENT_LOG: current final-state banner; historical authorizations remain dated history.
- DATA_SOURCES: full 7.2B identities/counts and contamination limitations.
- AI_ASSISTANCE: material research/learning/planning/code/analysis/paper/submission roles fully disclosed, distinct from the scratch-trained model.
- Evaluation protocol: “Future execution” changed to completed/frozen; corrected outer status path.
- Thermal controls: final run evidence added without individual-control causality.
- All six paper notes: EXP-017 pending/WSD promoted language corrected; EXP-018/019 rejection and final official results included.
- SOURCE_LEDGER/CODE_ATTRIBUTION: final pointers/disclosure added; historical source pins distinguished from evaluated snapshots. Unfinished citation/license checks remain unfinished.
- Twelve archival plans/drafts/storage/publication snapshots: clear archival warnings, no retroactive result edits.

## Every tracked document: classification and action

| Document | Class | Search-hit lines | Disposition |
|---|---|---:|---|
| [AGENTS.md](../../AGENTS.md) | canonical | 0 | Updated to final evidence; attribution checklists retain explicit limits |
| [AI_ASSISTANCE.md](../../AI_ASSISTANCE.md) | canonical | 2 | Updated to final evidence; attribution checklists retain explicit limits |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | canonical | 3 | Updated to final evidence; attribution checklists retain explicit limits |
| [CODE_ATTRIBUTION.md](../../CODE_ATTRIBUTION.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [DATA_SOURCES.md](../../DATA_SOURCES.md) | canonical | 3 | Updated to final evidence; attribution checklists retain explicit limits |
| [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md) | canonical | 11 | Added final-state banner; dated chronology preserved |
| [PROJECT_PLAN.md](../../PROJECT_PLAN.md) | canonical | 12 | Updated to final evidence; attribution checklists retain explicit limits |
| [README.md](../../README.md) | canonical | 21 | Updated to final evidence; attribution checklists retain explicit limits |
| [RESULTS.md](../../RESULTS.md) | canonical | 18 | Updated to final evidence; attribution checklists retain explicit limits |
| [SOURCE_LEDGER.md](../../SOURCE_LEDGER.md) | canonical | 6 | Updated to final evidence; attribution checklists retain explicit limits |
| [docs/EXP-012-INFERENCE-PUBLICATION.md](../../docs/EXP-012-INFERENCE-PUBLICATION.md) | archival | 6 | Added explicit archive notice; original content preserved |
| [docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md](../../docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md) | canonical | 6 | Updated to final evidence; attribution checklists retain explicit limits |
| [docs/OPERATIONAL_THERMAL_CONTROLS.md](../../docs/OPERATIONAL_THERMAL_CONTROLS.md) | canonical | 0 | Updated to final evidence; attribution checklists retain explicit limits |
| [docs/drafts/EXP-011-result-report.md](../../docs/drafts/EXP-011-result-report.md) | archival | 1 | Added explicit archive notice; original content preserved |
| [docs/drafts/OFFICIAL-EVALUATION-READINESS.md](../../docs/drafts/OFFICIAL-EVALUATION-READINESS.md) | archival | 6 | Added explicit archive notice; original content preserved |
| [docs/storage-manifest-2026-08-30.md](../../docs/storage-manifest-2026-08-30.md) | archival | 2 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-23-exp001a-implementation.md](../../docs/superpowers/plans/2026-08-23-exp001a-implementation.md) | archival | 2 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-24-exp005-architecture-preflight.md](../../docs/superpowers/plans/2026-08-24-exp005-architecture-preflight.md) | archival | 0 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-24-exp006-preparation.md](../../docs/superpowers/plans/2026-08-24-exp006-preparation.md) | archival | 0 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-25-exp007-preflight.md](../../docs/superpowers/plans/2026-08-25-exp007-preflight.md) | archival | 0 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-26-exp011-long-horizon-preparation.md](../../docs/superpowers/plans/2026-08-26-exp011-long-horizon-preparation.md) | archival | 0 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-28-exp013-wsd-ablation.md](../../docs/superpowers/plans/2026-08-28-exp013-wsd-ablation.md) | archival | 2 | Added explicit archive notice; original content preserved |
| [docs/superpowers/plans/2026-08-30-exp017a-thermal-recovery.md](../../docs/superpowers/plans/2026-08-30-exp017a-thermal-recovery.md) | archival | 3 | Added explicit archive notice; original content preserved |
| [docs/superpowers/specs/2026-08-23-exp001a-design.md](../../docs/superpowers/specs/2026-08-23-exp001a-design.md) | archival | 3 | Added explicit archive notice; original content preserved |
| [experiments/EXP-001.md](../../experiments/EXP-001.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-003.md](../../experiments/EXP-003.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-004.md](../../experiments/EXP-004.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-005.md](../../experiments/EXP-005.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-006.md](../../experiments/EXP-006.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-007.md](../../experiments/EXP-007.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-008.md](../../experiments/EXP-008.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-009.md](../../experiments/EXP-009.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-010.md](../../experiments/EXP-010.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-011.md](../../experiments/EXP-011.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-012-official-evaluation.md](../../experiments/EXP-012-official-evaluation.md) | historical | 30 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-012.md](../../experiments/EXP-012.md) | historical | 4 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-013-seed43-confirmation.md](../../experiments/EXP-013-seed43-confirmation.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-013-wsd-ablation.md](../../experiments/EXP-013-wsd-ablation.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-014-htsr-llr.md](../../experiments/EXP-014-htsr-llr.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-015-curriculum-wsd-phase.md](../../experiments/EXP-015-curriculum-wsd-phase.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-015-fixed-example-permutation.md](../../experiments/EXP-015-fixed-example-permutation.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-016-magma.md](../../experiments/EXP-016-magma.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [experiments/EXP-017A-2.4b-wsd.md](../../experiments/EXP-017A-2.4b-wsd.md) | historical | 7 | Preserved unchanged: experiment-specific truth, not current status |
| [paper/claims_and_evidence.md](../../paper/claims_and_evidence.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [paper/experiment_matrix.md](../../paper/experiment_matrix.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [paper/figures_plan.md](../../paper/figures_plan.md) | canonical | 0 | Updated to final evidence; attribution checklists retain explicit limits |
| [paper/outline.md](../../paper/outline.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [paper/related_work.md](../../paper/related_work.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [paper/reviewer_risks.md](../../paper/reviewer_risks.md) | canonical | 1 | Updated to final evidence; attribution checklists retain explicit limits |
| [results/EXP-001-summary.md](../../results/EXP-001-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-001A-summary.md](../../results/EXP-001A-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-001B-summary.md](../../results/EXP-001B-summary.md) | historical | 2 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-001C-summary.md](../../results/EXP-001C-summary.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-001D-summary.md](../../results/EXP-001D-summary.md) | historical | 8 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-002-summary.md](../../results/EXP-002-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-002A-summary.md](../../results/EXP-002A-summary.md) | historical | 6 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-003-summary.md](../../results/EXP-003-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-004-summary.md](../../results/EXP-004-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-004A-summary.md](../../results/EXP-004A-summary.md) | historical | 22 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-005-preflight.md](../../results/EXP-005-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-005-summary.md](../../results/EXP-005-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-005B-evaluation.md](../../results/EXP-005B-evaluation.md) | historical | 22 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-006-summary.md](../../results/EXP-006-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-006A-evaluation.md](../../results/EXP-006A-evaluation.md) | historical | 22 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-007-preflight.md](../../results/EXP-007-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-007-summary.md](../../results/EXP-007-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-008-preflight.md](../../results/EXP-008-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-008-summary.md](../../results/EXP-008-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-009-audit.md](../../results/EXP-009-audit.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-009-preflight.md](../../results/EXP-009-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-009-summary.md](../../results/EXP-009-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-010-preflight.md](../../results/EXP-010-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-010-summary.md](../../results/EXP-010-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-011-horizon-scaling-analysis.md](../../results/EXP-011-horizon-scaling-analysis.md) | historical | 3 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-011-preflight.md](../../results/EXP-011-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-011-summary.md](../../results/EXP-011-summary.md) | historical | 1 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-012-summary.md](../../results/EXP-012-summary.md) | historical | 3 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-013-summary.md](../../results/EXP-013-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-014-summary.md](../../results/EXP-014-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-015-summary.md](../../results/EXP-015-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-016-preflight.md](../../results/EXP-016-preflight.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/EXP-016-summary.md](../../results/EXP-016-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/SYS-001-summary.md](../../results/SYS-001-summary.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |
| [results/SYS-002-runtime-qualification.md](../../results/SYS-002-runtime-qualification.md) | historical | 0 | Preserved unchanged: experiment-specific truth, not current status |

## Historical numbers deliberately retained

EXP-001–019 scores and statuses are valid for their named experiments. EXP-012's 2.4B/35.939-PPL/earlier accuracy figures remain historical controls, not conflicting final-model claims. Old readiness and launch instructions are marked archived where not already unambiguously experiment-specific. Frozen JSON provenance and config/code default values are not rewritten to incorporate later knowledge.

The new [EXP-020 summary](../../results/EXP-020-summary.md), [RESULTS](../../RESULTS.md), [reproducibility guide](REPRODUCIBILITY.md) and [evidence audit](EVIDENCE_AUDIT.md) are the current story.

## Residual gaps, not hidden contradictions

The final all-tracked-text search covered 235 files, with 654 matching lines in 105 files. Outside the Markdown inventory, `pyproject.toml` still described EXP-012 as finalized; only its human-readable package description was updated to EXP-020. Dependencies, build/runtime settings and all frozen scientific YAML/JSON/code remain unchanged. Historical EXP-specific script names/docstrings, test expectations and bibliography entries retain their original role.

Update 2026-09-09: all four raw harness artifacts record `3733279`, resolved to full evaluator commit `37332797909df963ca7c77a945cea8752b60d481`. [Public model delivery and anonymous hash verification](PUBLIC_RELEASE.md) are complete. Video and final source publication remain human actions; PIQA evaluation/exclusion snapshot equivalence, PIQA/WikiText license limits and exploratory bibliography VERIFY items remain explicit. Full campaign energy/cost and continuous CPU temperature are unavailable, not invented. See [final reconciliation](FINAL_RECONCILIATION.md) for the new placeholder inventory. No scientific conflict between final hashes/scores was found.

Scratch files and their purposes are listed in [WORKING_FILES.md](WORKING_FILES.md). Judges need only the actual repository documentation, digest and exported figures; staging is not an alternate source of truth.
