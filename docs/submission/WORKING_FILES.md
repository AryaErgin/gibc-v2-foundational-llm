# Temporary working files — not the submission source of truth

The authoritative submission files are in `/home/aryae/gibc-v2-foundational-llm`. Newly created files are in that worktree for review (not staged/committed). `.submission-work` is only a temporary Windows-side drafting/deployment area; judges need none of it.

The files below are retained temporarily for human diff/recovery/reproduction review. They are not a second canonical repository. No new experimental results exist only here. Historical input copies and old render versions must never be mistaken for an altered model/evaluation artifact. They may be removed in a later expressly scoped scratch cleanup after review; this task does not delete them or other artifacts.

## Every retained scratch file

| File relative to Windows artifact-store root | Purpose / why temporarily retained |
|---|---|
| `.submission-work/tree/docs/submission/CHANGES.md` | Temporary deployment copy of complete authoritative change inventory |
| `.submission-work/tree/pyproject.toml` | Temporary deployment copy of description-only metadata correction; dependencies/runtime unchanged |
| `.submission-work/audit_docs.py` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/copy_figure_inputs.sh` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/deploy.sh` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/document-audit.json` | Temporary audit/input snapshot; judge evidence is in repository results/docs |
| `.submission-work/evidence.json` | Temporary audit/input snapshot; judge evidence is in repository results/docs |
| `.submission-work/figures-repro-check/decisions.png` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/decisions.svg` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/figure-provenance.json` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/pipeline.png` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/pipeline.svg` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/progression.png` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/progression.svg` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/trajectory.png` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-repro-check/trajectory.svg` | Independent rerender used to verify byte-reproducibility; not judge entry point |
| `.submission-work/figures-v2/decisions.png` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/decisions.svg` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/figure-provenance.json` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/pipeline.png` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/pipeline.svg` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/progression.png` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/progression.svg` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/trajectory.png` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/figures-v2/trajectory.svg` | Intermediate render export; same final figures delivered to docs/assets/exp020 |
| `.submission-work/final_audit.py` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/inspect_evidence.py` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/probe.py` | One-session inspection/deployment helper; not a production command or unique scientific evidence |
| `.submission-work/tree/AGENTS.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/AI_ASSISTANCE.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/ARCHITECTURE.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/CODE_ATTRIBUTION.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/DATA_SOURCES.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/EXPERIMENT_LOG.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/PROJECT_PLAN.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/README.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/RESULTS.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/SOURCE_LEDGER.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/EXP-012-INFERENCE-PUBLICATION.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/OPERATIONAL_THERMAL_CONTROLS.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/README.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/decisions.png` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/decisions.svg` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/figure-provenance.json` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/pipeline.png` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/pipeline.svg` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/progression.png` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/progression.svg` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/trajectory.png` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/assets/exp020/trajectory.svg` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/drafts/EXP-011-result-report.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/drafts/OFFICIAL-EVALUATION-READINESS.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/storage-manifest-2026-08-30.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/DEMO_STORYBOARD.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/DEVPOST_DRAFT.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/DOCUMENT_FRESHNESS.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/EVIDENCE_AUDIT.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/JUDGE_REVIEW.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/REPRODUCIBILITY.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/VALIDATION.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/WORKING_FILES.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/submission/WORK_PLAN.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-23-exp001a-implementation.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-24-exp005-architecture-preflight.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-24-exp006-preparation.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-25-exp007-preflight.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-26-exp011-long-horizon-preparation.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-28-exp013-wsd-ablation.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/plans/2026-08-30-exp017a-thermal-recovery.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/docs/superpowers/specs/2026-08-23-exp001a-design.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/claims_and_evidence.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/experiment_matrix.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/figures_plan.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/outline.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/related_work.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/paper/reviewer_risks.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/provenance/exp018-closure.json` | Read-only historical input copy for rendering; original tracked record unchanged |
| `.submission-work/tree/provenance/exp019-closure.json` | Read-only historical input copy for rendering; original tracked record unchanged |
| `.submission-work/tree/results/EXP-001D-summary.md` | Read-only historical input copy for rendering; original tracked record unchanged |
| `.submission-work/tree/results/EXP-020-summary.md` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/results/exp020-submission-evidence.json` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/scripts/export_submission_evidence.py` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/scripts/render_submission_figures.py` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/scripts/verify_submission_package.py` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/tests/test_submission_contract.py` | Temporary deployment copy; authoritative version exists in the WSL repository |
| `.submission-work/tree/tests/test_submission_package.py` | Temporary deployment copy; authoritative version exists in the WSL repository |

The repository's [freshness audit](DOCUMENT_FRESHNESS.md), [evidence digest](../../results/exp020-submission-evidence.json), [Devpost draft](DEVPOST_DRAFT.md), [storyboard](DEMO_STORYBOARD.md) and [final figures](../assets/exp020/README.md) are the actual deliverables. This inventory includes its own temporary deployment copy.
