# Final competition repository reconciliation — 2026-09-09

Authority: `/home/aryae/gibc-v2-foundational-llm`. The stale Windows checkout and all frozen artifacts are unchanged. Existing submission work remains unstaged for review; this is not a commit or publication action.

## Public delivery and current story

[Permanent EXP-020 model](https://huggingface.co/AryaErgin/gibc-v2-exp020), public/ungated Hub revision `80703cd304d1d1b16abf6539dcb0ebe83386f0ca`. The user reported a matching downloaded safetensors SHA; this pass independently streamed all 19 package files anonymously and matched every SHA and byte count. [Evidence](PUBLIC_RELEASE.md).

Root README now links the model prominently and embeds trajectory, decisions and pipeline; RESULTS embeds progression. All four are existing recorded-data assets, not new renders. The exact final parameter/token/hardware/runtime/approximate-compute story and frozen benchmark results are retained.

Current release instructions use a pinned download and exclude Hub-added `.gitattributes`, which is not in the exact package inventory. No loader or package file was altered. The packaged README remains an export-time snapshot; current source documentation records the later smoke/publication chronology.

## Files changed in this reconciliation

Updated current documentation or added an explicit historical-receipt cross-reference:

- `AI_ASSISTANCE.md`
- `DATA_SOURCES.md`
- `PROJECT_PLAN.md`
- `README.md`
- `docs/submission/CHANGES.md`
- `docs/submission/COMPLIANCE.md`
- `docs/submission/COMPLIANCE_VALIDATION.md`
- `docs/submission/DEMO_STORYBOARD.md`
- `docs/submission/DEVPOST_DRAFT.md`
- `docs/submission/DOCUMENT_FRESHNESS.md`
- `docs/submission/EVIDENCE_AUDIT.md`
- `docs/submission/EXP020_MODEL_CARD.md`
- `docs/submission/EXP020_RELEASE.md`
- `docs/submission/JUDGE_REVIEW.md`
- `docs/submission/PREPUBLICATION_SMOKE.md`
- `docs/submission/REPRODUCIBILITY.md`
- `docs/submission/WORK_PLAN.md`
- `paper/reviewer_risks.md`

New publication/reconciliation evidence:

- `docs/submission/PUBLIC_RELEASE.md`
- `results/exp020-public-release.json`
- `docs/submission/FINAL_RECONCILIATION.md` (this report)
- `results/exp020-placeholder-audit.json` (each search hit and disposition)
- `docs/submission/RECONCILIATION_WORKTREE.txt` (complete review inventory)

No experiment-specific record, config, model, tokenizer, source-data or benchmark artifact was rewritten. License/bibliography evidence is retained rather than inventing clearance or missing citation metadata.

## Remaining occurrence classes

The [occurrence-level audit](../../results/exp020-placeholder-audit.json) covers tracked and intended untracked Markdown/text/BibTeX/TOML, excluding egg-info and ignored artifact/scratch trees. Each matched line has a category and disposition; its dated line numbers are audit locations, not immutable document identities.

- **Manual submission actions:** video, team/eligibility, Devpost fields, final source commit/tag/push. Public model delivery is complete.
- **Explicit scientific/provenance limitations:** PIQA exclusion/evaluation snapshot equality is unproved; no continuous trustworthy CPU temperature/campaign-energy ledger or multi-seed final-scale claim.
- **License limits:** PIQA's source license is unknown; WikiText's pinned card has conflicting license-version statements. No benchmark text is redistributed; no legal clearance is invented.
- **Provisional research bibliography:** exploratory entries/metadata and full originality-audit checklist items remain marked, not relied on for final-model numerical claims. They are not blank final-result fields.
- **Historical evidence:** EXP-specific earlier pending/local-only statuses, archival notices, the original packaging plan and dated validation/smoke receipts remain true to their original scopes. Later completion links are explicit.
- **Resolved/instructional/search vocabulary:** positive public-release statements, negated pending claims and audit search terms are not unresolved placeholders.

No current-facing EXP-012-as-final assertion remains; explicit historical references and comparisons are retained.

## Verification

Executed successfully in the authoritative WSL repository:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest tests/test_inference_release.py tests/test_submission_package.py tests/test_submission_contract.py -q
.venv/bin/python -B scripts/verify_submission_package.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B artifacts/exp020-public-inference-20260908-release/verify.py
git diff --check
git status --short --untracked-files=all
```

- Complete targeted tests: **21 passed in 6.68 seconds**, zero failures.
- Submission verifier: **PASS** for frozen evidence, config identity, local links and all four PNG/SVG pairs with original renderer/input/output hashes.
- Packaged verifier: **PASS**, strict CPU FP32 loading, finite values, **49,860,480 parameters**, no forward/inference.
- Independent current-file checks: **PASS** for README's five rounded benchmark metrics against exact frozen JSON, hardware/37.94 h/parameter/token facts, and `6NT = 2,153,967,221,829,795,840` approximate theoretical FLOPs.
- All four actual image embeds resolve: trajectory/decisions/pipeline in README, progression in RESULTS.
- All **19** local release files remain equal to both the retained smoke copy and the independent anonymous remote hash receipt.
- Bibliography: **70 unique keys**, no duplicate IDs; provisional research metadata stays explicitly labeled.
- `git diff --check`: **PASS**. Git index empty; HEAD unchanged at `37332797909df963ca7c77a945cea8752b60d481`.
- Final worktree: **33 modified tracked files and 51 untracked files**, including earlier submission work and five intentionally untracked egg-info files. [Complete inventory](RECONCILIATION_WORKTREE.txt).

No model inference, training, benchmark requests, package changes, staging, commit, tag or push occurred. The full scientific/benchmark suite was not run; the entire relevant release/submission test set above was run.

## Commit readiness boundary

After successful checks, this is technically ready for the reviewed source-only competition commit, not a claim that the Devpost entry/video is complete or that every upstream rights uncertainty is resolved. Inspect the [full unstaged/untracked inventory](RECONCILIATION_WORKTREE.txt); keep egg-info and ignored model/data/cache files out of Git. No staging, commit, tag or push was performed.

Recommended commit: `submission:finalize-exp020-competition-release`

Recommended tag after review/commit: `gibc-v2-track01-exp020-v1.0.0`
