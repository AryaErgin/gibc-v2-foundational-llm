# Final compliance validation — 2026-09-08

> Historical packaging receipt. Subsequent [clean-environment smoke](PREPUBLICATION_SMOKE.md) and [public delivery](PUBLIC_RELEASE.md) passed. The original no-inference/no-publication observations below describe that earlier pass, not current release status.

This receipt covers the narrow compliance/release pass, not all earlier uncommitted submission work. Authority: `/home/aryae/gibc-v2-foundational-llm`. HEAD remains `37332797909df963ca7c77a945cea8752b60d481`. The stale Windows Git checkout was not modified; frozen Windows-hosted artifacts were read only.

## Results

- Test-first missing-module failure confirmed before implementation; after implementation **21 tests passed in 6.72 s**, zero failures.
- Exact original checkpoint SHA/598,447,091 bytes gated both exports. Every tensor is bitwise/numerically equal after safetensors reload: **83 tensors, 49,860,480 elements and trainable parameters**, CPU FP32, finite, strict loading, no forward call.
- Both exports yielded weight SHA `4c4f97801ce0c3d0cee52172bfe851b57690b81d153775a33107b8aaed1bc129`. The preliminary directory is explicitly marked NOT_FOR_PUBLICATION because its first prose draft incorrectly described duplicate tied keys. Only the corrected `-release` directory is publishable.
- Independent packaged CLI from `/tmp` passed: exact inventory, all hashes, config/tokenizer identities, strict CPU loading/count. **No inference executed.**
- Package: **200,166,267 bytes**, including manifests; weights: **199,450,256 bytes**. [Full release receipt](../../results/exp020-inference-release.json) lists every file/hash.
- Manifest SHA: `b3f9cdadce6757d7134834303b76c254caa3433c667fb2f191e4c21375c2fe2e`.
- All four original harness files rehashed to the frozen identities and record `3733279`; full SHA expansion verified. Frozen summary rehashed to `31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a`.
- Offline submission checker passed evidence/config/link checks and four existing 1280×720 PNG/SVG pairs bound to their original data/renderer hashes. No figures were regenerated.
- `git diff --check` passed. No Git index mutation/publication or package installation.
- Original scientific source/config and frozen artifacts were not changed. This pass did not execute the full scientific test suite because many tests perform forward passes; only relevant benchmark-free/no-inference tests were run.
- A clean-machine dependency installation and actual generation are **not executed** under the user's no-package-change/no-inference boundary. CPU strict loading and the unchanged generation source are verified, not represented as a new end-to-end generation result.

## Exact check commands

From the authoritative repo, using existing `.venv`:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/test_inference_release.py tests/test_submission_package.py tests/test_submission_contract.py -q
.venv/bin/python scripts/verify_submission_package.py
git diff --check
git status --short
git diff --stat
```

Independent package check (working directory `/tmp`):

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/aryae/gibc-v2-foundational-llm/.venv/bin/python -B \
/home/aryae/gibc-v2-foundational-llm/artifacts/exp020-public-inference-20260908-release/verify.py
```

The retained parameter-contract test calls the count script. Its generation-argument test exits at missing required arguments; it does not load weights or generate text. No test calls `lm_eval.simple_evaluate`, loads benchmark datasets, or executes official requests.

## Files changed in this pass

Updated existing current-facing files, preserving earlier edits:

- `README.md`: compute, setup/use, release status, precise evaluator source and PIQA caveat, credits/checklist links.
- `REFERENCES.bib`: verified full core dataset/benchmark author lists and software citations; unrelated exploratory entries remain explicitly provisional.
- `SOURCE_LEDGER.md`, `CODE_ATTRIBUTION.md`: scoped primary-license audit cross-references; no claim of exhaustive line-by-line originality.
- `PROJECT_PLAN.md`: precise manual publication gates.
- `docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md`, `results/EXP-020-summary.md`: raw short/full evaluator commit provenance correction.
- `docs/submission/REPRODUCIBILITY.md`, `EVIDENCE_AUDIT.md`, `JUDGE_REVIEW.md`: release/provenance issue status.
- `paper/reviewer_risks.md`: same narrow provenance/license correction.

Created:

- `src/gibc_llm/inference_release.py`
- `scripts/build_exp020_publication.py`
- `scripts/verify_exp020_publication.py`
- `scripts/exp020_release_generate.py`
- `tests/test_inference_release.py`
- `results/exp020-evaluation-source-provenance.json`
- `results/exp020-inference-release.json`
- `docs/submission/COMPLIANCE.md`
- `docs/submission/LICENSE_AUDIT.md`
- `docs/submission/EXP020_MODEL_CARD.md`
- `docs/submission/EXP020_RELEASE.md`
- this validation receipt
- `docs/superpowers/plans/2026-09-08-exp020-release-compliance.md`

Two ignored local package directories were created during development. Publish **only** `artifacts/exp020-public-inference-20260908-release`; the earlier draft carries a rejection marker. Temporary patch transport used the Windows system Temp directory, never the Windows source checkout. No frozen artifact was overwritten. The known egg-info remains untouched/untracked.

## Publication actions recorded at that time (superseded)

Choose/confirm a public model-host namespace, upload only the verified final package, record its immutable public revision/URL, and publish reviewed source under separate authorization. Video recording and Devpost team/eligibility fields remain manual. PIQA license/snapshot equivalence and WikiText card license-version ambiguity remain explicitly unresolved; no such dataset is redistributed. [Upload steps](EXP020_RELEASE.md) · [Six-component checklist](COMPLIANCE.md) · [License evidence](LICENSE_AUDIT.md).

Technically review-ready for a scoped source commit after human review, **not** a claim that the public Devpost submission is complete. Nothing has been staged, committed, tagged or pushed.

## Final read-only review

Independent review found **no critical or important issues** and judged the scoped release tooling ready for commit. One non-blocking generic helper comment mentions alias keys; EXP-020's actual 83-tensor inventory, model card and provenance correctly state one embedding key used directly for output projection. No scientific behavior depends on that comment. Review executed no model or benchmark.

Public dependency metadata was checked without installation: the [official cu132 wheel index](https://download.pytorch.org/whl/cu132/torch/) lists torch 2.13.0+cu132 CPython 3.11 Linux x86-64; PyPI metadata exists for the pinned [NumPy 2.4.6](https://pypi.org/project/numpy/2.4.6/), [safetensors 0.8.0](https://pypi.org/project/safetensors/0.8.0/) and [tokenizers 0.21.4](https://pypi.org/project/tokenizers/0.21.4/). Availability is not a claim that a clean environment was installed/tested here. Seventy bibliography keys are unique.
