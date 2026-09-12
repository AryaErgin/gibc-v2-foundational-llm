# Submission verification record

This is a repository-only submission verification record. It is not a model-training or benchmark execution record.

## Completed evidence extraction

From the authoritative WSL checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python scripts/export_submission_evidence.py \
  --artifact-root /mnt/c/Users/aryae/Desktop/gibc_v2/artifacts \
  --logs-root /mnt/c/Users/aryae/Desktop/gibc_v2/logs \
  --output results/exp020-submission-evidence.json --full-stream --checkpoint-tensors
```

Exit 0, PASS. Independently recomputed checkpoint/config/tokenizer/summary/manifest/index and official output identities, full corpus and both historical prefix hashes. Read checkpoint tensors on CPU (no model instantiated/executed). Verified all 219,726 train rows sequential/finite with exact tokens/cursor.

## Package checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_submission_package.py tests/test_submission_contract.py -q
python3 scripts/verify_submission_package.py
git diff --check
git status --short
git diff --stat
```

- Targeted tests: **12 passed in 5.76s** on the completed package. These exercise offline evidence/figure/link contracts and existing parameter-count/CLI-argument contracts. They do not train, run model inference, load datasets or invoke official evaluation. No full training/evaluation suite is claimed.
- Earlier test-first run: 7 failures from intentionally missing tools. First integrated run: 10 passed / 2 failed (missing scratch-inventory document and an overly broad AST guard that mistook `.strip()` after Git inspection for a subprocess launch). Both corrected without scientific code changes.
- Offline package checker: **PASS** for frozen evidence arithmetic, config SHA, current-document local links, four PNG/SVG pairs and render provenance.
- `git diff --check`: **PASS**, no output. Initial Windows CRLF warnings in derivative SVGs were corrected by explicit LF rendering, not changes to evidence.
- Four PNG/SVG pairs visually inspected at 1280×720. Second render to a fresh directory: all nine export/provenance files byte-identical (PowerShell `Compare-Object` over SHA-256 output returned no differences).
- All tracked-text search: 235 files; 105 files/654 lines matched requested phrases. All 84 tracked Markdown documents classified in [DOCUMENT_FRESHNESS](DOCUMENT_FRESHNESS.md).
- Twelve archive bodies compared to `git show HEAD:path`: **exact preservation PASS**.
- `git diff --exit-code HEAD -- src configs provenance experiments`: **PASS**. Every pre-existing tracked result file compared byte-for-byte with HEAD: **PASS**.
- Final rehash of checkpoint, tokenizer, training summary, data manifest and all five official result files: **PASS**, identical to the independently audited identities.
- The only `pyproject.toml` change is its human-readable EXP-020 description; no dependency or runtime setting changed.

The Python read-only preservation/hash helper and renderer staging are listed in [WORKING_FILES](WORKING_FILES.md). External URLs were not re-fetched; local Markdown destinations were checked. No weights or data were published.

## Handoff boundary

Starting HEAD remains `37332797909df963ca7c77a945cea8752b60d481`; no commit/tag/push/staging. New files are in the authoritative worktree for review. Known untracked egg-info remains untouched. No training, checkpoint resume, benchmark rerun or model/tokenizer/dataset/official-result modification occurred.
