# Code Attribution and External Implementation Review

This file records external codebases that were inspected, used as libraries/tools, or consulted while implementing project features. It exists to separate **conceptual/paper influence**, **behavioral code consultation**, **library use**, and **actual code reuse**.

## Policy

- Project-specific implementation should be independently written unless reuse is explicitly documented and license-compatible.
- Inspecting upstream code for behavioral interpretation is recorded even when no code is copied.
- If any upstream snippet is copied or closely adapted later, this file must be updated with exact path, commit/tag, license, and local destination.
- Absence of a license is treated conservatively: inspect for understanding only; do not copy/vend source.
- Scholarly attribution remains separate in `REFERENCES.bib` / `SOURCE_LEDGER.md`.

---

## 1. Layer-wise Learning Rate / HT-SR LLR

**Repository:** https://github.com/hed-ucas/Layer-wise-Learning-Rate  
**Pinned inspected commit:** `bbd0dcf86af80b8843866a9a041086a37de35897`  
**Relevant paper:** He et al., *One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs*, arXiv:2605.22297v3  
**Inspected paths:**

- `README.md`
- `olmo/LRUnbalance.py`
- functions/methods including `layerTempbalance.net_esd_estimator`, `get_layer_temps`, `step`

**Consulted for:** behavioral interpretation of the spectral statistic, positive linear mapping, embedding treatment, soft switching, and published command settings.

**License status at inspection:** no declared repository license observed.

**Reuse status:**

- upstream source copied verbatim: **NO**
- upstream source vendored: **NO**
- local implementation: **independent**
- local implementation path: `src/gibc_llm/llr.py`

The exact provenance record is `provenance/exp014-upstream-provenance.json`.

---

## 2. Magma

**Paper:** Joo et al., *On Surprising Effectiveness of Masking Updates in Adaptive Optimizers*, arXiv:2602.15322  
**Primary source:** https://arxiv.org/abs/2602.15322

**Consulted for:** algorithm definition, masking behavior, target parameter blocks, and experimental context.

**Reuse status:**

- third-party Magma package imported: **NO**
- upstream implementation copied: **NO**
- local implementation: **independent implementation from paper specification**

Exact project provenance: `provenance/exp016-magma-paper.json` and `experiments/EXP-016-magma.md`.

---

## 3. Cautious Weight Decay (CWD)

**Paper:** Chen et al., *Cautious Weight Decay*, arXiv:2510.12402v2  
**Primary source:** https://arxiv.org/abs/2510.12402

**Consulted for:** Algorithm 1 / entrywise cautious-decay semantics.

**Reuse status:**

- author-maintained optimizer implementation relied upon: **NO**
- upstream code copied: **NO**
- local implementation: **independent implementation from Algorithm 1**

Exact project provenance: `provenance/exp019-cwd-preregistration.json` and EXP-019 records.

---

## 4. Warmup-Stable-Decay / MiniCPM

**Project/paper:** MiniCPM, arXiv:2404.06395  
**Repository:** https://github.com/OpenBMB/MiniCPM

**Consulted for:** WSD schedule lineage and published small-model training strategy.

**Reuse status:** local scheduler implementation is project code; no MiniCPM source is vendored.

Additional interpretation papers are recorded in `SOURCE_LEDGER.md`.

---

## 5. EleutherAI Language Model Evaluation Harness

**Repository:** https://github.com/EleutherAI/lm-evaluation-harness  
**Project version:** `lm-eval==0.4.9.1`

**Role:** required benchmark evaluation framework and upstream task templates.

**Reuse status:** third-party dependency used normally; project evaluation wrappers/scripts invoke the installed package. No claim of authorship over harness code or templates.

When publishing results, cite/credit EleutherAI's evaluation harness and record the frozen version/protocol.

---

## 6. Pythia

**Repository:** https://github.com/EleutherAI/pythia  
**Paper:** arXiv:2304.01373

**Role:** comparator and research/provenance inspection only.

**Reuse status:** no Pythia model weights, tokenizer, training code, or pretrained initialization are used in the submitted model.

---

## 7. FineWeb / FineWeb-Edu and Hugging Face tooling

**Datasets/projects:**

- https://huggingface.co/datasets/HuggingFaceFW/fineweb
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

**Libraries:**

- Hugging Face `datasets`
- Hugging Face `tokenizers`

**Role:** actual public training data sources and dataset/tokenizer infrastructure.

**Reuse status:** data/library use under their applicable licenses/terms; the model tokenizer itself is trained from scratch. No pretrained model weights are imported.

Exact data revisions are in `DATA_SOURCES.md` and manifests.

---

## 8. Research repositories inspected during method selection

These repositories were consulted for research/implementation understanding. Unless separately documented above, they were **not vendored or copied into the submitted implementation**.

| Repository | Purpose | Local code copied/adapted? |
|---|---|---|
| https://github.com/mlfoundations/scaling | overtraining/scaling evidence and experimental context | No known copy |
| https://github.com/huggingface/datablations | data-constrained/repetition research context | No known copy |
| https://github.com/sail-sg/regmix | RegMix/data-mixture research context | No known copy |
| https://github.com/sail-sg/scaling-with-vocab | vocabulary-scaling research context | No known copy |
| https://github.com/google-research/deduplicate-text-datasets | deduplication research/implementation context | No known copy |
| https://github.com/mlexpos/adana | ADana algorithm/scale evidence | No known copy |
| https://github.com/mrflogs/LoRA-Pre | LoRA-Pre optimizer research | No known copy |
| https://github.com/K1seki221/MuonPlus | Muon+ research/implementation context | No known copy |
| https://github.com/OpenBMB/MiniCPM | WSD/MiniCPM research context | No known copy |

Before camera-ready release, re-check the final source tree against these repositories if any implementation was subsequently adapted.

---

## 9. Runtime / software dependencies

The project uses third-party software under their respective licenses and does not claim authorship of them. Direct Python package pins are recorded in `pyproject.toml`.

Material dependencies/tools include:

- PyTorch
- NVIDIA CUDA runtime/toolchain
- Python
- NumPy
- Hugging Face `datasets`
- Hugging Face `tokenizers`
- EleutherAI `lm-eval`
- `safetensors`
- PyYAML
- SQLite / Python `sqlite3`
- pytest
- git-filter-repo
- Git / GitHub / GitHub CLI
- WSL2 / Windows host tooling

Package usage should be acknowledged where relevant to reproducibility; not every utility requires an academic bibliography entry.

---

## 10. AI coding and research assistance

AI assistance is not third-party source-code authorship, but it is a required disclosure category for this project.

Used materially:

- ChatGPT — research synthesis, experiment design review, technical explanation, debugging advice, adversarial review, writing assistance.
- OpenAI Codex — implementation, tests, profiling, repository audits, debugging, documentation, and controlled code changes.
- Deep Research — broad literature discovery/synthesis.

Human review retained authority over experiment authorization, scientific controls, acceptance/rejection decisions, benchmark timing, claims, and publication. See `AI_ASSISTANCE.md`.

---

## 11. Pre-release code-attribution audit

Before final publication/submission:

- [ ] Search the final repository for copied/adapted third-party snippets.
- [ ] For every match, record upstream file + commit/tag + license + local path.
- [ ] Verify that every upstream repository with copied code has a compatible license and required notices.
- [ ] Re-check `hed-ucas/Layer-wise-Learning-Rate`; because no license was observed, retain independent implementation only.
- [ ] Ensure no pretrained weights/tokenizers/model code were silently introduced by examples or dependencies.
- [ ] Ensure evaluation task templates are credited to their benchmark/harness sources.
- [ ] Ensure comments/docs do not imply original invention of WSD, LLR, Magma, QK-Norm, CWD, RoPE, RMSNorm, SwiGLU, AdamW, or embedding tying.
- [ ] Ensure AI-assistance disclosure matches actual assistance through the deadline.

Last substantive reconstruction: 2026-09-04.
