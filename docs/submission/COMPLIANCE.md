# Final submission compliance — reconciled 2026-09-09

The 2026-09-08 [official rules](https://gibc-v2.devpost.com/rules) and [competition page](https://gibc-v2.devpost.com/) were rechecked. This audit does not submit the project.

## Track 01 evidence

| Requirement | Repository evidence / disposition |
|---|---|
| At most 50M, scratch weights, reproducible model/count | Root README; exact 49,860,480; config and count script; no pretrained initialization |
| Hardware, duration and approximate compute | Root README: RTX 5090 Laptop GPU, 37.94 h; independently calculated 6NT = 2.153967221829795840e18 FLOPs |
| Required results and methodology | README/RESULTS; four harness tasks and separate held-out WikiText; frozen CPU FP32 protocol/scripts |
| Setup, prerequisites and usage | README and inference release guide; separate clean-environment smoke passed; no benchmark rerun |
| Source/data/tools attribution and AI disclosure | README, SOURCE_LEDGER, CODE_ATTRIBUTION, AI_ASSISTANCE and license audit |
| Public reproducibility | [Public weights](https://huggingface.co/AryaErgin/gibc-v2-exp020) and anonymous hash verification complete; competition source published at `gibc-v2-track01-exp020-v1.0.0` / `69f56d7b1f4f377f94a30216a9945df8a6662cce` |

6NT is a conventional approximate theoretical estimate for the final training run only, not measured energy, exact hardware operations or total campaign compute.

## Six Devpost components

| Component | State / human action |
|---|---|
| Project description | [Draft](DEVPOST_DRAFT.md) ready to paste/review |
| Public source repository with setup/usage | Public source and frozen tag published; [judge quickstart](../../README.md#run-the-public-model--no-gpu-or-benchmark-download) |
| 2–5 minute English public/unlisted demo video | **PENDING**; [3:30 storyboard](DEMO_STORYBOARD.md) |
| Complete Built With list | Below; copy to Devpost, including AI tools |
| Team information | **MANUAL**: real names, invite all members; verify student status, age/guardian permission if applicable, and team size at most six |
| At least three high-quality images | Seven native 1800×1200 PNGs, each below 5 MB, plus editable SVGs; [ordered gallery and evidence captions](../assets/exp020-devpost/README.md). Original scientific figures remain preserved. |

Additional manual eligibility checks: one project/one track, valid student/age eligibility, public access without permission requests, English submission, and final submission before the official deadline. Repository files cannot prove the team's personal eligibility or that Devpost fields were completed.

## Built With — copyable inventory

Python; PyTorch; NVIDIA CUDA; NVIDIA GeForce RTX 5090 Laptop GPU; Windows; WSL2/Linux; OMEN thermal controls; Hugging Face datasets, tokenizers, safetensors and Hub; FineWeb; FineWeb-Edu; Common Crawl; EleutherAI lm-evaluation-harness; HellaSwag; ARC-Easy; PIQA; WinoGrande; WikiText-103; NumPy; PyArrow; PyYAML; SQLite; sacrebleu (environment dependency, not these task metrics); pytest; Git/GitHub; git-filter-repo; Pillow; ChatGPT; OpenAI Codex; Deep Research. Include the actual video editor only after the human chooses/uses it.

## Narrow outstanding issues

- Model hosting and its anonymous hash check are complete; source publication is complete; video, eligibility and Devpost field completion remain manual.
- [License audit](LICENSE_AUDIT.md): PIQA license unknown; WikiText source card has inconsistent version statements. No benchmark data is distributed.
- [Evaluator supplement](../../results/exp020-evaluation-source-provenance.json): raw short hash is now resolved, while PIQA cross-snapshot identity remains unproved.
- [Release guide](EXP020_RELEASE.md): exact public weights, pinned download and manifest checks; [clean-environment smoke](PREPUBLICATION_SMOKE.md) passed. The prior packaging-only no-inference receipt remains a historical record.
