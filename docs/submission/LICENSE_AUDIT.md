# Release license and attribution audit — 2026-09-08

Scope: the final inference package, its dependencies, training datasets and required evaluation sources. This is a provenance audit, not a legal opinion. Dataset cards and license files were read; no dataset was loaded and no benchmark was executed.

## Data and evaluation sources

| Source / citation | Primary license evidence | Release treatment |
|---|---|---|
| FineWeb — Penedo et al. (2024), arXiv:2406.17557 | [HuggingFaceFW card](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md): ODC-By 1.0; Common Crawl terms also apply | Credit HuggingFaceFW/Common Crawl; original webpage rights are not replaced by the database license. No corpus is redistributed. |
| FineWeb-Edu — same paper; HuggingFaceFW dataset | [Publisher card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md): ODC-By 1.0 and Common Crawl terms | Credit upstream filtering/classifier work. Source documents, not pretrained weights, were used. No corpus/classifier is bundled. |
| WikiText-103 — Merity et al. (2016), arXiv:1609.07843 | [Frozen-revision card](https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md) is internally inconsistent: metadata lists CC-BY-SA-3.0 and GFDL; prose links CC-BY-SA-4.0 | **Exact licensing-version discrepancy unresolved.** Credit Wikipedia/Salesforce; do not redistribute text under an assumed version. |
| HellaSwag — Zellers et al. (2019), arXiv:1905.07830 | [Author repository LICENSE](https://github.com/rowanz/hellaswag/blob/master/LICENSE): MIT | Retain attribution; this is not a claim that all underlying WikiHow/ActivityNet material is newly MIT-licensed. No questions/answers bundled. |
| ARC / ARC-Easy — Clark et al. (2018), arXiv:1803.05457 | [AllenAI dataset card](https://huggingface.co/datasets/allenai/ai2_arc/raw/main/README.md): CC-BY-SA-4.0 | Credit Allen Institute for AI; no benchmark text bundled. |
| PIQA — Bisk, Zellers, Le Bras, Gao & Choi (AAAI 2020; arXiv:1911.11641) | [ybisk source card](https://huggingface.co/datasets/ybisk/piqa/raw/main/README.md): license **unknown** | **Unresolved**; do not invent a permissive license for the official `baber/piqa` mirror. No benchmark text bundled. |
| WinoGrande — Sakaguchi, Le Bras, Bhagavatula & Choi (2019), arXiv:1907.10641 | [AllenAI LICENSE](https://github.com/allenai/winogrande/blob/master/LICENSE): Apache-2.0 | Credit AllenAI/authors; no benchmark text bundled. |

These license observations do not establish historical dataset byte equivalence. Frozen training revisions remain in the manifest/evidence; no revision or exclusion protocol was changed.

## Software and code

| Software | Verified license / evidence |
|---|---|
| Project code and exported project weights/tokenizer | Existing project Apache-2.0 policy; full `LICENSE` included, consistent with the earlier EXP-012 publication policy. This does not relicense external datasets or dependencies. |
| PyTorch 2.13.0+cu132 | [Project LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) includes BSD-3-Clause and third-party notices. Installed wheel expression: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT. |
| Hugging Face datasets 3.5.1 | Apache-2.0, installed distribution LICENSE/metadata; [upstream LICENSE](https://github.com/huggingface/datasets/blob/main/LICENSE). |
| Hugging Face tokenizers 0.21.4 | [Tagged LICENSE](https://github.com/huggingface/tokenizers/blob/v0.21.4/LICENSE): Apache-2.0. |
| EleutherAI lm-eval 0.4.9.1 | MIT, installed distribution LICENSE/metadata; [upstream LICENSE](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/LICENSE.md). Task datasets retain their own terms. |
| safetensors 0.8.0 | Apache-2.0, installed LICENSE; [upstream](https://github.com/huggingface/safetensors/blob/main/LICENSE). |
| NumPy 2.4.6 | Installed wheel expression BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; retain wheel notices, not just the main project label. |
| PyYAML 6.0.2 | MIT, installed LICENSE/metadata. |

Libraries and their binary dependencies are installed separately, not vendored in the release. Their distributions supply their notices. NVIDIA CUDA, Windows/WSL and OMEN are credited tools/hardware infrastructure, not project-owned or relicensed software.

The recorded code-attribution ledger identifies independent implementations of consulted research, not copied upstream source. A tracked `src/` and `scripts/` marker scan for copyright/copied/adapted/ported source found no additional marked snippet. This is not a proof of exhaustive line-by-line originality. The release bundles only the project's exact model/config/generation source files; their historical commit bytes are checked before export. No third-party model code, weights, tokenizer, benchmark text or library source is bundled.

The unlicensed LLR repository remains consultation-only according to its existing provenance; its source was not copied or vendored and LLR is absent from the final recipe. Upstream task templates are dependency code, credited to EleutherAI and dataset authors. Research concepts (RoPE, RMSNorm, SwiGLU, AdamW, tying, and rejected methods) are not claimed as original inventions.

## Explicit remaining limits

PIQA license and cross-snapshot identity, and the WikiText card's conflicting license-version statements, are unresolved. They do not justify rewriting frozen scores or relicensing/distributing benchmark data. The package redistributes none of those datasets. Exploratory bibliography entries marked PROVISIONAL remain historical research notes, not verified final-recipe citations or external numerical claims. Core dataset/benchmark and runtime citations are maintained in `REFERENCES.bib`.

AI assistance in research, implementation, review, documentation and packaging is disclosed in the source repository and model card; the submitted weights themselves were trained from scratch.
