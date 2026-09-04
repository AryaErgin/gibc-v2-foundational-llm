# Source Ledger

This file is the project's over-inclusive attribution and provenance ledger. It is intentionally broader than the eventual paper bibliography.

**Purpose:** prevent accidental plagiarism, omitted attribution, and false originality claims. A source belongs here if it materially influenced a project decision, experiment, implementation, comparison, compliance interpretation, research hypothesis, or written claim — even if it is never cited in the final paper.

**Source-of-truth rule:** for claims about our own experiments, use repository experiment/provenance records. For external scientific claims, cite the original paper/project rather than AI-generated research summaries. If metadata below is marked `VERIFY`, verify it against the primary source before camera-ready publication.

## Attribution rules

1. A named method we implemented or experimentally tested must cite the original method paper.
2. If upstream code was inspected, record repository, commit when known, license status, and whether code was copied or independently reimplemented.
3. Datasets and benchmarks must cite both their canonical publication/project and the exact version/revision actually used when available.
4. Search summaries, ChatGPT/Deep Research outputs, and our own repository are not substitutes for external citations.
5. Distinctive prose from a paper, repository, AI report, or documentation must never be reused verbatim without quotation and attribution.
6. A source considered and rejected does not necessarily belong in the final paper, but it remains in this ledger.

---

# A. Competition / compliance

| Source | URL | Project use | Status |
|---|---|---|---|
| Global Innovation Build Challenge V2 — Track 01 rules | https://gibc-v2.devpost.com/rules | Parameter cap, scratch-training requirement, public-repo requirement, required evaluations, reporting, AI-assistance disclosure, project period/deadline | **Authoritative external rule source** |

---

# B. Core architecture and optimization lineage

| Source | Identifier / URL | Project use | Relationship |
|---|---|---|---|
| Vaswani et al., *Attention Is All You Need* | arXiv:1706.03762 — https://arxiv.org/abs/1706.03762 | Transformer/self-attention lineage | Background architecture citation |
| Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* | arXiv:2104.09864 — https://arxiv.org/abs/2104.09864 | RoPE | Final architecture |
| Zhang & Sennrich, *Root Mean Square Layer Normalization* | arXiv:1910.07467 — https://arxiv.org/abs/1910.07467 | RMSNorm | Final architecture |
| Shazeer, *GLU Variants Improve Transformer* | arXiv:2002.05202 — https://arxiv.org/abs/2002.05202 | SwiGLU/GLU-family FFN rationale | Final architecture; local EXP-008 is stronger project-specific evidence |
| Press & Wolf, *Using the Output Embedding to Improve Language Models* | arXiv:1608.05859 — https://arxiv.org/abs/1608.05859 | Input/output embedding tying | Final architecture |
| Loshchilov & Hutter, *Decoupled Weight Decay Regularization* | arXiv:1711.05101 — https://arxiv.org/abs/1711.05101 | AdamW | Final optimizer lineage |
| Henry et al., *Query-Key Normalization for Transformers* | arXiv:2010.04245 — https://arxiv.org/abs/2010.04245 | QK normalization concept | EXP-018 background; our exact RMS/Q-gain variant is project-specific |

---

# C. Scaling, token horizon, and parameter-constrained training

| Source | Identifier / URL | Why it mattered |
|---|---|---|
| Kaplan et al., *Scaling Laws for Neural Language Models* | arXiv:2001.08361 — https://arxiv.org/abs/2001.08361 | Classical parameter/data/compute scaling baseline |
| Hoffmann et al., *Training Compute-Optimal Large Language Models* | arXiv:2203.15556 — https://arxiv.org/abs/2203.15556 | Chinchilla joint compute allocation; explicitly not treated as a fixed-parameter stopping rule |
| Gadre et al., *Language Models Scale Reliably with Over-training and on Downstream Tasks* | arXiv:2403.08540 — https://arxiv.org/abs/2403.08540 | Direct evidence down to very small models that deliberate overtraining remains predictable |
| Muennighoff et al., *Scaling Data-Constrained Language Models* | arXiv:2305.16264 — https://arxiv.org/abs/2305.16264 | Unique-vs-repeated data scaling; repetition useful only under genuine data scarcity |
| Xue et al., *To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis* | arXiv:2305.13230 — https://arxiv.org/abs/2305.13230 | Eventual degradation under excessive data repetition |
| Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling* | arXiv:2304.01373 — https://arxiv.org/abs/2304.01373 | 14M/31M/70M long-training comparator; parameter-accounting warning because of large untied vocabulary |
| Choshen, Zhang & Andreas, *A Hitchhiker's Guide to Scaling Law Estimation* | arXiv:2410.11840 — https://arxiv.org/abs/2410.11840 | Extrapolation caution; multiple scales/checkpoints |
| Tukenov, *SozKZ: Training Efficient Small Language Models for Kazakh from Scratch* | arXiv:2603.20854 — https://arxiv.org/abs/2603.20854 | Nominal 50M / 9B-token from-scratch precedent; not a performance comparator because language/domain differ |
| Chang et al., *Scaling Parameter-Constrained Language Models with Quality Data* | arXiv:2410.03083 — https://arxiv.org/abs/2410.03083 | Direct 25M/50M/75M/125M evidence that data quality materially affects small-model downstream performance |
| Multilingual small-model scaling-law study | **VERIFY exact paper metadata before citation** | Consulted in Deep Research as ~85M+ evidence that mixture/sampling relationships can transfer across scale |
| Decoder-only translation scaling study | **VERIFY exact paper metadata before citation** | Consulted as ~70M+ depth/width scaling evidence in a different task |

---

# D. Data quality, mixture, filtering, deduplication

| Source | Identifier / URL | Project use |
|---|---|---|
| Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* | arXiv:2406.17557 — https://arxiv.org/abs/2406.17557 | Primary FineWeb/FineWeb-Edu paper; filtering, MinHash dedup, Edu trade-offs, data-ablation evidence |
| HuggingFaceFW/FineWeb dataset | https://huggingface.co/datasets/HuggingFaceFW/fineweb | Actual training source. Project pin: `sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc` |
| HuggingFaceFW/FineWeb-Edu dataset | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | Actual training source. Project pin: revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Soldaini et al., *Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* | arXiv:2402.00159 — https://arxiv.org/abs/2402.00159 | Data construction/filtering/dedup comparison |
| Li et al., *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models* | arXiv:2406.11794 — https://arxiv.org/abs/2406.11794 | Model-based data-quality filtering; considered but rejected as default replacement under competition-integrity concerns |
| Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training* | arXiv:2407.01492 — https://arxiv.org/abs/2407.01492 | Evidence that mixture interactions are nontrivial and intuitive ratios can be unreliable |
| Marion et al., *When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale* | arXiv:2309.04564 — https://arxiv.org/abs/2309.04564 | Data-quality pruning evidence |
| Lee et al., *Deduplicating Training Data Makes Language Models Better* | arXiv:2107.06499 — https://arxiv.org/abs/2107.06499 | General deduplication evidence |
| Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* | arXiv:2112.11446 — https://arxiv.org/abs/2112.11446 | Filtering/repetition/data-quality background |
| Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* | arXiv:2502.02737 — https://arxiv.org/abs/2502.02737 | Larger-scale data-mixture/quality comparator; FineWeb-Edu trade-offs |
| Sedova et al., *Scaling Laws for Mixture Pretraining Under Data Constraints* | arXiv:2605.12715 — https://arxiv.org/abs/2605.12715 | Mixture/repetition scaling; used in EXP-015 interpretation and later data-strategy review |
| Apple ML Research summary for mixture-scaling work | https://machinelearning.apple.com/research/scaling-laws-mixture-pretraining | Institutional summary consulted alongside paper |
| DoReMi | **VERIFY exact paper metadata before citation** | Considered as learned-domain-weighting prior art; not implemented because it creates another optimization loop and scale mismatch |

---

# E. Tokenization and vocabulary allocation

| Source | Identifier / URL | Project use |
|---|---|---|
| Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* | arXiv:2407.13623 — https://arxiv.org/abs/2407.13623 | Main evidence that vocabulary optimum depends on model/compute scale; supports treating 8K as plausible rather than universally optimal |
| Schmidt et al., *Tokenization Is More Than Compression* | arXiv:2402.18376 — https://arxiv.org/abs/2402.18376 | Tokenizer quality beyond raw compression/fertility |
| Huang et al., *Over-Tokenized Transformer* | **VERIFY exact identifier before citation** | Larger-vocabulary mechanism explored and rejected under total-parameter cap |
| Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units* | arXiv:1508.07909 — https://arxiv.org/abs/1508.07909 | Canonical BPE reference |
| Length-MAX | **VERIFY exact paper title/identifier before citation** | Consulted as an 8K tokenizer redesign candidate; not implemented |

---

# F. Learning-rate schedules and optimizer research

## F1. Methods actually tested

| Source | Identifier / URL | Project relationship |
|---|---|---|
| Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* | arXiv:2404.06395 — https://arxiv.org/abs/2404.06395 | WSD source lineage; EXP-013 / EXP-017 |
| Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective* | arXiv:2410.05192 — https://arxiv.org/abs/2410.05192 | WSD interpretation/mechanistic research |
| Dremov et al., *Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler* | arXiv:2508.01483 — https://arxiv.org/abs/2508.01483 | Cooldown interpretation and bias/variance caution |
| He et al., *One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs* | arXiv:2605.22297v3 — https://arxiv.org/abs/2605.22297v3 | EXP-014 algorithmic source |
| `hed-ucas/Layer-wise-Learning-Rate` | https://github.com/hed-ucas/Layer-wise-Learning-Rate — inspected commit `bbd0dcf86af80b8843866a9a041086a37de35897` | Behavioral reference for EXP-014. No verbatim source vendored; local implementation recorded as independent. Upstream repo had no declared license when inspected. |
| Joo et al., *On Surprising Effectiveness of Masking Updates in Adaptive Optimizers* | arXiv:2602.15322 — https://arxiv.org/abs/2602.15322 | Magma source; EXP-016 independently implemented from paper |
| Chen et al., *Cautious Weight Decay* | arXiv:2510.12402v2 — https://arxiv.org/abs/2510.12402 | CWD source; EXP-019 independently implemented from Algorithm 1 |

## F2. Optimizer/schedule candidates researched but not promoted

| Source | Identifier / URL | Use |
|---|---|---|
| Zhao et al., *Deconstructing What Makes a Good Optimizer for Language Models* | arXiv:2407.07972 — https://arxiv.org/abs/2407.07972 | Evidence that fairly tuned adaptive optimizers can be closer than headline comparisons imply |
| Zhang et al., *How Does Critical Batch Size Scale in Pre-training?* | arXiv:2410.21676 — https://arxiv.org/abs/2410.21676 | Batch/LR interaction evidence; reason not to copy large-cluster batch sizes blindly |
| Wen et al., *Fantastic Pretraining Optimizers and Where to Find Them* | arXiv:2509.02046 — https://arxiv.org/abs/2509.02046 | Major evidence that optimizer rankings/speedups can shrink or flip with scale and training horizon |
| Liu et al., *Muon is Scalable for LLM Training* | arXiv:2502.16982 — https://arxiv.org/abs/2502.16982 | Muon scaling evidence |
| Li et al., *NorMuon: Making Muon More Efficient and Scalable* | arXiv:2510.05491 — https://arxiv.org/abs/2510.05491 | Candidate optimizer; not implemented |
| Zhang et al., *Muon+: Towards Better Muon via One Additional Normalization Step* | arXiv:2602.21545 — https://arxiv.org/abs/2602.21545 | Direct ~60M evidence that strengthened Muon-family candidacy during method search |
| Zhang et al., *TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon for Large Language Model Pre-Training* | arXiv:2601.23261 — https://arxiv.org/abs/2601.23261 | Direct ~60M optimizer evidence; not implemented |
| Ferbach et al., *Logarithmic-time Schedules for Scaling Language Models with Momentum* (ADana) | arXiv:2602.05298 — https://arxiv.org/abs/2602.05298 | Strict ≤50M optimizer/schedule comparator; rejected on transfer/engineering risk |
| `mlexpos/adana` | https://github.com/mlexpos/adana | Official/author implementation consulted during ADana evaluation |
| Glentis et al., *A Minimalist Optimizer Design for LLM Pretraining* | arXiv:2506.16659 — https://arxiv.org/abs/2506.16659 | SCALE optimizer candidate; research only |
| Wang et al., *Taming Momentum: Rethinking Optimizer States Through Low-Rank Approximation* (LoRA-Pre) | arXiv:2602.24283 — https://arxiv.org/abs/2602.24283 | Near-scale ~60M optimizer evidence; not implemented |
| `mrflogs/LoRA-Pre` | https://github.com/mrflogs/LoRA-Pre | Author/project code consulted during literature review |
| Sanyal et al., *Early Weight Averaging Meets High Learning Rates for LLM Pre-training* | arXiv:2306.03241 — https://arxiv.org/abs/2306.03241 | Checkpoint/weight-averaging candidate; not implemented |
| Song et al., *Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training* | arXiv:2507.09846 — https://arxiv.org/abs/2507.09846 | Schedule-free optimizer analysis; research only |
| Ma & Chen, *WSqD: A Horizon-Free Learning Rate Schedule for Large Model Training* | arXiv:2607.10959 — https://arxiv.org/abs/2607.10959 | WSD alternative retained only in literature ledger |
| Liu et al., *Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining* | arXiv:2608.24814 — https://arxiv.org/abs/2608.24814 | Diagnostic effective-LR interpretation during EXP-014 research |
| Apte et al., *Anytime Training with Schedule-Free Spectral Optimization* | arXiv:2605.23061 — https://arxiv.org/abs/2605.23061 | Schedule-free spectral candidate; not implemented |
| Grigorev, *IMU-1: Sample-Efficient Pre-training of Small Language Models* | arXiv:2602.02522 — https://arxiv.org/abs/2602.02522 | Bundled modern-recipe comparator; helped motivate QK-Norm/NorMuon/CWD research but not causal evidence for any single component |
| *The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training* | arXiv:2502.19002 — https://arxiv.org/abs/2502.19002 | Candidate optimizer/training-dynamics research; not implemented |
| Adam-mini | **VERIFY exact source metadata before citation** | Considered in optimizer search, not implemented |

---

# G. Curriculum and data-scheduling literature

| Source | Identifier / URL | Project use |
|---|---|---|
| Bergsma, Dey & Hestness, *Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs* | arXiv:2509.25380 — https://arxiv.org/abs/2509.25380 | TREC; mechanistic prior for data timing and explicit WSD interaction; EXP-015 interpretation |
| Luo et al., *How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining* | arXiv:2511.18903 — https://arxiv.org/abs/2511.18903 | Direct prior art for curriculum × LR interaction; prevented false novelty claim |
| Zhang et al., *Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning* | arXiv:2506.11300 — https://arxiv.org/abs/2506.11300 ; EACL 2026 DOI 10.18653/v1/2026.eacl-long.271 | Broader curriculum evidence; warmup/random sequencing and endpoint effects |
| *Sequence Length Matters in Data Scheduling for Accelerating Language Model Pretraining* | **VERIFY exact OpenReview/arXiv identifier before citation** | Direct ~60M scheduling evidence; supportive only, not implemented |
| RegMix | arXiv:2407.01492 | Also relevant here as evidence against intuitive manual mixture scheduling |

---

# H. Small-model / comparator projects consulted

| Source / project | Identifier / URL | Use / caveat |
|---|---|---|
| Pythia family | arXiv:2304.01373; https://github.com/EleutherAI/pythia | 14M/31M/70M/160M and dedup variants; long-training and embedding-accounting comparator, not apples-to-apples benchmark comparator |
| SozKZ | arXiv:2603.20854 | 50M/9B-token language-specific precedent |
| Gadre overtraining suite | arXiv:2403.08540 | Controlled 11M+ overtraining family |
| ADana / Enoki scaling family | arXiv:2602.05298; https://github.com/mlexpos/adana | Includes ~45.7M point, but embedding-heavy and much shorter horizon |
| TinyStories | arXiv:2305.07759 — https://arxiv.org/abs/2305.07759 | Tiny-LM capability comparator; synthetic teacher-generated corpus makes it unsuitable as a GIBC recipe comparator |
| IMU-1 | arXiv:2602.02522 | 430M bundled-recipe comparator |
| L20-Edu-135M | **VERIFY exact project/paper metadata before citation** | Recent single-GPU resource comparator; above cap and later post-training makes final scores non-comparable |
| LoRA-Pre small models | arXiv:2602.24283 | ~60M optimizer evidence |
| Muon+/TEON small models | arXiv:2602.21545; 2601.23261 | ~60M optimizer evidence |

---

# I. Multi-token prediction and other rejected candidate research

| Source | Identifier / URL | Use |
|---|---|---|
| *Babies Learn to Look Ahead: Multi-Token Prediction in Small LMs* | ACL Anthology: https://aclanthology.org/2025.babylm-main.41/ | Small-LM MTP evidence considered; MTP removed from execution path because of cap/complexity/evidence risk |
| Multi-token prediction / MTP broader literature | **VERIFY any exact additional papers used before final citation** | Candidate research only; no MTP model was trained in this project |

---

# J. Benchmarks and evaluation

The benchmark datasets were used both for final evaluation and, where public splits were available, for contamination-index construction. Exact Hugging Face revisions used by the project are recorded in `provenance/exp001-benchmark-revisions.json`.

| Benchmark | Canonical source | Exact project dataset revision |
|---|---|---|
| HellaSwag | Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?*, arXiv:1905.07830 — https://arxiv.org/abs/1905.07830 | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| ARC / ARC-Easy | Clark et al., *Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge*, arXiv:1803.05457 — https://arxiv.org/abs/1803.05457 | `210d026faf9955653af8916fad021475a3f00453` |
| PIQA | Bisk et al., *PIQA: Reasoning about Physical Commonsense in Natural Language*, arXiv:1911.11641 — https://arxiv.org/abs/1911.11641 | `2e8ac2dffd59bac8c3c6714948f4c551a0848bb0` |
| WinoGrande | Sakaguchi et al., *WinoGrande: An Adversarial Winograd Schema Challenge at Scale*, arXiv:1907.10641 — https://arxiv.org/abs/1907.10641 | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` |
| WikiText / WikiText-103 | Merity et al., *Pointer Sentinel Mixture Models*, arXiv:1609.07843 — https://arxiv.org/abs/1609.07843 | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| EleutherAI Language Model Evaluation Harness | https://github.com/EleutherAI/lm-evaluation-harness | Frozen project version `lm-eval==0.4.9.1` |

---

# K. Software / libraries / tools materially used

These are primarily reproducibility/acknowledgment items rather than scholarly claims, but they must be credited where appropriate.

| Software | Project role |
|---|---|
| PyTorch | Model/training runtime |
| NVIDIA CUDA | GPU runtime |
| Python 3.11 | Main runtime |
| NumPy | Array/memmap/data utilities |
| Hugging Face `datasets==3.5.1` | Dataset streaming/loading |
| Hugging Face `tokenizers==0.21.4` | Scratch tokenizer training/loading |
| `lm-eval==0.4.9.1` | Required benchmark evaluation |
| `safetensors==0.8.0` | Model serialization |
| `PyYAML==6.0.2` | Configuration loading |
| SQLite / Python `sqlite3` | Contamination n-gram index |
| `pytest==8.3.5` | Test suite |
| `git-filter-repo==2.47.0` | Repository maintenance |
| Git / GitHub | Version control and public repository |
| GitHub CLI | WSL authentication/push operations |
| WSL2 | Later scientifically qualified runtime environment |
| Windows host environment | Main host OS / operational environment |

---

# L. External code repositories explicitly inspected

See `CODE_ATTRIBUTION.md` for the code-focused version of this table.

- `https://github.com/hed-ucas/Layer-wise-Learning-Rate` — EXP-014 behavioral reference; pinned commit recorded.
- `https://github.com/EleutherAI/lm-evaluation-harness` — evaluation framework.
- `https://github.com/EleutherAI/pythia` — small-model comparator/provenance.
- `https://github.com/mlfoundations/scaling` — overtraining/scaling research code consulted in Deep Research.
- `https://github.com/huggingface/datablations` — data-constrained scaling project consulted in Deep Research.
- `https://github.com/sail-sg/regmix` — RegMix implementation/project.
- `https://github.com/sail-sg/scaling-with-vocab` — vocabulary-scaling implementation/project.
- `https://github.com/google-research/deduplicate-text-datasets` — deduplication implementation reference.
- `https://github.com/mlexpos/adana` — ADana implementation/project.
- `https://github.com/mrflogs/LoRA-Pre` — LoRA-Pre project.
- `https://github.com/K1seki221/MuonPlus` — Muon+ project consulted during optimizer review.
- `https://github.com/OpenBMB/MiniCPM` — MiniCPM/WSD project.
- Hugging Face `datasets`, `tokenizers`, FineWeb, and FineWeb-Edu repositories/resources.
- PyTorch project/documentation.

**Important:** repository inspection is not the same as source-code reuse. The project must preserve the exact distinction for every method.

---

# M. AI-assisted research and internal research artifacts

These are disclosure/provenance sources, **not scholarly citations**.

- ChatGPT research/engineering conversations for GIBC V2 Track 01.
- OpenAI Codex for implementation, profiling, testing, audits, and repository work.
- Deep Research report: *Adversarial literature review for GIBC V2 Track 01* (2026-09-01).
- Deep Research report: *Adversarial Review of the GIBC V2 ≤50M Foundational LM* (2026-09-01).
- `GIBC_LLM_MASTER_INSTRUCTIONS.txt` when present.
- `GIBCV2_EXTENDED_INSTRUCTIONS.txt`.
- This repository's own `README.md`, `PROJECT_PLAN.md`, `ARCHITECTURE.md`, `DATA_SOURCES.md`, `EXPERIMENT_LOG.md`, `RESULTS.md`, `AI_ASSISTANCE.md`, experiment records, configs, provenance JSON, manifests, hashes, and run artifacts.

When a Deep Research report supports a statement using an external paper, cite the **external paper**, not the AI report.

---

# N. Sources identified during exploratory research but requiring metadata cleanup

These were materially consulted or surfaced in decision-making, but exact bibliographic identity should be verified before they appear in a formal paper:

- Over-Tokenized Transformer.
- Length-MAX tokenizer work.
- Sequence Length Matters in Data Scheduling for Accelerating Language Model Pretraining.
- L20-Edu-135M project/paper.
- Multilingual scaling-law study (~85M+ family in Deep Research comparator table).
- Decoder-only translation scaling study (~70M+ family in Deep Research comparator table).
- Adam-mini source used during early optimizer consideration.
- Any additional multi-token-prediction paper beyond the BabyLM/ACL source above.
- Any standalone checkpoint-averaging/EMA source beyond Sanyal et al. that is mentioned in future drafts.

**Rule:** do not cite one of these in a camera-ready paper until exact title/authors/venue/identifier are resolved from a primary source.

---

# O. Maintenance checklist

Before any paper/submission release:

- [ ] Search `EXPERIMENT_LOG.md`, `PROJECT_PLAN.md`, `experiments/`, `provenance/`, and git history for every named external method/repository.
- [ ] Search final paper text for every capitalized/named method and ensure a citation exists.
- [ ] Search final paper text for external numerical claims and ensure primary-source citations exist.
- [ ] Verify every BibTeX author list/title/year/venue against primary metadata.
- [ ] Verify dataset revision/license/URL records.
- [ ] Verify code-license status for every repository actually copied/adapted; independent reimplementations must say so.
- [ ] Keep AI assistance disclosure synchronized with actual use.
- [ ] Never treat benchmark test results as model-selection evidence in retrospective writing.
- [ ] Keep rejected/null results attributed to their originating methods without implying those methods universally fail.

Last substantive reconstruction: 2026-09-04.
