# GIBC V2 Track 01 — from-scratch foundational language model

## 30-second summary

This repository contains the reproducible source, configurations, tests, and
provenance for a from-scratch, decoder-only GIBC Track 01 model. The final
EXP-012 checkpoint was selected by frozen validation before benchmarks,
contains exactly **49,860,480** trainable parameters, and completed the
required official evaluation on 2026-08-28. No pretrained weights,
fine-tuning, distillation, benchmark-answer training, or benchmark-driven
checkpoint selection was used.

The public source is intentionally separate from large local checkpoints,
datasets, caches, and benchmark outputs. The planned inference-only model
package is described in [the publication plan](docs/EXP-012-INFERENCE-PUBLICATION.md);
it requires an explicit publication approval before upload.

## Final model and official results

| Item | Final record |
|---|---|
| Architecture | Decoder-only causal Transformer; vocab 8,192; width 640; 9 layers; 20 heads × 32; SwiGLU `d_ff=1728`; RoPE; pre-RMSNorm; tied embedding/output; context 512 |
| Trainable parameters | 49,860,480 exactly; [checked-in count evidence](results/exp012-parameter-count.json) |
| Selected checkpoint SHA-256 | `cacb728b3963c10af8f4613149d8b879b0ef6e44558069726c621c1cb1bb981c` |
| Tokenizer SHA-256 | `c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14` |
| HellaSwag acc_norm | 0.28759211312487554 |
| ARC-Easy acc_norm | 0.36447811447811446 |
| PIQA acc_norm | 0.6022850924918389 |
| WinoGrande acc | 0.5035516969218626 |
| WikiText-103 held-out PPL / BPB | 35.93897257521639 / 1.4083853215598 |

The compact, safe-to-publish official provenance record is
[results/exp012-official-provenance.json](results/exp012-official-provenance.json).
Raw official evaluator artifacts remain local because they are large; their
SHA-256 values, protocol, exact metrics, and runtime are recorded in
[the EXP-012 official-evaluation record](experiments/EXP-012-official-evaluation.md).

## Training data, efficiency, and hardware

EXP-012 trained from scratch on the frozen Data Recipe v1: a deterministic 2:1
FineWeb/FineWeb-Edu mixture with global canonical-content SHA-256
deduplication and an indexed normalized 13-gram contamination screen. The
final run used 2,399,993,856 prediction tokens over 73,242 updates.

Training ran on Windows 10.0.26200 with an NVIDIA GeForce RTX 5090 Laptop GPU,
Python 3.11.9, PyTorch 2.13.0+cu132 / CUDA 13.2, BF16 autocast and FP32
parameters/optimizer state. Model-training wall time was 24,362.3826 seconds;
separate data preparation took 29,422.8518 seconds. Approximate training
compute was 717,989,073,943,265,280 FLOPs (`6 × parameters × prediction
tokens`). Full run evidence is in [results/EXP-012-summary.md](results/EXP-012-summary.md).

## Quick start

Prerequisites: Python 3.11 and a compatible PyTorch installation for the
intended platform. PyTorch is deliberately not pinned in `pyproject.toml`
because its build is platform/CUDA-specific; install the appropriate build from
[the official PyTorch selector](https://pytorch.org/get-started/locally/) first,
then install this project.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the platform-appropriate PyTorch build from pytorch.org.
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
```

### Verify the final parameter count

```powershell
.\.venv\Scripts\python.exe scripts\count_parameters.py `
  --config configs\exp012.yaml --expected-total 49860480 --json
```

This command must exit zero and print `"total": 49860480`. Its committed
output is [results/exp012-parameter-count.json](results/exp012-parameter-count.json).

### Final-model inference

The final public inference package is not uploaded yet; see the approval-gated
publication plan above. Once obtained, place its files in `FINAL_MODEL_DIR`
without changing their names or manifest, then run:

```powershell
.\.venv\Scripts\python.exe scripts\generate.py "A short prompt" `
  --config FINAL_MODEL_DIR\exp012.yaml `
  --checkpoint FINAL_MODEL_DIR\model_state.pt `
  --tokenizer FINAL_MODEL_DIR\tokenizer.json `
  --device auto --max-new-tokens 64 --temperature 0.0
```

`generate.py` accepts only explicit config, checkpoint, and tokenizer paths;
it loads the package's strict `model` state dictionary and performs local
generation only.

### Evaluation reproduction

The frozen official evaluators are present at
`scripts/run_exp012_cpu_official_sequence.py`,
`scripts/eval_exp012_cpu_task.py`, and
`scripts/eval_exp012_wikitext103.py`. Their CPU FP32, zero-shot, batch-16,
context-512 protocol and exact commands are recorded in
[experiments/EXP-012-official-evaluation.md](experiments/EXP-012-official-evaluation.md).
Do not use official results to select checkpoints or tune the model; no
benchmark command is run by the ordinary test suite.

## Evidence, limitations, and credits

- [RESULTS.md](RESULTS.md) records final results and the limited comparable
  EXP-006A-to-EXP-012 reasoning-task comparison.
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [PROJECT_PLAN.md](PROJECT_PLAN.md),
  [ARCHITECTURE.md](ARCHITECTURE.md), and
  [results/EXP-012-summary.md](results/EXP-012-summary.md) retain training and
  decision provenance.
- The contamination screen detects only indexed normalized 13-gram overlap; it
  does not prove absence of all lexical, semantic, or unknown-source overlap.
- Exact recreation of the full training run requires the original hardware,
  pinned data revisions, local data artifacts, and tokenizer/checkpoint files;
  those large artifacts are intentionally not committed. The publication plan
  makes inference reproduction possible once the model artifact is approved.
- Third-party components include PyTorch, Hugging Face `datasets`,
  `tokenizers`, `lm-evaluation-harness`, FineWeb, and FineWeb-Edu. See the
  locked dependency versions and experiment records for provenance.

## AI assistance disclosure

AI assistance supported experiment specification review, implementation,
testing, profiling, artifact audits, and documentation. Human research review
retained authority over architecture, data, checkpoint selection, benchmark
execution, and publication. AI did not provide pretrained weights, synthetic
training data, benchmark answers, or benchmark-driven model selection. See
[AI_ASSISTANCE.md](AI_ASSISTANCE.md).
