# gibc-v2-foundational-llm

Reproducible, from-scratch decoder-only language-model research infrastructure for GIBC Track 01.

## Current status

Near-Cap Architecture Recipe v3 is accepted: 49,860,480 trainable parameters; 640 width; 9 layers; 20 heads x 32; SiLU-gated SwiGLU d_ff 1,728; tied embeddings/output. EXP-011 completed its single authorized 1.5B-prediction-token long-horizon calibration with the frozen 45,777-step cosine schedule. This is calibration evidence, not automatic final-model promotion. Official benchmarks have not run; the next action is research review.

## Repository rules

Public source code and documentation belong in Git. Local environments, credentials, raw/tokenized data, caches, checkpoints, and large outputs are ignored.

See [RESULTS.md](RESULTS.md), [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [ARCHITECTURE.md](ARCHITECTURE.md), [DATA_SOURCES.md](DATA_SOURCES.md), and [results/EXP-011-summary.md](results/EXP-011-summary.md). Benchmark evaluation is permitted only after a checkpoint is selected by the frozen validation-based promotion rule; benchmark results must not select checkpoints or tune training.

## Commands

```text
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\count_parameters.py --config configs\exp011.yaml
```
