# gibc-v2-foundational-llm

Reproducible, from-scratch decoder-only language-model research infrastructure for GIBC Track 01.

## Status

EXP-001 training is complete at 3,052 updates / 100,007,936 prediction tokens. Exact local training evidence is recorded in `results/EXP-001-summary.md`; final-checkpoint benchmark evaluation is pending EXP-001D. The final local checkpoint must not be resumed or modified.

## Repository rules

Public source code and documentation belong in Git. Local environments, credentials, raw/tokenized data, caches, checkpoints, and large outputs are ignored.

See `AGENTS.md`, `PROJECT_PLAN.md`, `AI_ASSISTANCE.md`, and `results/EXP-001A-summary.md`. Local extended instructions are intentionally ignored.

## Commands

```text
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\count_parameters.py
.\.venv\Scripts\python.exe scripts\prepare_exp001.py
.\.venv\Scripts\python.exe scripts\prepare_exp001.py --full-data  # preparation only; no training
.\.venv\Scripts\python.exe scripts\train_exp001.py
.\.venv\Scripts\python.exe scripts\generate.py "The" --max-new-tokens 16
.\.venv\Scripts\python.exe scripts\eval_exp001.py --tasks hellaswag --limit 1
```
