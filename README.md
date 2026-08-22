# gibc-v2-foundational-llm

Reproducible, from-scratch decoder-only language-model research infrastructure for GIBC Track 01.

## Status

EXP-001A infrastructure is validated. The approved 100M-token EXP-001 training run must not start without research-chat review.

## Repository rules

Public source code and documentation belong in Git. Local environments, credentials, raw/tokenized data, caches, checkpoints, and large outputs are ignored.

See `AGENTS.md`, `PROJECT_PLAN.md`, `AI_ASSISTANCE.md`, and `results/EXP-001A-summary.md`. Local extended instructions are intentionally ignored.

## Commands

```text
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\count_parameters.py
.\.venv\Scripts\python.exe scripts\prepare_exp001.py
.\.venv\Scripts\python.exe scripts\train_exp001.py
.\.venv\Scripts\python.exe scripts\generate.py "The" --max-new-tokens 16
.\.venv\Scripts\python.exe scripts\eval_exp001.py --tasks hellaswag --limit 1
```
