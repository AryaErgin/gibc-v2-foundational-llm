"""Persistence wrapper for immutable lm-eval task results and execution metadata."""

from __future__ import annotations

from typing import Any


def evaluation_output_record(
    *,
    task: str,
    checkpoint: str,
    tokenizer: str,
    tokenizer_sha256: str,
    batch_size: int,
    lm_eval_version: str,
    num_fewshot: int,
    wall_seconds: float,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """Keep the harness result unchanged while persisting reproducibility metadata."""
    return {
        "metadata": {
            "task": task,
            "checkpoint": checkpoint,
            "tokenizer": tokenizer,
            "tokenizer_sha256": tokenizer_sha256,
            "batch_size": batch_size,
            "lm_eval_version": lm_eval_version,
            "num_fewshot": num_fewshot,
            "wall_seconds": wall_seconds,
        },
        "raw_lm_eval_result": raw_result,
    }
