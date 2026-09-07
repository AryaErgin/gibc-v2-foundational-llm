"""Run one future frozen EXP-020 lm-eval task in CPU FP32 after hard gates."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.exp020_official_evaluation import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PARAMS,
    EXPECTED_PREDICTION_TOKENS,
    EXPECTED_SELECTED_STEP,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SUMMARY_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    EXPECTED_LM_METRICS,
    atomic_write_json,
    load_exp020_cpu_model,
    validate_lm_task_record,
    validate_runtime_provenance,
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(EXPECTED_LM_METRICS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an EXP-020 official result: {args.output}")
    validate_runtime_provenance(args.runtime_provenance)
    started_at = _timestamp()
    started = time.perf_counter()
    torch, config, _, _, adapter = load_exp020_cpu_model(args)
    # Only the future execution path imports the harness and requests a task.
    import datasets
    import lm_eval

    raw_result = lm_eval.simple_evaluate(model=adapter, tasks=[args.task], num_fewshot=0, batch_size=args.batch_size, limit=None)
    record = {
        "metadata": {
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "tokenizer": str(args.tokenizer),
            "checkpoint_sha256": "95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89",
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "summary_sha256": EXPECTED_SUMMARY_SHA256,
            "trainable_parameters": EXPECTED_PARAMS,
            "selected_step": EXPECTED_SELECTED_STEP,
            "prediction_tokens": EXPECTED_PREDICTION_TOKENS,
            "training_source_commit": EXPECTED_SOURCE_COMMIT,
            "device": "cpu",
            "precision": "fp32",
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "datasets_version": datasets.__version__,
            "lm_eval_version": lm_eval.__version__,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "context_length": config.model.context_length,
            "command": sys.argv,
            "platform": platform.platform(),
            "started_at": started_at,
            "ended_at": _timestamp(),
            "wall_seconds": time.perf_counter() - started,
            "no_pretrained_weights": True,
        },
        "raw_lm_eval_result": raw_result,
    }
    validate_lm_task_record(record, args.task)
    atomic_write_json(args.output, record)
    print(f"EXP-020 official task completed: {args.task}")


if __name__ == "__main__":
    main()
