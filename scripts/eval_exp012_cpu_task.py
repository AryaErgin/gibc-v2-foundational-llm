"""Run exactly one frozen lm-eval task on the selected EXP-012 checkpoint in CPU FP32."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from gibc_llm.official_cpu_evaluation import (
    AMENDMENT_COMMIT,
    CHECKPOINT_SHA256,
    EXPECTED_LM_METRICS,
    EXPECTED_PARAMS,
    TOKENIZER_SHA256,
    enforce_cpu_isolation,
    validate_lm_task_record,
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _load_cpu_model(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('Official CPU evaluation requires CUDA_VISIBLE_DEVICES="" before importing PyTorch.')
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Official CPU evaluation unexpectedly sees CUDA.")
    from gibc_llm.evaluation import CustomCausalLM
    from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
    from gibc_llm.tokenizer import load_tokenizer
    from gibc_llm.utils import load_config, sha256_file

    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Selected checkpoint SHA-256 mismatch.")
    if sha256_file(args.tokenizer) != TOKENIZER_SHA256:
        raise RuntimeError("Frozen tokenizer SHA-256 mismatch.")
    config = load_config(args.config)
    if config.model.context_length != 512:
        raise RuntimeError("Official EXP-012 evaluation requires context length 512.")
    model = DecoderOnlyTransformer(config.model).to(torch.device("cpu"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if parameter_breakdown(model).total != EXPECTED_PARAMS:
        raise RuntimeError("Selected checkpoint parameter-count mismatch.")
    enforce_cpu_isolation(torch, model)
    tokenizer = load_tokenizer(args.tokenizer)
    adapter = CustomCausalLM(model, tokenizer, torch.device("cpu"), batch_size=args.batch_size)
    direct_logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))[0, -1]
    if not torch.isfinite(direct_logits).all():
        raise FloatingPointError("CPU pre-evaluation direct logits are non-finite.")
    direct_score = float(torch.log_softmax(direct_logits, dim=-1)[4])
    adapter_score, _ = adapter._loglikelihood_tokens([(None, [1, 2, 3], [4])])[0]
    if abs(direct_score - adapter_score) > 1e-6:
        raise RuntimeError("CPU evaluation adapter/direct-logit equivalence gate failed.")
    return torch, config, model, adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(EXPECTED_LM_METRICS), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/exp012.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an official result artifact: {args.output}")
    started_at = _timestamp()
    started = time.perf_counter()
    torch, config, model, adapter = _load_cpu_model(args)
    import datasets
    import lm_eval

    raw_result = lm_eval.simple_evaluate(
        model=adapter,
        tasks=[args.task],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=None,
    )
    wall_seconds = time.perf_counter() - started
    ended_at = _timestamp()
    record = {
        "metadata": {
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer": str(args.tokenizer),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "trainable_parameters": EXPECTED_PARAMS,
            "device": "cpu",
            "precision": "fp32",
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "lm_eval_version": lm_eval.__version__,
            "datasets_version": datasets.__version__,
            "torch_version": torch.__version__,
            "num_fewshot": 0,
            "batch_size": args.batch_size,
            "context_length": config.model.context_length,
            "amendment_commit": AMENDMENT_COMMIT,
            "command": sys.argv,
            "platform": platform.platform(),
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_seconds": wall_seconds,
            "no_pretrained_weights": True,
        },
        "raw_lm_eval_result": raw_result,
    }
    validate_lm_task_record(record, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"task": args.task, "status": "completed", "wall_seconds": wall_seconds}, sort_keys=True))


if __name__ == "__main__":
    main()
