"""Run limited, zero-shot lm-eval integration checks against a local checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lm_eval
import torch

from gibc_llm.evaluation import CustomCausalLM
from gibc_llm.evaluation_output import evaluation_output_record
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.tokenizer import load_tokenizer
from gibc_llm.train import CosineWithWarmup, build_optimizer, load_checkpoint
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande,wikitext")
    parser.add_argument("--limit", type=float, default=1)
    parser.add_argument("--full", action="store_true", help="Evaluate every available task example; overrides --limit.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/exp001a/smoke/checkpoint-final.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/exp001a/tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/exp001a/smoke/lm-eval-integration.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    schedule = CosineWithWarmup(optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    load_checkpoint(args.checkpoint, model, optimizer, schedule, device)
    started = time.perf_counter()
    limit = None if args.full else args.limit
    result = lm_eval.simple_evaluate(model=CustomCausalLM(model, load_tokenizer(args.tokenizer), device, args.batch_size), tasks=args.tasks.split(","), num_fewshot=args.num_fewshot, batch_size=args.batch_size, limit=limit)
    wall_seconds = time.perf_counter() - started
    output = evaluation_output_record(
        task=args.tasks,
        checkpoint=str(args.checkpoint),
        batch_size=args.batch_size,
        lm_eval_version=lm_eval.__version__,
        num_fewshot=args.num_fewshot,
        wall_seconds=wall_seconds,
        raw_result=result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"tasks": args.tasks, "limit": limit, **output["metadata"], "result_keys": list(result.keys())}, indent=2))


if __name__ == "__main__":
    main()
