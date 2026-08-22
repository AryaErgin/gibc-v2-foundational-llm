"""Run limited, zero-shot lm-eval integration checks against a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval
import torch

from gibc_llm.evaluation import CustomCausalLM
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.tokenizer import load_tokenizer
from gibc_llm.train import CosineWithWarmup, build_optimizer, load_checkpoint
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande,wikitext")
    parser.add_argument("--limit", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/exp001a/smoke/checkpoint-final.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/exp001a/tokenizer/tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/exp001a/smoke/lm-eval-integration.json"))
    args = parser.parse_args()
    config = load_config(Path("configs/exp001.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    schedule = CosineWithWarmup(optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    load_checkpoint(args.checkpoint, model, optimizer, schedule, device)
    result = lm_eval.simple_evaluate(model=CustomCausalLM(model, load_tokenizer(args.tokenizer), device, args.batch_size), tasks=args.tasks.split(","), num_fewshot=0, batch_size=args.batch_size, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"integration_only": True, "tasks": args.tasks, "limit": args.limit, "lm_eval_version": lm_eval.__version__, "result_keys": list(result.keys())}, indent=2))


if __name__ == "__main__":
    main()
