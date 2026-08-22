"""Generate text from a local EXP-001 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gibc_llm.generation import generate
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.tokenizer import load_tokenizer
from gibc_llm.train import CosineWithWarmup, build_optimizer, load_checkpoint
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/exp001a/smoke/checkpoint-final.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/exp001a/tokenizer/tokenizer.json"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(Path("configs/exp001.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    schedule = CosineWithWarmup(optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    load_checkpoint(args.checkpoint, model, optimizer, schedule, device)
    print(generate(model, load_tokenizer(args.tokenizer), args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
