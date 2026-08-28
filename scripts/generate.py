"""Generate text from an explicitly selected local decoder-only checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gibc_llm.generation import generate
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.tokenizer import load_tokenizer
from gibc_llm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = DecoderOnlyTransformer(config.model).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Inference checkpoint must contain a strict 'model' state dictionary.") from exc
    model.eval()
    print(generate(model, load_tokenizer(args.tokenizer), args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
