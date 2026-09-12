"""Manual local inference entry point; never invoked by packaging or verification."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from gibc_llm.inference_release import verify_release
from gibc_llm.generation import generate
from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.utils import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt")
    p.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    verify_release(args.package)
    model = DecoderOnlyTransformer(load_config(args.package/"config.yaml").model).cpu().float()
    model.load_state_dict(load_file(str(args.package/"model.safetensors"), device="cpu"), strict=True)
    model.eval()
    print(generate(model, Tokenizer.from_file(str(args.package/"tokenizer.json")),
                   args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed))
