"""Evaluate the frozen EXP-002 final checkpoint on EXP-003 edu_validation only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gibc_llm.exp003 import FROZEN_TOKENIZER_SHA256
from gibc_llm.full_run import load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import evaluate
from gibc_llm.utils import atomic_json_write, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/exp002-full/run/checkpoints/checkpoint-step-9156.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("EXP-003 baseline evaluation requires a BF16-capable CUDA device.")
    config = load_config(Path("configs/exp003.yaml"))
    artifact = load_full_run_artifact(args.artifact_dir, config)
    if artifact.manifest["tokenizer"]["sha256"] != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("EXP-003 artifact does not use the frozen EXP-001 tokenizer.")
    if artifact.edu_validation_inputs is None or artifact.edu_validation_targets is None:
        raise RuntimeError("EXP-003 artifact has no educational validation tensor.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("run_state", {})
    if state.get("step") != 9156 or state.get("tokens") != 300_023_808:
        raise RuntimeError("Checkpoint is not the final EXP-002 9,156-step / 300,023,808-token model.")
    device = torch.device("cuda")
    model = DecoderOnlyTransformer(config.model).to(device)
    if parameter_breakdown(model).total != 8_392_960:
        raise RuntimeError("Fixed EXP model parameter count invariant failed.")
    model.load_state_dict(payload["model"])
    started = time.perf_counter()
    result = evaluate(model, artifact.edu_validation_inputs, artifact.edu_validation_targets, args.batch_size, device)
    wall_seconds = time.perf_counter() - started
    output = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": 9156,
        "checkpoint_prediction_tokens": 300_023_808,
        "tokenizer_sha256": artifact.manifest["tokenizer"]["sha256"],
        "edu_validation_inputs_sha256": artifact.manifest["edu_validation"]["inputs_sha256"],
        "edu_validation_targets_sha256": artifact.manifest["edu_validation"]["targets_sha256"],
        "loss": result.loss,
        "perplexity": result.perplexity,
        "prediction_tokens": result.token_count,
        "batch_size": args.batch_size,
        "wall_seconds": wall_seconds,
    }
    output_path = args.output or args.artifact_dir / "exp002-on-edu-validation.json"
    atomic_json_write(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
