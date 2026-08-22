"""Run the separate full-model fixed-sample learnability check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import tiny_overfit
from gibc_llm.utils import atomic_json_write, load_config, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp001a"))
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda")
    data = torch.load(args.artifact_dir / "train.pt", map_location="cpu", weights_only=True)
    set_global_seed(config.training.seed)
    model = DecoderOnlyTransformer(config.model).to(device)
    inputs = data["inputs"][:1]
    targets = data["targets"][:1]
    losses = tiny_overfit(model, inputs, targets, args.steps, learning_rate=3e-3, device=device)
    report = {"steps": args.steps, "first_loss": losses[0], "final_loss": losses[-1], "trajectory": losses}
    atomic_json_write(args.artifact_dir / "overfit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
