"""Recover validation/resume evidence from a completed EXP-001A smoke checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, evaluate, load_checkpoint, optimizer_update
from gibc_llm.utils import atomic_json_write, load_config, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp001a"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/exp001a/smoke"))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda")
    train = torch.load(args.artifact_dir / "train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(args.artifact_dir / "validation.pt", map_location="cpu", weights_only=True)
    set_global_seed(config.training.seed)
    initial_model = DecoderOnlyTransformer(config.model).to(device)
    initial_validation = evaluate(initial_model, validation["inputs"], validation["targets"], 16, device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    schedule = CosineWithWarmup(optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    probe = validation["inputs"][:1].to(device)
    state = load_checkpoint(args.run_dir / "checkpoint-final.pt", model, optimizer, schedule, device)
    output_once = model(probe).detach().cpu()
    verifier = DecoderOnlyTransformer(config.model).to(device)
    verifier_optimizer = build_optimizer(verifier, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    verifier_schedule = CosineWithWarmup(verifier_optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    load_checkpoint(args.run_dir / "checkpoint-final.pt", verifier, verifier_optimizer, verifier_schedule, device)
    checkpoint_output_equal = bool(torch.equal(output_once, verifier(probe).detach().cpu()))
    final_validation = evaluate(model, validation["inputs"], validation["targets"], 16, device)
    resumed = optimizer_update(model, optimizer, schedule, [(train["inputs"][:8], train["targets"][:8])], device, config.training.gradient_clip_norm)
    report = {
        "initial_validation_loss_reconstructed": initial_validation.loss,
        "initial_validation_perplexity_reconstructed": initial_validation.perplexity,
        "final_validation_loss": final_validation.loss,
        "final_validation_perplexity": final_validation.perplexity,
        "checkpoint_step": state.step,
        "checkpoint_tokens": state.tokens,
        "checkpoint_output_finite": bool(torch.isfinite(output_once).all()),
        "checkpoint_output_round_trip_exact": checkpoint_output_equal,
        "resume_update_tokens": resumed["tokens"],
        "resume_schedule_step": schedule.step_count,
    }
    atomic_json_write(args.run_dir / "checkpoint-analysis.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
