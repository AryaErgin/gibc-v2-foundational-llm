"""Profile and run the bounded 60-update EXP-001A BF16 smoke validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import (
    CosineWithWarmup,
    JsonlLogger,
    RunState,
    build_optimizer,
    evaluate,
    load_checkpoint,
    profile_microbatches,
    save_checkpoint,
    train_smoke,
)
from gibc_llm.utils import atomic_json_write, collect_environment, load_config, set_global_seed


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/exp001a"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/exp001a/smoke"))
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("EXP-001A smoke requires a BF16-capable CUDA device.")
    device = torch.device("cuda")
    train = torch.load(args.artifact_dir / "train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(args.artifact_dir / "validation.pt", map_location="cpu", weights_only=True)
    manifest = json.loads((args.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    set_global_seed(config.training.seed)
    model = DecoderOnlyTransformer(config.model).to(device)
    profile = profile_microbatches(model, train["inputs"], train["targets"], device)
    stable = [record for record in profile if record["status"] == "ok"]
    if not stable:
        raise RuntimeError("No approved microbatch candidate completed the bounded CUDA profile.")
    selected = max(stable, key=lambda record: float(record["tokens_per_second"]))
    microbatch = int(selected["microbatch_sequences"])
    accumulation = int(selected["gradient_accumulation_steps"])
    set_global_seed(config.training.seed)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = build_optimizer(
        model,
        config.training.peak_learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
        config.training.eps,
    )
    schedule = CosineWithWarmup(
        optimizer,
        config.training.peak_learning_rate,
        config.training.min_learning_rate,
        config.training.warmup_steps,
        config.training.full_schedule_steps,
    )
    initial_validation = evaluate(model, validation["inputs"], validation["targets"], microbatch, device)
    initial_train = evaluate(model, train["inputs"][:64], train["targets"][:64], microbatch, device)
    torch.cuda.reset_peak_memory_stats(device)
    logger = JsonlLogger(args.run_dir / "metrics.jsonl")
    state = RunState()
    records = train_smoke(
        model,
        train["inputs"],
        train["targets"],
        validation["inputs"],
        validation["targets"],
        optimizer,
        schedule,
        state,
        device,
        microbatch,
        accumulation,
        config.training.smoke_steps,
        config.training.gradient_clip_norm,
        logger,
    )
    final_validation = evaluate(model, validation["inputs"], validation["targets"], microbatch, device)
    checkpoint = args.run_dir / "checkpoint-final.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, state, config.as_dict())
    sample_inputs = validation["inputs"][:1].to(device)
    sample_before = model(sample_inputs).detach().cpu()
    restored_model = DecoderOnlyTransformer(config.model).to(device)
    restored_optimizer = build_optimizer(restored_model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    restored_schedule = CosineWithWarmup(restored_optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    restored_state = load_checkpoint(checkpoint, restored_model, restored_optimizer, restored_schedule, device)
    checkpoint_exact = bool(torch.equal(sample_before, restored_model(sample_inputs).detach().cpu()) and restored_state == state)
    rolling = [float(item["tokens_per_second"]) for item in records[-10:]]
    summary = {
        "git_commit": _git_commit(),
        "environment": collect_environment(),
        "precision": "FP32 parameters and AdamW state; CUDA BF16 autocast forward/backward; no GradScaler",
        "parameter_count": parameter_breakdown(model).total,
        "data_manifest_sha256": manifest.get("manifest_sha256"),
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "profile_candidates": profile,
        "selected_microbatch_sequences": microbatch,
        "selected_gradient_accumulation_steps": accumulation,
        "effective_batch_tokens": microbatch * accumulation * 512,
        "initial_train_loss": initial_train.loss,
        "initial_validation_loss": initial_validation.loss,
        "initial_validation_perplexity": initial_validation.perplexity,
        "final_train_loss": float(records[-1]["loss"]),
        "final_validation_loss": final_validation.loss,
        "final_validation_perplexity": final_validation.perplexity,
        "smoke_steps": state.step,
        "smoke_prediction_tokens": state.tokens,
        "mean_tokens_per_second": sum(float(item["tokens_per_second"]) for item in records) / len(records),
        "rolling_tokens_per_second": sum(rolling) / len(rolling),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_round_trip_exact": checkpoint_exact,
        "estimated_training_flops": 6 * parameter_breakdown(model).total * state.tokens,
        "flops_label": "estimate: 6 x trainable_parameters x training_tokens",
    }
    atomic_json_write(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
