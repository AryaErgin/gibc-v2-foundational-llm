"""Run a controlled full-horizon experiment, or an explicit bounded full-path dry run."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from gibc_llm.full_run import (
    assert_physical_batch_control,
    assert_exp011_phase_capacity,
    dry_run_plan,
    expected_full_sequences,
    full_run_milestones,
    load_full_run_artifact,
    sequences_per_update,
)
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import (
    CosineWithWarmup,
    JsonlLogger,
    RunState,
    WarmupStableDecay,
    build_optimizer,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_smoke,
)
from gibc_llm.utils import atomic_json_write, collect_environment, load_config, set_global_seed


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd().resolve().as_posix()}", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_provenance(config: Any, tokenizer_hash: str, manifest_hash: str, commit: str | None) -> dict[str, Any]:
    payload = config.as_dict()
    payload["provenance"] = {
        "git_commit": commit,
        "tokenizer_sha256": tokenizer_hash,
        "data_manifest_sha256": manifest_hash,
        "full_stream_non_cycled": True,
    }
    return payload


def _verify_resume_provenance(path: Path, config: Any, artifact: Any, tokenizer_hash: str, manifest_hash: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload.get("config", {}).get("provenance", {})
    if provenance.get("tokenizer_sha256") != tokenizer_hash:
        raise RuntimeError("Checkpoint provenance does not match this exact full-run tokenizer/data manifest.")
    if config.experiment_id == "EXP-011":
        checkpoint_config = dict(payload.get("config", {}))
        checkpoint_config.pop("provenance", None)
        if checkpoint_config != config.as_dict():
            raise RuntimeError("EXP-011 checkpoint configuration differs from the exact frozen long-horizon control.")
        previous_manifest_hash = provenance.get("data_manifest_sha256")
        valid_exp006_to_exp011_resume = (
            artifact.manifest.get("experiment_id") == "EXP-011"
            and previous_manifest_hash == artifact.manifest.get("frozen_exp006_source", {}).get("manifest_sha256")
            and payload.get("run_state", {}).get("step") == 27_468
        )
        if previous_manifest_hash != manifest_hash and not valid_exp006_to_exp011_resume:
            raise RuntimeError("EXP-011 checkpoint may resume only the same artifact or the recorded exact EXP-006-to-EXP-011 boundary.")
    elif provenance.get("data_manifest_sha256") != manifest_hash:
        raise RuntimeError("Checkpoint provenance does not match this exact full-run tokenizer/data manifest.")
    if provenance.get("full_stream_non_cycled") is not True:
        raise RuntimeError("Checkpoint does not declare the required non-cycled full token stream.")


def _log_validation(logger: JsonlLogger, label: str, result: Any, state: RunState, common: dict[str, Any]) -> None:
    logger.log(
        {
            **common,
            "event": "validation",
            "label": label,
            "step": state.step,
            "tokens": state.tokens,
            "next_sequence_index": state.next_sequence_index,
            "validation_loss": result.loss,
            "validation_ppl": result.perplexity,
            "validation_tokens": result.token_count,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp001.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Explicit checkpoint from this exact artifact; never auto-resumed.")
    parser.add_argument("--max-steps", type=int, help="Explicit bounded override. Logs DRY RUN / INCOMPLETE TRAINING.")
    parser.add_argument("--microbatch-sequences", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--recorded-source-commit", help="Pre-training source/spec commit recorded by an external clean checkout.")
    args = parser.parse_args()
    config = load_config(args.config)
    fixed_milestone_experiment = config.experiment_id in {"EXP-003", "EXP-004", "EXP-005A", "EXP-005B", "EXP-006", "EXP-007A", "EXP-007B", "EXP-008A", "EXP-009A", "EXP-009B", "EXP-010A", "EXP-011", "EXP-012", "EXP-013-C", "EXP-013-W"}
    if args.checkpoint_interval is None:
        args.checkpoint_interval = full_run_milestones(config)[1] if fixed_milestone_experiment else 500
    if args.validation_interval is None:
        args.validation_interval = full_run_milestones(config)[1] if fixed_milestone_experiment else 500
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("EXP-001 full runner requires a BF16-capable CUDA device.")
    assert_physical_batch_control(config, args.microbatch_sequences, args.gradient_accumulation_steps)
    if args.checkpoint_interval <= 0 or args.validation_interval <= 0:
        raise ValueError("Checkpoint and validation intervals must be positive.")
    artifact = load_full_run_artifact(args.artifact_dir, config)
    device = torch.device("cuda")
    commit = args.recorded_source_commit or _git_commit()
    common = {
        "git_commit": commit,
        "tokenizer_sha256": artifact.manifest["tokenizer"]["sha256"],
        "data_manifest_sha256": artifact.manifest_sha256,
        "dry_run": args.max_steps is not None,
    }
    set_global_seed(config.training.seed)
    model = DecoderOnlyTransformer(config.model).to(device)
    parameters = parameter_breakdown(model).total
    expected_parameters = {"EXP-005A": 20_984_064, "EXP-005B": 20_848_512, "EXP-006": 20_848_512, "EXP-007A": 49_353_184, "EXP-007B": 49_491_840, "EXP-008A": 49_860_480, "EXP-009A": 49_860_480, "EXP-009B": 49_860_480, "EXP-010A": 49_985_504, "EXP-011": 49_860_480, "EXP-012": 49_860_480, "EXP-013-C": 49_860_480, "EXP-013-W": 49_860_480}.get(config.experiment_id, 8_392_960)
    if parameters != expected_parameters:
        raise RuntimeError(f"{config.experiment_id} model parameter invariant failed: {parameters} != {expected_parameters:,}.")
    optimizer = build_optimizer(
        model,
        config.training.peak_learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
        config.training.eps,
    )
    if config.training.schedule == "cosine_decay":
        schedule = CosineWithWarmup(
            optimizer,
            config.training.peak_learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
            config.training.full_schedule_steps,
        )
    elif config.training.schedule == "warmup_stable_decay":
        schedule = WarmupStableDecay(
            optimizer,
            config.training.peak_learning_rate,
            config.training.min_learning_rate,
            config.training.warmup_steps,
            config.training.full_schedule_steps,
            int(config.training.cooldown_steps),
        )
    else:
        raise RuntimeError(f"Unsupported controlled schedule: {config.training.schedule}")
    state = RunState()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(args.run_dir / "metrics.jsonl")
    if args.resume is not None:
        _verify_resume_provenance(args.resume, config, artifact, common["tokenizer_sha256"], artifact.manifest_sha256)
        state = load_checkpoint(args.resume, model, optimizer, schedule, device)
    elif (args.run_dir / "metrics.jsonl").exists():
        raise RuntimeError("Run directory already has metrics. Use --resume with an explicit matching checkpoint or choose a new run directory.")
    requested_steps, incomplete = dry_run_plan(config, state.step, args.max_steps)
    planned_end = state.step + requested_steps
    assert_exp011_phase_capacity(config, artifact.manifest.get("experiment_id"), artifact.train.token_count - 1, planned_end)
    if state.tokens != state.step * config.training.effective_batch_tokens or state.next_sequence_index != state.step * sequences_per_update(config):
        raise RuntimeError("Checkpoint RunState counters do not establish the exact sequential EXP-001 cursor.")
    provenance = _checkpoint_provenance(config, common["tokenizer_sha256"], artifact.manifest_sha256, commit)
    logger.log(
        {
            **common,
            "event": "run_start",
            "status": "DRY RUN / INCOMPLETE TRAINING" if incomplete else "FULL HORIZON REQUESTED",
            "start_step": state.step,
            "requested_steps": requested_steps,
            "schedule_horizon_steps": config.training.full_schedule_steps,
            "effective_batch_tokens": config.training.effective_batch_tokens,
            "microbatch_sequences": args.microbatch_sequences,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    initial_validation = evaluate(model, artifact.validation_inputs, artifact.validation_targets, args.microbatch_sequences, device)
    _log_validation(logger, "before_training", initial_validation, state, common)
    initial_edu_validation = None
    if artifact.edu_validation_inputs is not None and artifact.edu_validation_targets is not None:
        initial_edu_validation = evaluate(
            model, artifact.edu_validation_inputs, artifact.edu_validation_targets, args.microbatch_sequences, device
        )
        _log_validation(logger, "edu_before_training", initial_edu_validation, state, common)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    records: list[dict[str, float]] = []
    validation_records: list[dict[str, float]] = [{"step": float(state.step), "loss": initial_validation.loss, "ppl": initial_validation.perplexity}]
    edu_validation_records: list[dict[str, float]] = []
    if initial_edu_validation is not None:
        edu_validation_records.append({"step": float(state.step), "loss": initial_edu_validation.loss, "ppl": initial_edu_validation.perplexity})
    checkpoint_paths: list[str] = []
    while state.step < planned_end:
        next_validation = ((state.step // args.validation_interval) + 1) * args.validation_interval
        next_checkpoint = ((state.step // args.checkpoint_interval) + 1) * args.checkpoint_interval
        if config.experiment_id == "EXP-013-W" and state.step < 8_240:
            next_checkpoint = min(next_checkpoint, 8_240)
        boundary = min(planned_end, next_validation, next_checkpoint)
        chunk = train_smoke(
            model,
            artifact.train,
            None,
            artifact.validation_inputs,
            artifact.validation_targets,
            optimizer,
            schedule,
            state,
            device,
            args.microbatch_sequences,
            args.gradient_accumulation_steps,
            boundary - state.step,
            config.training.gradient_clip_norm,
        )
        for record in chunk:
            record_step = int(record["step"])
            enriched = {
                **common,
                "event": "train",
                "tokens": int(record["cumulative_tokens"]),
                "next_sequence_index": record_step * sequences_per_update(config),
                **record,
            }
            logger.log(enriched)
        records.extend(chunk)
        if state.step % args.validation_interval == 0 or state.step == planned_end:
            result = evaluate(model, artifact.validation_inputs, artifact.validation_targets, args.microbatch_sequences, device)
            _log_validation(logger, "milestone" if state.step % args.validation_interval == 0 else "end", result, state, common)
            validation_records.append({"step": float(state.step), "loss": result.loss, "ppl": result.perplexity})
            if artifact.edu_validation_inputs is not None and artifact.edu_validation_targets is not None:
                edu_result = evaluate(
                    model, artifact.edu_validation_inputs, artifact.edu_validation_targets, args.microbatch_sequences, device
                )
                _log_validation(logger, "edu_milestone" if state.step % args.validation_interval == 0 else "edu_end", edu_result, state, common)
                edu_validation_records.append({"step": float(state.step), "loss": edu_result.loss, "ppl": edu_result.perplexity})
        if state.step % args.checkpoint_interval == 0 or state.step == 8_240 or state.step == planned_end:
            checkpoint = args.run_dir / "checkpoints" / f"checkpoint-step-{state.step:04d}.pt"
            save_checkpoint(checkpoint, model, optimizer, schedule, state, provenance)
            checkpoint_paths.append(str(checkpoint))
    elapsed = time.perf_counter() - started
    final_validation = validation_records[-1]
    summary = {
        **common,
        "status": "DRY RUN / INCOMPLETE TRAINING" if incomplete else "FULL HORIZON COMPLETE",
        "environment": collect_environment(),
        "parameter_count": parameters,
        "default_full_steps": config.training.full_schedule_steps,
        "requested_steps": requested_steps,
        "final_step": state.step,
        "prediction_tokens": state.tokens,
        "next_sequence_index": state.next_sequence_index,
        "expected_full_sequences": expected_full_sequences(config),
        "effective_batch_tokens": config.training.effective_batch_tokens,
        "microbatch_sequences": args.microbatch_sequences,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "checkpoint_interval": args.checkpoint_interval,
        "validation_interval": args.validation_interval,
        "initial_validation_loss": initial_validation.loss,
        "initial_validation_ppl": initial_validation.perplexity,
        "final_validation_loss": final_validation["loss"],
        "final_validation_ppl": final_validation["ppl"],
        "first_train_loss": float(records[0]["loss"]) if records else None,
        "final_train_loss": float(records[-1]["loss"]) if records else None,
        "mean_tokens_per_second": sum(float(item["tokens_per_second"]) for item in records) / len(records) if records else 0.0,
        "final_tokens_per_second": float(records[-1]["tokens_per_second"]) if records else 0.0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "wall_seconds": elapsed,
        "checkpoints": checkpoint_paths,
        "validation_records": validation_records,
        "edu_validation_records": edu_validation_records,
    }
    atomic_json_write(args.run_dir / "summary.json", summary)
    logger.log({**common, "event": "run_end", **summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
