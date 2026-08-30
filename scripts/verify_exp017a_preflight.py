"""Non-training integrity preflight for the frozen EXP-017A WSD branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gibc_llm.full_run import expected_run_state, full_run_milestones, load_full_run_artifact
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import WarmupStableDecay, build_optimizer
from gibc_llm.utils import load_config


EXPECTED_REFERENCE = {
    "terminal_general_nll": 3.190959542989731,
    "terminal_edu_nll": 2.8499483168125153,
    "terminal_combined_nll": 3.020453929901123,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp017a-wsd.yaml"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("provenance/exp017a-exp012-reference.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    if config.experiment_id != "EXP-017A":
        raise RuntimeError("EXP-017A preflight requires configs/exp017a-wsd.yaml.")
    if (config.training.full_schedule_steps, config.training.full_training_tokens) != (73_242, 2_399_993_856):
        raise RuntimeError("EXP-017A horizon is not the frozen 2.4B-token schedule.")
    if (config.training.schedule, config.training.warmup_steps, config.training.cooldown_steps) != ("warmup_stable_decay", 100, 7_324):
        raise RuntimeError("EXP-017A WSD semantics differ from the preregistered schedule.")
    if config.training.seed != 42:
        raise RuntimeError("EXP-017A must use fresh seed 42.")

    model = DecoderOnlyTransformer(config.model)
    params = parameter_breakdown(model).total
    if params != 49_860_480:
        raise RuntimeError(f"EXP-017A parameter invariant failed: {params:,}.")
    optimizer = build_optimizer(model, config.training.peak_learning_rate, config.training.weight_decay, (config.training.beta1, config.training.beta2), config.training.eps)
    schedule = WarmupStableDecay(
        optimizer,
        config.training.peak_learning_rate,
        config.training.min_learning_rate,
        config.training.warmup_steps,
        config.training.full_schedule_steps,
        config.training.cooldown_steps,
    )
    if schedule.stable_end_step != 65_918:
        raise RuntimeError("EXP-017A stable boundary differs from scheduler-derived 65,918.")
    if schedule.lr_at_step(65_918) != 6.0e-4 or not (6.0e-5 < schedule.lr_at_step(65_919) < 6.0e-4) or schedule.lr_at_step(73_242) != 6.0e-5:
        raise RuntimeError("EXP-017A WSD learning-rate boundary is invalid.")
    if expected_run_state(config, 0, 73_242) != (73_242, 2_399_993_856, 4_687_488):
        raise RuntimeError("EXP-017A terminal step/token/cursor accounting is invalid.")
    if full_run_milestones(config) != (0, 9_156, 18_312, 27_468, 36_624, 45_780, 54_936, 64_092, 73_242):
        raise RuntimeError("EXP-017A full-horizon milestone accounting changed unexpectedly.")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_values = reference.get("reference", {})
    if any(reference_values.get(key) != value for key, value in {"experiment_id": "EXP-012", "schedule": "cosine_decay", **EXPECTED_REFERENCE}.items()):
        raise RuntimeError("EXP-017A reference is not the frozen exact EXP-012 terminal result.")

    artifact = load_full_run_artifact(args.artifact_dir, config)
    packed = artifact.manifest["packed"]
    print(json.dumps({
        "training_launch": False,
        "official_benchmark_invocation": False,
        "parameters": params,
        "updates": config.training.full_schedule_steps,
        "prediction_tokens": config.training.full_training_tokens,
        "terminal_cursor": 4_687_488,
        "stable_end_step": schedule.stable_end_step,
        "cooldown_start_step": schedule.stable_end_step + 1,
        "cooldown_updates": schedule.cooldown_steps,
        "stable_cursor": schedule.stable_end_step * 64,
        "stable_prediction_tokens": schedule.stable_end_step * config.training.effective_batch_tokens,
        "final_learning_rate": schedule.lr_at_step(config.training.full_schedule_steps),
        "stream_sha256": packed["train_stream_sha256"],
        "manifest_sha256": artifact.manifest_sha256,
        "tokenizer_sha256": artifact.manifest["tokenizer"]["sha256"],
        "general_validation_sha256": [artifact.manifest["general_validation"]["inputs_sha256"], artifact.manifest["general_validation"]["targets_sha256"]],
        "edu_validation_sha256": [artifact.manifest["edu_validation"]["inputs_sha256"], artifact.manifest["edu_validation"]["targets_sha256"]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
