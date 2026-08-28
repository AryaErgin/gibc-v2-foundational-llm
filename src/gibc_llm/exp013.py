"""Pure EXP-013 result arithmetic and checkpoint-integrity checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .model import DecoderOnlyTransformer, parameter_breakdown
from .train import WarmupStableDecay, build_optimizer, load_checkpoint
from .utils import ExperimentConfig, sha256_file


EXP013_TOKENS = 300_023_808
EXP013_STEPS = 9_156
EXP013_STABLE_STEP = 8_240
EXP013_PARAMETERS = 49_860_480


def combined_validation_loss(general: float, edu: float) -> float:
    """The frozen General and Edu validation sets have equal token counts."""
    return (general + edu) / 2.0


def classify(control_general: float, control_edu: float, wsd_general: float, wsd_edu: float) -> dict[str, Any]:
    control_combined = combined_validation_loss(control_general, control_edu)
    wsd_combined = combined_validation_loss(wsd_general, wsd_edu)
    general_delta = wsd_general - control_general
    edu_delta = wsd_edu - control_edu
    delta = wsd_combined - control_combined
    individual_regression = general_delta > 0.020 or edu_delta > 0.020
    if delta <= -0.020 and not individual_regression:
        classification = "CAPABILITY WIN"
        decision = "WSD eligible for confirmation; stop after both predeclared arms."
    elif delta <= 0.010:
        classification = "PERFORMANCE TIE"
        decision = "No capability claim; WSD is only an engineering/horizon-flexibility candidate."
    else:
        classification = "REJECT WSD"
        decision = "Reject WSD for this controlled 300M-token ablation."
    if individual_regression and classification == "CAPABILITY WIN":
        raise AssertionError("Capability classification must reject an individual validation regression above 0.020 nat.")
    return {
        "control": {"general_loss": control_general, "edu_loss": control_edu, "combined_loss": control_combined},
        "wsd": {"general_loss": wsd_general, "edu_loss": wsd_edu, "combined_loss": wsd_combined},
        "delta_combined": delta,
        "delta_general": general_delta,
        "delta_edu": edu_delta,
        "individual_regression_over_0_020": individual_regression,
        "classification": classification,
        "decision": decision,
    }


def assert_full_summary(summary: dict[str, Any], experiment_id: str) -> None:
    if summary.get("status") != "FULL HORIZON COMPLETE":
        raise RuntimeError(f"{experiment_id} is not a completed full-horizon run.")
    if summary.get("final_step") != EXP013_STEPS or summary.get("prediction_tokens") != EXP013_TOKENS:
        raise RuntimeError(f"{experiment_id} does not establish the exact EXP-013 step/token horizon.")
    if summary.get("next_sequence_index") != EXP013_TOKENS // 512:
        raise RuntimeError(f"{experiment_id} has an invalid deterministic data cursor.")
    if summary.get("parameter_count") != EXP013_PARAMETERS:
        raise RuntimeError(f"{experiment_id} violates the exact parameter invariant.")
    if len(summary.get("edu_validation_records", [])) == 0:
        raise RuntimeError(f"{experiment_id} is missing the frozen Edu validation curve.")
    if not isinstance(summary.get("final_validation_loss"), float) or not math.isfinite(summary["final_validation_loss"]):
        raise RuntimeError(f"{experiment_id} is missing a finite General validation result.")


def validate_stable_checkpoint(path: Path, config: ExperimentConfig) -> dict[str, Any]:
    """Load the actual WSD stable checkpoint and validate exact resume state."""
    if config.experiment_id != "EXP-013-W" or config.training.cooldown_steps != 916:
        raise ValueError("Stable checkpoint validation is only defined for the fixed EXP-013-W configuration.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {"model", "optimizer", "schedule", "run_state", "data_cursor", "rng", "config"}
    if set(payload) != expected_keys:
        raise RuntimeError("Stable checkpoint is missing required model/optimizer/scheduler/RNG/data-cursor state.")
    if payload["run_state"] != {
        "step": EXP013_STABLE_STEP,
        "tokens": EXP013_STABLE_STEP * 32768,
        "next_sequence_index": EXP013_STABLE_STEP * 64,
    }:
        raise RuntimeError("Stable checkpoint is not exactly the completed pre-cooldown update 8,240 state.")
    if payload["data_cursor"] != {"next_sequence_index": EXP013_STABLE_STEP * 64, "mechanism": "sequential_example_index"}:
        raise RuntimeError("Stable checkpoint lacks its exact sequential data cursor.")
    expected_schedule = {
        "type": "warmup_stable_decay",
        "step_count": EXP013_STABLE_STEP,
        "warmup_steps": 100,
        "total_steps": EXP013_STEPS,
        "cooldown_steps": 916,
    }
    if payload["schedule"] != expected_schedule:
        raise RuntimeError("Stable checkpoint scheduler state does not establish the pre-cooldown boundary.")
    model = DecoderOnlyTransformer(config.model)
    if parameter_breakdown(model).total != EXP013_PARAMETERS:
        raise RuntimeError("Reload model parameter count differs from the EXP-013 invariant.")
    optimizer = build_optimizer(
        model,
        config.training.peak_learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
        config.training.eps,
    )
    schedule = WarmupStableDecay(
        optimizer,
        config.training.peak_learning_rate,
        config.training.min_learning_rate,
        config.training.warmup_steps,
        config.training.full_schedule_steps,
        int(config.training.cooldown_steps),
    )
    state = load_checkpoint(path, model, optimizer, schedule, torch.device("cpu"))
    if state.step != EXP013_STABLE_STEP or schedule.step_count != EXP013_STABLE_STEP:
        raise RuntimeError("Stable checkpoint cannot be reloaded at the exact pre-cooldown state.")
    next_lr = schedule.step()
    if not (config.training.min_learning_rate < next_lr < config.training.peak_learning_rate):
        raise RuntimeError("Reloaded stable checkpoint does not advance into the WSD cooldown.")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "resume_gate": "PASSED",
        "next_update": EXP013_STABLE_STEP + 1,
        "next_learning_rate": next_lr,
    }
