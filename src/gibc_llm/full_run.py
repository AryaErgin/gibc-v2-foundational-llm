"""Validation and fixed arithmetic for the authorized EXP-001 full-run entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import TokenStreamDataset, tensor_sha256
from .utils import ExperimentConfig, sha256_file


@dataclass(frozen=True)
class FullRunArtifact:
    train: TokenStreamDataset
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    manifest: dict[str, Any]
    manifest_sha256: str


def sequences_per_update(config: ExperimentConfig) -> int:
    return config.training.effective_batch_tokens // config.training.sequence_predictions


def expected_full_sequences(config: ExperimentConfig) -> int:
    return config.training.full_training_tokens // config.training.sequence_predictions


def dry_run_plan(config: ExperimentConfig, start_step: int, max_steps: int | None) -> tuple[int, bool]:
    """Return requested updates and whether this invocation is explicitly incomplete."""
    remaining = config.training.full_schedule_steps - start_step
    if remaining < 0:
        raise ValueError("Checkpoint step exceeds the fixed EXP-001 schedule horizon.")
    if max_steps is None:
        return remaining, False
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive when supplied.")
    return min(max_steps, remaining), start_step + max_steps < config.training.full_schedule_steps


def expected_run_state(config: ExperimentConfig, start_step: int, requested_steps: int) -> tuple[int, int, int]:
    step = start_step + requested_steps
    sequences = sequences_per_update(config)
    return step, step * config.training.effective_batch_tokens, step * sequences


def load_full_run_artifact(artifact_dir: Path, config: ExperimentConfig) -> FullRunArtifact:
    """Load and strictly validate the complete non-cycled full-run artifact."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Full-run artifact is missing manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset", {})
    if dataset != {
        "repo": config.data.dataset_repo,
        "config": config.data.dataset_config,
        "revision": config.data.dataset_revision,
        "field": config.data.text_field,
    }:
        raise RuntimeError("Full-run manifest dataset provenance differs from the pinned EXP-001 configuration.")
    tokenizer = manifest.get("tokenizer", {})
    if (
        tokenizer.get("vocab_size") != config.data.tokenizer_vocab_size
        or tokenizer.get("special_tokens") != [config.data.eod_token]
        or not tokenizer.get("sha256")
    ):
        raise RuntimeError("Full-run manifest has no exact frozen 8192-entry tokenizer hash.")
    tokenizer_path = artifact_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != tokenizer["sha256"]:
        raise RuntimeError("Frozen tokenizer artifact is absent or does not match its manifest SHA-256.")
    packed = manifest.get("packed", {})
    expected_tokens = config.training.full_training_tokens
    expected_stored = expected_tokens + 1
    if (
        packed.get("representation") != "one-dimensional uint16 token stream with on-demand torch.long 513-token views"
        or packed.get("storage_dtype") != "uint16"
        or packed.get("context_length") != config.data.context_length
        or packed.get("prediction_tokens_per_example") != config.training.sequence_predictions
        or packed.get("train_prediction_tokens") != expected_tokens
        or packed.get("train_token_count_including_final_target") != expected_stored
        or packed.get("train_examples") != expected_full_sequences(config)
        or packed.get("non_cycled") is not True
    ):
        raise RuntimeError("Full-run manifest fails EXP-001 stream shape/non-cycling invariants.")
    stream_path = artifact_dir / packed.get("train_stream_file", "train-token-stream.uint16")
    if not stream_path.is_file() or stream_path.stat().st_size != expected_stored * 2:
        raise RuntimeError("Full-run uint16 token stream is missing or has the wrong exact size.")
    if packed.get("train_stream_bytes") != stream_path.stat().st_size or packed.get("train_stream_sha256") != sha256_file(stream_path):
        raise RuntimeError("Full-run uint16 token stream does not match manifest provenance.")
    validation = manifest.get("validation", {})
    validation_path = artifact_dir / validation.get("file", "")
    if not validation_path.is_file():
        raise RuntimeError("Full-run artifact is missing deterministic held-out validation data.")
    values = torch.load(validation_path, map_location="cpu", weights_only=True)
    inputs, targets = values["inputs"], values["targets"]
    if (
        inputs.shape != targets.shape
        or inputs.ndim != 2
        or inputs.shape[1] != config.data.context_length
        or validation.get("prediction_tokens") != int(targets.numel())
        or validation.get("prediction_tokens") != config.training.smoke_validation_tokens
        or validation.get("inputs_sha256") != tensor_sha256(inputs)
        or validation.get("targets_sha256") != tensor_sha256(targets)
    ):
        raise RuntimeError("Full-run validation material does not match required held-out manifest invariants.")
    return FullRunArtifact(
        train=TokenStreamDataset(stream_path, expected_stored, config.data.context_length),
        validation_inputs=inputs,
        validation_targets=targets,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
    )
