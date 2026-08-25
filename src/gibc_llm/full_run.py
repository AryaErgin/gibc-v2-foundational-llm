"""Validation and fixed arithmetic for the authorized EXP-001 full-run entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .data import TokenStreamDataset, tensor_sha256
from .utils import ExperimentConfig, sha256_file, sha256_file_prefix


EXP005_IDS = {"EXP-005A", "EXP-005B"}
EXP007_IDS = {"EXP-007A", "EXP-007B"}
EXP008_IDS = {"EXP-008A"}
EXP006_ID = "EXP-006"
EXP004_FROZEN_STREAM_SHA256 = "8e727fa2a2614751a1c34d7f9ac411dfebeb379a09f584ba4f7f418d1059cea1"
EXP004_PREFIX_BYTE_COUNT = 600_047_618


@dataclass(frozen=True)
class FullRunArtifact:
    train: TokenStreamDataset
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    edu_validation_inputs: torch.Tensor | None
    edu_validation_targets: torch.Tensor | None
    manifest: dict[str, Any]
    manifest_sha256: str


def sequences_per_update(config: ExperimentConfig) -> int:
    return config.training.effective_batch_tokens // config.training.sequence_predictions


def expected_full_sequences(config: ExperimentConfig) -> int:
    return config.training.full_training_tokens // config.training.sequence_predictions


def full_run_milestones(config: ExperimentConfig) -> tuple[int, int, int, int]:
    """Return the predeclared equal-token internal validation curve for a controlled full run."""
    if config.experiment_id == EXP006_ID:
        if config.training.full_schedule_steps != 27_468:
            raise RuntimeError("The EXP-006 milestone plan must remain 0/9156/18312/27468.")
        return (0, 9_156, 18_312, 27_468)
    if config.experiment_id not in {"EXP-003", "EXP-004", *EXP005_IDS, *EXP007_IDS, *EXP008_IDS}:
        raise ValueError("Only controlled dual-validation experiments declare these milestones.")
    interval = config.training.full_schedule_steps // 3
    if config.training.full_schedule_steps != 9_156 or interval != 3_052:
        raise RuntimeError("The controlled 300M milestone plan must remain 0/3052/6104/9156.")
    return (0, interval, 2 * interval, config.training.full_schedule_steps)


def assert_physical_batch_control(config: ExperimentConfig, microbatch_sequences: int, accumulation_steps: int) -> None:
    """Preserve the explicit physical batch where an experiment fixes it."""
    if microbatch_sequences * accumulation_steps != sequences_per_update(config):
        raise RuntimeError("Full runner must retain exactly 64 sequences / 32,768 prediction tokens per update.")
    if config.experiment_id in {"EXP-003", "EXP-004", *EXP005_IDS, *EXP007_IDS, *EXP008_IDS, EXP006_ID} and (
        microbatch_sequences != config.training.default_microbatch_sequences
        or accumulation_steps != config.training.default_gradient_accumulation_steps
    ):
        raise RuntimeError(f"{config.experiment_id} full runner requires the fixed physical batch: 32 sequences x 2 accumulation steps.")


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
    expected_manifest_id = "EXP-004" if config.experiment_id in {*EXP005_IDS, *EXP007_IDS, *EXP008_IDS} else config.experiment_id
    if manifest.get("experiment_id") != expected_manifest_id:
        raise RuntimeError("Full-run manifest experiment identity differs from the supplied configuration.")
    dual_validation_experiment = config.experiment_id in {"EXP-003", "EXP-004", *EXP005_IDS, *EXP007_IDS, *EXP008_IDS, EXP006_ID}
    if dual_validation_experiment and manifest.get("preparation_mode") != "full_stream":
        raise RuntimeError(f"{config.experiment_id} full runner requires a complete stream materialization, not validation-only preparation.")
    dataset = manifest.get("dataset", {})
    expected_dataset = {
        "repo": config.data.dataset_repo,
        "config": config.data.dataset_config,
        "revision": config.data.dataset_revision,
        "field": config.data.text_field,
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise RuntimeError("Full-run manifest dataset provenance differs from the pinned experiment configuration.")
    tokenizer = manifest.get("tokenizer", {})
    if (
        tokenizer.get("vocab_size") != config.data.tokenizer_vocab_size
        or tokenizer.get("special_tokens") != [config.data.eod_token]
        or not tokenizer.get("sha256")
    ):
        raise RuntimeError("Full-run manifest has no exact frozen 8192-entry tokenizer hash.")
    if dual_validation_experiment and tokenizer.get("sha256") != "c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14":
        raise RuntimeError(f"{config.experiment_id} full runner requires the exact frozen tokenizer hash.")
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
    if config.experiment_id in {*EXP005_IDS, *EXP007_IDS, *EXP008_IDS} and packed.get("train_stream_sha256") != EXP004_FROZEN_STREAM_SHA256:
        raise RuntimeError("EXP-005/EXP-007/EXP-008 requires the exact frozen EXP-004 stream SHA-256; rematerialized streams are forbidden.")
    if packed.get("train_stream_bytes") != stream_path.stat().st_size or packed.get("train_stream_sha256") != sha256_file(stream_path):
        raise RuntimeError("Full-run uint16 token stream does not match manifest provenance.")
    if config.experiment_id == EXP006_ID:
        prefix = manifest.get("exp004_prefix", {})
        if (
            prefix.get("byte_count") != EXP004_PREFIX_BYTE_COUNT
            or prefix.get("expected_sha256") != EXP004_FROZEN_STREAM_SHA256
            or prefix.get("observed_sha256") != EXP004_FROZEN_STREAM_SHA256
            or prefix.get("prefix_match") is not True
        ):
            raise RuntimeError("EXP-006 manifest lacks a verified exact EXP-004 byte prefix.")
        if sha256_file_prefix(stream_path, EXP004_PREFIX_BYTE_COUNT) != EXP004_FROZEN_STREAM_SHA256:
            raise RuntimeError("EXP-006 stream fails independent EXP-004 prefix verification.")
    if config.experiment_id == "EXP-002" and manifest.get("exp001_prefix", {}).get("prefix_match") is not True:
        raise RuntimeError("EXP-002 artifact lacks a verified byte-identical EXP-001 training prefix.")
    validation_key = "general_validation" if dual_validation_experiment else "validation"
    validation = manifest.get(validation_key, {})
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
    edu_inputs = edu_targets = None
    if dual_validation_experiment:
        if (
            validation.get("inputs_sha256") != "f721fda2a0a0ca11a580178dba6c2592af4dd9324da3ed7120624b7364d653f7"
            or validation.get("targets_sha256") != "2ca518affa0b36d15c7c427de84b4c7d761926e1e993c03724d75f396934f13e"
        ):
            raise RuntimeError(f"{config.experiment_id} general validation must be the frozen EXP-001/002 artifact.")
        edu_validation = manifest.get("edu_validation", {})
        edu_path = artifact_dir / edu_validation.get("file", "")
        if not edu_path.is_file() or edu_validation.get("contamination_screened") is not True:
            raise RuntimeError(f"{config.experiment_id} artifact lacks a contamination-screened educational validation set.")
        edu_values = torch.load(edu_path, map_location="cpu", weights_only=True)
        edu_inputs, edu_targets = edu_values["inputs"], edu_values["targets"]
        if (
            edu_inputs.shape != edu_targets.shape
            or edu_inputs.ndim != 2
            or edu_inputs.shape[1] != config.data.context_length
            or edu_validation.get("prediction_tokens") != int(edu_targets.numel())
            or edu_validation.get("prediction_tokens") != config.training.smoke_validation_tokens
            or edu_validation.get("inputs_sha256") != tensor_sha256(edu_inputs)
            or edu_validation.get("targets_sha256") != tensor_sha256(edu_targets)
        ):
            raise RuntimeError(f"{config.experiment_id} educational validation material does not match its held-out manifest invariants.")
        expected_edu_hashes = (
            ("cc75580b854b69846b1ff15385fbb87adf5bdf1701c1bfe9e8e8d5fdb651fb1a", "300608bc74e052f1580d78e3ad5e1174312360a766f3278c6ce2bdf3336a48b4")
            if config.experiment_id in {"EXP-004", *EXP005_IDS, *EXP007_IDS, *EXP008_IDS, EXP006_ID}
            else (edu_validation["inputs_sha256"], edu_validation["targets_sha256"])
        )
        if (edu_validation["inputs_sha256"], edu_validation["targets_sha256"]) != expected_edu_hashes:
            raise RuntimeError(f"{config.experiment_id} educational validation is not the frozen approved tensor.")
    if config.experiment_id in {"EXP-004", *EXP005_IDS, *EXP007_IDS, *EXP008_IDS, EXP006_ID}:
        mixture = manifest.get("mixture", {})
        expected_mixture = config.mixture or {}
        if (
            mixture.get("target_prediction_tokens") != expected_mixture.get("target_prediction_tokens")
            or mixture.get("global_deduplication") != "canonical_content_sha256"
            or mixture.get("sources") != expected_mixture.get("sources")
            or sum(mixture.get("actual_prediction_token_contributions", {}).values()) != expected_tokens
            or mixture.get("unique_document_count", 0) <= 0
        ):
            raise RuntimeError(f"{config.experiment_id} mixture provenance, deduplication, or source-token accounting is invalid.")
    return FullRunArtifact(
        train=TokenStreamDataset(stream_path, expected_stored, config.data.context_length),
        validation_inputs=inputs,
        validation_targets=targets,
        edu_validation_inputs=edu_inputs,
        edu_validation_targets=edu_targets,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
    )
