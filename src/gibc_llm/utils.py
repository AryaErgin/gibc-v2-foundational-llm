"""Configuration, reproducibility, and small provenance utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    d_ff: int
    activation: str
    gelu_approximate: str
    norm: str
    rmsnorm_eps: float
    norm_placement: str
    positional_encoding: str
    rope_theta: float
    rotary_dim: int
    rope_scaling: str
    attention: str
    causal: bool
    tie_input_output_embeddings: bool
    linear_bias: bool
    dropout: float
    context_length: int
    init_std: float


@dataclass(frozen=True)
class TrainingConfig:
    precision: str
    optimizer: str
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    peak_learning_rate: float
    min_learning_rate: float
    schedule: str
    warmup_steps: int
    full_schedule_steps: int
    gradient_clip_norm: float
    effective_batch_tokens: int
    sequence_predictions: int
    default_microbatch_sequences: int
    default_gradient_accumulation_steps: int
    full_training_tokens: int
    smoke_steps: int
    smoke_training_tokens: int
    smoke_validation_tokens: int
    seed: int


@dataclass(frozen=True)
class DataConfig:
    dataset_repo: str
    dataset_config: str
    dataset_revision: str | None
    text_field: str
    split_seed: int
    validation_bucket_modulus: int
    validation_bucket_cutoff: int
    tokenizer_training_text_bytes: int
    contamination_ngram_size: int
    context_length: int
    eod_token: str
    tokenizer_vocab_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(section: str, cls: type[Any], values: dict[str, Any]) -> Any:
    expected = set(cls.__dataclass_fields__)
    actual = set(values)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(f"Invalid {section} config; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return cls(**values)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load the complete explicit EXP-001 configuration and reject drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"experiment_id", "model", "training", "data"}:
        raise ValueError("Configuration must contain only experiment_id, model, training, and data.")
    config = ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        model=_construct("model", ModelConfig, raw["model"]),
        training=_construct("training", TrainingConfig, raw["training"]),
        data=_construct("data", DataConfig, raw["data"]),
    )
    _validate_controlled_experiment(config)
    return config


def _validate_controlled_experiment(config: ExperimentConfig) -> None:
    model, training, data = config.model, config.training, config.data
    horizons = {"EXP-001": (3052, 100_007_936), "EXP-002": (9156, 300_023_808), "EXP-003": (9156, 300_023_808)}
    if config.experiment_id not in horizons:
        raise ValueError("Only EXP-001, EXP-002, and EXP-003 controlled configurations are supported.")
    if (model.vocab_size, model.d_model, model.n_layers, model.n_heads, model.head_dim, model.d_ff) != (8192, 256, 8, 8, 32, 1024):
        raise ValueError("EXP-001 model dimensions differ from the approved control.")
    if model.d_model != model.n_heads * model.head_dim or model.rotary_dim != model.head_dim:
        raise ValueError("EXP-001 requires full-head RoPE with consistent attention dimensions.")
    if model.rope_theta != 10000.0 or model.rope_scaling != "none":
        raise ValueError("EXP-001 requires unscaled RoPE theta=10000.0.")
    if model.rmsnorm_eps != 1.0e-5 or model.gelu_approximate != "none":
        raise ValueError("EXP-001 requires RMSNorm eps=1e-5 and exact GELU.")
    if not model.causal or not model.tie_input_output_embeddings or model.linear_bias or model.dropout != 0.0:
        raise ValueError("EXP-001 causal/tied/bias/dropout invariants are violated.")
    if training.effective_batch_tokens != 32768 or training.sequence_predictions != 512:
        raise ValueError("EXP-001 effective batch is 64 x 512 prediction tokens.")
    if training.default_microbatch_sequences * training.default_gradient_accumulation_steps * 512 != 32768:
        raise ValueError("Configured microbatch/accumulation does not preserve effective batch tokens.")
    expected_steps, expected_tokens = horizons[config.experiment_id]
    if training.full_schedule_steps != expected_steps or training.full_training_tokens != expected_tokens or training.full_training_tokens != training.full_schedule_steps * training.effective_batch_tokens or training.smoke_steps != 60 or training.smoke_training_tokens != 1_966_080:
        raise ValueError(f"{config.experiment_id} full/smoke token budget invariant is violated.")
    if training.seed != 42 or data.split_seed != 42:
        raise ValueError("EXP-001 requires fixed seed 42.")
    expected_data = (
        ("HuggingFaceFW/fineweb-edu", "default", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
        if config.experiment_id == "EXP-003"
        else ("HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc")
    )
    if (data.dataset_repo, data.dataset_config, data.dataset_revision) != expected_data:
        raise ValueError(f"{config.experiment_id} dataset pin is invalid.")
    if data.tokenizer_vocab_size != 8192 or data.eod_token != "<|endoftext|>":
        raise ValueError("EXP-001 tokenizer invariants are violated.")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, target)


def collect_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }
