"""Configuration, reproducibility, and small provenance utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from dataclasses import MISSING, asdict, dataclass
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
    cooldown_steps: int | None = None


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
class LLRConfig:
    method: str
    min_multiplier: float
    max_multiplier: float
    recompute_interval_updates: int
    hill_k_fraction: float
    soft_switch_steps: int
    active_recompute_last_step: int
    multiplier_freeze_step: int


@dataclass(frozen=True)
class MagmaConfig:
    method: str
    survival_probability: float
    tau: float
    smoothing: float
    rng_seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    mixture: dict[str, Any] | None = None
    llr: LLRConfig | None = None
    magma: MagmaConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["llr"] is None:
            payload.pop("llr")
        if payload["magma"] is None:
            payload.pop("magma")
        return payload


def _construct(section: str, cls: type[Any], values: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__
    expected = set(fields)
    actual = set(values)
    missing = {
        name
        for name, field in fields.items()
        if name not in actual and field.default is MISSING and field.default_factory is MISSING
    }
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(f"Invalid {section} config; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return cls(**values)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load the complete explicit EXP-001 configuration and reject drift."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not {"experiment_id", "model", "training", "data"}.issubset(raw) or set(raw) - {"experiment_id", "model", "training", "data", "mixture", "llr", "magma"}:
        raise ValueError("Configuration must contain experiment_id, model, training, data, and optional mixture/llr/magma.")
    config = ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        model=_construct("model", ModelConfig, raw["model"]),
        training=_construct("training", TrainingConfig, raw["training"]),
        data=_construct("data", DataConfig, raw["data"]),
        mixture=raw.get("mixture"),
        llr=_construct("llr", LLRConfig, raw["llr"]) if raw.get("llr") is not None else None,
        magma=_construct("magma", MagmaConfig, raw["magma"]) if raw.get("magma") is not None else None,
    )
    _validate_controlled_experiment(config)
    return config


def _validate_controlled_experiment(config: ExperimentConfig) -> None:
    model, training, data = config.model, config.training, config.data
    horizons = {
        "EXP-001": (3052, 100_007_936),
        "EXP-002": (9156, 300_023_808),
        "EXP-003": (9156, 300_023_808),
        "EXP-004": (9156, 300_023_808),
        "EXP-005A": (9156, 300_023_808),
        "EXP-005B": (9156, 300_023_808),
        "EXP-006": (27_468, 900_071_424),
        "EXP-007A": (9156, 300_023_808),
        "EXP-007B": (9156, 300_023_808),
        "EXP-008A": (9156, 300_023_808),
        "EXP-009A": (9156, 300_023_808),
        "EXP-009B": (9156, 300_023_808),
        "EXP-010A": (9156, 300_023_808),
        "EXP-011": (45_777, 1_500_020_736),
        "EXP-012": (73_242, 2_399_993_856),
        "EXP-013-C": (9_156, 300_023_808),
        "EXP-013-W": (9_156, 300_023_808),
        "EXP-013-C43": (9_156, 300_023_808),
        "EXP-013-W43": (9_156, 300_023_808),
        "EXP-014": (9_156, 300_023_808),
        "EXP-016-C": (9_156, 300_023_808),
        "EXP-016-M": (9_156, 300_023_808),
        "EXP-017A": (73_242, 2_399_993_856),
    }
    if config.experiment_id not in horizons:
        raise ValueError("Only registered controlled experiment configurations are supported.")
    expected_dimensions = {
        "EXP-005A": (8192, 256, 24, 8, 32, 1024),
        "EXP-005B": (8192, 384, 10, 12, 32, 1536),
        "EXP-006": (8192, 384, 10, 12, 32, 1536),
        "EXP-007A": (8192, 608, 10, 19, 32, 2432),
        "EXP-007B": (8192, 640, 9, 20, 32, 2560),
        "EXP-008A": (8192, 640, 9, 20, 32, 1728),
        "EXP-009A": (8192, 640, 9, 20, 32, 1728),
        "EXP-009B": (8192, 640, 9, 20, 32, 1728),
        "EXP-010A": (8192, 608, 10, 19, 32, 1656),
        "EXP-011": (8192, 640, 9, 20, 32, 1728),
        "EXP-012": (8192, 640, 9, 20, 32, 1728),
        "EXP-013-C": (8192, 640, 9, 20, 32, 1728),
        "EXP-013-W": (8192, 640, 9, 20, 32, 1728),
        "EXP-013-C43": (8192, 640, 9, 20, 32, 1728),
        "EXP-013-W43": (8192, 640, 9, 20, 32, 1728),
        "EXP-014": (8192, 640, 9, 20, 32, 1728),
        "EXP-016-C": (8192, 640, 9, 20, 32, 1728),
        "EXP-016-M": (8192, 640, 9, 20, 32, 1728),
        "EXP-017A": (8192, 640, 9, 20, 32, 1728),
    }.get(config.experiment_id, (8192, 256, 8, 8, 32, 1024))
    if (model.vocab_size, model.d_model, model.n_layers, model.n_heads, model.head_dim, model.d_ff) != expected_dimensions:
        raise ValueError(f"{config.experiment_id} model dimensions differ from the approved allocation.")
    if model.d_model != model.n_heads * model.head_dim or model.rotary_dim != model.head_dim:
        raise ValueError("Controlled experiments require full-head RoPE with consistent attention dimensions.")
    if model.rope_theta != 10000.0 or model.rope_scaling != "none":
        raise ValueError("Controlled experiments require unscaled RoPE theta=10000.0.")
    if model.rmsnorm_eps != 1.0e-5 or model.gelu_approximate != "none":
        raise ValueError("Controlled experiments require RMSNorm eps=1e-5 and exact GELU.")
    if not model.causal or not model.tie_input_output_embeddings or model.linear_bias or model.dropout != 0.0:
        raise ValueError("Controlled causal/tied/bias/dropout invariants are violated.")
    if (
        model.architecture != "decoder_only_transformer"
        or model.activation != ("swiglu" if config.experiment_id in {"EXP-008A", "EXP-009A", "EXP-009B", "EXP-010A", "EXP-011", "EXP-012", "EXP-013-C", "EXP-013-W", "EXP-013-C43", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M", "EXP-017A"} else "gelu")
        or model.norm != "rmsnorm"
        or model.norm_placement != "pre_norm"
        or model.positional_encoding != "rope"
        or model.attention != "standard_multi_head_self_attention"
        or model.context_length != 512
        or model.init_std != 0.02
    ):
        raise ValueError("Controlled architecture invariants are violated.")
    if training.effective_batch_tokens != 32768 or training.sequence_predictions != 512:
        raise ValueError("Controlled effective batch is 64 x 512 prediction tokens.")
    if training.default_microbatch_sequences * training.default_gradient_accumulation_steps * 512 != 32768:
        raise ValueError("Configured microbatch/accumulation does not preserve effective batch tokens.")
    if config.experiment_id in {"EXP-005A", "EXP-005B", "EXP-006", "EXP-007A", "EXP-007B", "EXP-008A", "EXP-009A", "EXP-009B", "EXP-010A", "EXP-011", "EXP-012", "EXP-013-C", "EXP-013-W", "EXP-013-C43", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M", "EXP-017A"} and (
        training.default_microbatch_sequences,
        training.default_gradient_accumulation_steps,
    ) != (32, 2):
        raise ValueError("EXP-005 through EXP-011 must retain the measured 32-sequence x 2 physical batch.")
    expected_steps, expected_tokens = horizons[config.experiment_id]
    if training.full_schedule_steps != expected_steps or training.full_training_tokens != expected_tokens or training.full_training_tokens != training.full_schedule_steps * training.effective_batch_tokens or training.smoke_steps != 60 or training.smoke_training_tokens != 1_966_080:
        raise ValueError(f"{config.experiment_id} full/smoke token budget invariant is violated.")
    if (
        training.precision != "bf16_autocast_fp32_parameters"
        or training.optimizer != "adamw"
        or (training.beta1, training.beta2, training.eps) != (0.9, 0.95, 1.0e-8)
        or training.weight_decay != 0.1
        or (training.peak_learning_rate, training.min_learning_rate) != ({"EXP-009A": (4.0e-4, 4.0e-5), "EXP-009B": (8.0e-4, 8.0e-5)}.get(config.experiment_id, (6.0e-4, 6.0e-5)))
        or training.schedule != ("warmup_stable_decay" if config.experiment_id in {"EXP-013-W", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M", "EXP-017A"} else "cosine_decay")
        or training.warmup_steps != 100
        or training.gradient_clip_norm != 1.0
    ):
        raise ValueError("Controlled optimizer, precision, clipping, or schedule invariants are violated.")
    if (config.experiment_id in {"EXP-013-W", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M"} and training.cooldown_steps != 916) or (
        config.experiment_id == "EXP-017A" and training.cooldown_steps != 7_324
    ) or (
        config.experiment_id not in {"EXP-013-W", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M", "EXP-017A"} and training.cooldown_steps is not None
    ):
        raise ValueError("WSD cooldown configuration differs from the experiment's frozen horizon semantics.")
    if config.experiment_id == "EXP-014":
        if config.llr is None or config.llr != LLRConfig("htsr_pl_alpha_hill_linear", 1.0, 5.0, 100, 0.5, 50, 1800, 1850):
            raise ValueError("EXP-014 must use the pre-registered HT-SR LLR settings.")
    elif config.llr is not None:
        raise ValueError("Only EXP-014 may declare LLR settings.")
    if config.experiment_id == "EXP-016-M":
        if config.magma != MagmaConfig("magma", 0.5, 2.0, 0.9, 42):
            raise ValueError("EXP-016-M must use the preregistered Magma settings.")
    elif config.magma is not None:
        raise ValueError("Only EXP-016-M may declare Magma settings.")
    expected_seed = 43 if config.experiment_id in {"EXP-013-C43", "EXP-013-W43"} else 42
    if training.seed != expected_seed or data.split_seed != 42:
        raise ValueError("Controlled experiments require their exact training seed and fixed data split seed 42.")
    expected_data = (
        ("HuggingFaceFW/fineweb-edu", "default", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
        if config.experiment_id == "EXP-003"
        else ("HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc")
    )
    if (data.dataset_repo, data.dataset_config, data.dataset_revision) != expected_data:
        raise ValueError(f"{config.experiment_id} dataset pin is invalid.")
    if config.experiment_id in {"EXP-004", "EXP-005A", "EXP-005B", "EXP-006", "EXP-007A", "EXP-007B", "EXP-008A", "EXP-009A", "EXP-009B", "EXP-010A", "EXP-011", "EXP-012", "EXP-013-C", "EXP-013-W", "EXP-013-C43", "EXP-013-W43", "EXP-014", "EXP-016-C", "EXP-016-M", "EXP-017A"}:
        target_prediction_tokens = (
            {"fineweb": 600_047_616, "fineweb_edu": 300_023_808}
            if config.experiment_id == "EXP-006"
            else {"fineweb": 1_000_013_824, "fineweb_edu": 500_006_912}
            if config.experiment_id == "EXP-011"
            else {"fineweb": 1_599_995_904, "fineweb_edu": 799_997_952}
            if config.experiment_id in {"EXP-012", "EXP-017A"}
            else {"fineweb": 200_015_872, "fineweb_edu": 100_007_936}
        )
        expected_mixture = {
            "target_prediction_tokens": target_prediction_tokens,
            "global_deduplication": "canonical_content_sha256",
            "sources": {
                "fineweb": {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT", "revision": "9bb295ddab0e05d785b879661af7260fed5140fc", "field": "text"},
                "fineweb_edu": {"repo": "HuggingFaceFW/fineweb-edu", "config": "default", "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "field": "text"},
            },
        }
        if config.mixture != expected_mixture:
            raise ValueError(f"{config.experiment_id} mixture specification is not the approved deduplicated 2:1 data control.")
    elif config.mixture is not None:
        raise ValueError("Only EXP-004 through EXP-014 may declare the approved mixture data specification.")
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


def sha256_file_prefix(path: Path | str, byte_count: int) -> str:
    """Hash exactly ``byte_count`` raw bytes and reject short files instead of silently hashing a prefix."""
    if byte_count <= 0:
        raise ValueError("Prefix byte count must be positive.")
    digest = hashlib.sha256()
    remaining = byte_count
    with Path(path).open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"File ended before its required {byte_count}-byte prefix.")
            digest.update(chunk)
            remaining -= len(chunk)
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
