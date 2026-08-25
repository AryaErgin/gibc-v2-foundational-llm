"""Bounded, systems-only performance measurements for frozen EXP-007B."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

from .data import TokenStreamDataset
from .full_run import FullRunArtifact
from .model import DecoderOnlyTransformer, RotaryEmbedding, parameter_breakdown
from .train import CosineWithWarmup, RunState, _autocast, build_optimizer, evaluate, optimizer_update
from .utils import ExperimentConfig, set_global_seed


SYS001_SYSTEMS_CONTROL = "Windows native environment; OMEN Performance mode; original AC power connected"
SYS001_PARAMETER_COUNT = 49_491_840
SYS001_CONTEXT_LENGTH = 512
SYS001_EFFECTIVE_BATCH_TOKENS = 32_768
SYS001_SEED = 42


@dataclass(frozen=True)
class Sys001Phase:
    identifier: str
    microbatch_sequences: int
    accumulation_steps: int
    synchronize_each_update: bool
    rope_cache: bool = False
    audit_sdpa_backend: bool = False
    fused_adamw: bool = False
    context_length: int = SYS001_CONTEXT_LENGTH
    effective_batch_tokens: int = SYS001_EFFECTIVE_BATCH_TOKENS
    parameter_count: int = SYS001_PARAMETER_COUNT


def sys001_phase_plan() -> tuple[Sys001Phase, ...]:
    """Return the ordered ladder; each phase starts from a fresh seeded model and cursor."""
    return (
        Sys001Phase("baseline_32x2", 32, 2, True),
        Sys001Phase("microbatch_64x1", 64, 1, True),
        Sys001Phase("production_timing", 64, 1, False),
        Sys001Phase("rope_cache", 64, 1, False, rope_cache=True),
        Sys001Phase("sdpa_backend_audit", 64, 1, False, rope_cache=True, audit_sdpa_backend=True),
        Sys001Phase("fused_adamw", 64, 1, False, rope_cache=True, fused_adamw=True),
    )


def phase_by_identifier(identifier: str) -> Sys001Phase:
    for phase in sys001_phase_plan():
        if phase.identifier == identifier:
            return phase
    raise ValueError(f"Unknown SYS-001 phase: {identifier}")


def assert_sys001_controls(config: ExperimentConfig) -> None:
    """Fail before measuring if any scientific EXP-007B control drifted."""
    if config.experiment_id != "EXP-007B":
        raise RuntimeError("SYS-001 requires the frozen EXP-007B configuration.")
    expected_model = {
        "vocab_size": 8192,
        "d_model": 640,
        "n_layers": 9,
        "n_heads": 20,
        "head_dim": 32,
        "d_ff": 2560,
        "context_length": SYS001_CONTEXT_LENGTH,
    }
    for field, expected in expected_model.items():
        actual = getattr(config.model, field)
        if actual != expected:
            raise RuntimeError(f"SYS-001 frozen model control {field} is {actual!r}, not {expected!r}.")
    expected_training = {
        "effective_batch_tokens": SYS001_EFFECTIVE_BATCH_TOKENS,
        "default_microbatch_sequences": 32,
        "default_gradient_accumulation_steps": 2,
        "seed": SYS001_SEED,
        "gradient_clip_norm": 1.0,
        "peak_learning_rate": 6.0e-4,
        "min_learning_rate": 6.0e-5,
        "warmup_steps": 100,
        "full_schedule_steps": 9156,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1.0e-8,
    }
    for field, expected in expected_training.items():
        actual = getattr(config.training, field)
        if actual != expected:
            raise RuntimeError(f"SYS-001 frozen training control {field} is {actual!r}, not {expected!r}.")
    if config.data.context_length != SYS001_CONTEXT_LENGTH or config.data.tokenizer_vocab_size != 8192:
        raise RuntimeError("SYS-001 requires the frozen 512-context, 8192-entry tokenizer/data controls.")


def set_rope_cache(model: DecoderOnlyTransformer, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.set_cache_enabled(enabled)


def classify_sdpa_backend(operator_keys: Iterable[str]) -> str:
    keys = tuple(operator_keys)
    if any("flash_attention" in key for key in keys):
        return "flash"
    if any("efficient_attention" in key for key in keys):
        return "memory_efficient"
    if any("attention_math" in key for key in keys):
        return "math"
    return "unknown"


def audit_sdpa_backend(model: DecoderOnlyTransformer, inputs: Tensor, targets: Tensor, device: torch.device) -> dict[str, Any]:
    """Profile an actual 512-token BF16 SDPA forward/backward, not only enabled backend flags."""
    if device.type != "cuda":
        raise RuntimeError("SYS-001 SDPA audit requires CUDA.")
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with _autocast(device):
            loss = model.loss(inputs.to(device), targets.to(device))
        loss.backward()
    torch.cuda.synchronize(device)
    keys = tuple(item.key for item in profiler.key_averages())
    model.zero_grad(set_to_none=True)
    return {
        "actual_backend": classify_sdpa_backend(keys),
        "operator_keys": [key for key in keys if "scaled_dot_product" in key],
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "memory_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def fused_adamw_supported(device: torch.device) -> tuple[bool, str]:
    """Probe the actual CUDA fused implementation; never silently downgrade to unfused AdamW."""
    if device.type != "cuda":
        return False, "CUDA is required"
    parameter = torch.nn.Parameter(torch.ones(1, device=device))
    try:
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, fused=True)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.cuda.synchronize(device)
    except (RuntimeError, TypeError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "supported"


def _batches(
    dataset: TokenStreamDataset,
    state: RunState,
    phase: Sys001Phase,
) -> list[tuple[Tensor, Tensor]]:
    total_sequences = phase.microbatch_sequences * phase.accumulation_steps
    end = state.next_sequence_index + total_sequences
    if end > len(dataset):
        raise RuntimeError("SYS-001 would exhaust or cycle the frozen train stream.")
    batches = [
        dataset.get_contiguous_batch(start, phase.microbatch_sequences)
        for start in range(state.next_sequence_index, end, phase.microbatch_sequences)
    ]
    state.next_sequence_index = end
    return batches


def _run_updates(
    model: DecoderOnlyTransformer,
    dataset: TokenStreamDataset,
    optimizer: torch.optim.Optimizer,
    schedule: CosineWithWarmup,
    state: RunState,
    phase: Sys001Phase,
    device: torch.device,
    updates: int,
    *,
    time_updates: bool,
) -> dict[str, float]:
    if updates <= 0:
        return {"updates": 0.0, "tokens": 0.0, "wall_seconds": 0.0, "tokens_per_second": 0.0}
    if state.next_sequence_index != state.step * (phase.microbatch_sequences * phase.accumulation_steps):
        raise RuntimeError("SYS-001 sequential stream cursor drifted before an update.")
    if time_updates and not phase.synchronize_each_update:
        torch.cuda.synchronize(device)
    started = time.perf_counter() if time_updates else 0.0
    tokens = 0.0
    for _ in range(updates):
        batches = _batches(dataset, state, phase)
        if time_updates and phase.synchronize_each_update:
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
        metrics = optimizer_update(
            model,
            optimizer,
            schedule,
            batches,
            device,
            gradient_clip_norm=1.0,
            capture_scalars=False,
        )
        if time_updates and phase.synchronize_each_update:
            torch.cuda.synchronize(device)
            tokens += metrics["tokens"]
            _ = time.perf_counter() - update_started
        else:
            tokens += metrics["tokens"]
        state.step += 1
        state.tokens += int(metrics["tokens"])
    if time_updates:
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    else:
        elapsed = 0.0
    return {"updates": float(updates), "tokens": tokens, "wall_seconds": elapsed, "tokens_per_second": tokens / elapsed if elapsed else 0.0}


def run_sys001_phase(
    config: ExperimentConfig,
    artifact: FullRunArtifact,
    phase: Sys001Phase,
    device: torch.device,
    *,
    warmup_updates: int = 100,
    timed_updates: int = 100,
) -> dict[str, Any]:
    """Execute one fresh bounded phase while preserving all scientific controls."""
    assert_sys001_controls(config)
    if phase.microbatch_sequences * phase.accumulation_steps * SYS001_CONTEXT_LENGTH != SYS001_EFFECTIVE_BATCH_TOKENS:
        raise RuntimeError("SYS-001 phase violates the fixed effective prediction-token batch.")
    supported, reason = fused_adamw_supported(device) if phase.fused_adamw else (False, "not requested")
    if phase.fused_adamw and not supported:
        return {"phase": asdict(phase), "status": "SKIPPED_UNSUPPORTED", "fused_adamw": reason}
    set_global_seed(SYS001_SEED)
    model = DecoderOnlyTransformer(config.model).to(device)
    if parameter_breakdown(model).total != SYS001_PARAMETER_COUNT:
        raise RuntimeError("SYS-001 model parameter count is not the frozen 49,491,840.")
    set_rope_cache(model, phase.rope_cache)
    optimizer = build_optimizer(
        model,
        config.training.peak_learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
        config.training.eps,
        fused=True if phase.fused_adamw else None,
    )
    schedule = CosineWithWarmup(
        optimizer,
        config.training.peak_learning_rate,
        config.training.min_learning_rate,
        config.training.warmup_steps,
        config.training.full_schedule_steps,
    )
    state = RunState()
    audit: dict[str, Any] | None = None
    if phase.audit_sdpa_backend:
        inputs, targets = artifact.train.get_contiguous_batch(0, phase.microbatch_sequences)
        audit = audit_sdpa_backend(model, inputs, targets, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _run_updates(model, artifact.train, optimizer, schedule, state, phase, device, warmup_updates, time_updates=False)
    measurement = _run_updates(model, artifact.train, optimizer, schedule, state, phase, device, timed_updates, time_updates=True)
    return {
        "phase": asdict(phase),
        "status": "COMPLETE",
        "systems_control": SYS001_SYSTEMS_CONTROL,
        "parameter_count": parameter_breakdown(model).total,
        "context_length": config.model.context_length,
        "seed": config.training.seed,
        "warmup_updates": warmup_updates,
        "timed_updates": timed_updates,
        "measurement": measurement,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "sdpa_audit": audit,
        "fused_adamw": reason if phase.fused_adamw else "not requested",
    }


def run_stability_validation(
    config: ExperimentConfig,
    artifact: FullRunArtifact,
    phase: Sys001Phase,
    device: torch.device,
    updates: int = 500,
) -> dict[str, Any]:
    """Bounded fresh 500-update comparison endpoint for a floating-order-changing phase."""
    assert_sys001_controls(config)
    if updates != 500:
        raise ValueError("SYS-001 floating-order stability comparisons must use exactly 500 updates.")
    supported, reason = fused_adamw_supported(device) if phase.fused_adamw else (False, "not requested")
    if phase.fused_adamw and not supported:
        return {"phase": asdict(phase), "status": "SKIPPED_UNSUPPORTED", "fused_adamw": reason}
    set_global_seed(SYS001_SEED)
    model = DecoderOnlyTransformer(config.model).to(device)
    if parameter_breakdown(model).total != SYS001_PARAMETER_COUNT:
        raise RuntimeError("SYS-001 model parameter count is not the frozen 49,491,840.")
    set_rope_cache(model, phase.rope_cache)
    optimizer = build_optimizer(
        model,
        config.training.peak_learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
        config.training.eps,
        fused=True if phase.fused_adamw else None,
    )
    schedule = CosineWithWarmup(optimizer, config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps, config.training.full_schedule_steps)
    state = RunState()
    _run_updates(model, artifact.train, optimizer, schedule, state, phase, device, updates, time_updates=False)
    general = evaluate(model, artifact.validation_inputs, artifact.validation_targets, phase.microbatch_sequences, device)
    if artifact.edu_validation_inputs is None or artifact.edu_validation_targets is None:
        raise RuntimeError("SYS-001 stability comparison requires both frozen validations.")
    educational = evaluate(model, artifact.edu_validation_inputs, artifact.edu_validation_targets, phase.microbatch_sequences, device)
    return {
        "phase": asdict(phase),
        "status": "COMPLETE",
        "systems_control": SYS001_SYSTEMS_CONTROL,
        "parameter_count": parameter_breakdown(model).total,
        "stability_validation": {
            "updates": updates,
            "general_loss": general.loss,
            "educational_loss": educational.loss,
            "combined_loss": (general.loss + educational.loss) / 2,
        },
        "fused_adamw": reason if phase.fused_adamw else "not requested",
    }
