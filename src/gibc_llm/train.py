"""BF16-autocast training, validation, and resumable checkpoints for EXP-001."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from .data import TokenStreamDataset
from .model import DecoderOnlyTransformer, RMSNorm


class LRController(Protocol):
    def step(self, step: int, global_lr: float) -> dict[str, Any] | None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class RunState:
    step: int = 0
    tokens: int = 0
    next_sequence_index: int = 0


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    perplexity: float
    token_count: int


class CosineWithWarmup:
    """Approved full-horizon schedule; smoke runs use its early, uncompressed part."""

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        self._set_lr(self.lr_at_step(0))

    def lr_at_step(self, step: int) -> float:
        if step <= 0:
            return 0.0
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> float:
        self.step_count += 1
        value = self.lr_at_step(self.step_count)
        self._set_lr(value)
        return value

    def state_dict(self) -> dict[str, Any]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self._set_lr(self.lr_at_step(self.step_count))


class WarmupStableDecay:
    """Explicit WSD schedule with a fixed, checkpointable stable-stage boundary."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
        cooldown_steps: int,
    ) -> None:
        if warmup_steps <= 0 or cooldown_steps <= 0 or total_steps <= warmup_steps + cooldown_steps:
            raise ValueError("WSD requires positive warmup/cooldown and a non-empty stable interval.")
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cooldown_steps = cooldown_steps
        self.stable_end_step = total_steps - cooldown_steps
        self.step_count = 0
        self._set_lr(self.lr_at_step(0))

    def lr_at_step(self, step: int) -> float:
        if step <= 0:
            return 0.0
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        if step <= self.stable_end_step:
            return self.peak_lr
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.stable_end_step) / self.cooldown_steps
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> float:
        self.step_count += 1
        value = self.lr_at_step(self.step_count)
        self._set_lr(value)
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "warmup_stable_decay",
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "cooldown_steps": self.cooldown_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "type": "warmup_stable_decay",
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "cooldown_steps": self.cooldown_steps,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("WSD checkpoint scheduler parameters do not match the configured schedule.")
        self.step_count = int(state["step_count"])
        self._set_lr(self.lr_at_step(self.step_count))


class CautiousAdamW(torch.optim.Optimizer):
    """AdamW with the source-faithful Cautious Weight Decay mask.

    Chen et al. (arXiv:2510.12402v2, Algorithm 1) define the entrywise update
    x_next = x - lr * (u + weight_decay * I(u * x >= 0) * x). The Adam
    moments and bias corrections remain ordinary decoupled AdamW semantics;
    only the pre-update decay term is sign-selective.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 1.0e-2,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        fused: bool | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid AdamW betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if foreach is True or fused is True:
            raise ValueError("CautiousAdamW uses an eager source-faithful per-parameter update; foreach/fused are unsupported.")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
                "foreach": foreach,
                "fused": fused,
            },
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            learning_rate = group["lr"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("CautiousAdamW does not support sparse gradients.")
                if maximize:
                    gradient = -gradient
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(epsilon)
                update = exp_avg / denominator
                if weight_decay != 0.0:
                    decay_mask = update.mul(parameter).ge(0.0)
                    parameter.add_(parameter * decay_mask, alpha=-learning_rate * weight_decay)
                parameter.add_(update, alpha=-learning_rate / bias_correction1)
        return loss


def build_optimizer(
    model: nn.Module,
    peak_learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    *,
    fused: bool | None = None,
    cautious_weight_decay: bool = False,
) -> torch.optim.Optimizer:
    """Decay matrices (including tied embeddings), not one-dimensional RMSNorm scales."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    options: dict[str, Any] = {
        "lr": peak_learning_rate,
        "betas": betas,
        "eps": eps,
    }
    if fused is not None:
        options["fused"] = fused
    parameter_groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if cautious_weight_decay:
        return CautiousAdamW(parameter_groups, **options)
    return torch.optim.AdamW(parameter_groups, **options)


def _autocast(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: DecoderOnlyTransformer, inputs: Tensor, targets: Tensor, batch_size: int, device: torch.device) -> ValidationResult:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    token_count = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        with _autocast(device):
            loss = model.loss(batch_inputs, batch_targets)
        tokens = batch_targets.numel()
        total_loss += float(loss.detach().float()) * tokens
        token_count += tokens
    if was_training:
        model.train()
    average_loss = total_loss / token_count
    return ValidationResult(loss=average_loss, perplexity=math.exp(average_loss), token_count=token_count)


def optimizer_update(
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    schedule: CosineWithWarmup,
    microbatches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    gradient_clip_norm: float,
    *,
    capture_scalars: bool = True,
    lr_controller: LRController | None = None,
) -> dict[str, float]:
    batches = list(microbatches)
    if not batches:
        raise ValueError("An optimizer update requires at least one microbatch.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    tokens = 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        with _autocast(device):
            loss = model.loss(inputs, targets)
        (loss / len(batches)).backward()
        if capture_scalars:
            loss_sum += float(loss.detach().float())
        tokens += targets.numel()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    learning_rate = schedule.step()
    llr_telemetry = lr_controller.step(schedule.step_count, learning_rate) if lr_controller is not None else None
    optimizer.step()
    metrics = {"tokens": float(tokens)}
    if capture_scalars:
        metrics.update({"loss": loss_sum / len(batches), "gradient_norm": float(gradient_norm), "learning_rate": learning_rate})
    if llr_telemetry is not None:
        metrics["llr_telemetry"] = llr_telemetry
    return metrics


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    schedule: CosineWithWarmup,
    state: RunState,
    config: dict[str, Any],
    *,
    lr_controller: LRController | None = None,
    data_cursor: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "run_state": asdict(state),
        "data_cursor": data_cursor or {"next_sequence_index": state.next_sequence_index, "mechanism": "sequential_example_index"},
        "rng": _rng_state(),
        "config": config,
    }
    if lr_controller is not None:
        payload["llr"] = lr_controller.state_dict()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    schedule: CosineWithWarmup,
    device: torch.device,
    *,
    lr_controller: LRController | None = None,
) -> RunState:
    # Keep serialized CPU/CUDA RNG state byte tensors on CPU; move optimizer state separately.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for optimizer_state in optimizer.state.values():
        for name, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[name] = value.to(device)
    schedule.load_state_dict(payload["schedule"])
    if lr_controller is not None:
        if "llr" not in payload:
            raise RuntimeError("EXP-014 checkpoint lacks required LLR controller state.")
        lr_controller.load_state_dict(payload["llr"])
    _restore_rng(payload["rng"])
    saved_state = dict(payload["run_state"])
    if "next_sequence_index" not in saved_state:
        training = payload.get("config", {}).get("training", {})
        sequences_per_update = int(training.get("effective_batch_tokens", 0)) // int(
            training.get("sequence_predictions", 1)
        )
        if sequences_per_update <= 0:
            raise RuntimeError("Legacy checkpoint lacks a recoverable deterministic training-data cursor.")
        saved_state["next_sequence_index"] = int(saved_state["step"]) * sequences_per_update
    return RunState(**saved_state)


def tiny_overfit(
    model: DecoderOnlyTransformer, inputs: Tensor, targets: Tensor, steps: int, learning_rate: float, device: torch.device
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            loss = model.loss(inputs.to(device), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().float()))
    return losses


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class DurableProgressLogger:
    """Append-only, fsynced output-only progress telemetry for long runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def train_smoke(
    model: DecoderOnlyTransformer,
    train_inputs: Tensor | TokenStreamDataset,
    train_targets: Tensor | None,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    optimizer: torch.optim.Optimizer,
    schedule: CosineWithWarmup,
    state: RunState,
    device: torch.device,
    microbatch_sequences: int,
    accumulation_steps: int,
    steps: int,
    gradient_clip_norm: float,
    logger: JsonlLogger | None = None,
    lr_controller: LRController | None = None,
    sequence_schedule: np.ndarray | None = None,
    progress_logger: DurableProgressLogger | None = None,
    progress_interval_updates: int = 100,
    progress_metadata: dict[str, Any] | None = None,
    inter_update_sleep_seconds: float = 0.0,
) -> list[dict[str, float]]:
    if progress_logger is not None and not 0 < progress_interval_updates <= 100:
        raise ValueError("Durable progress interval must be in [1, 100] completed optimizer updates.")
    if inter_update_sleep_seconds < 0.0:
        raise ValueError("Inter-update sleep seconds must be non-negative.")
    if isinstance(train_inputs, TokenStreamDataset):
        if train_inputs.context_length != 512 or train_targets is not None:
            raise ValueError("TokenStreamDataset training expects 512-token views and no duplicate target tensor.")
        train_example_count = len(train_inputs)
    else:
        if train_targets is None:
            raise ValueError("Tensor training inputs require a matching target tensor.")
        train_example_count = train_inputs.shape[0]
    if microbatch_sequences * accumulation_steps * 512 != 32_768:
        raise ValueError("EXP-001 smoke updates must contain exactly 32,768 prediction tokens.")
    model.to(device)
    records: list[dict[str, float]] = []
    total_sequences = microbatch_sequences * accumulation_steps
    if sequence_schedule is not None:
        if sequence_schedule.dtype != np.uint32 or sequence_schedule.shape != (train_example_count,) or not np.array_equal(np.sort(sequence_schedule), np.arange(train_example_count, dtype=np.uint32)):
            raise ValueError("Sequence schedule must be an exact uint32 permutation of immutable training windows.")
    expected_cursor = state.step * total_sequences
    if state.next_sequence_index != expected_cursor:
        raise ValueError("RunState data cursor must equal global optimizer step times sequences per update.")
    for _ in range(steps):
        next_cursor = state.next_sequence_index + total_sequences
        if next_cursor > train_example_count:
            raise ValueError("Sequential training data exhausted; EXP-001 never cycles a prepared corpus.")
        indices = (sequence_schedule[state.next_sequence_index:next_cursor].tolist() if sequence_schedule is not None else list(range(state.next_sequence_index, next_cursor)))
        batches = []
        for offset in range(0, total_sequences, microbatch_sequences):
            batch_indices = indices[offset : offset + microbatch_sequences]
            if isinstance(train_inputs, TokenStreamDataset):
                batch_inputs, batch_targets = train_inputs.get_indexed_batch(batch_indices)
            else:
                assert train_targets is not None
                batch_inputs = train_inputs[batch_indices]
                batch_targets = train_targets[batch_indices]
            batches.append((batch_inputs, batch_targets))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        metrics = optimizer_update(model, optimizer, schedule, batches, device, gradient_clip_norm, lr_controller=lr_controller)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        state.step += 1
        state.tokens += int(metrics["tokens"])
        state.next_sequence_index = next_cursor
        metrics.update({"step": float(state.step), "cumulative_tokens": float(state.tokens), "wall_seconds": elapsed, "tokens_per_second": metrics["tokens"] / elapsed})
        if inter_update_sleep_seconds:
            time.sleep(inter_update_sleep_seconds)
        paced_wall_seconds = time.perf_counter() - start
        metrics.update(
            {
                "active_compute_tokens_per_second": metrics["tokens_per_second"],
                "inter_update_sleep_seconds": inter_update_sleep_seconds,
                "paced_wall_seconds": paced_wall_seconds,
                "paced_tokens_per_second": metrics["tokens"] / paced_wall_seconds,
            }
        )
        if device.type == "cuda":
            metrics["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
            metrics["peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))
        records.append(metrics)
        if logger is not None:
            logger.log(metrics)
        if progress_logger is not None and state.step % progress_interval_updates == 0:
            progress_logger.log(
                {
                    **(progress_metadata or {}),
                    "event": "durable_progress",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": state.step,
                    "prediction_tokens": state.tokens,
                    "next_sequence_index": state.next_sequence_index,
                    "train_loss": float(metrics["loss"]),
                    "learning_rate": float(metrics["learning_rate"]),
                    "tokens_per_second": float(metrics["tokens_per_second"]),
                    "active_compute_tokens_per_second": float(metrics["active_compute_tokens_per_second"]),
                    "paced_tokens_per_second": float(metrics["paced_tokens_per_second"]),
                }
            )
    return records


def profile_microbatches(
    model: DecoderOnlyTransformer,
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
    candidates: tuple[int, ...] = (8, 16, 32, 64),
) -> list[dict[str, float | int | str]]:
    """Bounded physical-batch comparison; each candidate preserves 64 x 512 tokens/update."""
    if device.type != "cuda":
        raise RuntimeError("EXP-001A microbatch profiling requires CUDA.")
    records: list[dict[str, float | int | str]] = []
    for microbatch_sequences in candidates:
        if 64 % microbatch_sequences:
            continue
        accumulation_steps = 64 // microbatch_sequences
        try:
            model.to(device)
            optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
            schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
            batches = [
                (inputs[offset : offset + microbatch_sequences], targets[offset : offset + microbatch_sequences])
                for offset in range(0, 64, microbatch_sequences)
            ]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer_update(model, optimizer, schedule, batches, device, 1.0)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            timed_updates = 3
            for _ in range(timed_updates):
                optimizer_update(model, optimizer, schedule, batches, device, 1.0)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "microbatch_sequences": microbatch_sequences,
                    "gradient_accumulation_steps": accumulation_steps,
                    "timed_updates": timed_updates,
                    "tokens_per_second": timed_updates * 32_768 / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "status": "ok",
                }
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            records.append(
                {
                    "microbatch_sequences": microbatch_sequences,
                    "gradient_accumulation_steps": accumulation_steps,
                    "status": "out_of_memory",
                }
            )
    return records
