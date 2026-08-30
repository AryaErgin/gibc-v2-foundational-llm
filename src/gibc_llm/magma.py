"""Independent Momentum-Aligned Gradient Masking (Magma) for AdamW.

This module follows Joo et al., arXiv:2602.15322v1, while making the
AdamW-specific operationalization explicit: AdamW updates its dense state and
produces a full parameter delta first; Magma then replaces each selected block
with ``old + s * mask * (adamw_new - old)``.  This retains dense moments and
includes AdamW's decoupled weight decay in the masked full delta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MagmaSettings:
    survival_probability: float = 0.5
    tau: float = 2.0
    smoothing: float = 0.9
    rng_seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.survival_probability <= 1.0:
            raise ValueError("Magma survival_probability must be in [0, 1].")
        if self.tau <= 0.0:
            raise ValueError("Magma tau must be positive.")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("Magma smoothing must be in [0, 1).")


@dataclass(frozen=True)
class MagmaBlock:
    name: str
    parameter: nn.Parameter


_ATTENTION_WEIGHTS = {"q_proj", "k_proj", "v_proj", "o_proj"}
_MLP_WEIGHTS = {"value_proj", "gate_proj", "out_proj"}


def magma_blocks(model: nn.Module) -> tuple[MagmaBlock, ...]:
    """Return Recipe-v3 attention and SwiGLU matrices in deterministic name order."""
    blocks: list[MagmaBlock] = []
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        is_attention = len(parts) == 5 and parts[0] == "blocks" and parts[2] == "attention" and parts[3] in _ATTENTION_WEIGHTS and parts[4] == "weight"
        is_mlp = len(parts) == 5 and parts[0] == "blocks" and parts[2] == "mlp" and parts[3] in _MLP_WEIGHTS and parts[4] == "weight"
        if is_attention or is_mlp:
            if parameter.ndim != 2:
                raise RuntimeError(f"Magma block {name} must be a matrix.")
            blocks.append(MagmaBlock(name, parameter))
    names = [block.name for block in blocks]
    if len(names) != len(set(names)):
        raise RuntimeError("Magma block mapping contains a duplicate matrix.")
    return tuple(blocks)


def masked_parameter_count(blocks: Iterable[MagmaBlock]) -> int:
    return sum(block.parameter.numel() for block in blocks)


def cosine_alignment_score(moment: Tensor, gradient: Tensor, tau: float) -> float:
    """Return sigmoid(cos(moment, gradient) / tau), using zero for a zero norm."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    left = moment.detach().float().reshape(-1)
    right = gradient.detach().float().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    cosine = torch.zeros((), device=left.device) if float(denominator) == 0.0 else torch.dot(left, right) / denominator
    return float(torch.sigmoid(cosine / tau))


def alignment_ema(previous: float, score: float, smoothing: float) -> float:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    return smoothing * previous + (1.0 - smoothing) * score


class MagmaAdamW:
    """A checkpointable wrapper that masks full AdamW deltas on selected matrices.

    The wrapper deliberately proxies AdamW's state and parameter groups so the
    existing WSD, gradient clipping, checkpoint, and resume paths remain the
    production paths. It adds no trainable tensor and uses an isolated generator
    for exactly one Bernoulli draw per selected block per optimizer update.
    """

    format_version = 1

    def __init__(self, base_optimizer: torch.optim.AdamW, blocks: Iterable[MagmaBlock], settings: MagmaSettings = MagmaSettings(), *, identity_mode: bool = False) -> None:
        self.base_optimizer = base_optimizer
        self.blocks = tuple(blocks)
        self.settings = settings
        self.identity_mode = identity_mode
        if not self.blocks:
            raise ValueError("Magma requires at least one selected matrix block.")
        all_parameters = {id(parameter) for group in base_optimizer.param_groups for parameter in group["params"]}
        if any(id(block.parameter) not in all_parameters for block in self.blocks):
            raise ValueError("Every Magma block must be owned by the base AdamW optimizer.")
        devices = {str(block.parameter.device) for block in self.blocks}
        if len(devices) != 1:
            raise ValueError("Magma blocks must reside on one device.")
        self._generator = torch.Generator(device=self.blocks[0].parameter.device)
        self._generator.manual_seed(settings.rng_seed)
        self.alignment = {block.name: 0.0 for block in self.blocks}
        self.last_masks: dict[str, int] = {}
        self.last_scores: dict[str, float] = {}
        self.mask_draw_count = 0

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    @property
    def state(self) -> dict[nn.Parameter, dict[str, Any]]:
        return self.base_optimizer.state

    @property
    def defaults(self) -> dict[str, Any]:
        return self.base_optimizer.defaults

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.base_optimizer.zero_grad(*args, **kwargs)

    def _draw_mask(self, parameter: nn.Parameter) -> int:
        self.mask_draw_count += 1
        return int(torch.rand((), device=parameter.device, generator=self._generator).item() < self.settings.survival_probability)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        if self.identity_mode:
            return self.base_optimizer.step(closure)
        old = {block.name: block.parameter.detach().clone(memory_format=torch.preserve_format) for block in self.blocks}
        gradients = {block.name: block.parameter.grad.detach() for block in self.blocks if block.parameter.grad is not None}
        result = self.base_optimizer.step(closure)
        self.last_masks = {}
        self.last_scores = {}
        for block in self.blocks:
            gradient = gradients.get(block.name)
            state = self.base_optimizer.state[block.parameter]
            if gradient is None or "exp_avg" not in state:
                raise RuntimeError(f"Magma requires dense AdamW first moment for {block.name}.")
            score = cosine_alignment_score(state["exp_avg"], gradient, self.settings.tau)
            alignment = alignment_ema(self.alignment[block.name], score, self.settings.smoothing)
            self.alignment[block.name] = alignment
            mask = self._draw_mask(block.parameter)
            self.last_masks[block.name] = mask
            self.last_scores[block.name] = score
            adamw_delta = block.parameter.detach() - old[block.name]
            block.parameter.copy_(old[block.name] + (alignment * mask) * adamw_delta)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "magma": {
                "format_version": self.format_version,
                "settings": asdict(self.settings),
                "block_names": [block.name for block in self.blocks],
                "alignment": dict(self.alignment),
                "generator_state": self._generator.get_state().cpu(),
                "mask_draw_count": self.mask_draw_count,
                "identity_mode": self.identity_mode,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != {"base_optimizer", "magma"}:
            raise RuntimeError("Magma checkpoint lacks the required wrapper state.")
        magma_state = state_dict["magma"]
        expected = {
            "format_version": self.format_version,
            "settings": asdict(self.settings),
            "block_names": [block.name for block in self.blocks],
            "identity_mode": self.identity_mode,
        }
        if any(magma_state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Magma checkpoint configuration or block mapping differs from this run.")
        alignment = magma_state.get("alignment")
        if not isinstance(alignment, dict) or list(alignment) != [block.name for block in self.blocks]:
            raise RuntimeError("Magma checkpoint alignment state does not match the block mapping.")
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.alignment = {name: float(value) for name, value in alignment.items()}
        self._generator.set_state(magma_state["generator_state"])
        self.mask_draw_count = int(magma_state["mask_draw_count"])
        self.last_masks = {}
        self.last_scores = {}
