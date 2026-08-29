"""HT-SR layerwise-LR controller for the pre-registered EXP-014 ablation.

This is deliberately independent from the OLMo-based upstream trainer: it only
uses the paper's weight-spectrum, positive linear mapping, embedding treatment,
and soft-transition semantics.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .model import DecoderOnlyTransformer, RMSNorm


def pl_alpha_hill(weight: torch.Tensor) -> float:
    """Hill power-law exponent on the top half of positive W^T W eigenvalues."""
    if weight.ndim != 2:
        raise ValueError("PL_Alpha_Hill requires a two-dimensional weight matrix.")
    values = torch.linalg.svdvals(weight.detach().float()).square()
    values = torch.sort(values[values > 0], descending=True).values
    n = int(values.numel())
    k = n // 2
    if k < 1 or k >= n:
        raise ValueError("PL_Alpha_Hill requires at least three non-zero singular values.")
    denominator = torch.log(values[:k] / values[k]).sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        raise ValueError("PL_Alpha_Hill is undefined for this degenerate spectrum.")
    return float(1.0 + k / denominator)


@dataclass(frozen=True)
class LLRSettings:
    min_multiplier: float = 1.0
    max_multiplier: float = 5.0
    recompute_interval: int = 100
    soft_switch_steps: int = 50
    active_recompute_last_step: int = 1800
    multiplier_freeze_step: int = 1850


def build_llr_optimizer(model: DecoderOnlyTransformer, peak_lr: float, weight_decay: float, betas: tuple[float, float], eps: float, *, fused: bool | None = None) -> torch.optim.AdamW:
    """One group per matrix; tied token/output storage is necessarily one group.

    RMSNorm scales are retained together at multiplier 1.0, matching the
    upstream default (its layernorm option is disabled).
    """
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim == 1:
            continue
        kind = "embedding_output" if parameter is model.token_embedding.weight else "matrix"
        groups.append({"params": [parameter], "weight_decay": weight_decay, "llr_name": name, "llr_kind": kind, "llr_multiplier": 1.0})
    norms = [p for p in model.parameters() if p.requires_grad and p.ndim == 1]
    groups.append({"params": norms, "weight_decay": 0.0, "llr_name": "rmsnorm", "llr_kind": "uniform", "llr_multiplier": 1.0})
    options: dict[str, Any] = {"lr": peak_lr, "betas": betas, "eps": eps}
    if fused is not None:
        options["fused"] = fused
    return torch.optim.AdamW(groups, **options)


class HTSRLLR:
    """Stateful controller; call after global WSD step and before optimizer.step."""
    def __init__(self, model: DecoderOnlyTransformer, optimizer: torch.optim.Optimizer, settings: LLRSettings = LLRSettings()) -> None:
        self.model, self.optimizer, self.settings = model, optimizer, settings
        self.current = {g["llr_name"]: 1.0 for g in optimizer.param_groups}
        self.start = dict(self.current)
        self.target = dict(self.current)
        self.switch_start = 0
        self.last_recompute_step = 0
        self.frozen = False
        self._parameters = {name: p for name, p in model.named_parameters()}
        if sum(g.get("llr_kind") == "embedding_output" for g in optimizer.param_groups) != 1:
            raise ValueError("The tied embedding/output matrix must be represented exactly once.")

    def _actualize(self, global_lr: float) -> None:
        for group in self.optimizer.param_groups:
            name = group["llr_name"]
            multiplier = self.current[name]
            group["llr_multiplier"] = multiplier
            group["lr"] = global_lr * multiplier

    def _recompute(self, step: int) -> dict[str, Any]:
        started = time.perf_counter()
        alphas: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            name, kind = group["llr_name"], group["llr_kind"]
            if kind == "matrix":
                alphas[name] = pl_alpha_hill(self._parameters[name])
        lo, hi = min(alphas.values()), max(alphas.values())
        self.start = dict(self.current)
        for group in self.optimizer.param_groups:
            name, kind = group["llr_name"], group["llr_kind"]
            if kind == "embedding_output":
                self.target[name] = self.settings.max_multiplier
            elif kind == "uniform" or hi == lo:
                self.target[name] = 1.0
            else:
                self.target[name] = self.settings.min_multiplier + (alphas[name] - lo) * (self.settings.max_multiplier - self.settings.min_multiplier) / (hi - lo)
        self.switch_start, self.last_recompute_step = step, step
        return {"event": "llr_recompute", "step": step, "pl_alpha_hill": alphas, "target_multipliers": dict(self.target), "spectral_seconds": time.perf_counter() - started}

    def step(self, step: int, global_lr: float) -> dict[str, Any] | None:
        telemetry = None
        if step <= self.settings.active_recompute_last_step and step % self.settings.recompute_interval == 0:
            telemetry = self._recompute(step)
        elapsed = step - self.switch_start
        if self.switch_start and elapsed <= self.settings.soft_switch_steps:
            ratio = elapsed / self.settings.soft_switch_steps
            self.current = {n: self.start[n] + ratio * (self.target[n] - self.start[n]) for n in self.current}
        elif self.switch_start:
            self.current = dict(self.target)
        if step >= self.settings.multiplier_freeze_step:
            self.frozen = True
        self._actualize(global_lr)
        if telemetry is not None:
            telemetry["actual_lrs"] = {g["llr_name"]: g["lr"] for g in self.optimizer.param_groups}
            telemetry["parameter_norms"] = {n: float(self._parameters[n].detach().float().norm()) for n in telemetry["pl_alpha_hill"]}
        return telemetry

    def state_dict(self) -> dict[str, Any]:
        return {"settings": asdict(self.settings), "current": self.current, "start": self.start, "target": self.target, "switch_start": self.switch_start, "last_recompute_step": self.last_recompute_step, "frozen": self.frozen}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("settings") != asdict(self.settings):
            raise RuntimeError("LLR checkpoint settings do not match the pre-registered configuration.")
        for name in ("current", "start", "target"):
            setattr(self, name, {str(k): float(v) for k, v in state[name].items()})
        self.switch_start, self.last_recompute_step, self.frozen = int(state["switch_start"]), int(state["last_recompute_step"]), bool(state["frozen"])
