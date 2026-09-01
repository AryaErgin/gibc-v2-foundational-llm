"""EXP-019 Cautious Weight Decay optimizer contracts."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import (
    CautiousAdamW,
    CosineWithWarmup,
    RunState,
    build_optimizer,
    load_checkpoint,
    optimizer_update,
    save_checkpoint,
    train_smoke,
)
from gibc_llm.utils import load_config, set_global_seed


def _tiny_model(context_length: int = 8) -> DecoderOnlyTransformer:
    base = load_config(Path("configs/exp001.yaml")).model
    return DecoderOnlyTransformer(
        replace(base, vocab_size=32, d_model=16, n_layers=1, n_heads=4, head_dim=4, d_ff=64, rotary_dim=4, context_length=context_length)
    )


def _single(values: list[float], *, betas: tuple[float, float] = (0.0, 0.0), weight_decay: float = 0.1) -> tuple[nn.Parameter, CautiousAdamW]:
    parameter = nn.Parameter(torch.tensor(values, dtype=torch.float64))
    return parameter, CautiousAdamW([parameter], lr=0.1, betas=betas, eps=1.0e-12, weight_decay=weight_decay)


def test_cwd_exact_coordinate_mask_uses_preupdate_parameter_and_ge_zero_boundary() -> None:
    """Breaks if CWD masks raw gradients, uses post-update values, or changes >= to >."""
    parameter, optimizer = _single([1.0, 1.0, -1.0, -1.0, 1.0])
    parameter.grad = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.0], dtype=torch.float64)
    optimizer.step()

    # Algorithm 1: x <- x - lr * (u + wd * I(u*x >= 0) * x), entrywise.
    assert torch.allclose(
        parameter.detach(),
        torch.tensor([0.89, 1.10, -1.10, -0.89, 0.99], dtype=torch.float64),
    )
    state = optimizer.state[parameter]
    assert torch.equal(state["exp_avg"][-1:], torch.zeros(1, dtype=torch.float64))


def test_cwd_masks_using_adam_update_direction_not_raw_gradient() -> None:
    """Breaks if the mask substitutes the instantaneous gradient for Adam's update direction."""
    parameter, optimizer = _single([1.0], betas=(0.9, 0.9))
    parameter.grad = torch.tensor([1.0], dtype=torch.float64)
    optimizer.step()
    parameter.grad = torch.tensor([-0.01], dtype=torch.float64)
    optimizer.step()

    # The second raw gradient is negative but the bias-corrected first moment remains positive.
    assert optimizer.state[parameter]["exp_avg"].item() > 0.0
    # CWD therefore applies the second 0.1*0.1 decoupled shrink too: value < raw-gradient-mask result.
    assert parameter.item() < 0.821


def test_cwd_disabled_is_direct_adamw_and_preserves_parameter_groups() -> None:
    """Breaks if default-off CWD changes any existing AdamW path or decay exclusions."""
    set_global_seed(42)
    model = _tiny_model()
    reference = _tiny_model()
    reference.load_state_dict(model.state_dict())
    disabled = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False)
    decay, no_decay = [], []
    for parameter in reference.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    direct = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=False,
    )
    assert type(disabled) is torch.optim.AdamW
    assert {group["weight_decay"] for group in disabled.param_groups} == {0.0, 0.1}

    for index, (candidate, plain) in enumerate(zip(model.parameters(), reference.parameters(), strict=True)):
        gradient = torch.full_like(candidate, (index + 1) * 1.0e-3)
        candidate.grad = gradient.clone()
        plain.grad = gradient.clone()
    disabled.step()
    direct.step()
    for candidate, plain in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(candidate, plain)


def test_cwd_uses_only_existing_decay_group_and_nominal_weight_decay() -> None:
    """Breaks if CWD changes matrix/vector grouping or changes the frozen 0.1 coefficient."""
    model = _tiny_model()
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False, cautious_weight_decay=True)
    assert isinstance(optimizer, CautiousAdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    for before_parameter, after_parameter in zip(before, model.parameters(), strict=True):
        if after_parameter.ndim >= 2:
            assert torch.allclose(after_parameter, before_parameter * (1.0 - 6e-5), rtol=0.0, atol=1.0e-8)
        else:
            assert torch.equal(after_parameter, before_parameter)


def test_cwd_state_is_finite_and_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    """Breaks if CWD cannot use normal GIBC FP32 state/checkpoint semantics."""
    set_global_seed(42)
    model = _tiny_model()
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False, cautious_weight_decay=True)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(8, dtype=torch.long).reshape(1, 8) % 32
    targets = (inputs + 1) % 32
    optimizer_update(model, optimizer, schedule, [(inputs, targets)], torch.device("cpu"), 1.0)
    assert all(torch.isfinite(value).all() for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor))
    checkpoint = tmp_path / "cwd.pt"
    state = RunState(step=1, tokens=8, next_sequence_index=1)
    output = model(inputs).detach().clone()
    save_checkpoint(checkpoint, model, optimizer, schedule, state, {"training": {"cautious_weight_decay": True}})
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert raw_checkpoint["config"]["training"]["cautious_weight_decay"] is True
    restored_model = _tiny_model()
    restored_optimizer = build_optimizer(restored_model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False, cautious_weight_decay=True)
    restored_schedule = CosineWithWarmup(restored_optimizer, 6e-4, 6e-5, 100, 3052)
    restored_state = load_checkpoint(checkpoint, restored_model, restored_optimizer, restored_schedule, torch.device("cpu"))
    assert restored_state == state
    assert torch.equal(restored_model(inputs), output)
    first, second = io.BytesIO(), io.BytesIO()
    torch.save(optimizer.state_dict(), first)
    torch.save(restored_optimizer.state_dict(), second)
    assert first.getvalue() == second.getvalue()


def _run_two_updates(cwd: bool) -> tuple[RunState, dict[str, object], torch.Tensor]:
    set_global_seed(42)
    model = _tiny_model(context_length=512)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False, cautious_weight_decay=cwd)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(128 * 512, dtype=torch.long).reshape(128, 512) % 32
    targets = (inputs + 1) % 32
    state = RunState()
    train_smoke(model, inputs, targets, inputs[:1], targets[:1], optimizer, schedule, state, torch.device("cpu"), 32, 2, 2, 1.0)
    return state, schedule.state_dict(), torch.get_rng_state().clone()


def test_cwd_does_not_change_scheduler_data_cursor_rng_or_pacing_semantics() -> None:
    """Breaks if CWD selection affects non-optimizer scientific state."""
    ordinary_state, ordinary_schedule, ordinary_rng = _run_two_updates(False)
    cwd_state, cwd_schedule, cwd_rng = _run_two_updates(True)
    assert ordinary_state == cwd_state == RunState(step=2, tokens=65_536, next_sequence_index=128)
    assert ordinary_schedule == cwd_schedule == {"step_count": 2}
    assert torch.equal(ordinary_rng, cwd_rng)


def test_exp019_model_stays_qk_norm_off_and_has_no_added_parameters() -> None:
    """Breaks if CWD accidentally changes model semantics or its 49,860,480 parameter cap."""
    config = load_config(Path("configs/exp011.yaml"))
    assert config.model.qk_norm is False
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 49_860_480


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 production path")
def test_cwd_bf16_production_shaped_update_is_finite() -> None:
    """Breaks if the CWD optimizer cannot execute a 32x2 BF16 production-shaped update."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    set_global_seed(42)
    model = _tiny_model(context_length=512).to(device)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False, cautious_weight_decay=True)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 45_777)
    inputs = torch.arange(64 * 512, dtype=torch.long).reshape(64, 512) % 32
    targets = (inputs + 1) % 32
    state = RunState()
    records = train_smoke(model, inputs, targets, inputs[:1], targets[:1], optimizer, schedule, state, device, 32, 2, 1, 1.0)
    assert state == RunState(step=1, tokens=32_768, next_sequence_index=64)
    assert torch.isfinite(torch.tensor(records[0]["loss"]))
    assert all(torch.isfinite(value).all() for optimizer_state in optimizer.state.values() for value in optimizer_state.values() if isinstance(value, torch.Tensor))
