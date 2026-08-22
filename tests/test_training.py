from dataclasses import replace
from pathlib import Path

import torch
import pytest

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, build_optimizer, evaluate, optimizer_update, tiny_overfit
from gibc_llm.utils import load_config, set_global_seed


def _tiny_model() -> DecoderOnlyTransformer:
    config = load_config(Path("configs/exp001.yaml")).model
    tiny_config = replace(config, vocab_size=32, d_model=16, n_layers=1, n_heads=4, head_dim=4, d_ff=64, rotary_dim=4, context_length=8)
    set_global_seed(42)
    return DecoderOnlyTransformer(tiny_config)


def test_schedule_uses_full_horizon_warmup_and_nonzero_cosine_floor() -> None:
    """Breaks if smoke training compresses the approved full-run LR schedule."""
    model = _tiny_model()
    optimizer = build_optimizer(model, peak_learning_rate=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    schedule = CosineWithWarmup(optimizer, peak_lr=6e-4, min_lr=6e-5, warmup_steps=100, total_steps=3052)

    assert schedule.lr_at_step(0) == 0.0
    assert schedule.lr_at_step(1) == pytest.approx(6e-6)
    assert schedule.lr_at_step(100) == pytest.approx(6e-4)
    assert schedule.lr_at_step(3052) == pytest.approx(6e-5)


def test_optimizer_decays_matrices_but_not_rmsnorm_scales() -> None:
    """Breaks if EXP-001 applies 0.1 AdamW decay to RMSNorm scale vectors."""
    optimizer = build_optimizer(_tiny_model(), peak_learning_rate=6e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    decays = {group["weight_decay"] for group in optimizer.param_groups}

    assert decays == {0.0, 0.1}
    assert sum(parameter.numel() for group in optimizer.param_groups if group["weight_decay"] == 0.0 for parameter in group["params"]) == 48


def test_one_update_and_validation_use_explicit_next_token_targets() -> None:
    """Breaks if training/validation silently shift or miscount the supplied 512-style targets."""
    model = _tiny_model()
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(16, dtype=torch.long).reshape(2, 8) % 32
    targets = (inputs + 1) % 32

    before = evaluate(model, inputs, targets, batch_size=2, device=torch.device("cpu"))
    metrics = optimizer_update(model, optimizer, schedule, [(inputs, targets)], device=torch.device("cpu"), gradient_clip_norm=1.0)
    after = evaluate(model, inputs, targets, batch_size=2, device=torch.device("cpu"))

    assert metrics["tokens"] == 16
    assert metrics["gradient_norm"] >= 0.0
    assert before.token_count == after.token_count == 16


def test_tiny_fixed_batch_overfits_substantially() -> None:
    """Breaks if gradients, targets, optimizer updates, or causal model wiring cannot learn a fixed sample."""
    model = _tiny_model()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    targets = (inputs + 1) % 32

    trajectory = tiny_overfit(model, inputs, targets, steps=80, learning_rate=3e-3, device=torch.device("cpu"))

    assert trajectory[-1] < trajectory[0] * 0.45
