"""Focused regression tests for default-off operational inter-update pacing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import io

import pytest
import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, train_smoke
from gibc_llm.utils import load_config, set_global_seed


def _run(inter_update_sleep_seconds: float) -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer, CosineWithWarmup, RunState, torch.Tensor]:
    set_global_seed(42)
    base = load_config(Path("configs/exp001.yaml")).model
    model = DecoderOnlyTransformer(
        replace(base, vocab_size=32, d_model=8, n_layers=1, n_heads=1, head_dim=8, d_ff=16, rotary_dim=8, context_length=512)
    )
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(128 * 512, dtype=torch.long).reshape(128, 512) % 32
    targets = (inputs + 1) % 32
    state = RunState()
    train_smoke(
        model, inputs, targets, inputs[:1], targets[:1], optimizer, schedule, state,
        torch.device("cpu"), 32, 2, 2, 1.0,
        inter_update_sleep_seconds=inter_update_sleep_seconds,
    )
    return model, optimizer, schedule, state, torch.get_rng_state().clone()


def test_inter_update_pacing_sleeps_once_per_completed_update_without_changing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Breaks if pacing is skipped, occurs per microbatch, or changes completed-update state."""
    import gibc_llm.train as train

    sleep_calls: list[float] = []
    monkeypatch.setattr(train.time, "sleep", sleep_calls.append)

    unpaced_model, unpaced_optimizer, unpaced_schedule, unpaced_state, unpaced_rng = _run(0.0)
    paced_model, paced_optimizer, paced_schedule, paced_state, paced_rng = _run(0.300)

    assert sleep_calls == [0.300, 0.300]
    assert unpaced_state == paced_state == RunState(step=2, tokens=65_536, next_sequence_index=128)
    assert unpaced_schedule.state_dict() == paced_schedule.state_dict()
    unpaced_optimizer_bytes, paced_optimizer_bytes = io.BytesIO(), io.BytesIO()
    torch.save(unpaced_optimizer.state_dict(), unpaced_optimizer_bytes)
    torch.save(paced_optimizer.state_dict(), paced_optimizer_bytes)
    assert unpaced_optimizer_bytes.getvalue() == paced_optimizer_bytes.getvalue()
    assert torch.equal(unpaced_rng, paced_rng)
    for expected, actual in zip(unpaced_model.parameters(), paced_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    probe = torch.arange(512, dtype=torch.long).reshape(1, 512) % 32
    with torch.no_grad():
        assert torch.equal(unpaced_model.eval()(probe), paced_model.eval()(probe))
