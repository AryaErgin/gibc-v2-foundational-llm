"""Focused durability regression checks for long-horizon full-run chunks."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, save_checkpoint, train_smoke
from gibc_llm.utils import load_config, set_global_seed


def _tiny_model() -> DecoderOnlyTransformer:
    base = load_config(Path("configs/exp001.yaml")).model
    return DecoderOnlyTransformer(
        replace(base, vocab_size=32, d_model=8, n_layers=1, n_heads=1, head_dim=8, d_ff=16, rotary_dim=8, context_length=512)
    )


def _run(tmp_path: Path, progress_path: Path | None) -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer, CosineWithWarmup, RunState, Path | None]:
    set_global_seed(42)
    model = _tiny_model()
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(128 * 512, dtype=torch.long).reshape(128, 512) % 32
    targets = (inputs + 1) % 32
    kwargs = {}
    if progress_path is not None:
        train = importlib.import_module("gibc_llm.train")
        kwargs = {"progress_logger": train.DurableProgressLogger(progress_path), "progress_interval_updates": 2}
    state = RunState()
    train_smoke(
        model,
        inputs,
        targets,
        inputs[:1],
        targets[:1],
        optimizer,
        schedule,
        state,
        torch.device("cpu"),
        32,
        2,
        2,
        1.0,
        **kwargs,
    )
    checkpoint = tmp_path / ("logged.pt" if progress_path is not None else "unlogged.pt")
    save_checkpoint(checkpoint, model, optimizer, schedule, state, {"name": "progress-regression"})
    return model, optimizer, schedule, state, checkpoint


def _assert_optimizer_states_equal(left: dict, right: dict) -> None:
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for parameter_id, left_state in left["state"].items():
        for key, value in left_state.items():
            actual = right["state"][parameter_id][key]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual


def _assert_rng_states_equal(left: dict, right: dict) -> None:
    assert left["python"] == right["python"]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][0] == right["numpy"][0]
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch_cpu"], right["torch_cpu"])
    assert (left["torch_cuda"] is None) == (right["torch_cuda"] is None)
    if left["torch_cuda"] is not None:
        assert right["torch_cuda"] is not None
        assert len(left["torch_cuda"]) == len(right["torch_cuda"])
        for expected, actual in zip(left["torch_cuda"], right["torch_cuda"], strict=True):
            assert torch.equal(expected, actual)


def test_durable_progress_records_completed_update_state(tmp_path: Path) -> None:
    """Breaks if a completed long-run chunk cannot emit a fsynced progress record."""
    train = importlib.import_module("gibc_llm.train")
    assert hasattr(train, "DurableProgressLogger")
    progress_path = tmp_path / "progress.jsonl"
    _, _, _, state, _ = _run(tmp_path, progress_path)
    records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == [2]
    assert records[-1]["prediction_tokens"] == 65_536
    assert records[-1]["next_sequence_index"] == 128
    assert {"train_loss", "learning_rate", "timestamp_utc", "tokens_per_second"} <= records[-1].keys()
    assert state == RunState(step=2, tokens=65_536, next_sequence_index=128)


def test_durable_progress_does_not_change_training_or_checkpoint_state(tmp_path: Path) -> None:
    """Breaks if output-only progress telemetry perturbs the deterministic CPU training path."""
    unlogged_model, unlogged_optimizer, unlogged_schedule, unlogged_state, unlogged_checkpoint = _run(tmp_path, None)
    logged_model, logged_optimizer, logged_schedule, logged_state, logged_checkpoint = _run(tmp_path, tmp_path / "progress.jsonl")
    assert unlogged_state == logged_state
    assert unlogged_schedule.state_dict() == logged_schedule.state_dict()
    _assert_optimizer_states_equal(unlogged_optimizer.state_dict(), logged_optimizer.state_dict())
    for expected, actual in zip(unlogged_model.parameters(), logged_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    unlogged_payload = torch.load(unlogged_checkpoint, map_location="cpu", weights_only=False)
    logged_payload = torch.load(logged_checkpoint, map_location="cpu", weights_only=False)
    assert unlogged_payload["model"].keys() == logged_payload["model"].keys()
    for name, expected in unlogged_payload["model"].items():
        assert torch.equal(expected, logged_payload["model"][name])
    _assert_optimizer_states_equal(unlogged_payload["optimizer"], logged_payload["optimizer"])
    assert unlogged_payload["run_state"] == logged_payload["run_state"]
    assert unlogged_payload["schedule"] == logged_payload["schedule"]
    _assert_rng_states_equal(unlogged_payload["rng"], logged_payload["rng"])
