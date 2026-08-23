import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, load_checkpoint, optimizer_update, save_checkpoint, train_smoke
from gibc_llm.utils import load_config, set_global_seed


def _tiny_model() -> DecoderOnlyTransformer:
    config = load_config(Path("configs/exp001.yaml")).model
    tiny_config = replace(config, vocab_size=32, d_model=16, n_layers=1, n_heads=4, head_dim=4, d_ff=64, rotary_dim=4, context_length=8)
    return DecoderOnlyTransformer(tiny_config)


def test_checkpoint_round_trip_restores_outputs_counters_rng_and_resume(tmp_path: Path) -> None:
    """Breaks if a checkpoint cannot exactly restore model/training/RNG state for continuation."""
    set_global_seed(42)
    model = _tiny_model()
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(8, dtype=torch.long).unsqueeze(0) % 32
    targets = (inputs + 1) % 32
    optimizer_update(model, optimizer, schedule, [(inputs, targets)], torch.device("cpu"), 1.0)
    state = RunState(step=1, tokens=8)
    output_before = model(inputs).detach().clone()
    checkpoint_path = tmp_path / "state.pt"
    save_checkpoint(checkpoint_path, model, optimizer, schedule, state, {"name": "test"})
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(1).item())

    _ = (random.random(), np.random.rand(), torch.rand(1))
    restored_model = _tiny_model()
    restored_optimizer = build_optimizer(restored_model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    restored_schedule = CosineWithWarmup(restored_optimizer, 6e-4, 6e-5, 100, 3052)
    restored = load_checkpoint(checkpoint_path, restored_model, restored_optimizer, restored_schedule, torch.device("cpu"))

    assert torch.equal(output_before, restored_model(inputs))
    assert restored.step == 1
    assert restored.tokens == 8
    assert (random.random(), float(np.random.rand()), torch.rand(1).item()) == expected_rng
    optimizer_update(restored_model, restored_optimizer, restored_schedule, [(inputs, targets)], torch.device("cpu"), 1.0)
    assert restored_schedule.step_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint regression requires the EXP-001A GPU")
def test_checkpoint_round_trip_keeps_cpu_rng_state_when_model_is_cuda(tmp_path: Path) -> None:
    """Breaks if CUDA map-location moves the CPU RNG state and prevents checkpoint restore."""
    device = torch.device("cuda")
    set_global_seed(42)
    model = _tiny_model().to(device)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
    inputs = torch.arange(8, dtype=torch.long).unsqueeze(0) % 32
    targets = (inputs + 1) % 32
    optimizer_update(model, optimizer, schedule, [(inputs, targets)], device, 1.0)
    expected = model(inputs.to(device)).detach().cpu()
    path = tmp_path / "cuda-state.pt"
    save_checkpoint(path, model, optimizer, schedule, RunState(step=1, tokens=8), {"name": "cuda"})
    restored_model = _tiny_model().to(device)
    restored_optimizer = build_optimizer(restored_model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    restored_schedule = CosineWithWarmup(restored_optimizer, 6e-4, 6e-5, 100, 3052)

    restored = load_checkpoint(path, restored_model, restored_optimizer, restored_schedule, device)

    assert restored == RunState(step=1, tokens=8)
    assert torch.equal(expected, restored_model(inputs.to(device)).detach().cpu())
    optimizer_update(restored_model, restored_optimizer, restored_schedule, [(inputs, targets)], device, 1.0)


def test_uninterrupted_and_reconstructed_resume_consume_identical_sequences_and_states(tmp_path: Path) -> None:
    """Breaks if resume restarts at local step zero rather than the checkpointed next data cursor."""
    device = torch.device("cpu")
    set_global_seed(7)
    inputs = torch.arange(64 * 3 * 512, dtype=torch.long).reshape(64 * 3, 512) % 32
    targets = (inputs + 1) % 32

    def build():
        model_config = replace(_tiny_model().config, context_length=512)
        model = DecoderOnlyTransformer(model_config)
        optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
        schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 3052)
        return model, optimizer, schedule

    set_global_seed(42)
    full_model, full_optimizer, full_schedule = build()
    full_state = RunState()
    train_smoke(full_model, inputs, targets, inputs[:64], targets[:64], full_optimizer, full_schedule, full_state, device, 64, 1, 3, 1.0)

    set_global_seed(42)
    first_model, first_optimizer, first_schedule = build()
    first_state = RunState()
    train_smoke(first_model, inputs, targets, inputs[:64], targets[:64], first_optimizer, first_schedule, first_state, device, 64, 1, 1, 1.0)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, first_model, first_optimizer, first_schedule, first_state, {"training": {"effective_batch_tokens": 32768, "sequence_predictions": 512}})

    resumed_model, resumed_optimizer, resumed_schedule = build()
    resumed_state = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, resumed_schedule, device)
    train_smoke(resumed_model, inputs, targets, inputs[:64], targets[:64], resumed_optimizer, resumed_schedule, resumed_state, device, 64, 1, 2, 1.0)

    assert resumed_state == full_state == RunState(step=3, tokens=98_304, next_sequence_index=192)
    assert full_schedule.state_dict() == resumed_schedule.state_dict()
    full_optimizer_state = full_optimizer.state_dict()
    resumed_optimizer_state = resumed_optimizer.state_dict()
    assert full_optimizer_state["param_groups"] == resumed_optimizer_state["param_groups"]
    assert full_optimizer_state["state"].keys() == resumed_optimizer_state["state"].keys()
    for parameter_id in full_optimizer_state["state"]:
        for key, value in full_optimizer_state["state"][parameter_id].items():
            resumed_value = resumed_optimizer_state["state"][parameter_id][key]
            assert torch.equal(value, resumed_value) if isinstance(value, torch.Tensor) else value == resumed_value
    for uninterrupted, resumed in zip(full_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(uninterrupted, resumed)
