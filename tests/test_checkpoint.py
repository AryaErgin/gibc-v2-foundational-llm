import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gibc_llm.model import DecoderOnlyTransformer
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, load_checkpoint, optimizer_update, save_checkpoint
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
