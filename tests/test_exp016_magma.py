"""EXP-016 Magma preflight tests; no benchmark or full run."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gibc_llm.data import write_token_stream
from gibc_llm.magma import MagmaAdamW, MagmaSettings, alignment_ema, cosine_alignment_score, magma_blocks, masked_parameter_count
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import RunState, WarmupStableDecay, build_optimizer, load_checkpoint, optimizer_update, save_checkpoint, train_smoke
from gibc_llm.utils import load_config, set_global_seed


def tiny_model(context=8):
    base = load_config(Path("configs/exp016-magma.yaml")).model
    return DecoderOnlyTransformer(replace(base, vocab_size=64, d_model=16, n_layers=1, n_heads=4, head_dim=4, d_ff=32, rotary_dim=4, context_length=context))


def build(device=torch.device("cpu"), settings=MagmaSettings(), identity=False, context=8):
    model = tiny_model(context).to(device)
    base = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False)
    optimizer = MagmaAdamW(base, magma_blocks(model), settings, identity_mode=identity)
    return model, optimizer, WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 9156, 916)


def update(model, optimizer, schedule, device=torch.device("cpu"), context=8):
    inputs = torch.arange(2 * context, dtype=torch.long).reshape(2, context) % 64
    return optimizer_update(model, optimizer, schedule, [(inputs, (inputs + 1) % 64)], device, 1.0)


def equal_tree(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            equal_tree(left[key], right[key])
        elif isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def test_recipe_config_and_exact_block_mapping():
    control = load_config(Path("configs/exp016-control.yaml"))
    magma = load_config(Path("configs/exp016-magma.yaml"))
    assert control.magma is None
    assert magma.magma is not None and magma.magma.survival_probability == 0.5 and magma.magma.tau == 2.0 and magma.magma.smoothing == 0.9
    model = DecoderOnlyTransformer(magma.model)
    blocks = magma_blocks(model)
    names = [block.name for block in blocks]
    assert parameter_breakdown(model).total == 49_860_480
    assert len(blocks) == len(set(names)) == 63
    assert masked_parameter_count(blocks) == 44_605_440
    assert all(("attention." in name or ".mlp." in name) and "norm" not in name for name in names)
    assert "token_embedding.weight" not in names
    assert sum(parameter.numel() for parameter in model.parameters()) - masked_parameter_count(blocks) == 5_255_040


def test_mask_zero_keeps_dense_adamw_moments():
    set_global_seed(9)
    model, optimizer, schedule = build(settings=MagmaSettings(survival_probability=0.0))
    block = optimizer.blocks[0]
    before = block.parameter.detach().clone()
    update(model, optimizer, schedule)
    state = optimizer.base_optimizer.state[block.parameter]
    assert torch.equal(block.parameter, before)
    assert torch.count_nonzero(state["exp_avg"]) > 0 and torch.count_nonzero(state["exp_avg_sq"]) > 0
    assert optimizer.last_masks[block.name] == 0


def test_identity_mode_matches_dense_adamw_wsd():
    set_global_seed(17)
    ordinary = tiny_model()
    ordinary_opt = build_optimizer(ordinary, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False)
    ordinary_wsd = WarmupStableDecay(ordinary_opt, 6e-4, 6e-5, 100, 9156, 916)
    set_global_seed(17)
    magma, wrapped, magma_wsd = build(identity=True)
    for _ in range(3):
        update(ordinary, ordinary_opt, ordinary_wsd)
        update(magma, wrapped, magma_wsd)
    for expected, actual in zip(ordinary.parameters(), magma.parameters(), strict=True):
        assert torch.equal(expected, actual)
    equal_tree(ordinary_opt.state_dict(), wrapped.base_optimizer.state_dict())
    assert ordinary_wsd.state_dict() == magma_wsd.state_dict()


def test_alignment_ema_and_deterministic_bernoulli_diagnostic():
    aligned = cosine_alignment_score(torch.tensor([2.0, 0.0]), torch.tensor([3.0, 0.0]), 2.0)
    opposite = cosine_alignment_score(torch.tensor([2.0, 0.0]), torch.tensor([-3.0, 0.0]), 2.0)
    orthogonal = cosine_alignment_score(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 3.0]), 2.0)
    assert aligned == pytest.approx(float(torch.sigmoid(torch.tensor(0.5))), abs=1e-7)
    assert opposite == pytest.approx(float(torch.sigmoid(torch.tensor(-0.5))), abs=1e-7)
    assert orthogonal == pytest.approx(0.5, abs=1e-7) and opposite < orthogonal < aligned
    assert alignment_ema(alignment_ema(0.0, aligned, 0.9), opposite, 0.9) == pytest.approx(0.09 * aligned + 0.1 * opposite)
    _, optimizer, _ = build()
    frequency = sum(optimizer._draw_mask(optimizer.blocks[0].parameter) for _ in range(4096)) / 4096
    assert 0.45 < frequency < 0.55


def test_dedicated_generator_does_not_perturb_global_torch_rng():
    model = tiny_model()
    torch.manual_seed(12345)
    expected = torch.rand(8)
    torch.manual_seed(12345)
    optimizer = MagmaAdamW(build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8, fused=False), magma_blocks(model))
    for _ in range(32):
        optimizer._draw_mask(optimizer.blocks[0].parameter)
    assert torch.equal(torch.rand(8), expected)


def test_checkpoint_resume_reproduces_next_mask_alignment_optimizer_and_data(tmp_path):
    device, context = torch.device("cpu"), 512
    dataset = write_token_stream(tmp_path / "stream.uint16", (torch.arange(128 * context + 1) % 64).tolist(), 128 * context + 1, context_length=context)
    order = np.arange(128, dtype=np.uint32)
    set_global_seed(42)
    model, optimizer, schedule = build(context=context)
    state, zero = RunState(), torch.zeros((1, context), dtype=torch.long)
    train_smoke(model, dataset, None, zero, zero, optimizer, schedule, state, device, 64, 1, 1, 1.0, sequence_schedule=order)
    checkpoint = tmp_path / "magma.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, state, {"experiment_id": "EXP-016-M"}, data_cursor={"mechanism": "fixed_example_index_permutation", "next_schedule_cursor": state.next_sequence_index})
    next_expected = dataset.get_indexed_batch(order[64:128].tolist())
    train_smoke(model, dataset, None, zero, zero, optimizer, schedule, state, device, 64, 1, 1, 1.0, sequence_schedule=order)
    expected_masks, expected_alignment, expected_state = dict(optimizer.last_masks), dict(optimizer.alignment), optimizer.state_dict()
    set_global_seed(999)
    restored_model, restored_optimizer, restored_schedule = build(context=context)
    restored = load_checkpoint(checkpoint, restored_model, restored_optimizer, restored_schedule, device)
    next_actual = dataset.get_indexed_batch(order[restored.next_sequence_index:restored.next_sequence_index + 64].tolist())
    assert torch.equal(next_expected[0], next_actual[0]) and torch.equal(next_expected[1], next_actual[1])
    train_smoke(restored_model, dataset, None, zero, zero, restored_optimizer, restored_schedule, restored, device, 64, 1, 1, 1.0, sequence_schedule=order)
    assert restored_optimizer.last_masks == expected_masks and restored_optimizer.alignment == expected_alignment
    equal_tree(expected_state, restored_optimizer.state_dict())
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires qualified WSL CUDA")
def test_gpu_resume_reproduces_mask_and_alignment(tmp_path):
    device = torch.device("cuda")
    set_global_seed(42)
    model, optimizer, schedule = build(device)
    update(model, optimizer, schedule, device)
    checkpoint = tmp_path / "gpu.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, RunState(step=1, tokens=32768, next_sequence_index=64), {"experiment_id": "EXP-016-M"})
    update(model, optimizer, schedule, device)
    masks, alignment = dict(optimizer.last_masks), dict(optimizer.alignment)
    restored_model, restored_optimizer, restored_schedule = build(device)
    load_checkpoint(checkpoint, restored_model, restored_optimizer, restored_schedule, device)
    update(restored_model, restored_optimizer, restored_schedule, device)
    assert restored_optimizer.last_masks == masks and restored_optimizer.alignment == alignment
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_wsd_boundaries_remain_unchanged():
    _, optimizer, schedule = build()
    assert schedule.lr_at_step(100) == 6e-4 and schedule.lr_at_step(8240) == 6e-4
    assert schedule.lr_at_step(8241) < 6e-4 and schedule.lr_at_step(9156) == 6e-5
    assert all(group["lr"] == 0.0 for group in optimizer.param_groups)
