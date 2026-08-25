"""EXP-008 SwiGLU ablation controls against the frozen EXP-007B recipe."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from gibc_llm.full_run import assert_physical_batch_control, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, FeedForward, SwiGLU, parameter_breakdown
from gibc_llm.train import CosineWithWarmup, RunState, build_optimizer, load_checkpoint, optimizer_update, save_checkpoint
from gibc_llm.utils import load_config, set_global_seed


def test_exp008a_has_the_independently_derived_near_cap_swiglu_count_and_only_mlp_changes() -> None:
    """Breaks if EXP-008A drifts frozen EXP-007B controls or exceeds the 50M cap."""
    control = load_config(Path("configs/exp007b.yaml"))
    candidate = load_config(Path("configs/exp008a.yaml"))

    assert candidate.experiment_id == "EXP-008A"
    assert candidate.training == control.training
    assert candidate.data == control.data
    assert candidate.mixture == control.mixture
    assert candidate.model == replace(control.model, activation="swiglu", d_ff=1728)
    assert parameter_breakdown(DecoderOnlyTransformer(control.model)).total == 49_491_840
    assert parameter_breakdown(DecoderOnlyTransformer(candidate.model)).total == 49_860_480
    assert parameter_breakdown(DecoderOnlyTransformer(candidate.model)).total <= 50_000_000
    assert full_run_milestones(candidate) == (0, 3_052, 6_104, 9_156)
    assert_physical_batch_control(candidate, 32, 2)


def test_swiglu_has_three_unbiased_projections_and_gelu_stays_backward_compatible() -> None:
    """Breaks if SwiGLU shapes/gating drift or GELU changes for the EXP-007B control."""
    control = load_config(Path("configs/exp007b.yaml"))
    candidate = load_config(Path("configs/exp008a.yaml"))
    gelu = DecoderOnlyTransformer(control.model)
    swiglu = DecoderOnlyTransformer(candidate.model)

    assert isinstance(gelu.blocks[0].mlp, FeedForward)
    assert isinstance(swiglu.blocks[0].mlp, SwiGLU)
    assert swiglu.blocks[0].mlp.value_proj.weight.shape == (1728, 640)
    assert swiglu.blocks[0].mlp.gate_proj.weight.shape == (1728, 640)
    assert swiglu.blocks[0].mlp.out_proj.weight.shape == (640, 1728)
    assert all(projection.bias is None for projection in (swiglu.blocks[0].mlp.value_proj, swiglu.blocks[0].mlp.gate_proj, swiglu.blocks[0].mlp.out_proj))

    inputs = torch.arange(8, dtype=torch.long).unsqueeze(0) % candidate.model.vocab_size
    targets = (inputs + 1) % candidate.model.vocab_size
    loss = swiglu.loss(inputs, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert swiglu.blocks[0].mlp.gate_proj.weight.grad is not None
    assert torch.isfinite(swiglu.blocks[0].mlp.gate_proj.weight.grad).all()


def test_swiglu_checkpoint_resume_preserves_next_update_on_a_small_structural_fixture(tmp_path: Path) -> None:
    """Breaks if the new three-projection MLP cannot checkpoint and resume safely."""
    candidate = load_config(Path("configs/exp008a.yaml"))
    tiny = replace(candidate.model, vocab_size=32, d_model=16, n_layers=1, n_heads=4, head_dim=4, rotary_dim=4, d_ff=12, context_length=8)
    set_global_seed(42)
    model = DecoderOnlyTransformer(tiny)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 9_156)
    inputs = torch.arange(8, dtype=torch.long).unsqueeze(0) % 32
    targets = (inputs + 1) % 32
    optimizer_update(model, optimizer, schedule, [(inputs, targets)], torch.device("cpu"), 1.0)
    expected = model(inputs).detach().clone()
    checkpoint = tmp_path / "swiglu.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, RunState(step=1, tokens=8), {"activation": "swiglu"})

    restored = DecoderOnlyTransformer(tiny)
    restored_optimizer = build_optimizer(restored, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    restored_schedule = CosineWithWarmup(restored_optimizer, 6e-4, 6e-5, 100, 9_156)
    restored_state = load_checkpoint(checkpoint, restored, restored_optimizer, restored_schedule, torch.device("cpu"))

    assert restored_state == RunState(step=1, tokens=8)
    assert torch.equal(expected, restored(inputs))
    optimizer_update(restored, restored_optimizer, restored_schedule, [(inputs, targets)], torch.device("cpu"), 1.0)
    assert restored_schedule.step_count == 2
