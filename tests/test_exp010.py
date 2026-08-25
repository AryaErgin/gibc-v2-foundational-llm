"""EXP-010 SwiGLU depth/width allocation controls."""

from dataclasses import replace
from pathlib import Path

from gibc_llm.full_run import assert_physical_batch_control, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp010_candidate_changes_only_the_predeclared_swiglu_depth_width_allocation() -> None:
    """Breaks if EXP-010 mutates a frozen control or its exact allocation."""
    control = load_config(Path("configs/exp008a.yaml"))
    candidate = load_config(Path("configs/exp010a.yaml"))

    assert candidate.experiment_id == "EXP-010A"
    assert candidate.data == control.data
    assert candidate.mixture == control.mixture
    assert candidate.training == control.training
    assert candidate.model == replace(control.model, d_model=608, n_layers=10, n_heads=19, d_ff=1656)
    assert parameter_breakdown(DecoderOnlyTransformer(candidate.model)).total == 49_985_504
    assert full_run_milestones(candidate) == (0, 3_052, 6_104, 9_156)
    assert_physical_batch_control(candidate, 32, 2)


def test_exp010_candidate_preserves_the_full_horizon_and_fixed_batch_controls() -> None:
    """Breaks if a proxy horizon, batch, warmup, or LR is substituted."""
    candidate = load_config(Path("configs/exp010a.yaml"))

    assert candidate.training.full_schedule_steps == 9_156
    assert candidate.training.full_training_tokens == 300_023_808
    assert candidate.training.warmup_steps == 100
    assert (candidate.training.peak_learning_rate, candidate.training.min_learning_rate) == (6.0e-4, 6.0e-5)
    assert (candidate.training.default_microbatch_sequences, candidate.training.default_gradient_accumulation_steps) == (32, 2)
