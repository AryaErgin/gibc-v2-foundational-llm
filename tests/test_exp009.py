"""EXP-009 learning-rate calibration controls for frozen Near-Cap Recipe v3."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gibc_llm.full_run import assert_physical_batch_control, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp009_candidates_change_only_the_predeclared_learning_rate_amplitude() -> None:
    """Breaks if EXP-009 mutates Recipe v3, data order, batch, or cosine shape."""
    control = load_config(Path("configs/exp008a.yaml"))
    low = load_config(Path("configs/exp009a.yaml"))
    high = load_config(Path("configs/exp009b.yaml"))

    assert low.experiment_id == "EXP-009A"
    assert high.experiment_id == "EXP-009B"
    assert low.model == high.model == control.model
    assert low.data == high.data == control.data
    assert low.mixture == high.mixture == control.mixture
    assert low.training == replace(control.training, peak_learning_rate=4.0e-4, min_learning_rate=4.0e-5)
    assert high.training == replace(control.training, peak_learning_rate=8.0e-4, min_learning_rate=8.0e-5)
    for candidate in (low, high):
        assert parameter_breakdown(DecoderOnlyTransformer(candidate.model)).total == 49_860_480
        assert candidate.training.peak_learning_rate / candidate.training.min_learning_rate == 10.0
        assert full_run_milestones(candidate) == (0, 3_052, 6_104, 9_156)
        assert_physical_batch_control(candidate, 32, 2)


@pytest.mark.parametrize("path", (Path("configs/exp009a.yaml"), Path("configs/exp009b.yaml")))
def test_exp009_schedule_remains_the_full_9156_update_cosine_with_100_step_warmup(path: Path) -> None:
    """Breaks if a candidate uses a proxy horizon or changes the predeclared warmup/cosine controls."""
    config = load_config(path)
    assert config.training.warmup_steps == 100
    assert config.training.full_schedule_steps == 9_156
    assert config.training.full_training_tokens == 300_023_808
    assert config.training.effective_batch_tokens == 32_768
