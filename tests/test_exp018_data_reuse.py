"""EXP-018 accepts only the independently verified EXP-011 prefix of EXP-012."""

from __future__ import annotations

from pathlib import Path

import pytest

from gibc_llm.full_run import (
    EXP011_PREDICTION_TOKENS,
    EXP012_PREDICTION_TOKENS,
    assert_exp011_phase_capacity,
    expected_artifact_sequences,
)
from gibc_llm.utils import load_config


def test_exp018_prefix_fallback_is_exactly_bounded_and_cannot_expand_horizon() -> None:
    config = load_config(Path("configs/exp018-qk-norm.yaml"))
    assert expected_artifact_sequences(config, "EXP-012", EXP012_PREDICTION_TOKENS) == 4_687_488
    assert_exp011_phase_capacity(config, "EXP-012", EXP011_PREDICTION_TOKENS, planned_end_step=45_777)
    with pytest.raises(RuntimeError):
        expected_artifact_sequences(config, "EXP-012", EXP011_PREDICTION_TOKENS)
    with pytest.raises(RuntimeError):
        assert_exp011_phase_capacity(config, "EXP-012", EXP012_PREDICTION_TOKENS, planned_end_step=45_777)
    with pytest.raises(RuntimeError):
        assert_exp011_phase_capacity(config, "EXP-012", EXP011_PREDICTION_TOKENS, planned_end_step=45_778)
