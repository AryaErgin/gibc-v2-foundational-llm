"""EXP-019 matched-CWD configuration and bounded data-reuse contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gibc_llm.full_run import (
    EXP011_PREDICTION_TOKENS,
    EXP012_PREDICTION_TOKENS,
    assert_exp011_phase_capacity,
    expected_artifact_sequences,
    full_run_milestones,
)
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp019_matches_exp011_except_for_explicit_cwd() -> None:
    """Breaks if EXP-019 alters any scientific control other than CWD."""
    control = load_config(Path("configs/exp011.yaml"))
    candidate = load_config(Path("configs/exp019-cwd.yaml"))
    assert candidate.experiment_id == "EXP-019"
    assert candidate.model == control.model
    assert candidate.model.qk_norm is False
    candidate_training, control_training = asdict(candidate.training), asdict(control.training)
    candidate_training.pop("cautious_weight_decay")
    control_training.pop("cautious_weight_decay")
    assert candidate_training == control_training
    assert candidate.training.cautious_weight_decay is True
    assert candidate.data == control.data
    assert candidate.mixture == control.mixture
    assert parameter_breakdown(DecoderOnlyTransformer(candidate.model)).total == 49_860_480


def test_exp019_accepts_exact_exp012_source_but_exposes_only_exp011_prefix() -> None:
    """Breaks if the 2.4B source can silently extend the 1.5B training cursor."""
    candidate = load_config(Path("configs/exp019-cwd.yaml"))
    assert full_run_milestones(candidate) == (0, 9_156, 18_312, 27_468, 36_624, 45_777)
    assert expected_artifact_sequences(candidate, "EXP-012", EXP012_PREDICTION_TOKENS) == EXP012_PREDICTION_TOKENS // 512
    assert_exp011_phase_capacity(candidate, "EXP-012", EXP011_PREDICTION_TOKENS, 45_777)
