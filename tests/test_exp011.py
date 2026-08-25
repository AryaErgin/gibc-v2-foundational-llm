"""EXP-011 controls: long-horizon Recipe v3 configuration and two-phase stream gates."""

from __future__ import annotations

from pathlib import Path

import hashlib
import pytest

from gibc_llm.full_run import assert_exp011_phase_capacity, assert_physical_batch_control, expected_artifact_sequences, expected_full_sequences, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp011_config_freezes_recipe_v3_and_exact_long_horizon() -> None:
    """Breaks if EXP-011 changes Recipe v3 or its 45,777-step schedule from step zero."""
    config = load_config(Path("configs/exp011.yaml"))

    assert config.experiment_id == "EXP-011"
    assert (config.model.vocab_size, config.model.d_model, config.model.n_layers, config.model.n_heads, config.model.head_dim, config.model.d_ff) == (
        8192,
        640,
        9,
        20,
        32,
        1728,
    )
    assert config.model.activation == "swiglu"
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 49_860_480
    assert config.training.full_schedule_steps == 45_777
    assert config.training.full_training_tokens == 1_500_020_736
    assert config.training.full_schedule_steps * config.training.effective_batch_tokens == config.training.full_training_tokens
    assert expected_full_sequences(config) == 2_929_728
    assert full_run_milestones(config) == (0, 9_156, 18_312, 27_468, 36_624, 45_777)
    assert config.mixture is not None
    assert config.mixture["target_prediction_tokens"] == {"fineweb": 1_000_013_824, "fineweb_edu": 500_006_912}
    assert_physical_batch_control(config, 32, 2)


def test_exp011_prefix_verifier_hard_fails_on_any_exp006_byte_difference(tmp_path: Path) -> None:
    """Breaks if a 1.5B extension can be authorized without exact 900M raw-byte identity."""
    from gibc_llm.exp011 import EXP006_PREFIX_BYTE_COUNT, verify_exp006_prefix

    stream = tmp_path / "stream.uint16"
    stream.write_bytes(b"abcdefghi")
    expected = hashlib.sha256(b"abcdef").hexdigest()

    assert verify_exp006_prefix(stream, byte_count=6, expected_sha256=expected) == expected
    with pytest.raises(RuntimeError, match="EXP-006 prefix SHA-256 mismatch"):
        verify_exp006_prefix(stream, byte_count=6, expected_sha256=hashlib.sha256(b"abcdeg").hexdigest())
    assert EXP006_PREFIX_BYTE_COUNT == 1_800_142_850


def test_exp011_phase_capacity_only_allows_exp006_before_900m_and_exp011_afterward() -> None:
    """Breaks if a short artifact can silently service the 1.5B phase, or an arbitrary artifact can resume it."""
    config = load_config(Path("configs/exp011.yaml"))

    assert_exp011_phase_capacity(config, "EXP-006", 900_071_424, planned_end_step=27_468)
    with pytest.raises(RuntimeError, match="only through step 27,468"):
        assert_exp011_phase_capacity(config, "EXP-006", 900_071_424, planned_end_step=27_469)
    assert_exp011_phase_capacity(config, "EXP-011", 1_500_020_736, planned_end_step=45_777)
    with pytest.raises(RuntimeError, match="requires either the validated EXP-006 900M artifact or EXP-011 1.5B artifact"):
        assert_exp011_phase_capacity(config, "EXP-004", 300_023_808, planned_end_step=1)


def test_exp011_900m_phase_uses_the_900m_stream_example_count() -> None:
    """Breaks if phase-one validation mistakenly compares an EXP-006 stream to the 1.5B total."""
    config = load_config(Path("configs/exp011.yaml"))

    assert expected_artifact_sequences(config, "EXP-006", 900_071_424) == 1_757_952
    assert expected_artifact_sequences(config, "EXP-011", 1_500_020_736) == 2_929_728
