"""EXP-020 final 7.2B cosine configuration and provenance contracts."""

from __future__ import annotations

from pathlib import Path
import math

import torch

import pytest

from gibc_llm.full_run import expected_run_state, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config


def test_exp020_final_config_freezes_promoted_adamw_cosine_controls() -> None:
    config = load_config(Path("configs/exp020-final-7p2b-cosine.yaml"))

    assert config.experiment_id == "EXP-020"
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 49_860_480
    assert config.model.qk_norm is False
    assert config.training.cautious_weight_decay is False
    assert config.training.optimizer == "adamw"
    assert config.training.schedule == "cosine_decay"
    assert (config.training.beta1, config.training.beta2, config.training.eps, config.training.weight_decay) == (0.9, 0.95, 1e-8, 0.1)
    assert (config.training.peak_learning_rate, config.training.min_learning_rate, config.training.warmup_steps) == (6e-4, 6e-5, 100)
    assert (config.training.default_microbatch_sequences, config.training.default_gradient_accumulation_steps) == (32, 2)
    assert config.training.effective_batch_tokens == 32_768
    assert config.training.seed == 42
    assert config.training.full_schedule_steps == 219_726
    assert config.training.full_training_tokens == 7_199_981_568
    assert config.training.full_training_tokens == config.training.full_schedule_steps * config.training.effective_batch_tokens
    assert config.mixture["target_prediction_tokens"] == {"fineweb": 4_799_987_712, "fineweb_edu": 2_399_993_856}


def test_exp020_milestones_and_cursor_are_exact() -> None:
    config = load_config(Path("configs/exp020-final-7p2b-cosine.yaml"))
    milestones = (0, 45_777, 73_242, 91_553, 109_863, 128_174, 146_484, 164_795, 183_105, 201_416, 219_726)

    assert full_run_milestones(config) == milestones
    assert tuple(step * 32_768 for step in milestones) == (
        0, 1_500_020_736, 2_399_993_856, 3_000_008_704, 3_599_990_784,
        4_200_005_632, 4_799_987_712, 5_400_002_560, 5_999_984_640,
        6_599_999_488, 7_199_981_568,
    )
    assert expected_run_state(config, 0, 219_726) == (219_726, 7_199_981_568, 14_062_464)


def test_exp020_prefix_provenance_rejects_missing_exp012_chain(tmp_path: Path) -> None:
    from gibc_llm.exp020 import assert_exp020_prefix_provenance

    stream = tmp_path / "stream.uint16"
    stream.write_bytes(b"\x00" * 4)
    with pytest.raises(RuntimeError, match="EXP-020 manifest lacks"):
        assert_exp020_prefix_provenance({}, stream)


def test_exp020_prefix_provenance_rehashes_inherited_exp011_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import gibc_llm.exp020 as exp020

    stream = tmp_path / "stream.uint16"
    stream.write_bytes(b"\x00" * 4)
    manifest = {
        "frozen_exp012_source": {
            "manifest_sha256": exp020.EXP012_MANIFEST_SHA256,
            "stream_sha256": exp020.EXP012_STREAM_SHA256,
            "stored_token_ids": exp020.EXP012_STORED_TOKEN_IDS,
            "prediction_tokens": exp020.EXP012_PREDICTION_TOKENS,
        },
        "exp012_prefix": {
            "byte_count": exp020.EXP012_PREFIX_BYTE_COUNT,
            "expected_sha256": exp020.EXP012_STREAM_SHA256,
            "observed_sha256": exp020.EXP012_STREAM_SHA256,
            "prefix_match": True,
        },
        "exp011_prefix": {
            "byte_count": exp020.EXP011_PREFIX_BYTE_COUNT,
            "expected_sha256": exp020.EXP011_STREAM_SHA256,
            "observed_sha256": exp020.EXP011_STREAM_SHA256,
            "prefix_match": True,
        },
    }

    def observed_prefix(_: Path, byte_count: int) -> str:
        return exp020.EXP012_STREAM_SHA256 if byte_count == exp020.EXP012_PREFIX_BYTE_COUNT else "wrong"

    monkeypatch.setattr(exp020, "sha256_file_prefix", observed_prefix)
    with pytest.raises(RuntimeError, match="EXP-020 manifest lacks the inherited verified EXP-011"):
        exp020.assert_exp020_prefix_provenance(manifest, stream)


def test_exp020_cosine_is_explicitly_parameterized_by_full_219726_step_horizon() -> None:
    from gibc_llm.train import CosineWithWarmup

    config = load_config(Path("configs/exp020-final-7p2b-cosine.yaml"))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    schedule = CosineWithWarmup(
        optimizer,
        config.training.peak_learning_rate,
        config.training.min_learning_rate,
        config.training.warmup_steps,
        config.training.full_schedule_steps,
    )
    assert schedule.total_steps == 219_726
    assert schedule.lr_at_step(0) == 0.0
    assert schedule.lr_at_step(100) == pytest.approx(6e-4)
    for step in (45_777, 73_242, 146_484, 183_105):
        progress = (step - 100) / (219_726 - 100)
        expected = 6e-5 + 0.5 * (6e-4 - 6e-5) * (1.0 + math.cos(math.pi * progress))
        assert schedule.lr_at_step(step) == pytest.approx(expected)
    assert schedule.lr_at_step(219_726) == pytest.approx(6e-5)
