"""Focused, single-threadable EXP-017A preflight checks; no training or benchmark work."""

from pathlib import Path

import pytest

from gibc_llm.full_run import expected_run_state, full_run_milestones
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import WarmupStableDecay, build_optimizer
from gibc_llm.utils import load_config


def test_exp017a_config_and_2_4b_wsd_semantics() -> None:
    config = load_config(Path("configs/exp017a-wsd.yaml"))
    cosine = load_config(Path("configs/exp012.yaml"))
    assert config.experiment_id == "EXP-017A"
    assert config.model == cosine.model
    assert config.data == cosine.data
    assert config.mixture == cosine.mixture
    assert config.training.seed == cosine.training.seed == 42
    assert config.training.effective_batch_tokens == cosine.training.effective_batch_tokens == 32_768
    assert (config.training.seed, config.training.full_schedule_steps, config.training.full_training_tokens) == (42, 73_242, 2_399_993_856)
    assert (config.training.schedule, config.training.warmup_steps, config.training.cooldown_steps) == ("warmup_stable_decay", 100, 7_324)
    assert parameter_breakdown(DecoderOnlyTransformer(config.model)).total == 49_860_480
    assert expected_run_state(config, 0, 73_242) == (73_242, 2_399_993_856, 4_687_488)
    assert full_run_milestones(config) == (0, 9_156, 18_312, 27_468, 36_624, 45_780, 54_936, 64_092, 73_242)


def test_exp017a_scheduler_boundary_and_stable_trunk_cursor() -> None:
    config = load_config(Path("configs/exp017a-wsd.yaml"))
    model = DecoderOnlyTransformer(config.model)
    optimizer = build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)
    schedule = WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 73_242, 7_324)
    assert schedule.stable_end_step == 65_918
    assert schedule.lr_at_step(65_918) == 6e-4
    assert schedule.lr_at_step(65_919) < 6e-4
    assert schedule.lr_at_step(65_919) > 6e-5
    assert schedule.lr_at_step(73_242) == 6e-5
    assert (65_918 * 64, 65_918 * 32_768) == (4_218_752, 2_160_001_024)
