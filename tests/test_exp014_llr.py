"""EXP-014 preflight mechanics only; no data preparation or benchmark code."""
from pathlib import Path

import pytest
import torch

from gibc_llm.llr import HTSRLLR, LLRSettings, build_llr_optimizer, pl_alpha_hill
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import RunState, WarmupStableDecay, load_checkpoint, save_checkpoint
from gibc_llm.utils import load_config


def _setup():
    config = load_config(Path("configs/exp014-llr.yaml"))
    model = DecoderOnlyTransformer(config.model)
    optimizer = build_llr_optimizer(model, 6e-4, .1, (.9, .95), 1e-8, fused=False)
    schedule = WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 9156, 916)
    controller = HTSRLLR(model, optimizer)
    return config, model, optimizer, schedule, controller


def test_exp014_config_and_parameter_invariants():
    config, model, optimizer, _, _ = _setup()
    assert config.training.seed == 42
    assert parameter_breakdown(model).total == 49_860_480
    assert sum(group["llr_kind"] == "embedding_output" for group in optimizer.param_groups) == 1
    flattened = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(flattened) == len(set(flattened)) == len(list(model.parameters()))


def test_hill_known_diagonal_matrix():
    diagonal = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0]))
    expected = 1.0 + 2.0 / torch.log(torch.tensor([64.0 / 4.0, 16.0 / 4.0])).sum()
    assert pl_alpha_hill(diagonal) == pytest.approx(float(expected), abs=1e-7)


def test_cadence_soft_switch_freeze_and_wsd_interaction():
    _, _, optimizer, schedule, controller = _setup()
    for step in range(1, 101):
        global_lr = schedule.step()
        controller.step(step, global_lr)
    assert controller.last_recompute_step == 100
    assert all(1.0 <= value <= 5.0 for value in controller.target.values())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(schedule.lr_at_step(100) * controller.current[optimizer.param_groups[0]["llr_name"]])
    for step in range(101, 1851):
        controller.step(step, schedule.lr_at_step(step))
    assert controller.frozen is True
    frozen = dict(controller.current)
    controller.step(8241, schedule.lr_at_step(8241))
    assert controller.current == frozen
    assert all(group["lr"] == pytest.approx(schedule.lr_at_step(8241) * frozen[group["llr_name"]]) for group in optimizer.param_groups)
    assert schedule.lr_at_step(9156) == 6e-5


def test_s_equals_one_is_a_wsd_golden_control():
    _, _, optimizer, schedule, controller = _setup()
    controller = HTSRLLR(controller.model, optimizer, LLRSettings(min_multiplier=1.0, max_multiplier=1.0))
    for step in (1, 100, 101, 8240, 8241, 9156):
        global_lr = schedule.lr_at_step(step)
        controller.step(step, global_lr)
        assert all(group["lr"] == global_lr for group in optimizer.param_groups)


def test_controller_checkpoint_resume(tmp_path):
    _, model, optimizer, schedule, controller = _setup()
    for step in range(1, 101):
        controller.step(step, schedule.step())
    state = RunState(step=100, tokens=100 * 32768, next_sequence_index=6400)
    checkpoint = tmp_path / "llr.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, state, {"experiment_id": "EXP-014"}, lr_controller=controller)
    _, restored_model, restored_optimizer, restored_schedule, restored_controller = _setup()
    restored = load_checkpoint(checkpoint, restored_model, restored_optimizer, restored_schedule, torch.device("cpu"), lr_controller=restored_controller)
    assert restored == state
    assert restored_controller.state_dict() == controller.state_dict()
    assert [g["llr_name"] for g in restored_optimizer.param_groups] == [g["llr_name"] for g in optimizer.param_groups]
