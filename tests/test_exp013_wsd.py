"""EXP-013 scheduler and fixed-control invariants; no benchmark code is invoked."""

from pathlib import Path

import pytest
import torch

from gibc_llm.full_run import expected_run_state, full_run_milestones
from gibc_llm.exp013 import classify
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.train import CosineWithWarmup, RunState, WarmupStableDecay, build_optimizer, load_checkpoint, save_checkpoint
from gibc_llm.utils import load_config, set_global_seed


def _optimizer() -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer]:
    config = load_config(Path("configs/exp001.yaml")).model
    # The scheduler is model-size independent; use the approved model only for
    # its real optimizer/checkpoint interface, never for a benchmark.
    model = DecoderOnlyTransformer(config)
    return model, build_optimizer(model, 6e-4, 0.1, (0.9, 0.95), 1e-8)


def _wsd() -> tuple[DecoderOnlyTransformer, torch.optim.Optimizer, WarmupStableDecay]:
    model, optimizer = _optimizer()
    return model, optimizer, WarmupStableDecay(optimizer, 6e-4, 6e-5, 100, 9156, 916)


def test_existing_cosine_control_golden_values_are_unchanged() -> None:
    """Guard the established 9,156-step control schedule against EXP-013 changes."""
    _, optimizer = _optimizer()
    schedule = CosineWithWarmup(optimizer, 6e-4, 6e-5, 100, 9156)

    assert schedule.lr_at_step(0) == 0.0
    assert schedule.lr_at_step(100) == 6e-4
    assert schedule.lr_at_step(101) == pytest.approx(0.0005999999837534675, abs=0.0)
    assert schedule.lr_at_step(3052) == pytest.approx(0.00047037116601798384, abs=0.0)
    assert schedule.lr_at_step(6104) == pytest.approx(0.00019771284042821013, abs=0.0)
    assert schedule.lr_at_step(8240) == pytest.approx(0.00007351742948488332, abs=0.0)
    assert schedule.lr_at_step(9156) == 6e-5


def test_wsd_warmup_stable_cooldown_and_final_lr() -> None:
    _, _, schedule = _wsd()

    assert schedule.stable_end_step == 8240
    assert schedule.lr_at_step(0) == 0.0
    assert schedule.lr_at_step(1) == pytest.approx(6e-6)
    assert schedule.lr_at_step(100) == 6e-4
    assert schedule.lr_at_step(101) == 6e-4
    assert schedule.lr_at_step(8240) == 6e-4
    assert schedule.lr_at_step(8241) < 6e-4
    assert schedule.lr_at_step(8241) > 6e-5
    assert schedule.lr_at_step(9156) == 6e-5


def test_wsd_step_accounting_and_resume_at_stable_checkpoint(tmp_path: Path) -> None:
    """The stored WSD state must resume the exact pre-cooldown update boundary."""
    set_global_seed(42)
    model, optimizer, schedule = _wsd()
    for _ in range(8240):
        schedule.step()
    state = RunState(step=8240, tokens=8240 * 32768, next_sequence_index=8240 * 64)
    checkpoint = tmp_path / "checkpoint-step-8240.pt"
    save_checkpoint(checkpoint, model, optimizer, schedule, state, {"experiment_id": "EXP-013-W"})

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(payload) == {"model", "optimizer", "schedule", "run_state", "data_cursor", "rng", "config"}
    assert payload["run_state"] == {"step": 8240, "tokens": 8240 * 32768, "next_sequence_index": 8240 * 64}
    assert payload["data_cursor"] == {"next_sequence_index": 8240 * 64, "mechanism": "sequential_example_index"}
    assert payload["schedule"] == {
        "type": "warmup_stable_decay",
        "step_count": 8240,
        "warmup_steps": 100,
        "total_steps": 9156,
        "cooldown_steps": 916,
    }

    resumed_model, resumed_optimizer, resumed_schedule = _wsd()
    resumed = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, resumed_schedule, torch.device("cpu"))
    assert resumed == state
    assert resumed_schedule.step_count == 8240
    assert resumed_schedule.step() == resumed_schedule.lr_at_step(8241)
    assert resumed_schedule.lr_at_step(8241) < 6e-4


def test_exp013_configs_preserve_all_non_scheduler_controls() -> None:
    cosine = load_config(Path("configs/exp013-cosine.yaml"))
    wsd = load_config(Path("configs/exp013-wsd.yaml"))

    assert cosine.experiment_id == "EXP-013-C"
    assert wsd.experiment_id == "EXP-013-W"
    assert cosine.model == wsd.model
    assert cosine.data == wsd.data
    assert cosine.mixture == wsd.mixture
    assert cosine.training.schedule == "cosine_decay"
    assert cosine.training.cooldown_steps is None
    assert wsd.training.schedule == "warmup_stable_decay"
    assert wsd.training.cooldown_steps == 916
    assert full_run_milestones(cosine) == full_run_milestones(wsd) == (0, 3052, 6104, 9156)
    assert expected_run_state(wsd, 0, 9156) == (9156, 300023808, 585984)


def test_exp013_parameter_count_is_exact() -> None:
    config = load_config(Path("configs/exp013-wsd.yaml"))
    model = DecoderOnlyTransformer(config.model)
    assert parameter_breakdown(model).total == 49_860_480


def test_exp013_predeclared_decision_boundaries_and_regression_guard() -> None:
    assert classify(4.0, 4.0, 3.98, 3.98)["classification"] == "CAPABILITY WIN"
    assert classify(4.0, 4.0, 4.009, 4.009)["classification"] == "PERFORMANCE TIE"
    assert classify(4.0, 4.0, 4.011, 4.011)["classification"] == "REJECT WSD"
    guarded = classify(4.0, 4.0, 4.021, 3.939)
    assert guarded["delta_combined"] < -0.020
    assert guarded["classification"] == "PERFORMANCE TIE"
    assert guarded["individual_regression_over_0_020"] is True
