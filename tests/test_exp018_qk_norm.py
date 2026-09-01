"""Focused EXP-018 QK-Norm ablation contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import types

import torch

import gibc_llm.model as model_module
from gibc_llm.model import DecoderOnlyTransformer, parameter_breakdown
from gibc_llm.utils import load_config, set_global_seed


def _tiny_qk_model() -> DecoderOnlyTransformer:
    base = load_config(Path("configs/exp001.yaml")).model
    config = replace(
        base,
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        d_ff=32,
        rotary_dim=8,
        context_length=8,
        qk_norm=True,
        qk_norm_epsilon=1.0e-6,
    )
    set_global_seed(42)
    return DecoderOnlyTransformer(config)


def test_qk_norm_defaults_off_and_disabled_recipe_v3_is_state_dict_compatible() -> None:
    """Breaks if default-off QK-Norm changes preexisting Recipe-v3 parameters or deterministic outputs."""
    control = load_config(Path("configs/exp011.yaml"))
    assert control.model.qk_norm is False
    assert control.model.qk_norm_epsilon == 1.0e-6

    set_global_seed(42)
    first = DecoderOnlyTransformer(control.model).eval()
    set_global_seed(42)
    second = DecoderOnlyTransformer(replace(control.model, qk_norm=False)).eval()
    baseline_source = subprocess.check_output(["git", "show", "HEAD:src/gibc_llm/model.py"], text=True)
    baseline_source = baseline_source.replace("from .utils import ModelConfig", "from gibc_llm.utils import ModelConfig")
    baseline_module = types.ModuleType("exp018_head_model")
    sys.modules[baseline_module.__name__] = baseline_module
    exec(compile(baseline_source, "<exp018-head-model>", "exec"), baseline_module.__dict__)
    set_global_seed(42)
    baseline = baseline_module.DecoderOnlyTransformer(control.model).eval()
    assert parameter_breakdown(first).total == baseline_module.parameter_breakdown(baseline).total == 49_860_480
    assert list(first.state_dict()) == list(second.state_dict()) == list(baseline.state_dict())
    assert not any("qk_norm_gain" in name for name in first.state_dict())
    for current, historical in zip(first.parameters(), baseline.parameters(), strict=True):
        assert torch.equal(current, historical)
    ids = torch.arange(16, dtype=torch.long).reshape(2, 8) % control.model.vocab_size
    with torch.no_grad():
        assert torch.equal(first(ids), second(ids))
        assert torch.equal(first(ids), baseline(ids))


def test_exp018_config_enables_only_qk_norm_and_adds_exactly_nine_parameters() -> None:
    """Breaks if EXP-018 drifts from EXP-011 except for nine one-scalar-per-layer QK gains."""
    control = load_config(Path("configs/exp011.yaml"))
    candidate = load_config(Path("configs/exp018-qk-norm.yaml"))
    assert candidate.experiment_id == "EXP-018"
    assert candidate.model.qk_norm is True
    assert candidate.model.qk_norm_epsilon == 1.0e-6
    assert {key: value for key, value in candidate.model.__dict__.items() if key not in {"qk_norm", "qk_norm_epsilon"}} == {key: value for key, value in control.model.__dict__.items() if key not in {"qk_norm", "qk_norm_epsilon"}}
    assert candidate.training == control.training
    assert candidate.data == control.data
    assert candidate.mixture == control.mixture
    breakdown = parameter_breakdown(DecoderOnlyTransformer(candidate.model))
    assert breakdown.attention == 14_745_609
    assert breakdown.total == 49_860_489
    assert breakdown.total <= 50_000_000


def test_qk_norm_initialization_rms_dtype_and_gain_contract() -> None:
    """Breaks if Q/K are not FP32-RMS-normalized, recast, or initialized with one scalar gain per layer."""
    model = _tiny_qk_model()
    attention = model.blocks[0].attention
    assert attention.qk_norm_gain is not None
    assert attention.qk_norm_gain.shape == torch.Size([])
    assert attention.qk_norm_gain.item() == 1.0
    queries = torch.randn(2, 2, 3, 8, dtype=torch.bfloat16)
    keys = torch.randn(2, 2, 3, 8, dtype=torch.bfloat16)
    fp32_queries, fp32_keys = attention.rms_normalize_qk(queries, keys, attention.qk_norm_epsilon)
    assert fp32_queries.dtype == fp32_keys.dtype == torch.float32
    assert torch.allclose(fp32_queries.square().mean(dim=-1).sqrt(), torch.ones(2, 2, 3), atol=2e-5, rtol=2e-5)
    assert torch.allclose(fp32_keys.square().mean(dim=-1).sqrt(), torch.ones(2, 2, 3), atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        attention.qk_norm_gain.fill_(2.0)
    normalized_queries, normalized_keys = attention.normalize_qk(queries, keys)
    assert normalized_queries.dtype == normalized_keys.dtype == torch.bfloat16
    assert torch.equal(normalized_queries, (fp32_queries * 2.0).to(torch.bfloat16))
    assert torch.equal(normalized_keys, fp32_keys.to(torch.bfloat16))


def test_qk_norm_is_after_rope_before_sdpa_and_preserves_default_sdpa_scale(monkeypatch) -> None:
    """Breaks if QK-Norm moves before RoPE, reaches attention late, or changes SDPA scaling semantics."""
    model = _tiny_qk_model().eval()
    attention = model.blocks[0].attention
    events: list[str] = []
    original_rope = attention.rope.forward
    original_normalize = attention.normalize_qk
    original_sdpa = model_module.functional.scaled_dot_product_attention

    def rope_wrapper(queries, keys):
        events.append("rope")
        return original_rope(queries, keys)

    def normalize_wrapper(queries, keys):
        assert events == ["rope"]
        events.append("normalize")
        return original_normalize(queries, keys)

    def sdpa_wrapper(queries, keys, values, **kwargs):
        assert events == ["rope", "normalize"]
        assert "scale" not in kwargs
        events.append("sdpa")
        return original_sdpa(queries, keys, values, **kwargs)

    monkeypatch.setattr(attention.rope, "forward", rope_wrapper)
    monkeypatch.setattr(attention, "normalize_qk", normalize_wrapper)
    monkeypatch.setattr(model_module.functional, "scaled_dot_product_attention", sdpa_wrapper)
    hidden = torch.randn(2, 8, 16)
    assert attention(hidden).shape == (2, 8, 16)
    assert events == ["rope", "normalize", "sdpa"]


def test_qk_norm_gradients_are_finite_nonzero_and_checkpoint_round_trips(tmp_path: Path) -> None:
    """Breaks if QK gain is disconnected, unstable, or omitted from serialization."""
    first = _tiny_qk_model()
    inputs = torch.arange(16, dtype=torch.long).reshape(2, 8) % 64
    targets = (inputs + 1) % 64
    loss = first.loss(inputs, targets)
    loss.backward()
    for block in first.blocks:
        gradient = block.attention.qk_norm_gain.grad
        assert gradient is not None and torch.isfinite(gradient) and gradient.abs() > 0
    checkpoint = tmp_path / "qk-norm-state.pt"
    torch.save(first.state_dict(), checkpoint)
    set_global_seed(7)
    second = DecoderOnlyTransformer(first.config)
    second.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        assert torch.equal(first.eval()(inputs), second.eval()(inputs))
