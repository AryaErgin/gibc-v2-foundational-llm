"""SYS-001 systems-ladder guards for the frozen EXP-007B recipe."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gibc_llm.model import RotaryEmbedding
from gibc_llm.sys001 import assert_sys001_controls, classify_sdpa_backend, sys001_phase_plan
from gibc_llm.utils import load_config


def test_sys001_phase_plan_is_a_fresh_sequential_ladder_with_only_the_authorized_changes() -> None:
    """Breaks if SYS-001 changes a scientific control or merges optimization variables."""
    config = load_config(Path("configs/exp007b.yaml"))
    assert_sys001_controls(config)

    phases = sys001_phase_plan()
    assert [(phase.identifier, phase.microbatch_sequences, phase.accumulation_steps) for phase in phases] == [
        ("baseline_32x2", 32, 2),
        ("microbatch_64x1", 64, 1),
        ("production_timing", 64, 1),
        ("rope_cache", 64, 1),
        ("sdpa_backend_audit", 64, 1),
        ("fused_adamw", 64, 1),
    ]
    assert phases[0].synchronize_each_update is True
    assert phases[1].synchronize_each_update is True
    assert phases[2].synchronize_each_update is False
    assert phases[3].rope_cache is True
    assert phases[4].audit_sdpa_backend is True
    assert phases[5].fused_adamw is True
    assert all(phase.context_length == 512 for phase in phases)
    assert all(phase.effective_batch_tokens == 32_768 for phase in phases)
    assert all(phase.parameter_count == 49_491_840 for phase in phases)

    changed = replace(config, model=replace(config.model, d_model=641))
    with pytest.raises(RuntimeError, match="d_model"):
        assert_sys001_controls(changed)


def test_rope_cache_reuses_the_same_tables_without_changing_values_or_parameters() -> None:
    """Breaks if cached RoPE changes the rotation or adds a trainable tensor."""
    rope = RotaryEmbedding(head_dim=32, theta=10_000.0)
    queries = torch.randn(2, 3, 8, 32)
    keys = torch.randn(2, 3, 8, 32)

    first_queries, first_keys = rope(queries, keys)
    first_cache = rope.cache_identity
    second_queries, second_keys = rope(queries, keys)

    assert first_cache == rope.cache_identity
    assert torch.equal(first_queries, second_queries)
    assert torch.equal(first_keys, second_keys)
    assert sum(parameter.numel() for parameter in rope.parameters()) == 0


@pytest.mark.parametrize(
    ("operator_keys", "expected"),
    [
        (("aten::_scaled_dot_product_flash_attention_for_cuda",), "flash"),
        (("aten::_scaled_dot_product_efficient_attention",), "memory_efficient"),
        (("aten::_scaled_dot_product_attention_math",), "math"),
        (("aten::matmul",), "unknown"),
    ],
)
def test_sdpa_backend_audit_classifies_actual_profiler_operators(operator_keys: tuple[str, ...], expected: str) -> None:
    """Breaks if SYS-001 reports enabled flags instead of the executed SDPA backend."""
    assert classify_sdpa_backend(operator_keys) == expected
