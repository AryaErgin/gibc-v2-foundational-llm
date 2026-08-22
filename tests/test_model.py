from pathlib import Path

import torch

from gibc_llm.model import (
    DecoderOnlyTransformer,
    RotaryEmbedding,
    parameter_breakdown,
)
from gibc_llm.utils import load_config, set_global_seed


def _model() -> DecoderOnlyTransformer:
    set_global_seed(42)
    return DecoderOnlyTransformer(load_config(Path("configs/exp001.yaml")).model)


def test_parameter_count_matches_approved_breakdown() -> None:
    """Breaks if a matrix, norm scale, tie, or dimensions drift."""
    breakdown = parameter_breakdown(_model())

    assert breakdown.embedding == 2_097_152
    assert breakdown.attention == 2_097_152
    assert breakdown.mlp == 4_194_304
    assert breakdown.norms == 4_352
    assert breakdown.output_head_additional == 0
    assert breakdown.total == 8_392_960


def test_output_projection_uses_embedding_weight_storage() -> None:
    """Breaks if a separate LM-head parameter is introduced."""
    model = _model()

    assert model.token_embedding.weight.data_ptr() == model.output_weight.data_ptr()
    assert model.token_embedding.weight is model.output_weight


def test_future_tokens_cannot_change_logits_at_or_before_prefix() -> None:
    """Breaks if the attention mask stops being causal."""
    model = _model().eval()
    first = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    second = torch.tensor([[3, 4, 5, 6, 7, 8, 111, 222]], dtype=torch.long)

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    assert torch.allclose(first_logits[:, :6], second_logits[:, :6], atol=1e-5, rtol=1e-5)


def test_next_token_example_has_512_predictions_not_511() -> None:
    """Breaks if input tokens are trained against themselves or one target is dropped."""
    stream = torch.arange(513, dtype=torch.long)
    inputs, targets = DecoderOnlyTransformer.next_token_example(stream, context_length=512)

    assert inputs.shape == targets.shape == (512,)
    assert torch.equal(inputs[:4], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets[:4], torch.tensor([1, 2, 3, 4]))
    assert targets[-1].item() == 512


def test_attention_and_residual_shapes_are_explicit() -> None:
    """Breaks if QKV/head/attention/MLP tensor layout changes incorrectly."""
    model = _model().eval()
    ids = torch.randint(0, 8192, (2, 7), dtype=torch.long)
    hidden = model.token_embedding(ids)
    attention = model.blocks[0].attention

    q, k, v = attention.project_qkv(hidden)
    assert hidden.shape == (2, 7, 256)
    assert q.shape == k.shape == v.shape == (2, 8, 7, 32)
    assert attention(hidden).shape == (2, 7, 256)
    assert model.blocks[0](hidden).shape == (2, 7, 256)
    assert model(ids).shape == (2, 7, 8192)


def test_rope_preserves_shape_norm_and_has_no_trainable_parameters() -> None:
    """Breaks if RoPE becomes trainable or stops being an orthogonal pair rotation."""
    rope = RotaryEmbedding(head_dim=32, theta=10000.0)
    q = torch.randn(2, 8, 5, 32)
    k = torch.randn(2, 8, 5, 32)

    rotated_q, rotated_k = rope(q, k)

    assert rotated_q.shape == q.shape
    assert rotated_k.shape == k.shape
    assert torch.allclose(rotated_q.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert list(rope.parameters()) == []
