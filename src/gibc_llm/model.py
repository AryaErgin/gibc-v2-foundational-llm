"""Transparent decoder-only Transformer for the fixed EXP-001 control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .utils import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm with one learned scale and no bias or mean subtraction."""

    def __init__(self, dimension: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, values: Tensor) -> Tensor:
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values * inverse_rms * self.scale


class RotaryEmbedding(nn.Module):
    """Full-head, parameter-free canonical adjacent-pair RoPE."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_enabled = False
        self._cached_cosine: Tensor | None = None
        self._cached_sine: Tensor | None = None
        self._cached_key: tuple[str, int | None, torch.dtype] | None = None

    @property
    def cache_identity(self) -> tuple[int, int] | None:
        """Expose cache reuse for SYS-001 without serializing non-trainable tables."""
        if self._cached_cosine is None or self._cached_sine is None:
            return None
        return (id(self._cached_cosine), id(self._cached_sine))

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or clear the non-persistent RoPE lookup cache for a systems-only phase."""
        self._cache_enabled = enabled
        if not enabled:
            self._cached_cosine = None
            self._cached_sine = None
            self._cached_key = None

    def _cos_sin(self, sequence_length: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cache_key = (device.type, device.index, dtype)
        if (
            self._cache_enabled
            and self._cached_key == cache_key
            and self._cached_cosine is not None
            and self._cached_sine is not None
            and self._cached_cosine.shape[-2] >= sequence_length
        ):
            return self._cached_cosine[:, :, :sequence_length], self._cached_sine[:, :, :sequence_length]
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        cosine, sine = angles.cos().to(dtype=dtype)[None, None, :, :], angles.sin().to(dtype=dtype)[None, None, :, :]
        if self._cache_enabled:
            self._cached_key = cache_key
            self._cached_cosine = cosine
            self._cached_sine = sine
        return cosine, sine

    @staticmethod
    def _rotate(values: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
        pairs = values.reshape(*values.shape[:-1], -1, 2)
        even, odd = pairs.unbind(dim=-1)
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(start_dim=-2)

    def forward(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        if queries.shape != keys.shape or queries.shape[-1] != self.inv_freq.numel() * 2:
            raise ValueError("RoPE expects matching Q/K tensors with the configured full head dimension.")
        cosine, sine = self._cos_sin(queries.shape[-2], queries.device, queries.dtype)
        return self._rotate(queries, cosine, sine), self._rotate(keys, cosine, sine)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.rotary_dim, config.rope_theta)

    def project_qkv(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, sequence, _ = hidden.shape

        def split_heads(values: Tensor) -> Tensor:
            return values.view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.q_proj(hidden))
        keys = split_heads(self.k_proj(hidden))
        values = split_heads(self.v_proj(hidden))
        queries, keys = self.rope(queries, keys)
        return queries, keys, values

    def forward(self, hidden: Tensor) -> Tensor:
        queries, keys, values = self.project_qkv(hidden)
        attended = functional.scaled_dot_product_attention(queries, keys, values, dropout_p=0.0, is_causal=True)
        batch, _, sequence, _ = attended.shape
        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        return self.o_proj(merged)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.fc2(functional.gelu(self.fc1(hidden), approximate="none"))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.norm1(hidden))
        return hidden + self.mlp(self.norm2(hidden))


@dataclass(frozen=True)
class ParameterBreakdown:
    embedding: int
    attention: int
    mlp: int
    norms: int
    output_head_additional: int
    total: int


class DecoderOnlyTransformer(nn.Module):
    """The exact 8,392,960-trainable-parameter EXP-001 model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

    @property
    def output_weight(self) -> nn.Parameter:
        """The LM projection's only weight: the input embedding storage."""
        return self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds the approved context length.")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return functional.linear(hidden, self.output_weight)

    def loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        if input_ids.shape != target_ids.shape:
            raise ValueError("Input and next-token target tensors must have matching [B, 512] shape.")
        logits = self(input_ids)
        return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))

    @staticmethod
    def next_token_example(stream: Tensor, context_length: int) -> tuple[Tensor, Tensor]:
        if stream.ndim != 1 or stream.numel() < context_length + 1:
            raise ValueError("A next-token example requires at least context_length + 1 stream tokens.")
        return stream[:context_length].clone(), stream[1 : context_length + 1].clone()


def parameter_breakdown(model: DecoderOnlyTransformer) -> ParameterBreakdown:
    attention = sum(parameter.numel() for block in model.blocks for parameter in block.attention.parameters())
    mlp = sum(parameter.numel() for block in model.blocks for parameter in block.mlp.parameters())
    norms = sum(parameter.numel() for module in model.modules() if isinstance(module, RMSNorm) for parameter in module.parameters())
    embedding = model.token_embedding.weight.numel()
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ParameterBreakdown(
        embedding=embedding,
        attention=attention,
        mlp=mlp,
        norms=norms,
        output_head_additional=0,
        total=total,
    )
