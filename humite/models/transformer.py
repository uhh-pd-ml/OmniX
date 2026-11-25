"""Decoder-only transformer: pre-norm blocks, rotary positions, one shared key/value head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransformerConfig:
    embedding_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: float = 4.0


@dataclass
class KeyValueCache:
    """Keys and values of one layer for incremental decoding: `[B, 1, max_len, head_dim]`,
    zero beyond the `length` positions written so far."""

    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0


def rotary_tables(length: int, head_dim: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sin and cos of position times frequency, each `[1, length, 1, head_dim]`."""
    frequency = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    angle = torch.einsum("i,j->ij", torch.arange(length, device=device, dtype=dtype), frequency)
    sin = torch.repeat_interleave(angle.sin()[None, :, None, :], 2, dim=-1)
    cos = torch.repeat_interleave(angle.cos()[None, :, None, :], 2, dim=-1)
    return sin, cos


def rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate each pair (x[2i], x[2i+1]) of `[B, T, H, D]` by its position's angle."""
    partner = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + partner * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Multi-query attention: `num_heads` query heads share one key and one value head."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: Optional[KeyValueCache] = None,
    ) -> torch.Tensor:
        """`attn_mask` is additive (-inf = no attention), `[B, 1 or H, T, keys]`.

        Without a cache, or with an empty one (which is then filled), the `T` positions
        attend among themselves. With a filled cache `x` is the one next position: it is
        appended and attends to the whole cache, `attn_mask` covering its full length.
        """
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, 1, self.head_dim)
        v = self.v_proj(x).view(batch, length, 1, self.head_dim)
        q, k = rotate(q, sin, cos).transpose(1, 2), rotate(k, sin, cos).transpose(1, 2)
        v = v.transpose(1, 2)
        if cache is not None and cache.length > 0:
            index = torch.full((batch, 1, 1, self.head_dim), cache.length, device=x.device)
            cache.keys.scatter_(2, index, k)
            cache.values.scatter_(2, index, v)
            cache.length += 1
            k = cache.keys.expand(-1, self.num_heads, -1, -1)
            v = cache.values.expand(-1, self.num_heads, -1, -1)
        else:
            if cache is not None:
                cache.keys[:, :, :length] = k
                cache.values[:, :, :length] = v
                cache.length = length
            # the fused kernel needs keys and values with real strides for an explicit mask
            k = k.expand(-1, self.num_heads, -1, -1).contiguous()
            v = v.expand(-1, self.num_heads, -1, -1).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype))
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ffn = SwiGLU(dim, int(mlp_ratio * dim))

    def forward(self, x, sin, cos, attn_mask, cache=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sin, cos, attn_mask, cache)
        return x + self.ffn(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dim // config.num_heads
        self.layers = nn.ModuleList(
            Block(config.embedding_dim, config.num_heads, config.mlp_ratio)
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)

    def new_cache(self, batch_size: int, max_len: int, device, dtype) -> List[KeyValueCache]:
        shape = (batch_size, 1, max_len, self.head_dim)
        return [
            KeyValueCache(
                torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: Optional[List[KeyValueCache]] = None,
    ) -> torch.Tensor:
        """Return the hidden states of `x` (`[B, T, dim]`). See Attention for the mask and the
        cache."""
        start = cache[0].length if cache is not None else 0
        sin, cos = rotary_tables(start + x.shape[1], self.head_dim, x.device, x.dtype)
        sin, cos = sin[:, start:], cos[:, start:]
        for i, layer in enumerate(self.layers):
            x = layer(x, sin, cos, attn_mask, None if cache is None else cache[i])
        return self.final_norm(x)


def attention_mask(
    valid: torch.Tensor, causal: bool, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Build the additive attention mask `[B, 1 or H, T, T]` from the `valid` keys (`[B, T]`): -inf
    on invalid keys and, if `causal`, on later positions; elsewhere `bias` (`[B, H, T, T]`) or
    zero."""
    batch, length = valid.shape
    allowed = valid[:, None, None, :].to(torch.bool)
    if causal:
        position = torch.arange(length, device=valid.device)
        allowed = allowed & (position[None, :] <= position[:, None])
    else:
        allowed = allowed.expand(batch, 1, length, length)
    if bias is None:
        bias = torch.zeros((batch, 1, length, length), device=valid.device)
    return bias.masked_fill(~allowed, float("-inf"))
