"""Pooling of the particle states into one event vector for classification."""

from __future__ import annotations

import torch
from torch import nn

from humite.models.transformer import SwiGLU


class PoolingBlock(nn.Module):
    """The class token attends to the particle states (cross-attention, then an MLP). Both
    output projections start at zero, so the block starts as the identity."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, int(mlp_ratio * dim))
        for projection in (self.attn.out_proj, self.ffn.w3):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self, query: torch.Tensor, states: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        states = self.norm_kv(states)
        attended, _ = self.attn(
            self.norm_q(query), states, states, key_padding_mask=padding, need_weights=False
        )
        query = query + attended
        return query + self.ffn(self.norm2(query))


class ReadHead(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, n_blocks: int = 2, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            PoolingBlock(dim, num_heads, mlp_ratio) for _ in range(n_blocks)
        )

    def forward(
        self, states: torch.Tensor, valid: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """Pool the particle states `states` (`[B, T, dim]`, with `valid` marking the particle
        positions) into the class token's state `query` (`[B, dim]`). Returns `[B, dim]`."""
        query = query[:, None, :]
        for block in self.blocks:
            query = block(query, states, ~valid)
        return query[:, 0]
