"""Jet-flavour decoder: one query per planned jet, decided jointly with the other jets."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from humite.data.readers import wrap_phi
from humite.models.transformer import SwiGLU

# how the particle at a position relates to a jet query
OWN_JET, OTHER_JET, OUT_OF_JET, NO_PARTICLE = 0, 1, 2, 3


class BiasedAttention(nn.Module):
    """Multi-head attention with an additive bias `[B, H, queries, keys]`."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        batch, n_queries, dim = queries.shape

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, x.shape[1], self.num_heads, dim // self.num_heads).transpose(1, 2)

        q, k, v = heads(self.q(queries)), heads(self.k(keys)), heads(self.v(keys))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.out(out.transpose(1, 2).reshape(batch, n_queries, dim))


class DecoderBlock(nn.Module):
    """Attention among the jets, attention from the jets to the particles, MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = BiasedAttention(dim, num_heads)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_keys = nn.LayerNorm(dim)
        self.cross_attn = BiasedAttention(dim, num_heads)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, int(mlp_ratio * dim))
        nn.init.zeros_(self.ffn.w3.weight)
        nn.init.zeros_(self.ffn.w3.bias)

    def forward(self, jets, particles, jet_bias, particle_bias) -> torch.Tensor:
        normed = self.norm_self(jets)
        jets = jets + self.self_attn(normed, normed, jet_bias)
        jets = jets + self.cross_attn(
            self.norm_cross(jets), self.norm_keys(particles), particle_bias
        )
        return jets + self.ffn(self.norm_ffn(jets))


class JetAssignmentDecoder(nn.Module):
    """Each jet query starts from the jet's planned kinematics. The attention among the jets
    is biased by pair features of their plans, the attention to the particle states by
    whether a particle belongs to the query's jet, to another jet or to none. No slot index
    enters, so the decoder treats the jets as a set."""

    def __init__(
        self, dim: int, num_classes: int, n_layers: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.seed_mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.unlabeled = nn.Parameter(torch.empty(dim).uniform_(-0.5, 0.5))
        self.membership_bias = nn.Parameter(torch.zeros(n_layers, num_heads, 4))
        self.pair_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, n_layers * num_heads)
        )
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)
        self.blocks = nn.ModuleList(
            DecoderBlock(dim, num_heads, mlp_ratio) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def jet_features(
        pt: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor, n_const: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """From the planned jets (`[B, K]` each) compute the query seeds `[B, K, 6]`, the mask of
        the slots that hold a jet `[B, K]`, and the pair features `[B, K, K, 3]` (dR, ln m^2, ln
        pT ratio)."""
        pt, eta, phi, n_const = (t.float() for t in (pt, eta, phi, n_const))
        valid = n_const >= 0.5
        weight = valid.to(pt.dtype)
        seeds = torch.stack(
            [
                torch.log1p(pt.clamp_min(0.0)) / 5.0,
                eta / 2.5,
                torch.cos(phi),
                torch.sin(phi),
                torch.log1p(n_const.clamp_min(0.0)) / 3.0,
                weight,
            ],
            dim=-1,
        ) * weight.unsqueeze(-1)
        d_eta = eta[:, :, None] - eta[:, None, :]
        d_phi = wrap_phi(phi[:, :, None] - phi[:, None, :])
        distance = torch.sqrt(d_eta**2 + d_phi**2 + 1e-12)
        m2 = 2.0 * pt[:, :, None] * pt[:, None, :] * (torch.cosh(d_eta) - torch.cos(d_phi))
        ratio = torch.log(pt[:, :, None].clamp_min(0.0) + 1.0) - torch.log(
            pt[:, None, :].clamp_min(0.0) + 1.0
        )
        both = (valid[:, :, None] & valid[:, None, :]).to(pt.dtype)
        pairs = torch.stack(
            [distance / 2.0, torch.log1p(m2.clamp_min(0.0)) / 10.0, ratio], dim=-1
        ) * both.unsqueeze(-1)
        return seeds, valid, pairs

    @staticmethod
    def membership(slot: torch.Tensor, complete: torch.Tensor, n_jets: int) -> torch.Tensor:
        """Return how the particle that is complete at each position relates to each jet query,
        `[B, K, T]`. `slot` (`[B, T]`) is the particle's jet slot, 0 for outside every jet."""
        jet = torch.arange(1, n_jets + 1, device=slot.device).view(1, -1, 1)
        slot, complete = slot.unsqueeze(1), complete.unsqueeze(1)
        out = torch.full((slot.shape[0], n_jets, slot.shape[2]), NO_PARTICLE, device=slot.device)
        out = torch.where(complete & (slot >= 1), torch.full_like(out, OTHER_JET), out)
        out = torch.where(complete & (slot == 0), torch.full_like(out, OUT_OF_JET), out)
        return torch.where(complete & (slot == jet), torch.full_like(out, OWN_JET), out)

    def forward(
        self,
        keys: torch.Tensor,
        key_valid: torch.Tensor,
        membership: torch.Tensor,
        seeds: torch.Tensor,
        jet_valid: torch.Tensor,
        pair_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the jet states `[B, K, dim]` from the particle states `keys` (`[B, T, dim]`) and
        the outputs of `jet_features` and `membership`."""
        batch, n_jets = jet_valid.shape
        dtype = self.unlabeled.dtype
        jets = self.seed_mlp(seeds.to(dtype))
        jets = jets + self.unlabeled.to(jets.dtype)
        lowest = torch.finfo(torch.float32).min
        itself = torch.eye(n_jets, dtype=torch.bool, device=jet_valid.device)
        jet_mask = torch.where(jet_valid[:, None, :] | itself[None], 0.0, lowest)[:, None]
        pair_bias = self.pair_mlp(pair_feats.to(dtype))
        pair_bias = pair_bias.view(batch, n_jets, n_jets, self.n_layers, self.num_heads)
        pair_bias = pair_bias.permute(0, 3, 4, 1, 2)
        key_mask = torch.where(key_valid.bool(), 0.0, lowest)[:, None, None, :]
        one_hot = F.one_hot(membership, 4).to(torch.float32)[:, None]
        for layer, block in enumerate(self.blocks):
            weights = self.membership_bias[layer].to(torch.float32)[None, :, None, None, :]
            particle_bias = (one_hot * weights).sum(-1) + key_mask
            jets = block(jets, keys, pair_bias[:, layer] + jet_mask, particle_bias)
        return self.norm(jets)
