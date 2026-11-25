"""Attention bias from the kinematics of particle pairs.

The pair features ln dR, ln kT, ln z and ln m^2 and their use as a bias on the attention
logits follow the Particle Transformer (H. Qu, C. Li, S. Qian, "Particle Transformer for Jet
Tagging", arXiv:2202.03772). Here the bias is computed between the particles that are complete
at each position of the autoregressive sequence, and the jet-membership flags are added.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# per position: pT, eta, phi, jet slot of the particle (0 = outside every jet) and whether
# there is a particle
Streams = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class PairwiseAttentionBias(nn.Module):
    """An MLP of pair features gives one bias per attention head for every pair of positions.

    The features of a pair are ln dR, ln kT = ln(min(pT) dR), ln z = ln(min(pT) / sum(pT))
    and ln m^2 (standardised with `feature_mean` / `feature_std`), a flag for pairs closer
    than `delta_min` (where dR is floored), and three flags for the jets of the two
    particles: same jet, both outside every jet, exactly one outside. The last layer starts
    at zero, so the bias is switched on by training.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        n_hidden_layers: int,
        delta_min: float,
        feature_mean: Sequence[float],
        feature_std: Sequence[float],
        rows_per_chunk: int = 64,
    ) -> None:
        super().__init__()
        self.n_bias_heads = num_heads
        self.delta_min = float(delta_min)
        self.rows_per_chunk = rows_per_chunk
        self.register_buffer("mean", torch.tensor([*feature_mean, 0.0]), persistent=False)
        self.register_buffer("std", torch.tensor([*feature_std, 1.0]), persistent=False)
        layers = [nn.Linear(8, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, num_heads))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def pair_features(
        self, query: Sequence[torch.Tensor], key: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the eight pair features, `[B, P, Q, 8]`, from (pT, eta, phi, slot) of `P` query
        and `Q` key positions."""
        tiny = 1e-12
        pt_i = query[0][:, :, None].clamp_min(tiny)
        pt_j = key[0][:, None, :].clamp_min(tiny)
        d_eta = query[1][:, :, None] - key[1][:, None, :]
        d_phi_raw = query[2][:, :, None] - key[2][:, None, :]
        d_phi = torch.where(
            d_phi_raw > math.pi,
            d_phi_raw - 2.0 * math.pi,
            torch.where(d_phi_raw <= -math.pi, d_phi_raw + 2.0 * math.pi, d_phi_raw),
        )
        distance = torch.sqrt(d_eta * d_eta + d_phi * d_phi + tiny)
        coincident = (distance < self.delta_min).to(distance.dtype)
        distance = distance.clamp_min(self.delta_min)
        pt_min = torch.minimum(pt_i, pt_j)
        kt = pt_min * distance
        z = pt_min / (pt_i + pt_j).clamp_min(tiny)
        # m^2 = 2 pT_i pT_j (cosh d_eta - cos d_phi), in the form that is exact for small angles
        m2 = (
            4.0
            * pt_i
            * pt_j
            * (torch.sinh(0.5 * d_eta).square() + torch.sin(0.5 * d_phi_raw).square())
        )
        kinematic = torch.stack(
            [
                torch.log(distance),
                torch.log(kt.clamp_min(tiny)),
                torch.log(z.clamp_min(tiny)),
                torch.log(m2.clamp_min(tiny)),
                coincident,
            ],
            dim=-1,
        )
        kinematic = (kinematic - self.mean) / self.std
        slot_i, slot_j = query[3][:, :, None], key[3][:, None, :]
        membership = torch.stack(
            [
                (slot_i == slot_j) & (slot_i >= 1),
                (slot_i == 0) & (slot_j == 0),
                (slot_i == 0) != (slot_j == 0),
            ],
            dim=-1,
        )
        return torch.cat([kinematic, membership.to(kinematic.dtype)], dim=-1)

    def _rows(self, *streams: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pair_features(streams[:4], streams[4:]))

    def forward(
        self,
        query: Streams,
        key: Streams,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the bias `[B, heads, P, Q]`. It is zero for a pair with a missing particle and
        for a position paired with itself."""
        chunks = []
        for start in range(0, query[0].shape[1], self.rows_per_chunk):
            rows = tuple(stream[:, start : start + self.rows_per_chunk] for stream in query[:4])
            if self.training and torch.is_grad_enabled():
                # the pair features are recomputed in the backward pass instead of stored
                chunks.append(checkpoint(self._rows, *rows, *key[:4], use_reentrant=False))
            else:
                chunks.append(self._rows(*rows, *key[:4]))
        bias = torch.cat(chunks, dim=1).permute(0, 3, 1, 2)
        keep = query[4][:, :, None] & key[4][:, None, :]
        keep = keep & ~(query_positions[:, None] == key_positions[None, :])[None]
        return (bias * keep[:, None].to(query[0].dtype)).contiguous()
