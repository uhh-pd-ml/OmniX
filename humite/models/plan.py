"""The jet plan: embeddings and prediction heads of the plan tokens.

The plan holds `n_fields` fields for each of `n_slots` jets. Its positions are staggered
like the particle features: position `p` carries field `f` of jet `p - f`, so the plan
takes `n_slots + n_fields - 1` positions and a jet's fields are predicted one after the
other.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from humite.data.codec import FeatureCodec
from humite.models.heads import TokenHead


class PlanHead(nn.Module):
    def __init__(self, codecs: Mapping[str, FeatureCodec], n_slots: int, dim: int) -> None:
        super().__init__()
        self.codecs = dict(codecs)
        self.field_names = tuple(self.codecs)
        self.n_slots = int(n_slots)
        self.n_positions = self.n_slots + len(self.field_names) - 1
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(codec.vocab_size, dim) for name, codec in self.codecs.items()}
        )
        self.slot_embeddings = nn.ModuleDict(
            {name: nn.Embedding(self.n_slots, dim) for name in self.field_names}
        )
        self.heads = nn.ModuleDict(
            {name: TokenHead(dim, codec.vocab_size) for name, codec in self.codecs.items()}
        )

    def embed(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Embed the plan tokens (`[B, n_slots]` per field) as `[B, n_positions, dim]`."""
        slots = torch.arange(self.n_slots, device=next(iter(tokens.values())).device)
        out = 0.0
        for offset, name in enumerate(self.field_names):
            embedded = self.embeddings[name](tokens[name]) + self.slot_embeddings[name](slots)
            out = out + F.pad(embedded, (0, 0, offset, self.n_positions - offset - self.n_slots))
        return out

    def embed_position(self, tokens: Mapping[str, torch.Tensor], position: int) -> torch.Tensor:
        """Embed one plan position, `[B, 1, dim]`, from the tokens sampled so far."""
        out = 0.0
        for offset, name in enumerate(self.field_names):
            slot = position - offset
            if 0 <= slot < self.n_slots:
                index = torch.full_like(tokens[name][:, slot], slot)
                out = out + (
                    self.embeddings[name](tokens[name][:, slot])
                    + self.slot_embeddings[name](index)
                ).unsqueeze(1)
        return out

    def logits(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the logits of every field, `[B, n_slots, vocab]`. Entry `p` of `hidden` (`[B,
        n_positions, dim]`) is the state that predicts plan position `p`."""
        return {
            name: self.heads[name](hidden[:, offset : offset + self.n_slots])
            for offset, name in enumerate(self.field_names)
        }

    def decode(
        self, tokens: Mapping[str, torch.Tensor], sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Decode plan tokens to values: bin centres, or a draw within each bin."""
        return {
            name: self.codecs[name].decode(
                tokens[name].clamp(1, self.codecs[name].nbins), sample=sample
            )
            for name in self.field_names
        }
