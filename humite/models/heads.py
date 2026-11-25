"""Output heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TokenHead(nn.Module):
    """Linear layer to token logits, computed in float32 with the mean over the vocabulary
    subtracted. The softmax does not see a common offset of the logits, so nothing keeps it
    small during training; in bfloat16 a large offset would round the differences away."""

    def __init__(self, dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.proj.weight.float(), self.proj.bias.float())
            return logits - logits.mean(dim=-1, keepdim=True)


class ClassificationHead(nn.Module):
    """Three-layer MLP to class logits."""

    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
