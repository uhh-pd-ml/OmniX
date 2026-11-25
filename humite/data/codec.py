"""Tokenization of one particle feature: values <-> token ids.

Token layout of every feature: 0 = PAD, 1..nbins = the bins, nbins + 1 = MISSING.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

PAD_ID = 0


def bin_edges(
    binning: str,
    value_min: float,
    value_max: float,
    nbins: int,
    linear_scale: Optional[float] = None,
    sentinel: Optional[float] = None,
) -> np.ndarray:
    """The `nbins + 1` edges of a binning.

    `linear`, `log` (geometric) and `quadratic` (denser towards `value_min`) are spaced
    over [value_min, value_max]. `symlog` is for a signed quantity peaked at zero: linear
    inside +-`linear_scale` and geometric outside. With a `sentinel` (a value below
    `value_min` that marks "no value") the first bin is reserved for it.
    """
    lo, hi = float(value_min), float(value_max)
    if binning == "linear":
        return np.linspace(lo, hi, nbins + 1)
    if binning == "log":
        return np.geomspace(lo, hi, nbins + 1)
    if binning == "quadratic":
        return lo + (hi - lo) * np.linspace(0.0, 1.0, nbins + 1) ** 2
    if binning != "symlog":
        raise ValueError(f"unknown binning {binning!r}")

    scale = float(linear_scale)
    n_physical = nbins - (1 if sentinel is not None else 0)

    def forward(x):
        return np.sign(x) * np.log1p(np.abs(x) / scale)

    def inverse(u):
        return np.sign(u) * scale * np.expm1(np.abs(u))

    edges = inverse(np.linspace(forward(np.array(lo)), forward(np.array(hi)), n_physical + 1))
    edges[0], edges[-1] = lo, hi
    if sentinel is not None:
        edges[0] = lo - max(abs(lo), 1.0) * 1e-6
        edges = np.concatenate([[float(sentinel) - 1.0], edges])
    return edges.astype(np.float64)


def sample_in_bin(binning: str, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Draw one value per bin [low, high): log-uniform for `log`, sqrt-uniform for `quadratic`,
    uniform otherwise."""
    u = torch.rand_like(low)
    if binning == "log":
        log_low, log_high = low.clamp_min(1e-30).log(), high.clamp_min(1e-30).log()
        return torch.exp(log_low + u * (log_high - log_low))
    if binning == "quadratic":
        root_low, root_high = low.clamp_min(0.0).sqrt(), high.clamp_min(0.0).sqrt()
        root = root_low + u * (root_high - root_low)
        return root * root
    return low + u * (high - low)


class FeatureCodec:
    """Encode and decode one feature.

    `index` features hold integer ids (jet slot, particle type): the token is the id.
    Continuous features are binned; decoding returns the bin centre or, with `sample=True`,
    a draw within the bin. A feature may declare a second binning for particles outside every
    jet (`out_of_jet`); which one applies to a particle follows from its jet slot.
    """

    def __init__(self, name: str, entry: Mapping[str, Any]) -> None:
        self.name = name
        self.binning = str(entry["binning"])
        self.vocab_size = int(entry["vocab_size"])
        self.is_index = self.binning == "index"
        self.nbins = self.vocab_size - 2
        self.missing_id = self.nbins + 1
        self.value_min = float(entry["value_min"])
        self.value_max = float(entry["value_max"])
        self.sentinel = None if entry.get("sentinel") is None else float(entry["sentinel"])
        self.bin_edges: Optional[torch.Tensor] = None
        self.bin_centers: Optional[torch.Tensor] = None
        if not self.is_index:
            edges = bin_edges(
                self.binning,
                self.value_min,
                self.value_max,
                self.nbins,
                entry.get("linear_scale"),
                self.sentinel,
            )
            self.bin_edges = torch.as_tensor(edges, dtype=torch.float32)
            self.bin_centers = torch.as_tensor(0.5 * (edges[:-1] + edges[1:]), dtype=torch.float32)
        self.out_of_jet: Optional[FeatureCodec] = None
        if entry.get("out_of_jet"):
            self.out_of_jet = FeatureCodec(
                name, {**dict(entry["out_of_jet"]), "vocab_size": self.vocab_size}
            )
        self._tables: Dict[tuple, torch.Tensor] = {}

    def _table(self, which: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (which, str(device), dtype)
        if key not in self._tables:
            self._tables[key] = getattr(self, which).to(device=device, dtype=dtype)
        return self._tables[key]

    def encode(
        self, values: torch.Tensor, out_of_jet: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode values as token ids in 1..nbins. `out_of_jet` (bool, same shape) selects the
        out-of-jet binning."""
        if self.is_index:
            index = (values - self.value_min).floor().to(torch.long)
            return index.clamp(0, self.nbins - 1) + 1
        low, high = float(self.bin_edges[0]), float(self.bin_edges[-1])
        clipped = torch.nan_to_num(values, nan=low, posinf=high, neginf=low)
        edges = self._table("bin_edges", values.device, values.dtype)
        index = torch.bucketize(clipped, edges[1:-1], right=False).to(torch.long)
        tokens = index.clamp(0, self.nbins - 1) + 1
        if self.out_of_jet is None or out_of_jet is None:
            return tokens
        return torch.where(out_of_jet.bool(), self.out_of_jet.encode(values), tokens)

    def decode(
        self,
        tokens: torch.Tensor,
        out_of_jet: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> torch.Tensor:
        """Decode token ids to values. PAD and MISSING decode to 0."""
        tokens = tokens.to(torch.long)
        valid = (tokens >= 1) & (tokens <= self.nbins)
        index = (tokens - 1).clamp(0, self.nbins - 1)
        if self.is_index:
            values = index.to(torch.float32) + self.value_min
            return torch.where(valid, values, torch.zeros_like(values))
        if sample:
            edges = self._table("bin_edges", tokens.device, torch.float32)
            values = sample_in_bin(
                self.binning, edges[index], edges[(index + 1).clamp(0, edges.numel() - 1)]
            )
        else:
            values = self._table("bin_centers", tokens.device, torch.float32)[index]
        if self.sentinel is not None:
            values = torch.where(index == 0, torch.full_like(values, self.sentinel), values)
        values = torch.where(valid, values, torch.zeros_like(values))
        if self.out_of_jet is None or out_of_jet is None:
            return values
        return torch.where(
            out_of_jet.bool(), self.out_of_jet.decode(tokens, sample=sample), values
        )

    def is_missing(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens == PAD_ID) | (tokens == self.missing_id)


def build_codecs(features: Sequence[Mapping[str, Any]]) -> Dict[str, FeatureCodec]:
    """One codec per entry of `dataset.particle_features`, in declaration order."""
    return {str(entry["name"]): FeatureCodec(str(entry["name"]), entry) for entry in features}


def undo_delay(stream: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Remove a per-row delay from a `[B, L]` stream: `out[b, i] = stream[b, i + shift[b]]`,
    zero where that runs past the end."""
    length = stream.shape[1]
    source = torch.arange(length, device=stream.device).unsqueeze(0) + shift.view(-1, 1)
    out = stream.gather(1, source.clamp(max=length - 1))
    return torch.where(source < length, out, torch.zeros_like(out))
