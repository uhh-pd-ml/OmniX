"""From padded events to the model's token streams, and back.

An event becomes

* the plan: the particle count and, for each of the `n_jet_slots` jet slots, the jet's
  (pT, rapidity, phi, mass) and its number of constituents, computed from the particles;
* one token stream per particle feature. A particle in a jet stores eta and phi relative to
  its jet axis; a particle outside every jet stores them in the lab frame, on its own binning.

The streams are staggered: the features of a particle are emitted one position after the
other, in an order drawn per event from a fixed set of orders. `shifts[event, feature]` in
1..n_features is the delay of a feature, so particle `i` has its token of feature `f` at
position `i + shifts[f]`, and is complete at position `i + n_features`.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch

from humite.data.codec import FeatureCodec, build_codecs, undo_delay
from humite.data.config import DataConfig
from humite.data.particle_types import charge_type_from_pid, pid_from_charge_type
from humite.data.readers import PARTICLE_COLUMNS, wrap_phi

# features with a continuous value stream next to their tokens
KINEMATICS: Tuple[str, ...] = ("pt", "eta", "phi")


def feature_orders(
    base_order: Sequence[str],
    precedence: Sequence[Sequence[str]],
    n_orders: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    """Return `n_orders` feature orders: `base_order` first, then a fixed random sample of the
    permutations that respect every `[first, second]` precedence pair."""
    base = tuple(base_order)

    def allowed(order: Tuple[str, ...]) -> bool:
        position = {name: i for i, name in enumerate(order)}
        return all(position[first] < position[second] for first, second in precedence)

    if not allowed(base):
        raise ValueError(f"the base order {list(base)} violates its precedence constraints")
    others = sorted(p for p in itertools.permutations(base) if allowed(p) and p != base)
    if n_orders - 1 >= len(others):
        return [base] + others
    return [base] + sorted(random.Random(seed).sample(others, n_orders - 1))


def shift_table(data: DataConfig) -> torch.Tensor:
    """Return the delay (1..n_features) of every feature under each feature order, as an
    `[n_feature_orders, n_features]` tensor with the columns in the order of
    `data.feature_names`."""
    orders = feature_orders(
        data.feature_order,
        data.feature_order_precedence,
        data.n_feature_orders,
        data.feature_order_seed,
    )
    return torch.tensor(
        [[order.index(name) + 1 for name in data.feature_names] for order in orders],
        dtype=torch.long,
    )


def jet_four_vectors(
    pt: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    jet_idx: torch.Tensor,
    mask: torch.Tensor,
    n_slots: int,
) -> Dict[str, torch.Tensor]:
    """Sum the massless constituents of every jet slot. Returns `[B, n_slots]` tensors for jet_pt,
    jet_rapidity, jet_dphi, jet_m and the constituent count n_const. Empty slots are zero in
    every field."""
    weight = mask.to(torch.float32) * (jet_idx >= 0.5).to(torch.float32)
    slot = (jet_idx.round().long() - 1).clamp(0, n_slots - 1)

    def per_slot(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros((pt.shape[0], n_slots)).scatter_add_(1, slot, values * weight)

    n_const = per_slot(torch.ones_like(pt))
    eta64 = torch.where(weight > 0, eta, torch.zeros_like(eta)).to(torch.float64)
    pt64, phi64 = pt.to(torch.float64), phi.to(torch.float64)
    px, py = per_slot(pt64 * torch.cos(phi64)), per_slot(pt64 * torch.sin(phi64))
    e_plus_pz = per_slot(pt64 * torch.exp(eta64))
    e_minus_pz = per_slot(pt64 * torch.exp(-eta64))
    jet_pt = torch.hypot(px, py)
    rapidity = torch.where(
        e_minus_pz > 0,
        0.5 * torch.log(e_plus_pz.clamp_min(1e-300) / e_minus_pz.clamp_min(1e-300)),
        torch.zeros_like(e_plus_pz),
    )
    mass = (e_plus_pz * e_minus_pz - jet_pt * jet_pt).clamp_min(0.0).sqrt()
    return {
        "n_const": n_const,
        "jet_pt": jet_pt.to(pt.dtype),
        "jet_rapidity": rapidity.to(pt.dtype),
        "jet_dphi": torch.atan2(py, px).to(pt.dtype),
        "jet_m": mass.to(pt.dtype),
    }


class EventTokenizer:
    """Tokenizes the batches of one member and decodes generated token streams."""

    def __init__(self, data: DataConfig, pad_length: int, exact_kinematics: bool = False) -> None:
        """With `exact_kinematics` the continuous (pT, eta, phi) inputs are the true values. By
        default they are drawn within the bin of the particle's token, as in training and in
        generation."""
        self.pad_length = int(pad_length)
        self.exact_kinematics = bool(exact_kinematics)
        self.n_jet_slots = int(data.n_jet_slots)
        self.max_particles = int(data.max_particles)
        self.codecs: Dict[str, FeatureCodec] = build_codecs(data.particle_features)
        self.plan_codecs: Dict[str, FeatureCodec] = build_codecs(data.plan)
        self.feature_names: Tuple[str, ...] = tuple(self.codecs)
        self.optional_features: Tuple[str, ...] = tuple(data.optional_features)
        self.shift_table = shift_table(data)

    def shifts(self, feature_order: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the delay of every feature under each event's feature order, as `[B]` tensors."""
        rows = self.shift_table.to(feature_order.device)[feature_order]
        return {name: rows[:, j] for j, name in enumerate(self.feature_names)}

    def collate(self, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad a list of events (humite.data.dataset.EventDataset) to `pad_length`."""
        n_particles = [len(event["particles"]) for event in events]
        if max(n_particles) + len(self.feature_names) > self.pad_length:
            raise ValueError(
                f"an event has {max(n_particles)} particles, too many for pad_length "
                f"{self.pad_length} with {len(self.feature_names)} staggered features"
            )
        particles = torch.zeros((len(events), self.pad_length, len(PARTICLE_COLUMNS)))
        mask = torch.zeros((len(events), self.pad_length), dtype=torch.bool)
        for i, event in enumerate(events):
            particles[i, : n_particles[i]] = torch.from_numpy(event["particles"])
            mask[i, : n_particles[i]] = True
        batch: Dict[str, Any] = {
            "particles": particles,
            "particle_mask": mask,
            "n_particles": torch.tensor(n_particles, dtype=torch.int32),
            "feature_order": torch.tensor([e["feature_order"] for e in events]),
            "condition": torch.tensor([e["condition"] for e in events]),
            "feature_mode": torch.tensor([e["feature_mode"] for e in events]),
            "event_values": {
                key: torch.tensor([e["values"][key] for e in events], dtype=torch.float32)
                for key in events[0]["values"]
            },
        }
        if "jet_flavor" in events[0]:
            batch["jet_flavor"] = torch.as_tensor(
                [e["jet_flavor"].tolist() for e in events], dtype=torch.long
            )
        return batch

    def _delay(
        self, stream: torch.Tensor, shift: torch.Tensor, valid: torch.Tensor, fill: float
    ) -> torch.Tensor:
        """Move `stream[b, i]` to position `i + shift[b]` and write `fill` where `valid` is
        false."""
        length = stream.shape[1]
        source = torch.arange(length, device=stream.device).unsqueeze(0) - shift.view(-1, 1)
        delayed = stream.gather(1, source.clamp(0, length - 1))
        return torch.where(valid, delayed, torch.full_like(delayed, fill))

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Add the token streams, their masks and the plan tokens to a collated batch."""
        particles, mask = batch["particles"], batch["particle_mask"]
        n_particles = batch["n_particles"]
        values = {name: particles[..., j] for j, name in enumerate(PARTICLE_COLUMNS)}

        jets = jet_four_vectors(
            values["pt"], values["eta"], values["phi"], values["jet_idx"], mask, self.n_jet_slots
        )
        for name, value in jets.items():
            batch[f"plan_{name}"] = value
            batch[f"plan_{name}_token"] = self.plan_codecs[name].encode(value)
        count_vocab = self.max_particles + 2
        batch["count_token"] = n_particles.to(torch.long).clamp(1, count_vocab - 2)

        in_jet = mask & (values["jet_idx"] >= 0.5)
        out_of_jet = mask & (values["jet_idx"] < 0.5)
        slot = (values["jet_idx"].round().long() - 1).clamp(0, self.n_jet_slots - 1)
        encoded = dict(values)
        encoded["pid"] = pid_from_charge_type(values["charge"], values["pf_type"], mask)
        if bool((encoded["pid"][mask] > self.codecs["pid"].value_max).any()):
            raise ValueError("a particle type outside the pid vocabulary occurs")
        relative_eta = values["eta"] - jets["jet_rapidity"].gather(1, slot)
        relative_phi = wrap_phi(values["phi"] - jets["jet_dphi"].gather(1, slot))
        encoded["eta"] = torch.where(in_jet, relative_eta, values["eta"])
        encoded["phi"] = torch.where(in_jet, relative_phi, values["phi"])

        shifts = self.shifts(batch["feature_order"].reshape(-1).long())
        length = mask.shape[1]
        position = torch.arange(length, device=mask.device).unsqueeze(0)
        attention_mask = position < (n_particles + len(self.feature_names)).unsqueeze(1)
        valid_masks: Dict[str, torch.Tensor] = {}
        for name, codec in self.codecs.items():
            start = shifts[name].view(-1, 1)
            valid = (position >= start) & (position < start + n_particles.unsqueeze(1))
            valid = valid & attention_mask
            flag = out_of_jet if codec.out_of_jet is not None else None
            tokens = self._delay(
                codec.encode(encoded[name], flag), shifts[name], valid, codec.missing_id
            )
            tokens = torch.where(attention_mask, tokens, torch.zeros_like(tokens))
            valid_masks[name] = valid
            batch[f"{name}_token"] = tokens
            if name in KINEMATICS and self.exact_kinematics:
                value = self._delay(encoded[name], shifts[name], valid, 0.0)
                batch[f"{name}_value"] = torch.where(
                    attention_mask, value, torch.zeros_like(value)
                )
            elif name in KINEMATICS:
                if flag is not None:
                    flag = self._delay(flag.long(), shifts[name], valid, 0).bool() & attention_mask
                value = codec.decode(tokens, flag, sample=True).to(torch.float32)
                batch[f"{name}_value"] = torch.where(
                    attention_mask, value, torch.zeros_like(value)
                )

        all_features = batch["feature_mode"].reshape(-1).bool().view(-1, 1)
        for name in self.optional_features:
            tokens = batch[f"{name}_token"]
            hidden = torch.where(
                attention_mask,
                torch.full_like(tokens, self.codecs[name].missing_id),
                torch.zeros_like(tokens),
            )
            batch[f"{name}_token"] = torch.where(all_features, tokens, hidden)
            valid_masks[name] = valid_masks[name] & all_features
        batch["attention_mask"] = attention_mask
        batch["feature_valid"] = valid_masks
        return batch

    def decode(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generated token streams and plan -> particles in the lab frame.

        `batch` holds the delayed streams `{feature}_token`, `feature_order`,
        `feature_mode`, `particle_mask` (which particles exist) and the plan values
        `plan_jet_rapidity` / `plan_jet_dphi`. Returns the particles
        `[B, L, len(PARTICLE_COLUMNS)]`, with a draw within every decoded bin, and their
        mask. On a kinematics-only row the optional features are 0.
        """
        shifts = self.shifts(batch["feature_order"].reshape(-1).long())
        tokens = {
            name: undo_delay(batch[f"{name}_token"], shifts[name]) for name in self.feature_names
        }
        all_features = batch["feature_mode"].reshape(-1).bool().view(-1, 1)
        slot_codec = self.codecs["jet_idx"]
        slot_valid = (tokens["jet_idx"] >= 1) & (tokens["jet_idx"] <= slot_codec.nbins)
        out_of_jet = slot_valid & (slot_codec.decode(tokens["jet_idx"]) < 0.5)

        values: Dict[str, torch.Tensor] = {}
        missing = torch.zeros_like(batch["particle_mask"], dtype=torch.bool)
        for name, codec in self.codecs.items():
            flag = out_of_jet if codec.out_of_jet is not None else None
            values[name] = codec.decode(tokens[name], flag, sample=True)
            absent = codec.is_missing(tokens[name])
            if name in self.optional_features:
                absent = absent & all_features
                values[name] = torch.where(
                    all_features, values[name], torch.zeros_like(values[name])
                )
            missing = missing | absent
        values["charge"], values["pf_type"] = charge_type_from_pid(values.pop("pid"))

        mask = batch["particle_mask"].to(torch.bool) & ~missing
        particles = torch.stack([values[name] for name in PARTICLE_COLUMNS], dim=-1)
        particles[~mask] = 0.0
        jet_idx = particles[..., PARTICLE_COLUMNS.index("jet_idx")].round().long()
        in_jet = mask & (jet_idx >= 1)
        slot = (jet_idx - 1).clamp(0, self.n_jet_slots - 1)
        i_eta, i_phi = PARTICLE_COLUMNS.index("eta"), PARTICLE_COLUMNS.index("phi")
        axis_eta = batch["plan_jet_rapidity"].to(particles).gather(1, slot)
        axis_phi = batch["plan_jet_dphi"].to(particles).gather(1, slot)
        particles[..., i_eta] = torch.where(
            in_jet, particles[..., i_eta] + axis_eta, particles[..., i_eta]
        )
        particles[..., i_phi] = torch.where(
            in_jet, wrap_phi(particles[..., i_phi] + axis_phi), particles[..., i_phi]
        )
        return particles, mask
