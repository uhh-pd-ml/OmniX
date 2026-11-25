"""The OmniX model: one autoregressive transformer over the tokens of a jet or an event.

Sequence layout::

    SOURCE  FEATURE_MODE  ORDER  CONDITION  count  plan (n_jet_slots + 4 positions)  particles ...

The first four tokens are given: the dataset, whether the particles carry all features, the
order of the particle features, and an optional class (jet type or process). The model then
predicts the particle count, the jet plan and the particle tokens. Position `p` of the
particle part carries one token of each feature, feature `f` belonging to particle
`p - shift[f]` (see humite.data.tokenizer).

Besides the token embeddings a position sees the particle that is complete there (all its
features emitted), as continuous kinematics and through an attention bias computed from
pairs of complete particles, and how much of its planned jet is still to be generated.

Tagging reads the same network without the causal mask: a class token pooled over the
particles gives the event class, and a decoder with one query per planned jet gives the jet
flavours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from humite.data.codec import FeatureCodec
from humite.data.particle_types import PID_PAIRS
from humite.data.readers import wrap_phi
from humite.data.tokenizer import KINEMATICS
from humite.models.heads import ClassificationHead, TokenHead
from humite.models.jet_decoder import JetAssignmentDecoder
from humite.models.pairwise_bias import PairwiseAttentionBias
from humite.models.plan import PlanHead
from humite.models.read_head import ReadHead
from humite.models.transformer import (
    KeyValueCache,
    Transformer,
    TransformerConfig,
    attention_mask,
)

# width of each feature's token embedding before the projection to the model width
FEATURE_EMBEDDING_DIM = 64


class FeatureEmbedding(nn.Module):
    """One embedding table and layer norm per particle feature, concatenated."""

    def __init__(self, vocab_sizes: Mapping[str, int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, FEATURE_EMBEDDING_DIM) for name, size in vocab_sizes.items()}
        )
        self.norms = nn.ModuleDict(
            {name: nn.LayerNorm(FEATURE_EMBEDDING_DIM) for name in vocab_sizes}
        )

    def forward(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [self.norms[name](self.embeddings[name](tokens[name])) for name in self.embeddings],
            dim=-1,
        )


class OmniX(nn.Module):
    def __init__(
        self,
        transformer: TransformerConfig,
        feature_codecs: Mapping[str, FeatureCodec],
        plan_codecs: Mapping[str, FeatureCodec],
        n_jet_slots: int,
        max_particles: int,
        shift_table: torch.Tensor,
        optional_features: Sequence[str],
        n_sources: int,
        n_condition_codes: int,
        tasks: Sequence[Mapping[str, Any]],
        pair_bias: Mapping[str, Any],
        jet_decoder: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """`tasks` lists the classification tasks, each with a name, a mode (`event` or `per_jet`)
        and a number of classes. `pair_bias` holds the settings of PairwiseAttentionBias, and
        `jet_decoder` those of the decoder for the `per_jet` task."""
        super().__init__()
        dim = transformer.embedding_dim
        self.dim = dim
        self.codecs = dict(feature_codecs)
        self.feature_names: Tuple[str, ...] = tuple(self.codecs)
        self.n_features = len(self.feature_names)
        self.optional_features = tuple(optional_features)
        self.n_jet_slots = int(n_jet_slots)
        self.max_particles = int(max_particles)
        self.register_buffer("shift_table", shift_table, persistent=False)

        self.transformer = Transformer(transformer)

        # the given tokens and the particle count
        self.source_embedding = nn.Embedding(n_sources, dim)
        self.feature_mode_embedding = nn.Embedding(2, dim)
        self.order_embedding = nn.Embedding(shift_table.shape[0], dim)
        self.condition_embedding = nn.Embedding(n_condition_codes + 1, dim)
        self.count_embedding = nn.Embedding(max_particles + 2, dim)
        self.count_tag = nn.Parameter(torch.randn(dim))
        self.count_head = TokenHead(dim, max_particles + 2)
        self.plan_head = PlanHead(plan_codecs, n_jet_slots, dim)
        self.prefix_len = 5 + self.plan_head.n_positions

        # particle inputs
        vocab_sizes = {name: codec.vocab_size for name, codec in self.codecs.items()}
        self.feature_embedding = FeatureEmbedding(vocab_sizes)
        self.feature_projection = nn.Linear(self.n_features * FEATURE_EMBEDDING_DIM, dim)
        self.kinematics_mlp = nn.Sequential(
            nn.Linear(len(KINEMATICS), 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, dim),
        )
        self.jet_remainder_mlp = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, dim))
        nn.init.zeros_(self.jet_remainder_mlp[-1].weight)
        nn.init.zeros_(self.jet_remainder_mlp[-1].bias)
        finest_bin = min(
            float((self.codecs[name].bin_edges[1:] - self.codecs[name].bin_edges[:-1]).min())
            for name in ("eta", "phi")
        )
        self.pair_bias = PairwiseAttentionBias(
            num_heads=transformer.num_heads, delta_min=0.5 * finest_bin, **pair_bias
        )

        self.feature_heads = nn.ModuleDict(
            {name: TokenHead(dim, size) for name, size in vocab_sizes.items()}
        )
        # impact parameters follow the particle type: neutral particles carry the sentinel
        # (token 1), charged particles any other bin
        charged = torch.full((vocab_sizes["pid"],), -1, dtype=torch.long)
        for token in range(1, self.codecs["pid"].nbins + 1):
            charged[token] = int(PID_PAIRS[token - 1][0] != 0)
        self.register_buffer("pid_token_is_charged", charged, persistent=False)

        # tagging
        self.tasks = [dict(task) for task in tasks]
        self.event_tasks = [task for task in self.tasks if task["mode"] == "event"]
        self.jet_task = next((task for task in self.tasks if task["mode"] == "per_jet"), None)
        self.class_query = nn.ParameterDict(
            {task["name"]: nn.Parameter(torch.zeros(dim)) for task in self.event_tasks}
        )
        for query in self.class_query.values():
            nn.init.normal_(query, std=0.02)
        self.class_heads = nn.ModuleDict(
            {
                task["name"]: ClassificationHead(dim, int(task["num_classes"]))
                for task in self.tasks
            }
        )
        self.read_head = ReadHead(dim, transformer.num_heads) if self.event_tasks else None
        self.jet_decoder = (
            JetAssignmentDecoder(dim, int(self.jet_task["num_classes"]), **jet_decoder)
            if self.jet_task is not None
            else None
        )

    def shifts(self, feature_order: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the delay of every feature under each event's feature order, as `[B, 1]`
        tensors."""
        rows = self.shift_table[feature_order.reshape(-1).long()]
        return {name: rows[:, j : j + 1] for j, name in enumerate(self.feature_names)}

    def complete_particles(
        self, batch: Mapping[str, Any], names: Sequence[str]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return the features `names` of the particle that is complete at each position, and a
        mask of the positions that have one. Kinematics are read from the value streams, the
        other features from the bin centres of their tokens."""
        shifts = self.shifts(batch["feature_order"])
        length = batch[f"{names[0]}_token"].shape[1]
        position = torch.arange(length, device=self.shift_table.device).unsqueeze(0)
        complete = position >= self.n_features
        values: Dict[str, torch.Tensor] = {}
        for name in names:
            source = position - self.n_features + shifts[name]
            index = source.clamp(0, length - 1)
            tokens = batch[f"{name}_token"].long().gather(1, index)
            if name in KINEMATICS:
                value = batch[f"{name}_value"].to(torch.float32).gather(1, index)
            else:
                value = self.codecs[name].decode(tokens)
            valid = (source >= 0) & (tokens >= 1) & (tokens <= self.codecs[name].nbins)
            values[name] = torch.where(valid, value, torch.zeros_like(value))
            complete = complete & valid
        return values, complete

    def plan_values(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Return the plan at bin centres, one `[B, n_jet_slots]` tensor per field."""
        tokens = {name: batch[f"plan_{name}_token"].long() for name in self.plan_head.field_names}
        return {
            name: value.to(torch.float32) for name, value in self.plan_head.decode(tokens).items()
        }

    def embed_kinematics(
        self, batch: Mapping[str, Any], position: Optional[int] = None
    ) -> torch.Tensor:
        """Embed (pT, eta, phi) of the complete particle at every position, or at one position,
        with an MLP. Positions without a complete particle get zero."""
        values, complete = self.complete_particles(batch, KINEMATICS)
        pt = values["pt"]
        features = torch.stack(
            [
                torch.sign(pt) * torch.log1p(pt.abs() + 1e-5) - 1.0,
                values["eta"],
                0.5 * values["phi"],
            ],
            dim=-1,
        )
        if position is not None:
            features = features[:, position : position + 1]
            complete = complete[:, position : position + 1]
        embedded = self.kinematics_mlp(features.to(self.kinematics_mlp[0].weight.dtype))
        return embedded * complete.unsqueeze(-1).to(embedded.dtype)

    def jet_remainder(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Return what is left to generate of a jet, as seven features per position.

        They refer to the jet of the particle whose pT is predicted next: the planned four-momentum
        minus that of the constituents emitted so far (in the jet's frame, relative to the plan),
        the emitted and the remaining constituent count, and a flag that the particle is in a jet.
        The momentum and the remaining count reach zero when the jet is complete.
        """
        shifts = self.shifts(batch["feature_order"])
        slot_codec = self.codecs["jet_idx"]
        slot_tokens = batch["jet_idx_token"].long()
        batch_size, length = slot_tokens.shape
        device = slot_tokens.device
        particle = torch.arange(length, device=device).unsqueeze(0)

        def of_particle(name: str, stream: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            source = particle + shifts[name]
            return stream.gather(1, source.clamp(max=length - 1)), source <= length - 1

        emitted = torch.ones((batch_size, length), dtype=torch.bool, device=device)
        values = {}
        for name in KINEMATICS:
            tokens, inside = of_particle(name, batch[f"{name}_token"].long())
            emitted = emitted & inside & (tokens >= 1) & (tokens <= self.codecs[name].nbins)
            values[name] = of_particle(name, batch[f"{name}_value"].float())[0]
        tokens, inside = of_particle("jet_idx", slot_tokens)
        slot = slot_codec.decode(tokens).round().long()
        emitted = emitted & inside & (tokens >= 1) & (tokens <= slot_codec.nbins) & (slot >= 1)

        plan = self.plan_values(batch)
        weight = emitted.to(torch.float32)
        pt, deta, dphi = values["pt"] * weight, values["eta"] * weight, values["phi"] * weight
        components = torch.stack(
            [
                pt * torch.cos(dphi),
                pt * torch.sin(dphi),
                pt * torch.exp(deta),
                pt * torch.exp(-deta),
                weight,
            ],
            dim=-1,
        )
        one_hot = F.one_hot((slot - 1).clamp(0, self.n_jet_slots - 1), self.n_jet_slots)
        one_hot = one_hot.to(torch.float32) * weight.unsqueeze(-1)
        sums = (one_hot.unsqueeze(-1) * components.unsqueeze(-2)).cumsum(dim=1)
        # the last particle that is complete at each position
        last = particle - torch.maximum(
            torch.maximum(shifts["jet_idx"], shifts["pt"]),
            torch.maximum(shifts["eta"], shifts["phi"]),
        )
        index = (
            last.clamp(min=0).view(batch_size, length, 1, 1).expand(-1, -1, self.n_jet_slots, 5)
        )
        sums = sums.gather(1, index) * (last >= 0).to(torch.float32).view(batch_size, length, 1, 1)

        # the particle whose pT is predicted next, and its jet
        upcoming = particle + 1 - (shifts["pt"] - shifts["jet_idx"])
        tokens = slot_tokens.gather(1, upcoming.clamp(0, length - 1))
        slot = slot_codec.decode(tokens).round().long()
        in_jet = (upcoming >= 0) & (tokens >= 1) & (tokens <= slot_codec.nbins) & (slot >= 1)
        slot = (slot - 1).clamp(0, self.n_jet_slots - 1)
        emitted_sum = sums.gather(
            2, slot.view(batch_size, length, 1, 1).expand(-1, -1, 1, 5)
        ).squeeze(2)
        jet_pt = plan["jet_pt"].gather(1, slot)
        jet_m = plan["jet_m"].gather(1, slot)
        n_const = plan["n_const"].gather(1, slot)
        in_jet = in_jet & (jet_pt > 0)
        pt_scale = jet_pt.clamp_min(1e-3)
        mt = torch.sqrt(pt_scale * pt_scale + jet_m * jet_m)
        features = torch.stack(
            [
                (jet_pt - emitted_sum[..., 0]) / pt_scale,
                -emitted_sum[..., 1] / pt_scale,
                (mt - emitted_sum[..., 2]) / mt,
                (mt - emitted_sum[..., 3]) / mt,
                torch.log1p(emitted_sum[..., 4]) * 0.25,
                (n_const - emitted_sum[..., 4]) * 0.125,
            ],
            dim=-1,
        )
        features = torch.where(in_jet.unsqueeze(-1), features, torch.zeros_like(features))
        return torch.cat([features, in_jet.to(features.dtype).unsqueeze(-1)], dim=-1)

    def embed_jet_remainder(
        self, batch: Mapping[str, Any], position: Optional[int] = None
    ) -> torch.Tensor:
        features = self.jet_remainder(batch)
        if position is not None:
            features = features[:, position : position + 1]
        return self.jet_remainder_mlp(features.to(self.jet_remainder_mlp[0].weight.dtype))

    def pair_streams(self, batch: Mapping[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Return (pT, eta, phi, jet slot, exists) of the complete particle at every position. Eta
        and phi are in the lab frame: for particles in jets the planned jet axis is added back."""
        values, complete = self.complete_particles(batch, (*KINEMATICS, "jet_idx"))
        plan = self.plan_values(batch)
        slot = values["jet_idx"].round().long()
        in_jet = complete & (slot >= 1) & (slot <= self.n_jet_slots)
        column = (slot - 1).clamp(0, self.n_jet_slots - 1)
        eta, phi = values["eta"].float(), values["phi"].float()
        lab_eta = torch.where(in_jet, eta + plan["jet_rapidity"].gather(1, column), eta)
        lab_phi = torch.where(in_jet, wrap_phi(phi + plan["jet_dphi"].gather(1, column)), phi)
        zero = torch.zeros_like(eta)
        return (
            values["pt"],
            torch.where(complete, lab_eta, zero),
            torch.where(complete, lab_phi, zero),
            torch.where(complete, slot, torch.zeros_like(slot)),
            complete,
        )

    def pair_bias_matrix(self, batch: Mapping[str, Any], total_len: int) -> torch.Tensor:
        """Return the attention bias between complete particles, `[B, heads, total_len,
        total_len]`. It is zero for all other positions."""
        pt, eta, phi, slot, exists = self.pair_streams(batch)
        batch_size, length = pt.shape
        first = self.prefix_len
        last = (exists.any(dim=0) * torch.arange(1, length + 1, device=pt.device)).max()
        # rounded up to a multiple of 64, so that a compiled forward sees few distinct shapes
        last = torch.div(last + 63, 64, rounding_mode="floor") * 64
        span = max(0, min(int(last.item()), length, total_len - first))
        bias = pt.new_zeros(batch_size, self.pair_bias.n_bias_heads, total_len, total_len)
        if span > 0:
            position = torch.arange(span, device=pt.device)
            streams = (
                pt[:, :span],
                eta[:, :span],
                phi[:, :span],
                slot[:, :span],
                exists[:, :span],
            )
            bias[:, :, first : first + span, first : first + span] = self.pair_bias(
                streams, streams, position, position
            )
        return bias

    def pair_bias_row(
        self, batch: Mapping[str, Any], position: int, total_len: int
    ) -> torch.Tensor:
        """Return one row of the bias matrix, `[B, heads, 1, total_len]`, for incremental
        decoding."""
        pt, eta, phi, slot, exists = self.pair_streams(batch)
        first = self.prefix_len
        span = min(pt.shape[1], total_len - first)
        key = (pt[:, :span], eta[:, :span], phi[:, :span], slot[:, :span], exists[:, :span])
        query = tuple(stream[:, position : position + 1] for stream in key)
        positions = torch.arange(span, device=pt.device)
        row = pt.new_zeros(pt.shape[0], self.pair_bias.n_bias_heads, 1, total_len)
        row[:, :, :, first : first + span] = self.pair_bias(
            query, key, positions[position : position + 1], positions
        )
        return row

    def embed_particles(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Return the input embeddings of all particle positions, `[B, L, dim]`."""
        tokens = {name: batch[f"{name}_token"].long() for name in self.feature_names}
        embedded = self.feature_projection(self.feature_embedding(tokens))
        embedded = embedded + self.embed_kinematics(batch)
        return embedded + self.embed_jet_remainder(batch)

    def embed_particle_position(self, batch: Mapping[str, Any], position: int) -> torch.Tensor:
        """Return the input embedding of one particle position, `[B, 1, dim]`."""
        tokens = {
            name: batch[f"{name}_token"][:, position : position + 1].long()
            for name in self.feature_names
        }
        embedded = self.feature_projection(self.feature_embedding(tokens))
        embedded = embedded + self.embed_kinematics(batch, position)
        return embedded + self.embed_jet_remainder(batch, position)

    def embed_given(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Embed the four given tokens, `[B, 4, dim]`."""
        return torch.stack(
            [
                self.source_embedding(batch["source"].reshape(-1).long()),
                self.feature_mode_embedding(batch["feature_mode"].reshape(-1).long()),
                self.order_embedding(batch["feature_order"].reshape(-1).long()),
                self.condition_embedding(batch["condition"].reshape(-1).long()),
            ],
            dim=1,
        )

    def embed_count(self, count_tokens: torch.Tensor) -> torch.Tensor:
        """Embed the particle-count token, `[B, 1, dim]`."""
        return (self.count_embedding(count_tokens) + self.count_tag).unsqueeze(1)

    def embed_prefix(
        self, batch: Mapping[str, Any], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Embed the given tokens, the particle count and the plan, `[B, prefix_len, dim]`. The
        token embeddings are cast to `dtype`, the precision of the particle embeddings."""
        plan_tokens = {
            name: batch[f"plan_{name}_token"].long() for name in self.plan_head.field_names
        }
        count = self.count_embedding(batch["count_token"].reshape(-1).long())
        return torch.cat(
            [
                self.embed_given(batch).to(dtype),
                (count.to(dtype) + self.count_tag).unsqueeze(1),
                self.plan_head.embed(plan_tokens).to(dtype),
            ],
            dim=1,
        )

    def feature_logits(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(hidden) for name, head in self.feature_heads.items()}

    def impact_parameter_support(self, feature: str, pid_tokens: torch.Tensor) -> torch.Tensor:
        """Return which d0 or dz tokens a particle with the given pid token can have, as a `[...,
        vocab]` mask."""
        vocab = self.codecs[feature].vocab_size
        charged = self.pid_token_is_charged[
            pid_tokens.long().clamp(0, self.pid_token_is_charged.numel() - 1)
        ]
        token = torch.arange(vocab, device=pid_tokens.device)
        measured = (token >= 2) & (token <= vocab - 2)
        sentinel = token == 1
        neutral = (charged == 0).unsqueeze(-1)
        return ~((neutral & measured) | ((charged == 1).unsqueeze(-1) & sentinel))

    def pid_of_predicted(
        self, feature: str, batch: Mapping[str, Any], position: Optional[int] = None
    ) -> torch.Tensor:
        """Return the pid token of the particle whose `feature` token is predicted at each
        position, or at `position`. It is PAD where the pid is not emitted yet."""
        shifts = self.shifts(batch["feature_order"])
        offset = shifts["pid"] - shifts[feature] + 1
        pid_tokens = batch["pid_token"]
        if position is not None:
            source = position + offset
        else:
            source = (
                torch.arange(pid_tokens.shape[1], device=pid_tokens.device).unsqueeze(0) + offset
            )
        tokens = pid_tokens.gather(1, source.clamp(min=0))
        tokens = torch.where(source >= 0, tokens, torch.zeros_like(tokens))
        return tokens.view(-1) if position is not None else tokens

    def forward(self, batch: Mapping[str, Any], tagging: bool = False) -> Dict[str, Any]:
        """One pass over a tokenized batch.

        `tagging=False`: the generative pass with causal attention. Returns the logits of
        every predicted token: `count` `[B, vocab]`, `plan` `{field: [B, slots, vocab]}`
        and `features` `{feature: [B, L, vocab]}` (position `p` predicts `p + 1`).

        `tagging=True`: attention in both directions, with the source and the condition
        not given. Returns `classes` `{task: logits}` (`[B, classes]` for an event
        task, `[B, slots, classes]` for the jet task) and `jet_valid`.
        """
        mask = batch["attention_mask"].to(torch.bool)
        if tagging:
            batch = {
                **batch,
                "source": torch.zeros_like(batch["source"]),
                "condition": torch.zeros_like(batch["condition"]),
            }
        particles = self.embed_particles(batch)
        batch_size, length = mask.shape
        end = mask.sum(dim=1).clamp_(0, length)
        rows = torch.arange(batch_size, device=mask.device)

        # the class token of each event task sits one position after the event's tokens
        n_queries = len(self.event_tasks)
        embedded = torch.cat([particles, particles.new_zeros(batch_size, n_queries, self.dim)], 1)
        valid = torch.cat([mask, mask.new_zeros(batch_size, n_queries)], dim=1)
        for i, task in enumerate(self.event_tasks):
            embedded[rows, end + i] = self.class_query[task["name"]].to(embedded.dtype)
            valid[rows, end + i] = True
        embedded = torch.cat([self.embed_prefix(batch, embedded.dtype), embedded], dim=1)
        valid = torch.cat([valid.new_ones(batch_size, self.prefix_len), valid], dim=1)

        bias = self.pair_bias_matrix(batch, embedded.shape[1])
        hidden = self.transformer(embedded, attention_mask(valid, causal=not tagging, bias=bias))
        prefix_hidden, hidden = hidden[:, : self.prefix_len], hidden[:, self.prefix_len :]

        if not tagging:
            return {
                "count": self.count_head(prefix_hidden[:, 3]),
                "plan": self.plan_head.logits(prefix_hidden[:, 4:]),
                "features": self.feature_logits(hidden[:, :length]),
            }
        classes: Dict[str, torch.Tensor] = {}
        position = torch.arange(hidden.shape[1], device=mask.device)
        for i, task in enumerate(self.event_tasks):
            pooled = self.read_head(
                hidden, position[None, :] < end[:, None], hidden[rows, end + i]
            )
            classes[task["name"]] = self.class_heads[task["name"]](pooled)
        out: Dict[str, Any] = {"classes": classes}
        if self.jet_task is not None:
            inputs = self.jet_decoder_inputs(batch, hidden[:, :length], mask)
            classes[self.jet_task["name"]] = self.decode_jets(inputs)
            out["jet_valid"] = inputs["jet_valid"]
        return out

    def jet_decoder_inputs(
        self, batch: Mapping[str, Any], hidden: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        plan = self.plan_values(batch)
        seeds, jet_valid, pairs = JetAssignmentDecoder.jet_features(
            plan["jet_pt"], plan["jet_rapidity"], plan["jet_dphi"], plan["n_const"]
        )
        values, complete = self.complete_particles(batch, ("jet_idx",))
        slot = values["jet_idx"].round().long()
        return {
            "keys": hidden,
            "key_valid": mask,
            "membership": JetAssignmentDecoder.membership(slot, complete, self.n_jet_slots),
            "seeds": seeds,
            "jet_valid": jet_valid,
            "pair_feats": pairs,
        }

    def decode_jets(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the flavour logits of the planned jets, `[B, n_jet_slots, classes]`."""
        return self.class_heads[self.jet_task["name"]](self.jet_decoder(**inputs))

    def new_cache(self, batch_size: int, max_len: int, device, dtype) -> List[KeyValueCache]:
        return self.transformer.new_cache(batch_size, self.prefix_len + max_len, device, dtype)
