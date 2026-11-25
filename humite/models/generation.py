"""Sampling jets and events from an OmniX model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from humite.data.readers import wrap_phi
from humite.data.tokenizer import KINEMATICS, EventTokenizer
from humite.models.omnix import OmniX
from humite.models.transformer import attention_mask


def sample_tokens(
    logits: torch.Tensor, temperature: float = 1.0, forbidden: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Draw one token per row of `logits` (`[B, vocab]`) at the given temperature. The token ids in
    `forbidden` are never drawn."""
    logits = logits / temperature
    if forbidden is not None:
        logits.index_fill_(-1, forbidden, float("-inf"))
    probabilities = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0).clamp(min=0.0)
    total = probabilities.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
    probabilities = torch.where(total > 0, probabilities / total.clamp_min(1e-8), uniform)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: OmniX,
    tokenizer: EventTokenizer,
    batch_size: int,
    source: int,
    condition: int = 0,
    all_features: bool = True,
    temperature: float = 1.0,
    out_of_jet_exclusion: Optional[float] = None,
    max_len: int = 704,
) -> Dict[str, Any]:
    """Sample `batch_size` jets or events.

    `source`: 1 = JetClass jets, 2 = COLLIDE-2V events. `condition`: 0 = no class given,
    1-10 a JetClass jet type, 11-15 a COLLIDE-2V process. `all_features=False` generates
    the kinematics (pT, eta, phi, jet slot) only. `out_of_jet_exclusion`: a radius R; a
    particle outside every jet is then not placed within R of a planned jet axis (the paper's
    event samples use R = 0.4). `max_len` bounds the particle positions.

    Returns `particles` `[B, L, 8]` (columns: humite.data.readers.PARTICLE_COLUMNS, lab
    frame; `L` is the largest particle count of the batch), `mask` `[B, L]` and `jets`,
    the planned (jet_pt, jet_rapidity, jet_dphi, jet_m, n_const) of each jet slot.
    """
    device = model.shift_table.device
    names = model.feature_names
    plan = model.plan_head

    state: Dict[str, Any] = {
        "feature_order": torch.randint(model.shift_table.shape[0], (batch_size,), device=device),
        "source": torch.full((batch_size,), source, device=device),
        "condition": torch.full((batch_size,), condition, device=device),
        "feature_mode": torch.full((batch_size,), int(all_features), device=device),
    }
    shifts = {name: shift.view(-1) for name, shift in model.shifts(state["feature_order"]).items()}
    position = torch.arange(max_len, device=device).unsqueeze(0)
    for name in names:
        tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        tokens[position < shifts[name].view(-1, 1)] = model.codecs[name].missing_id
        state[f"{name}_token"] = tokens
    for name in KINEMATICS:
        state[f"{name}_value"] = torch.zeros((batch_size, max_len), device=device)

    # the particle count and the plan, one token after the other
    columns = [model.embed_given(state)]

    def last_hidden() -> torch.Tensor:
        sequence = torch.cat(columns, dim=1)
        valid = torch.ones(sequence.shape[:2], dtype=torch.bool, device=device)
        return model.transformer(sequence, attention_mask(valid, causal=True))[:, -1]

    count = sample_tokens(model.count_head(last_hidden()), temperature)
    state["count_token"] = count
    n_particles = count.clamp(1, max_len)
    plan_tokens = {
        name: torch.ones((batch_size, plan.n_slots), dtype=torch.long, device=device)
        for name in plan.field_names
    }
    for plan_position in range(plan.n_positions):
        columns.append(
            model.embed_count(count)
            if plan_position == 0
            else plan.embed_position(plan_tokens, plan_position - 1)
        )
        hidden = last_hidden()
        for offset, name in enumerate(plan.field_names):
            slot = plan_position - offset
            if 0 <= slot < plan.n_slots:
                tokens = sample_tokens(plan.heads[name](hidden), temperature)
                plan_tokens[name][:, slot] = tokens.clamp(1, plan.codecs[name].nbins)
    for name, tokens in plan_tokens.items():
        state[f"plan_{name}_token"] = tokens
    jets = plan.decode(plan_tokens, sample=True)

    if out_of_jet_exclusion is not None:
        centres = plan.decode(plan_tokens)
        axis = {"eta": centres["jet_rapidity"].float(), "phi": centres["jet_dphi"].float()}
        planned = centres["n_const"] > 0.5
        grid = {name: model.codecs[name].out_of_jet.bin_centers.to(device) for name in axis}

    def earlier_token(feature: str, other: str, t: int):
        """For the particle whose `feature` token is sampled next, return whether its `other` token
        is emitted already, and that token."""
        lag = shifts[feature] - shifts[other]
        source_position = (t + 1) - lag
        tokens = state[f"{other}_token"].gather(
            1, source_position.clamp(0, max_len - 1).view(-1, 1)
        )
        return (lag >= 1) & (source_position >= 0), tokens.view(-1)

    def exclusion(feature: str, t: int) -> Optional[torch.Tensor]:
        """Return the eta or phi bins allowed for a particle outside every jet whose other
        coordinate is known: those not within R of a planned jet axis. The mask is `[B, vocab]`,
        or None if no row is affected."""
        other = "phi" if feature == "eta" else "eta"
        slot_known, slot_token = earlier_token(feature, "jet_idx", t)
        other_known, other_token = earlier_token(feature, other, t)
        n_bins = grid[other].numel()
        applies = slot_known & (slot_token == 1) & other_known
        applies = applies & (other_token >= 1) & (other_token <= n_bins)
        if not bool(applies.any()):
            return None
        other_value = grid[other][(other_token - 1).clamp(0, n_bins - 1)]
        distance = grid[feature].view(1, 1, -1) - axis[feature].unsqueeze(-1)
        other_distance = other_value.view(-1, 1) - axis[other]
        if feature == "phi":
            distance = wrap_phi(distance)
        else:
            other_distance = wrap_phi(other_distance)
        inside = (distance**2 + other_distance.unsqueeze(-1) ** 2) < out_of_jet_exclusion**2
        excluded = (inside & planned.unsqueeze(-1)).any(dim=1) & applies.view(-1, 1)
        excluded &= ~excluded.all(dim=-1, keepdim=True)
        allowed = torch.ones(
            (batch_size, model.codecs[feature].vocab_size), dtype=torch.bool, device=device
        )
        allowed[:, 1 : grid[feature].numel() + 1] = ~excluded
        return allowed

    # the particles: position t + 1 is sampled from the state at position t
    total_len = model.prefix_len + max_len
    cache = model.new_cache(batch_size, max_len, device, model.count_tag.dtype)
    valid = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
    valid[:, : model.prefix_len] = True
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    stop = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    for t in range(max_len - 1):
        if not active.any():
            break
        valid[:, model.prefix_len + t] |= active
        if t == 0:
            first = torch.cat(
                [model.embed_prefix(state), model.embed_particles(state)[:, :1]], dim=1
            )
            bias = model.pair_bias_matrix(state, first.shape[1])
            mask = attention_mask(valid[:, : first.shape[1]], causal=True, bias=bias)
            hidden = model.transformer(first, mask, cache)[:, -1:]
        else:
            mask = torch.zeros((batch_size, 1, 1, total_len), device=device)
            mask = mask.masked_fill_(~valid[:, None, None, :], float("-inf"))
            mask = mask + model.pair_bias_row(state, t, total_len)
            hidden = model.transformer(model.embed_particle_position(state, t), mask, cache)
        logits = model.feature_logits(hidden)

        new_tokens = {}
        for name in names:
            feature_logits = logits[name][:, 0].clone()
            codec = model.codecs[name]
            if name in ("d0", "dz"):
                pid = model.pid_of_predicted(name, state, t)
                feature_logits = feature_logits.masked_fill(
                    ~model.impact_parameter_support(name, pid), float("-inf")
                )
            if out_of_jet_exclusion is not None and name in ("eta", "phi"):
                allowed = exclusion(name, t)
                if allowed is not None:
                    feature_logits = feature_logits.masked_fill(~allowed, float("-inf"))
            forbidden = torch.tensor([0, codec.missing_id], device=device)
            tokens = sample_tokens(feature_logits, temperature, forbidden)
            # after the last particle a feature's stream ends with MISSING
            ended = (stop >= 0) & (t + 1 - stop >= 1) & (t + 1 - stop > shifts[name] - 1)
            tokens = torch.where(ended, torch.full_like(tokens, codec.missing_id), tokens)
            if name in model.optional_features:
                hidden_row = state["feature_mode"] == 0
                tokens = torch.where(hidden_row, torch.full_like(tokens, codec.missing_id), tokens)
            new_tokens[name] = tokens

        for name in names:
            codec = model.codecs[name]
            tokens = new_tokens[name]
            tokens = torch.where(
                shifts[name] > t + 1, torch.full_like(tokens, codec.missing_id), tokens
            )
            stream = state[f"{name}_token"]
            stream[:, t + 1] = torch.where(active, tokens, stream[:, t + 1])
            if name in KINEMATICS:
                out_of_jet = None
                if codec.out_of_jet is not None:
                    slot_position = (t + 1) - (shifts[name] - shifts["jet_idx"])
                    slot_tokens = state["jet_idx_token"].gather(
                        1, slot_position.clamp(min=0).view(-1, 1)
                    )
                    slot_tokens = slot_tokens.view(-1)
                    slot_codec = model.codecs["jet_idx"]
                    known = (
                        (slot_tokens >= 1)
                        & (slot_tokens <= slot_codec.nbins)
                        & (slot_position >= 0)
                    )
                    out_of_jet = known & (slot_codec.decode(slot_tokens) < 0.5)
                value = codec.decode(tokens, out_of_jet, sample=True).float()
                values = state[f"{name}_value"]
                values[:, t + 1] = torch.where(active, value, values[:, t + 1])

        # an event ends when its count is reached and the delayed features are emitted
        reached = active & (stop < 0) & (t + 2 >= n_particles)
        stop = torch.where(reached, n_particles, stop)
        active = active & ~((stop >= 0) & (t + 1 - stop >= model.n_features))

    stop = torch.where(stop < 0, torch.full_like(stop, max_len), stop)
    state["particle_mask"] = position < stop.unsqueeze(1)
    for name, values in jets.items():
        state[f"plan_{name}"] = values
    particles, mask = tokenizer.decode(state)
    width = int(mask.sum(dim=1).max())
    return {"particles": particles[:, :width], "mask": mask[:, :width], "jets": jets}
