"""The dataset configuration: the token design shared by all members, and the members."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class MemberConfig:
    """One dataset (or one process of it) of a training set."""

    # "jetclass" or "collide2v" (humite.data.readers)
    reader: str
    # split ("train", "val", "test") -> glob patterns of the converted parquet files
    files: Dict[str, List[str]]
    # batch width; at least the largest particle count plus the number of particle features
    pad_length: int
    # index of the SOURCE token (0 is reserved for "not given")
    source_id: int
    data_dir: str = ""
    # COLLIDE-2V: the file name prefix of the member's process
    process: Optional[str] = None
    # event value that can be given in the conditioning slot, and the offset of its codes
    condition_label: Optional[str] = None
    condition_offset: int = 0
    reader_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    # one entry per particle feature: name, binning, vocab_size, value_min, value_max, ...
    particle_features: List[Dict[str, Any]]
    # one entry per jet field of the plan: name, binning, vocab_size, value_min, value_max
    plan: List[Dict[str, Any]]
    n_jet_slots: int
    # the largest particle count of the training set; sizes the particle-count vocabulary
    max_particles: int
    # base order in which a particle's features are emitted, and the constraints
    # [first, second] every random order must respect
    feature_order: List[str]
    feature_order_precedence: List[List[str]]
    n_feature_orders: int
    feature_order_seed: int
    # features hidden on kinematics-only rows, and the training fraction of complete rows
    optional_features: List[str]
    p_all_features: float
    # training probability of giving the condition label
    p_condition: float
    n_sources: int
    members: Dict[str, Any]
    # relative number of training / validation batches per member
    mix_weights: Dict[str, float]
    val_mix_weights: Dict[str, float]
    seed: int = 44

    def __post_init__(self) -> None:
        plain = lambda node: OmegaConf.to_container(node, resolve=True)  # noqa: E731
        for name in ("particle_features", "plan", "members", "mix_weights", "val_mix_weights"):
            value = getattr(self, name)
            if OmegaConf.is_config(value):
                setattr(self, name, plain(value))
        self.members = {
            name: member if isinstance(member, MemberConfig) else MemberConfig(**member)
            for name, member in self.members.items()
        }
        for weights in (self.mix_weights, self.val_mix_weights):
            if set(weights) != set(self.members):
                raise ValueError(f"mix weights {sorted(weights)} do not name the members")
        for name, member in self.members.items():
            if not 1 <= member.source_id < self.n_sources:
                raise ValueError(f"member {name}: source_id must lie in 1..{self.n_sources - 1}")

    @property
    def feature_names(self) -> List[str]:
        return [str(entry["name"]) for entry in self.particle_features]
