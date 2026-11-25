"""Particle types shared by JetClass and COLLIDE-2V.

A particle carries a charge and one of eight particle-flow types. The model sees both as one
signed id, `pid`; of the 13 possible ids only the first eight occur in the two datasets.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# impact-parameter value of a particle without a track
NO_TRACK = -999.0
# range of the impact parameters d0 and dz in cm; values beyond are clipped
D0_CLIP = 25.0
DZ_CLIP = 60.0

# particle-flow types
CHARGED_HADRON, NEUTRAL_HADRON, PHOTON, ELECTRON, MUON, HF_HADRON, HF_EM, OTHER = range(8)
N_PF_TYPES = 8

# pid -> (charge, particle-flow type)
PID_PAIRS: Tuple[Tuple[int, int], ...] = (
    (+1, CHARGED_HADRON),
    (-1, CHARGED_HADRON),
    (0, NEUTRAL_HADRON),
    (0, PHOTON),
    (+1, ELECTRON),
    (-1, ELECTRON),
    (+1, MUON),
    (-1, MUON),
    (0, HF_HADRON),
    (0, HF_EM),
    (+1, OTHER),
    (-1, OTHER),
    (0, OTHER),
)
PID_NAMES = ("h+", "h-", "h0", "photon", "e+", "e-", "mu+", "mu-")

_PID_OF_PAIR = np.full(3 * N_PF_TYPES, -1, dtype=np.int64)
for _pid, (_charge, _type) in enumerate(PID_PAIRS):
    _PID_OF_PAIR[(_charge + 1) * N_PF_TYPES + _type] = _pid
_CHARGE_OF_PID = np.array([charge for charge, _ in PID_PAIRS], dtype=np.float32)
_TYPE_OF_PID = np.array([pf_type for _, pf_type in PID_PAIRS], dtype=np.float32)

# COLLIDE-2V stores signed PDG ids; these are its charged hadrons
C2V_CHARGED_HADRONS = (211, 321, 2212, 3112, 3222, 3312, 3334)
# charge of the positive-id particle of each charged species
C2V_CHARGE_OF_POSITIVE_ID = {
    11: -1,
    13: -1,
    211: 1,
    321: 1,
    2212: 1,
    3112: -1,
    3222: 1,
    3312: -1,
    3334: -1,
}


# the PDG ids that occur in COLLIDE-2V (0 = Delphes neutral hadron)
C2V_PDG_SPECIES = (
    0,
    22,
    *(sign * pdg_id for pdg_id in C2V_CHARGE_OF_POSITIVE_ID for sign in (1, -1)),
)


def pid_from_charge_type(
    charge: torch.Tensor, pf_type: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Map (charge, particle-flow type) to the pid, as float. Raises if a real particle (`mask`)
    has a pair that is not in `PID_PAIRS`."""
    q = charge.round().to(torch.long)
    t = pf_type.round().to(torch.long)
    in_range = (q >= -1) & (q <= 1) & (t >= 0) & (t < N_PF_TYPES)
    flat = ((q + 1) * N_PF_TYPES + t).clamp(0, 3 * N_PF_TYPES - 1)
    pid = torch.as_tensor(_PID_OF_PAIR, device=flat.device)[flat]
    if bool((mask.to(torch.bool) & (~in_range | (pid < 0))).any()):
        raise ValueError("a particle has a (charge, type) pair that is not a pid")
    return pid.clamp_min(0).to(charge.dtype)


def charge_type_from_pid(pid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map the pid back to (charge, particle-flow type), as float."""
    index = pid.round().to(torch.long).clamp(0, len(PID_PAIRS) - 1)
    charge = torch.as_tensor(_CHARGE_OF_PID, device=index.device)[index]
    pf_type = torch.as_tensor(_TYPE_OF_PID, device=index.device)[index]
    return charge.to(pid.dtype), pf_type.to(pid.dtype)


def pf_type_from_pdg_id(pdg_id: np.ndarray) -> np.ndarray:
    """Map a COLLIDE-2V PDG id to the particle-flow type. Delphes stores neutral hadrons as id
    0."""
    ids = np.asarray(pdg_id).astype(np.int64)
    abs_ids = np.abs(ids)
    out = np.full(ids.shape, OTHER, dtype=np.int8)
    out[np.isin(abs_ids, C2V_CHARGED_HADRONS)] = CHARGED_HADRON
    out[ids == 0] = NEUTRAL_HADRON
    out[abs_ids == 22] = PHOTON
    out[abs_ids == 11] = ELECTRON
    out[abs_ids == 13] = MUON
    return out


def charge_from_pdg_id(pdg_id: np.ndarray) -> np.ndarray:
    """Return the charge of a COLLIDE-2V PDG id of a known species, and NaN for any other id."""
    ids = np.asarray(pdg_id).astype(np.int64)
    out = np.full(ids.shape, np.nan, dtype=np.float32)
    out[(ids == 0) | (ids == 22)] = 0.0
    for abs_id, charge in C2V_CHARGE_OF_POSITIVE_ID.items():
        out[ids == abs_id] = charge
        out[ids == -abs_id] = -charge
    return out
