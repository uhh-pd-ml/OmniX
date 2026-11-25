"""Readers of the converted datasets.

A reader turns a row range of one parquet file into an EventBlock: the particles of
every event in the layout the tokenizer expects, plus the event values and jet flavours the
conditioning slot and the classification tasks read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from humite.data.particle_types import NO_TRACK, charge_from_pdg_id, pf_type_from_pdg_id

log = logging.getLogger(__name__)

# particle columns of an event block, in this order
PARTICLE_COLUMNS: Tuple[str, ...] = (
    "pt",
    "eta",
    "phi",
    "jet_idx",
    "charge",
    "pf_type",
    "d0",
    "dz",
)


@dataclass
class EventBlock:
    # ragged record array with the fields `PARTICLE_COLUMNS` (float32)
    particles: ak.Array
    # per-event values, each of length n_events
    values: Dict[str, np.ndarray]
    # flavour class of every jet slot (2 = b, 1 = c, 0 = light, -1 = no jet), or None
    jet_flavor: Optional[np.ndarray] = None


def row_count(path: str) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def read_rows(path: str, start: int, stop: int, step: int, columns: Sequence[str]) -> ak.Array:
    """Read the rows `start:stop:step` of a parquet file, loading only the row groups they
    touch."""
    file = pq.ParquetFile(path)
    group_rows = [file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)]
    edges = np.cumsum([0, *group_rows])
    first = max(0, int(np.searchsorted(edges, start, side="right")) - 1)
    last = max(first + 1, min(file.num_row_groups, int(np.searchsorted(edges, stop, side="left"))))
    rows = ak.from_parquet(path, row_groups=list(range(first, last)), columns=list(columns))
    offset = int(edges[first])
    return rows[start - offset : stop - offset : step]


def wrap_phi(phi):
    """Map an angle (numpy, awkward or torch) to [-pi, pi)."""
    return (phi + math.pi) % (2.0 * math.pi) - math.pi


class JetClassReader:
    """JetClass jets (`scripts/data/convert_jetclass.py`), one jet per event.

    The jet is written as an event with a single jet in slot 1: every constituent within
    `window` of the jet axis in eta and phi is kept (the leading one always), eta is the lab
    pseudorapidity, and phi is measured from the summed transverse momentum of the kept
    constituents.
    """

    COLUMNS = ("pt", "eta", "phi", "src_jet_eta", "jc_label", "charge", "pf_type", "d0", "dz")
    # the jet types, in the order of `jc_label`
    CLASSES: Tuple[str, ...] = (
        "QCD",
        "HToBB",
        "HToCC",
        "HToGG",
        "HToWW4Q",
        "HToWW2Q1L",
        "ZToQQ",
        "WToQQ",
        "TTBar",
        "TTBarLep",
    )

    def __init__(self, window: float = 1.3) -> None:
        self.window = float(window)

    def read(self, path: str, start: int, stop: int, step: int = 1) -> EventBlock:
        rows = read_rows(path, start, stop, step, self.COLUMNS)
        pt = ak.values_astype(rows["pt"], np.float32)
        deta = ak.values_astype(rows["eta"] - rows["src_jet_eta"], np.float32)
        dphi = ak.values_astype(rows["phi"], np.float32)
        keep = (abs(deta) <= self.window) & (abs(dphi) <= self.window)
        keep = keep | (ak.local_index(pt, axis=1) == ak.argmax(pt, axis=1, keepdims=True))
        pt, deta, dphi = pt[keep], deta[keep], dphi[keep]

        jet_eta = np.asarray(ak.to_numpy(rows["src_jet_eta"]), dtype=np.float32)
        eta = ak.values_astype(deta + ak.Array(jet_eta)[:, np.newaxis], np.float32)
        jet_idx = ak.ones_like(pt)
        in_jet = jet_idx == 1
        sum_x = ak.sum(ak.where(in_jet, pt * np.cos(dphi), 0.0), axis=1)
        sum_y = ak.sum(ak.where(in_jet, pt * np.sin(dphi), 0.0), axis=1)
        axis_phi = np.arctan2(np.asarray(ak.to_numpy(sum_y)), np.asarray(ak.to_numpy(sum_x)))
        phi = dphi - ak.Array(axis_phi.astype(np.float32))[:, np.newaxis]
        phi = ak.values_astype(wrap_phi(phi), np.float32)

        fields = {"pt": pt, "eta": eta, "phi": phi, "jet_idx": jet_idx}
        for name in ("charge", "pf_type", "d0", "dz"):
            fields[name] = ak.values_astype(rows[name], np.float32)[keep]
        values = {"jc_label": np.asarray(ak.to_numpy(rows["jc_label"]), dtype=np.float32)}
        return EventBlock(ak.zip(fields), values)


class Collide2VReader:
    """COLLIDE-2V events (`scripts/data/convert_c2v_fullevent.py`).

    For every block of rows:

    1. particles with |eta| above `particle_abs_eta_max` are dropped, and with them events
       left without particles;
    2. jets: the stored anti-kT R = 0.4 jets with pT > `jet_pt_min` and |eta| <
       `jet_abs_eta_max`, at most `max_jets`, numbered 1..k by pT; particles of other jets
       and unclustered particles get jet_idx 0;
    3. phi is rotated so that the leading selected jet sits at phi = 0; an event without a
       jet has its highest-pT particle rotated there;
    4. d0 and dz of charged particles are smeared with the CMS Phase-2 track resolution
       (`phase2_track_resolution.json`). The draws depend only on the file name, the row
       and the particle's stored index, so an event gets the same values on every read;
    5. the particle type is reduced to the eight particle-flow types.

    Event values: `c2v_process` (index into `PROCESSES`). Jet flavours: 2 = b, 1 = c,
    0 = light for each selected jet.
    """

    FORMAT = "c2v_fullevent"
    COLUMN_TYPES: Dict[str, str] = {
        **{c: "var * float32" for c in ("pt", "eta", "phi", "d0", "dz")},
        "charge": "var * int8",
        "pdg_id": "var * int32",
        "is_pu": "var * int8",
        "ak4_idx": "var * int8",
        "ak8_idx": "var * int8",
        "process": "string",
        "n_pu": "int16",
        "event_weight": "float32",
        **{f"ak4_{q}": "var * float32" for q in ("pt", "eta", "phi", "m")},
        **{f"ak8_{q}": "var * float32" for q in ("pt", "eta", "phi", "m")},
        "ak4_flavor": "var * int8",
        "ak8_flavor": "var * int8",
    }
    # the converter's definitions this reader relies on
    DEFINITIONS: Dict[str, Any] = {
        "phi_frame": "absolute",
        "eta_phi_storage": "float32, dequantized from float16",
        "charged_hadron_eta": "corrected: rapidity of the stored (PT, Eta, Mass)",
        "d0_dz_unit": "cm",
        "d0_dz_clipped": False,
        "jet_flavor": "Delphes Flavor, raw code",
        "no_track": NO_TRACK,
    }
    PROCESSES: Tuple[str, ...] = (
        "QCD_HT50toInf",
        "QCD_HT50tobb",
        "tt0123j_5f_ckm_LO_MLM_leptonic",
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
        "tt0123j_5f_ckm_LO_MLM_hadronic",
    )
    TRACK_TABLE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phase2_track_resolution.json"
    )
    TRACK_TABLE_SHA256 = "e1ac6a88a32270b544f1ce00cd83100e7b8e0d2628d9bec0e2e0152b8804dedd"
    TRACK_TABLE_FORMAT = "humite track resolution table, version 1"
    READ_COLUMNS = (
        "pt",
        "eta",
        "phi",
        "charge",
        "pdg_id",
        "d0",
        "dz",
        "ak4_idx",
        "ak4_flavor",
        "process",
        "ak4_pt",
        "ak4_eta",
        "ak4_phi",
    )
    PARTICLE_FIELDS = ("pt", "eta", "phi", "charge", "pdg_id", "d0", "dz", "ak4_idx")

    def __init__(
        self,
        n_jet_slots: int,
        d0_range: Tuple[float, float],
        dz_range: Tuple[float, float],
        split: Optional[str] = None,
        jet_pt_min: float = 15.0,
        jet_abs_eta_max: float = 4.0,
        max_jets: int = 20,
        particle_abs_eta_max: float = 4.0,
    ) -> None:
        if max_jets > n_jet_slots:
            raise ValueError(f"max_jets {max_jets} exceeds the {n_jet_slots} jet slots")
        self.n_jet_slots = int(n_jet_slots)
        self.clip = {"d0": tuple(d0_range), "dz": tuple(dz_range)}
        self.split = split
        self.jet_pt_min = float(jet_pt_min)
        self.jet_abs_eta_max = float(jet_abs_eta_max)
        self.max_jets = int(max_jets)
        self.particle_abs_eta_max = float(particle_abs_eta_max)
        self.track_table = self.read_track_table(self.TRACK_TABLE)
        self._checked: set = set()

    @classmethod
    def read_track_table(cls, path: str) -> Dict[str, np.ndarray]:
        """The d0 / dz widths (cm) per bin of |eta| (rows) and pT (columns), with the lower pT
        edges and the upper |eta| edges of the bins."""
        with open(path, "rb") as file:
            data = file.read()
        if hashlib.sha256(data).hexdigest() != cls.TRACK_TABLE_SHA256:
            raise ValueError(f"{path} is not the track resolution table this reader expects")
        raw = json.loads(data)
        if raw.get("format") != cls.TRACK_TABLE_FORMAT:
            raise ValueError(f"{path}: not a '{cls.TRACK_TABLE_FORMAT}'")
        return {
            "pt_edges": np.asarray(raw["pt_lower_edges"], dtype=np.float64),
            "abs_eta_edges": np.asarray(raw["abs_eta_upper_edges"], dtype=np.float64),
            "d0": np.asarray(raw["d0"], dtype=np.float64),
            "dz": np.asarray(raw["dz"], dtype=np.float64),
        }

    def check_file(self, path: str) -> None:
        """Check the format, columns, converter definitions and split of a file, once per file."""
        if path in self._checked:
            return
        schema = pq.read_schema(path)
        metadata = schema.metadata or {}
        if b"conversion" not in metadata:
            raise ValueError(f"{path}: no 'conversion' metadata, so not a {self.FORMAT} file")
        block = json.loads(metadata[b"conversion"])
        problems = []
        if block.get("format") != self.FORMAT:
            problems.append(f"format {block.get('format')!r}, expected {self.FORMAT!r}")
        missing = sorted(set(self.COLUMN_TYPES) - set(schema.names))
        if missing:
            problems.append(f"columns missing: {missing}")
        stored = dict(block.get("definitions") or {})
        if stored != self.DEFINITIONS:
            problems.append(f"definitions {stored}, expected {self.DEFINITIONS}")
        split = (block.get("info") or {}).get("split")
        folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if split != folder or (self.split is not None and split != self.split):
            problems.append(f"split {split!r} in folder {folder!r}, loading {self.split!r}")
        if problems:
            raise ValueError(f"{path} does not match the reader:\n  - " + "\n  - ".join(problems))
        self._checked.add(path)

    @staticmethod
    def _splitmix64(x: np.ndarray) -> np.ndarray:
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    @classmethod
    def fixed_normals(
        cls, file_name: str, row: np.ndarray, particle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two standard normal draws per particle that depend only on the file name, the
        event's row and the particle's stored index (hash, then Box-Muller)."""
        digest = hashlib.sha1(f"track smearing:{file_name}".encode()).digest()
        file_key = np.uint64(int.from_bytes(digest[:8], "big"))
        key = row.astype(np.uint64) << np.uint64(17)
        key |= particle.astype(np.uint64) << np.uint64(1)
        uniform = []
        for axis in (0, 1):
            bits = cls._splitmix64(cls._splitmix64(key | np.uint64(axis)) ^ file_key)
            uniform.append((bits >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53))
        radius = np.sqrt(-2.0 * np.log1p(-uniform[0]))
        angle = 2.0 * np.pi * uniform[1]
        return radius * np.cos(angle), radius * np.sin(angle)

    def track_widths(self, pt: np.ndarray, abs_eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The d0 and dz resolution (cm) at each track's pT and |eta|; values beyond the table
        use its first or last bin."""
        table = self.track_table
        i_pt = np.searchsorted(table["pt_edges"][1:], pt, side="left")
        i_eta = np.searchsorted(table["abs_eta_edges"], abs_eta, side="left")
        return table["d0"][i_eta, i_pt], table["dz"][i_eta, i_pt]

    def select_jets(self, rows: ak.Array, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(rank, event)` of every stored jet, flat over the block. The rank is 1..k among
        the selected jets of its event in pT order, and 0 for a jet that is not selected."""
        n_jets = np.asarray(ak.to_numpy(ak.num(rows["ak4_pt"])), dtype=np.int64)
        event = np.repeat(np.arange(len(rows)), n_jets)
        pt = np.asarray(ak.to_numpy(ak.flatten(rows["ak4_pt"])), dtype=np.float64)
        eta = np.asarray(ak.to_numpy(ak.flatten(rows["ak4_eta"])), dtype=np.float64)
        if np.any(np.diff(pt)[event[1:] == event[:-1]] > 0):
            raise ValueError(f"{path}: the stored jets are not ordered by pT")
        passing = (pt > self.jet_pt_min) & (np.abs(eta) < self.jet_abs_eta_max)
        first = np.concatenate([[0], np.cumsum(n_jets)])[:-1]
        cumulative = np.concatenate([[0], np.cumsum(passing)])
        rank = cumulative[1:] - cumulative[first][event]
        rank = np.where(passing & (rank <= self.max_jets), rank, 0).astype(np.int64)
        return rank, event

    def read(self, path: str, start: int, stop: int, step: int = 1) -> EventBlock:
        self.check_file(path)
        rows = read_rows(path, start, stop, step, self.READ_COLUMNS)
        wrong = [c for c in rows.fields if str(rows[c].type.content) != self.COLUMN_TYPES[c]]
        if wrong:
            raise ValueError(f"{path}: unexpected types of the columns {wrong}")

        file_row = start + step * np.arange(len(rows))
        stored_index = ak.local_index(rows["pt"], axis=1)
        keep = np.abs(rows["eta"]) <= self.particle_abs_eta_max
        for column in self.PARTICLE_FIELDS:
            rows = ak.with_field(rows, rows[column][keep], column)
        stored_index = stored_index[keep]
        nonempty = np.asarray(ak.to_numpy(ak.num(rows["pt"]))) > 0
        if not nonempty.all():
            rows, file_row, stored_index = (
                rows[nonempty],
                file_row[nonempty],
                stored_index[nonempty],
            )

        n_events = len(rows)
        counts = np.asarray(ak.to_numpy(ak.num(rows["pt"])), dtype=np.int64)
        if n_events == 0:
            raise ValueError(f"{path} rows {start}..{stop}: no event with particles")
        event = np.repeat(np.arange(n_events), counts)
        flat = {c: np.asarray(ak.to_numpy(ak.flatten(rows[c]))) for c in self.PARTICLE_FIELDS}

        rank, jet_event = self.select_jets(rows, path)
        n_stored = np.bincount(jet_event, minlength=n_events)
        stored_jet = flat["ak4_idx"].astype(np.int64)
        if np.any(stored_jet < 0) or np.any(stored_jet > n_stored[event]):
            raise ValueError(f"{path}: ak4_idx points outside the event's jet list")
        first_jet = np.concatenate([[0], np.cumsum(n_stored)])[:-1]
        jet_idx = np.zeros_like(stored_jet)
        in_jet = stored_jet > 0
        jet_idx[in_jet] = rank[first_jet[event[in_jet]] + stored_jet[in_jet] - 1]

        phi = flat["phi"].astype(np.float64)
        jet_phi = np.asarray(ak.to_numpy(ak.flatten(rows["ak4_phi"])), dtype=np.float64)
        leading = rank == 1
        shift = np.zeros(n_events)
        shift[jet_event[leading]] = 0.0 - jet_phi[leading]
        no_jet = np.ones(n_events, dtype=bool)
        no_jet[jet_event[leading]] = False
        first_particle = np.concatenate([[0], np.cumsum(counts)])[:-1]
        leading_particle = first_particle + np.asarray(
            ak.to_numpy(ak.fill_none(ak.argmax(rows["pt"], axis=1), 0)), dtype=np.int64
        )
        shift[no_jet] = 0.0 - phi[leading_particle[no_jet]]
        phi = wrap_phi(phi + shift[event]).astype(np.float32)

        charged = flat["charge"] != 0
        has_track = (flat["d0"] != NO_TRACK) & (flat["dz"] != NO_TRACK)
        no_track = (flat["d0"] == NO_TRACK) & (flat["dz"] == NO_TRACK)
        if np.any(np.where(charged, ~has_track, ~no_track)):
            raise ValueError(f"{path}: charge, d0 and dz disagree on which particles are tracks")
        particle = np.asarray(ak.to_numpy(ak.flatten(stored_index)), dtype=np.int64)
        if particle.max() >= 2**16:
            raise ValueError(f"{path}: a particle index does not fit the smearing key")
        normal_d0, normal_dz = self.fixed_normals(
            os.path.basename(path), file_row[event][charged], particle[charged]
        )
        width_d0, width_dz = self.track_widths(
            flat["pt"][charged].astype(np.float64),
            np.abs(flat["eta"][charged].astype(np.float64)),
        )
        track = {}
        for axis, noise in (("d0", width_d0 * normal_d0), ("dz", width_dz * normal_dz)):
            values = flat[axis].astype(np.float64)
            values[charged] += noise
            values = values.astype(np.float32)
            low, high = self.clip[axis]
            track[axis] = np.where(values != NO_TRACK, np.clip(values, low, high), values)

        expected_charge = charge_from_pdg_id(flat["pdg_id"])
        if np.any(np.isfinite(expected_charge) & (expected_charge != flat["charge"])):
            raise ValueError(f"{path}: a particle's charge disagrees with its PDG id")

        def column(values: np.ndarray) -> ak.Array:
            return ak.unflatten(np.asarray(values, dtype=np.float32), counts)

        particles = ak.zip(
            {
                "pt": column(flat["pt"]),
                "eta": column(flat["eta"]),
                "phi": column(phi),
                "jet_idx": column(jet_idx),
                "charge": column(flat["charge"]),
                "pf_type": column(pf_type_from_pdg_id(flat["pdg_id"])),
                "d0": column(track["d0"]),
                "dz": column(track["dz"]),
            }
        )

        selected = rank > 0
        code = np.asarray(ak.to_numpy(ak.flatten(rows["ak4_flavor"])), dtype=np.int64)
        flavor_class = np.where(code == 5, 2, np.where(code == 4, 1, 0))
        jet_flavor = np.full((n_events, self.n_jet_slots), -1, dtype=np.int64)
        jet_flavor[jet_event[selected], rank[selected] - 1] = flavor_class[selected]
        process = ak.to_list(rows["process"])
        unknown = set(process) - set(self.PROCESSES)
        if unknown:
            raise ValueError(f"{path}: unknown processes {sorted(unknown)}")
        label = {name: index for index, name in enumerate(self.PROCESSES)}
        values = {"c2v_process": np.asarray([label[p] for p in process], dtype=np.float32)}
        return EventBlock(particles, values, jet_flavor)


def condition_codes(data) -> Dict[str, int]:
    """Return the code of the conditioning token for every jet type and process of the members of
    `data`, keyed by class name."""
    names = {"jetclass": JetClassReader.CLASSES, "collide2v": Collide2VReader.PROCESSES}
    codes: Dict[str, int] = {}
    for member in data.members.values():
        for index, name in enumerate(names[member.reader]):
            codes[name] = 1 + member.condition_offset + index
    return codes


def build_reader(
    name: str, options: Mapping[str, Any], codecs: Mapping[str, Any], n_jet_slots: int, split: str
):
    if name == "jetclass":
        return JetClassReader(**options)
    if name == "collide2v":
        return Collide2VReader(
            n_jet_slots=n_jet_slots,
            d0_range=(codecs["d0"].value_min, codecs["d0"].value_max),
            dz_range=(codecs["dz"].value_min, codecs["dz"].value_max),
            split=split,
            **options,
        )
    raise ValueError(f"unknown reader {name!r}")
