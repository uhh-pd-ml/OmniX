#!/usr/bin/env python3
"""Raw COLLIDE-2V parquet (collide2v_v1.0) -> canonical event-cloud parquet ("c2v_fullevent").

One source file (one process, one generator seed) -> one output file per split:
`<out>/<split>/<source stem>_<split>.parquet` (split = train, val or test, in the folder AND the
file name, so files of different splits never share a name). The rule throughout is "the
converter stores, the reader cuts": apart from the particle selection below, no particle, jet or
value range is cut here. A value that is known to be wrong in the source and can be undone
exactly is corrected here, so that no known error reaches the output files.

INPUT: raw files of the full COLLIDE-2V dataset, named `<process>-NEVENT<n>-RS<seed>.parquet`,
whose parquet metadata says `dataset_version = collide2v_v1.0`. The converter stops on any other
version, on a missing input column (the constituent lists of both jet views and PuppiW included)
and on eta / phi not stored as float16. A column declared nullable is accepted if it holds no null;
one with a null stops the conversion.

PARTICLES (lists, in source order): the PUPPI candidates (`FullReco_PUPPIPart_*`) with
PuppiW > 0, minus those flagged IsRecoPU = 1. The flagged ones are pileup electrons and muons:
Delphes' charged pileup subtraction flags pileup tracks outside a z window around the hard
scatter, PUPPI gives flagged hadrons weight 0, but electrons and muons bypass PUPPI and are merged
back with weight 1. The card removes them again before it builds its jets and MET, one step after
the stored collection, so the stored list still contains them; no stored jet contains one.
  * pt        float32, PT * PuppiW (GeV).
  * eta, phi  float32, absolute (not rotated). The source stores them as float16. Each value is
              replaced by a uniform draw over its float16 rounding interval (dequantizing), so no
              storage lattice reaches the model; the draws are fixed per (source file name,
              source row, the particle's index in the source PUPPI list, axis), so the output is
              reproducible. phi is then wrapped to [-pi, pi).
              Charged hadrons (charge != 0, |PDG id| not 11 or 13) then get a corrected eta.
              Delphes' PUPPI rebuilds every particle it processes with fastjet's
              reset_PtYPhiM(pt, eta, phi, m), passing the pseudorapidity as the rapidity, and the
              rebuilt vector is stored: its rapidity is the true eta, and its stored Eta is too
              large in magnitude for a massive particle. The true eta is the rapidity of the stored
              (PT, Eta, Mass): asinh(PT sinh(Eta) / sqrt(PT^2 + Mass^2)). Electrons and muons
              bypass PUPPI and are stored correctly; neutral particles are massless.
  * charge    int8, as stored. Must agree with the species of a known PDG id (else stop).
  * pdg_id    int32, the signed PDG id as stored: 211 = pi+, -211 = pi-, 22 = photon, 11 = e-,
              -11 = e+, and so on; 0 = Delphes' neutral hadron (a calorimeter deposit carries no
              species). Charged particles carry their TRUE species: Delphes takes each track
              directly from a generator particle and has no particle-identification step. An id
              outside the 20 known species (`C2V_PDG_SPECIES`) is kept as it is and counted in
              the manifest.
  * is_pu     int8, IsPU as stored: 1 = the particle comes from a pileup collision (generator
              truth). Not read by the model.
  * d0, dz    float32, cm (the source stores mm). dz is measured from the generated hard scatter,
              the vertex with Vertex_Index 0 (exactly one per event, else stop). The source applies
              no impact-parameter smearing, so prompt hard-scatter tracks have d0 = dz = 0 exactly.
              Neutral particles have no track: -999 (NO_TRACK) for both. Not clipped; the reader
              clips to the range of the d0 / dz tokens.
  * ak4_idx, ak8_idx   int8, the particle's jet rank in that view (1 = highest pT), 0 = in no
              jet. Every stored PUPPI jet is ranked (no pT / eta cut, no cap).

JETS (per event, one entry per stored jet, in rank order): `<view>_pt`, `_eta`, `_phi`,
`_m` (float32) and `<view>_flavor` (int8), for the views ak4 (`FullReco_JetPuppiAK4`) and
ak8 (`FullReco_JetPuppiAK8`).
  * Membership: the jet's constituent list holds Delphes' unique ids (fUniqueID), matched to the
    kept particles by (event, id). Every constituent must be a kept particle and belong to one jet
    of the view only (else stop).
  * Quantities: ALWAYS recomputed from the constituents, never Delphes' stored jet values (those
    are sums of PUPPI's rebuilt, massive vectors): massless 4-vectors with the pt, eta, phi stored
    above, summed with the `vector` package in float64. eta = pseudorapidity of the sum
    (rapidity follows from pt, eta, m); m = 0 exactly for a one-constituent jet (no rounding
    noise).
  * Order: descending recomputed pT (ties: stored order).
  * Flavour: Delphes' `Flavor` code as stored: 0 = no parton nearby, 1-3 = light quark, 4 = c,
    5 = b, 21 = gluons only (stop on any other code).

PER EVENT: `process` (string, from the file name); `n_pu` (int16) = number of generated
vertices - 1; `event_weight` (float32) = Event_Weight as stored (one per event, else stop).
Events without a kept particle are dropped (counted).

EVENTS WITH WEIGHT 0 are dropped (counted), never written. In these MLM-matched samples weight 0
marks an event that failed the matching at the central matching scale; with the generator's
systematics switched on, MadGraph's Pythia interface writes such events with weight 0 instead of
removing them. They carry more and harder jets than the weighted events.

METADATA: every output file carries a parquet key-value entry `conversion` (JSON, see
`conversion_block`), built from the same values the code uses: `definitions` (the reader must
agree with each entry and raises otherwise), `info` (how the file was made, including the file's
own `source_file`, `process` and `split`) and `provenance` (git commit + sha256 of this
file). `read_conversion(path)` returns it.

MANIFESTS: one JSON per source file in `<out>/_manifests/` (counts + the block), merged by
`--merge-manifests` into `<out>/manifest.json`. A source file whose manifest exists is skipped,
so an interrupted run can be restarted. Particle and jet counts cover the converted events (after
dropping the weight-0 events), event counts say what happened to every source event.

SPLITS: sha1 of "<source file name>:<source row>" (no run/lumi/event exists upstream), fractions
0.8 / 0.1 / 0.1.

    python scripts/data/convert_c2v_fullevent.py <source files or folders> --out <out dir>
        [--processes P ...] [--part I N] [--jobs J]
    python scripts/data/convert_c2v_fullevent.py --out <out dir> --merge-manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import vector

from humite.data.particle_types import (
    C2V_PDG_SPECIES,
    D0_CLIP,
    DZ_CLIP,
    NO_TRACK,
    charge_from_pdg_id,
)

vector.register_awkward()

FORMAT = "c2v_fullevent"
# The only source format this converter knows.
DATASET_VERSION = "collide2v_v1.0"
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
# The five training processes; `--processes` overrides.
TRAINING_PROCESSES = (
    "QCD_HT50toInf",
    "QCD_HT50tobb",
    "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt0123j_5f_ckm_LO_MLM_leptonic",
    "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
)
MM_TO_CM = 0.1
VIEWS = ("ak4", "ak8")
JET_BRANCH = {"ak4": "FullReco_JetPuppiAK4", "ak8": "FullReco_JetPuppiAK8"}
# Delphes' jet flavour: 0 no parton nearby, 1-3 light quarks, 4 c, 5 b, 21 gluons only.
FLAVOR_CODES = (0, 1, 2, 3, 4, 5, 21)
# Electrons and muons bypass PUPPI, so their stored eta needs no correction.
LEPTONS = (11, 13)
# The generated hard scatter in Delphes' vertex list.
HARD_SCATTER_INDEX = 0
# Bits for the particle's index in the dequantizing key: at most 2**16 PUPPI rows per event.
PARTICLE_BITS = 16
# Reference pad length for the manifest count only (the real one is a reader/config value).
PAD_LENGTH_REFERENCE = 640

PARTICLE_COLUMNS = [
    f"FullReco_PUPPIPart_{c}"
    for c in (
        "PT",
        "Eta",
        "Phi",
        "Mass",
        "Charge",
        "PID",
        "D0",
        "DZ",
        "fUniqueID",
        "PuppiW",
        "IsPU",
        "IsRecoPU",
    )
]
# The dequantizing assumes float16 storage of these.
FLOAT16_COLUMNS = ("FullReco_PUPPIPart_Eta", "FullReco_PUPPIPart_Phi")
COLUMNS = (
    PARTICLE_COLUMNS
    + [f"{JET_BRANCH[v]}_{c}" for v in VIEWS for c in ("Constituents", "Flavor")]
    + ["Vertex_Index", "Vertex_Z", "Event_Weight"]
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def conversion_block(git_commit: str) -> dict:
    """The `conversion` metadata written to the parquet files, built from the values the code
    uses (no value typed twice). Per-file entries (source file, process, split) are added by
    `convert_file`."""
    return {
        "format": FORMAT,
        "definitions": {
            "phi_frame": "absolute",
            "eta_phi_storage": "float32, dequantized from float16",
            "charged_hadron_eta": "corrected: rapidity of the stored (PT, Eta, Mass)",
            "d0_dz_unit": "cm",
            "d0_dz_clipped": False,
            "no_track": NO_TRACK,
            "jet_flavor": "Delphes Flavor, raw code",
        },
        "info": {
            "dataset_version": DATASET_VERSION,
            "particles": "PUPPI candidates with PuppiW > 0 and IsRecoPU == 0, source order",
            "pt": "PT * PuppiW",
            "dequantizing": (
                "uniform over the float16 rounding interval, fixed per (source file name, source "
                "row, index in the source PUPPI list, axis); before the eta correction"
            ),
            "charged_hadrons": f"charge != 0 and |PDG id| not in {LEPTONS}",
            "pdg_id": "signed PDG id as stored; 0 = Delphes neutral hadron",
            "dz_reference": f"Vertex_Z of the vertex with Vertex_Index {HARD_SCATTER_INDEX}",
            "jet_membership": (
                "constituent lists matched to the kept particles by (event, fUniqueID)"
            ),
            "jet_sums": (
                "massless constituent 4-vectors (vector package, float64), every constituent; "
                "eta = pseudorapidity of the sum; m = 0 for one constituent; never Delphes' "
                "stored jet values"
            ),
            "jet_order": "descending recomputed pt (ties: stored order); every stored jet",
            "flavor_codes": list(FLAVOR_CODES),
            "n_pu": "number of vertices - 1",
            "event_weight": "Event_Weight as stored",
            "zero_weight_events": "dropped",
            "split_key": "sha1('<source file name>:<source row>')",
            "split_fractions": SPLIT_FRACTIONS,
        },
        "provenance": {
            "git_commit": git_commit,
            "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def read_conversion(path) -> dict:
    """The `conversion` block of an output file (reads only the parquet footer)."""
    meta = pq.read_schema(path).metadata or {}
    if b"conversion" not in meta:
        raise RuntimeError(f"{path}: no 'conversion' metadata (not written by this converter)")
    return json.loads(meta[b"conversion"])


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _flat(x: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(x))


def _parse_name(path: Path) -> str:
    m = re.match(r"(.+)-NEVENT\d+-RS\d+\.parquet$", path.name)
    if not m:
        raise RuntimeError(f"unexpected source file name {path.name}")
    return m.group(1)


def _split_of(file_name: str, rows: np.ndarray) -> np.ndarray:
    edges = np.cumsum([SPLIT_FRACTIONS[s] for s in SPLITS])
    r = np.array(
        [
            int.from_bytes(hashlib.sha1(f"{file_name}:{int(i)}".encode()).digest()[:8], "big")
            / 2**64
            for i in rows
        ]
    )
    return np.where(r < edges[0], "train", np.where(r < edges[1], "val", "test"))


def _add(stats: dict, key: str, value) -> None:
    stats[key] = stats.get(key, 0) + value


def _count_codes(stats: dict, key: str, codes: np.ndarray) -> None:
    """Add the occurrences of each value of `codes` to `stats[key]` ({value: count})."""
    agg = stats.setdefault(key, {})
    values, counts = np.unique(codes, return_counts=True)
    for v, c in zip(values.tolist(), counts.tolist()):
        agg[str(v)] = agg.get(str(v), 0) + c


def _has_option(layout) -> bool:
    """True if a column's type has an option ("may be missing") at any list depth."""
    while True:
        if layout.is_option:
            return True
        if not layout.is_list:
            return False
        layout = layout.content


def _without_option_types(a: ak.Array, file_name: str) -> ak.Array:
    """A few source files declare a column nullable although it holds no null. awkward then reads
    it with an option type, and numpy code gets masked arrays. Such a column loses its option type
    here, so every file is converted with the same types; a column with a real null stops."""
    for c in a.fields:
        x = a[c]
        if not _has_option(x.layout):
            continue
        n_null = sum(int(ak.sum(ak.is_none(x, axis=d), axis=None)) for d in range(x.ndim))
        if n_null:
            raise RuntimeError(f"{file_name}: {c} holds {n_null} nulls")
        x = ak.drop_none(x)
        if _has_option(x.layout):
            raise RuntimeError(f"{file_name}: {c} keeps an option type without holding a null")
        a = ak.with_field(a, x, c)
    return a


# ── dequantizing and the eta correction ─────────────────────────────────────────────────────────


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer on a uint64 array (arithmetic wraps modulo 2**64)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _file_key(file_name: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha1(file_name.encode()).digest()[:8], "big"))


def _uniform(file_key: np.uint64, row: np.ndarray, particle: np.ndarray, axis: int) -> np.ndarray:
    """Uniform [0, 1) per (file, row, particle, axis), the same on every call."""
    key = (
        (row.astype(np.uint64) << np.uint64(PARTICLE_BITS + 1))
        | (particle.astype(np.uint64) << np.uint64(1))
        | np.uint64(axis)
    )
    bits = _splitmix64(_splitmix64(key) ^ file_key)
    return (bits >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def _dequantize_f16(v16: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Each float16 value v -> a draw at quantile u over [v - half the step below, v + half the
    step above], i.e. exactly the real numbers that round to v (the two steps differ at powers
    of 2). float64."""
    v = v16.astype(np.float64)
    up = np.nextafter(v16, np.float16(np.inf)).astype(np.float64)
    down = np.nextafter(v16, np.float16(-np.inf)).astype(np.float64)
    low = v - 0.5 * (v - down)
    high = v + 0.5 * (up - v)
    return low + u * (high - low)


def _corrected_eta(pt: np.ndarray, eta: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """True eta of a vector PUPPI rebuilt with its pseudorapidity as rapidity: the rapidity of
    the stored (PT, Eta, Mass)."""
    return np.arcsinh(pt * np.sinh(eta) / np.sqrt(pt**2 + mass**2))


# ── conversion ──────────────────────────────────────────────────────────────────────────────────


def _event_weight(a: ak.Array) -> np.ndarray:
    """Event_Weight per event (float32); stops unless every event has exactly one entry."""
    n_w = ak.to_numpy(ak.num(a["Event_Weight"]))
    if np.any(n_w != 1):
        raise RuntimeError(
            f"Event_Weight with {sorted(set(n_w.tolist()) - {1})} entries in an event (expected 1)"
        )
    return _flat(a["Event_Weight"]).astype(np.float32)


def _jets(view: str, a: ak.Array, part: dict, stats: dict) -> dict:
    """One jet view: per-particle rank, per-jet 4-vector sums and flavour, all in rank order."""
    n = len(a)
    con = a[f"{JET_BRANCH[view]}_Constituents"]
    flavor_lists = a[f"{JET_BRANCH[view]}_Flavor"]
    nj = ak.to_numpy(ak.num(con, axis=1))
    if not np.array_equal(ak.to_numpy(ak.num(flavor_lists, axis=1)), nj):
        raise RuntimeError(f"{view}: Flavor and Constituents list different jets")
    nc = _flat(ak.num(con, axis=2))
    n_jets = len(nc)
    if n_jets and nc.min() == 0:
        raise RuntimeError(f"{view}: a stored jet without constituents")
    jet_ev = np.repeat(np.arange(n, dtype=np.int64), nj)
    starts = np.r_[0, np.cumsum(nj)[:-1]]
    stored_local = np.arange(n_jets) - np.repeat(starts, nj)

    # --- constituent -> kept particle by (event, fUniqueID).
    con_uid = ak.to_numpy(ak.flatten(con, axis=None)).astype(np.int64)
    con_jet = np.repeat(np.arange(n_jets, dtype=np.int64), nc)
    ckey = (jet_ev[con_jet] << 32) | con_uid
    order = np.argsort(ckey, kind="stable")
    ckey, con_jet = ckey[order], con_jet[order]
    if np.any(ckey[1:] == ckey[:-1]):
        raise RuntimeError(f"{view}: a particle is a constituent of two jets")
    pkey = (part["ev"] << 32) | part["uid"]
    missing = ~np.isin(ckey, pkey)
    if missing.any():
        raise RuntimeError(f"{view}: {int(missing.sum())} constituents are not kept particles")
    pjet = np.full(len(pkey), -1, np.int64)
    if len(ckey):
        pos = np.clip(np.searchsorted(ckey, pkey), 0, len(ckey) - 1)
        pjet = np.where(ckey[pos] == pkey, con_jet[pos], -1)

    # --- jet 4-vectors: massless constituents, grouped as [event][jet][constituent] (stored jet
    #     order) and summed with vector (float64).
    sel = np.flatnonzero(pjet >= 0)
    sel = sel[np.argsort(pjet[sel], kind="stable")]
    per_jet = np.bincount(pjet[sel], minlength=n_jets)
    if not np.array_equal(per_jet, nc):
        raise RuntimeError(f"{view}: constituent lists and matched particles disagree")

    def nested(x: np.ndarray) -> ak.Array:
        return ak.unflatten(ak.unflatten(x, per_jet), nj)

    p4 = ak.zip(
        {
            "pt": nested(part["pt"][sel].astype(np.float64)),
            "eta": nested(part["eta"][sel].astype(np.float64)),
            "phi": nested(part["phi"][sel].astype(np.float64)),
            "mass": nested(np.zeros(len(sel))),
        },
        with_name="Momentum4D",
    )
    jet_p4 = ak.sum(p4, axis=2)
    jpt = _flat(jet_p4.pt)
    jeta = _flat(jet_p4.eta)
    jphi = _flat(jet_p4.phi)
    jm = np.sqrt(np.maximum(_flat(jet_p4.mass2), 0.0))
    jm = np.where(nc == 1, 0.0, jm)  # one massless constituent: exactly 0, no rounding noise

    # --- rank by recomputed pt (ties: stored order); rank 1 = hardest.
    o = np.lexsort((stored_local, -jpt, jet_ev))
    rank = np.empty(n_jets, np.int64)
    rank[o] = np.arange(n_jets) - starts[jet_ev[o]] + 1
    if n_jets and rank.max() > np.iinfo(np.int8).max:
        raise RuntimeError(f"{view}: {rank.max()} jets in one event do not fit int8")
    pidx = np.zeros(len(pjet), np.int8)
    if n_jets:
        pidx = np.where(pjet >= 0, rank[np.maximum(pjet, 0)], 0).astype(np.int8)

    # --- Delphes' flavour code, as stored.
    flavor = _flat(flavor_lists).astype(np.int64)
    unknown = ~np.isin(flavor, FLAVOR_CODES)
    if unknown.any():
        raise RuntimeError(
            f"{view}: Flavor codes {sorted(set(flavor[unknown].tolist()))} are not Delphes jet "
            f"flavours {FLAVOR_CODES}"
        )

    _add(stats, f"jets_{view}", n_jets)
    _add(stats, f"constituents_{view}", len(sel))
    stats[f"max_jets_{view}"] = max(stats.get(f"max_jets_{view}", 0), int(nj.max(initial=0)))
    _count_codes(stats, f"flavor_codes_{view}", flavor)

    def per_event(x: np.ndarray, dtype) -> ak.Array:
        return ak.unflatten(x[o].astype(dtype), nj)

    return {
        f"{view}_idx": pidx,
        f"{view}_pt": per_event(jpt, np.float32),
        f"{view}_eta": per_event(jeta, np.float32),
        f"{view}_phi": per_event(jphi, np.float32),
        f"{view}_m": per_event(jm, np.float32),
        f"{view}_flavor": per_event(flavor, np.int8),
    }


def transform(
    a: ak.Array, rows: np.ndarray, process: str, file_key: np.uint64, stats: dict
) -> ak.Array:
    """Source events -> canonical cloud (every event given; the caller drops the 0-particle
    ones). `rows`: the events' row numbers in the source file (the dequantizing key)."""
    P = lambda c: a["FullReco_PUPPIPart_" + c]  # noqa: E731
    n = len(a)

    # --- per event: the weight and the hard-scatter z.
    weight = _event_weight(a)
    hs = a["Vertex_Index"] == HARD_SCATTER_INDEX
    n_hs = ak.to_numpy(ak.sum(hs, axis=1))
    if np.any(n_hs != 1):
        raise RuntimeError(
            f"{int((n_hs != 1).sum())} events without exactly one vertex with Vertex_Index "
            f"{HARD_SCATTER_INDEX}"
        )
    z_hs = _flat(a["Vertex_Z"][hs]).astype(np.float32)
    n_pu = (ak.to_numpy(ak.num(a["Vertex_Index"])) - 1).astype(np.int16)

    # --- kept particles: PuppiW > 0 and not flagged IsRecoPU.
    n_rows = ak.to_numpy(ak.num(P("PT")))
    if n_rows.max(initial=0) >= 2**PARTICLE_BITS:
        raise RuntimeError(f"{n_rows.max()} PUPPI rows in one event exceed the dequantizing key")
    puppi_w = _flat(P("PuppiW")).astype(np.float32)
    recopu = _flat(P("IsRecoPU"))
    keep = (puppi_w > 0) & (recopu == 0)
    _add(stats, "particles_in", len(keep))
    _add(stats, "particles_dropped_puppi_weight_0", int((puppi_w <= 0).sum()))
    _add(stats, "particles_dropped_isrecopu", int(((puppi_w > 0) & (recopu != 0)).sum()))
    _add(stats, "particles_kept", int(keep.sum()))

    ev_all = np.repeat(np.arange(n, dtype=np.int64), n_rows)
    counts = np.bincount(ev_all[keep], minlength=n)
    ev = ev_all[keep]
    src_row = np.asarray(rows, dtype=np.int64)[ev]
    src_index = _flat(ak.local_index(P("PT")))[keep]  # the particle's index in the source list

    pt_stored = _flat(P("PT"))[keep]
    pt = (pt_stored.astype(np.float32) * puppi_w[keep]).astype(np.float32)
    charge = _flat(P("Charge"))[keep].astype(np.int8)
    pdg = _flat(P("PID"))[keep].astype(np.int32)

    # --- eta / phi: dequantize the float16 values, then correct the charged-hadron eta.
    eta = _dequantize_f16(_flat(P("Eta"))[keep], _uniform(file_key, src_row, src_index, 0))
    phi = _dequantize_f16(_flat(P("Phi"))[keep], _uniform(file_key, src_row, src_index, 1))
    phi = _wrap_pi(phi).astype(np.float32)
    hadron = (charge != 0) & ~np.isin(np.abs(pdg), LEPTONS)
    mass = _flat(P("Mass"))[keep].astype(np.float64)
    eta = np.where(hadron, _corrected_eta(pt_stored.astype(np.float64), eta, mass), eta)
    eta = eta.astype(np.float32)
    _add(stats, "charged_hadrons_eta_corrected", int(hadron.sum()))

    # --- species: the stored charge must follow from a known PDG id; unknown ids kept, as long
    #     as their charge is -1, 0 or 1.
    expected_charge = charge_from_pdg_id(pdg)
    contradicts = np.isfinite(expected_charge) & (expected_charge != charge)
    contradicts |= np.abs(charge) > 1
    if contradicts.any():
        i = int(np.flatnonzero(contradicts)[0])
        raise RuntimeError(
            f"{int(contradicts.sum())} particles whose charge contradicts their PDG id "
            f"(first: id {pdg[i]}, charge {charge[i]})"
        )
    unknown = ~np.isin(pdg, C2V_PDG_SPECIES)
    _add(stats, "particles_unknown_pdg", int(unknown.sum()))
    stats["unknown_pdg_ids"] = sorted(
        set(stats.get("unknown_pdg_ids", [])) | set(np.unique(pdg[unknown]).tolist())
    )

    # --- d0 / dz: mm -> cm, dz relative to the hard scatter, NO_TRACK for neutrals.
    charged = charge != 0
    d0_mm = _flat(P("D0"))[keep].astype(np.float32)
    dz_mm = _flat(P("DZ"))[keep].astype(np.float32)
    d0 = np.where(charged, d0_mm * np.float32(MM_TO_CM), np.float32(NO_TRACK)).astype(np.float32)
    dz = np.where(charged, (dz_mm - z_hs[ev]) * np.float32(MM_TO_CM), np.float32(NO_TRACK)).astype(
        np.float32
    )
    _add(stats, "charged", int(charged.sum()))
    _add(stats, "d0_beyond_clip", int((np.abs(d0[charged]) > D0_CLIP).sum()))
    _add(stats, "dz_beyond_clip", int((np.abs(dz[charged]) > DZ_CLIP).sum()))
    _add(stats, "d0_exact_zero", int((d0[charged] == 0).sum()))
    _add(stats, "dz_exact_zero", int((dz[charged] == 0).sum()))

    # --- jets, from the stored (float32) particle values.
    part = {
        "ev": ev,
        "uid": _flat(P("fUniqueID"))[keep].astype(np.int64),
        "pt": pt,
        "eta": eta,
        "phi": phi,
    }
    jets = {}
    for view in VIEWS:
        jets.update(_jets(view, a, part, stats))

    def per_particle(x: np.ndarray) -> ak.Array:
        return ak.unflatten(x, counts)

    return ak.Array(
        {
            "pt": per_particle(pt),
            "eta": per_particle(eta),
            "phi": per_particle(phi),
            "charge": per_particle(charge),
            "pdg_id": per_particle(pdg),
            "is_pu": per_particle(_flat(P("IsPU"))[keep].astype(np.int8)),
            "d0": per_particle(d0),
            "dz": per_particle(dz),
            "ak4_idx": per_particle(jets.pop("ak4_idx")),
            "ak8_idx": per_particle(jets.pop("ak8_idx")),
            "process": ak.Array(np.full(n, process)),
            "n_pu": n_pu,
            "event_weight": weight,
            **jets,
        }
    )


def _check_source(path: Path, pf: pq.ParquetFile) -> None:
    """Stop on another dataset version, a missing column or non-float16 eta / phi."""
    meta = pf.metadata.metadata or {}
    version = meta.get(b"dataset_version", b"absent").decode()
    if version != DATASET_VERSION:
        raise RuntimeError(
            f"{path.name}: dataset_version {version!r}, expected {DATASET_VERSION!r}"
        )
    names = set(pf.schema_arrow.names)
    missing = [c for c in COLUMNS if c not in names]
    if missing:
        raise RuntimeError(f"{path.name}: missing columns {missing}")
    for c in FLOAT16_COLUMNS:
        t = pf.schema_arrow.field(c).type
        while pa.types.is_list(t) or pa.types.is_large_list(t):
            t = t.value_type
        if not pa.types.is_float16(t):
            raise RuntimeError(f"{path.name}: {c} is {t}, the dequantizing assumes float16")


def _write(arr: ak.Array, path: Path, block: dict, row_group: int) -> None:
    """Write with the `conversion` block in the parquet key-value metadata (atomic rename)."""
    table = ak.to_arrow_table(
        arr, extensionarray=True, string_to32=True, bytestring_to32=True, count_nulls=True
    )
    meta = dict(table.schema.metadata or {})
    meta[b"conversion"] = json.dumps(block).encode()
    table = table.replace_schema_metadata(meta)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, row_group_size=row_group, compression="zstd")
    tmp.rename(path)


def convert_file(path: Path, out: Path, block: dict, read_batch: int, row_group: int) -> dict:
    process = _parse_name(path)
    pf = pq.ParquetFile(path)
    _check_source(path, pf)
    file_key = _file_key(path.name)

    stats: dict = {}
    parts: dict = {s: [] for s in SPLITS}
    row0 = n_in = n_zero_weight = n_no_particles = 0
    for rb in pf.iter_batches(batch_size=read_batch, columns=COLUMNS):
        a = _without_option_types(ak.from_arrow(rb), path.name)
        rows = np.arange(row0, row0 + len(a))
        row0 += len(a)
        n_in += len(a)
        zero = _event_weight(a) == 0
        n_zero_weight += int(zero.sum())
        a, rows = a[~zero], rows[~zero]
        if len(a) == 0:
            continue
        cloud = transform(a, rows, process, file_key, stats)
        n_particles = ak.to_numpy(ak.num(cloud.pt))
        has = n_particles >= 1
        n_no_particles += int((~has).sum())
        cloud, rows, n_particles = cloud[has], rows[has], n_particles[has]
        stats["max_particles"] = max(
            stats.get("max_particles", 0), int(n_particles.max(initial=0))
        )
        _add(stats, "events_over_pad_reference", int((n_particles > PAD_LENGTH_REFERENCE).sum()))
        split = _split_of(path.name, rows)
        for s in SPLITS:
            sel = split == s
            if sel.any():
                parts[s].append(cloud[sel])
    if n_in != pf.metadata.num_rows:
        raise RuntimeError(f"{path.name}: read {n_in} of {pf.metadata.num_rows} rows")

    written = {}
    for s in SPLITS:
        written[s] = sum(len(p) for p in parts[s])
        if not written[s]:
            continue
        merged = ak.concatenate(parts[s]) if len(parts[s]) > 1 else parts[s][0]
        d = out / s
        d.mkdir(parents=True, exist_ok=True)
        file_block = json.loads(json.dumps(block))
        file_block["info"].update({"source_file": path.name, "process": process, "split": s})
        _write(merged, d / f"{path.stem}_{s}.parquet", file_block, row_group)
    return {
        "file": path.name,
        "process": process,
        "n_events_in": n_in,
        "n_events_dropped_weight_0": n_zero_weight,
        "n_events_dropped_no_particles": n_no_particles,
        "n_written": written,
        **stats,
        "conversion": block,
    }


def _run_one(job: tuple) -> str:
    f, out, block, read_batch, row_group = job
    man = out / "_manifests" / f"{f.stem}.json"
    if man.exists():
        return f"{f.name}: skip (manifest exists)"
    t0 = time.time()
    res = convert_file(f, out, block, read_batch, row_group)
    res["seconds"] = round(time.time() - t0, 1)
    man.write_text(json.dumps(res) + "\n")
    return (
        f"{f.name}: {sum(res['n_written'].values())}/{res['n_events_in']} written "
        f"{res['n_written']} ({res['n_events_dropped_weight_0']} with weight 0 dropped) "
        f"in {res['seconds']} s"
    )


def _merge_into(agg: dict, m: dict) -> None:
    """Add one manifest's counts into `agg`: numbers add (`max_*` take the maximum), dicts
    merge key by key, lists become their sorted union; strings and the block are skipped."""
    for k, v in m.items():
        if k in ("conversion", "file", "process", "seconds") or isinstance(v, str):
            continue
        if isinstance(v, dict):
            _merge_into(agg.setdefault(k, {}), v)
        elif isinstance(v, list):
            agg[k] = sorted(set(agg.get(k, [])) | set(v))
        elif k.startswith("max_"):
            agg[k] = max(agg.get(k, 0), v)
        else:
            agg[k] = agg.get(k, 0) + v


def _merge_manifests(out: Path) -> None:
    """Sum the per-file manifests per process and overall -> <out>/manifest.json. All files must
    carry the same `conversion` block (a mixed output directory is a problem, not a merge)."""
    loaded = [json.loads(f.read_text()) for f in sorted((out / "_manifests").glob("*.json"))]
    if not loaded:
        raise SystemExit(f"no manifests under {out / '_manifests'}")
    blocks = {json.dumps(m["conversion"], sort_keys=True) for m in loaded}
    problems = [] if len(blocks) == 1 else [f"{len(blocks)} different conversion blocks"]
    per: dict = {}
    total: dict = {"n_files": len(loaded)}
    for m in loaded:
        agg = per.setdefault(m["process"], {"n_files": 0})
        agg["n_files"] += 1
        _merge_into(agg, m)
        _merge_into(total, m)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "conversion": loaded[0]["conversion"],
                "total": total,
                "problems": problems,
                "per_process": per,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"merged {len(loaded)} file manifests: written {total['n_written']}; {problems}")


def _source_files(inputs: list[Path]) -> list[Path]:
    """Files as given; folders searched recursively for *.parquet."""
    files = []
    for p in inputs:
        if p.is_dir():
            files += p.rglob("*.parquet")
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"{p}: no such file or folder")
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "inputs", nargs="*", type=Path, help="source files and/or folders (searched recursively)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge-manifests", action="store_true", help="only merge -> manifest.json")
    ap.add_argument("--processes", nargs="*", default=list(TRAINING_PROCESSES))
    ap.add_argument(
        "--part",
        type=int,
        nargs=2,
        metavar=("I", "N"),
        default=(0, 1),
        help="only every N-th file, starting at I (SLURM arrays)",
    )
    ap.add_argument("--jobs", type=int, default=1, help="parallel files (finished ones skipped)")
    ap.add_argument("--read-batch", type=int, default=2000)
    ap.add_argument("--row-group", type=int, default=8192)
    args = ap.parse_args()
    if args.merge_manifests:
        _merge_manifests(args.out)
        return
    if not args.inputs:
        ap.error("give source files or folders")

    found = _source_files(args.inputs)
    files = [f for f in found if _parse_name(f) in args.processes]
    if len(files) < len(found):
        print(f"{len(found) - len(files)} source files of other processes left out")
    i, n_parts = args.part
    files = files[i::n_parts]
    block = conversion_block(_git_commit())
    (args.out / "_manifests").mkdir(parents=True, exist_ok=True)
    jobs = [(f, args.out, block, args.read_batch, args.row_group) for f in files]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for k, msg in enumerate(pool.imap_unordered(_run_one, jobs), 1):
                print(f"[{k}/{len(jobs)}] {msg}", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            print(f"[{k}/{len(jobs)}] {_run_one(job)}", flush=True)


if __name__ == "__main__":
    main()
