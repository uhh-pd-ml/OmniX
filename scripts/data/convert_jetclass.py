#!/usr/bin/env python3
"""Convert the JetClass ROOT files to the parquet files that the JetClass reader loads.

One row is one jet; the splits are JetClass's own train_100M / val_5M / test_20M. Per jet:

  * pt, eta     pt = hypot(px, py), eta = arcsinh(pz / pt), from the momenta. (Not
                jet_eta + part_deta: JetClass stores deta = (eta - jet_eta) * sign(jet_eta),
                mirrored into one hemisphere.)
  * phi         part_dphi, the azimuth relative to the jet axis.
  * jet_idx     1 for every constituent.
  * pf_type     from the five species flags: 0 charged hadron, 1 neutral hadron, 2 photon,
                3 electron, 4 muon. charge is kept as -1, 0 or 1.
  * d0, dz      mm -> cm, clipped to +-D0_CLIP / +-DZ_CLIP; then NO_TRACK (-999) for neutral
                constituents, which JetClass stores with a literal 0.
  * jc_label    0..9, the jet type in the order of CLASS_ORDER.

Also carried, although the model does not read them: part_energy, the impact-parameter
uncertainties, and the source jet's pt, eta, phi, soft-drop mass and tau1-4.

    python scripts/data/convert_jetclass.py --src <JetClass/Pythia> --out <converted> --jobs 16

Follow with interleave_jetclass.py: the files written here hold one jet type each.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from humite.data.particle_types import D0_CLIP, DZ_CLIP, NO_TRACK

# filename stem -> jc_label. APPEND ONLY: the integer is a training target and, once a
# classifier is trained, renumbering silently relabels every class.
CLASS_ORDER: Dict[str, int] = {
    "ZJetsToNuNu": 0,  # QCD
    "HToBB": 1,
    "HToCC": 2,
    "HToGG": 3,
    "HToWW4Q": 4,
    "HToWW2Q1L": 5,
    "ZToQQ": 6,
    "WToQQ": 7,
    "TTBar": 8,  # t -> bqq
    "TTBarLep": 9,  # t -> b l nu
}
LABEL_BRANCHES = (
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
)
SPLITS = ("train_100M", "val_5M", "test_20M")

PART_BRANCHES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)
JET_BRANCHES = (
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_sdmass",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
    "aux_genpart_pt",
    "aux_genpart_eta",
    "aux_genpart_phi",
    "aux_genpart_pid",
    "aux_truth_match",
)


def _clip(x: ak.Array, lim: float) -> ak.Array:
    # np.clip is not a ufunc and cannot act on ragged arrays.
    return np.minimum(np.maximum(x, -lim), lim)


def _wrap(a: ak.Array) -> ak.Array:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def convert_one(job: Tuple[Path, Path, int]) -> Dict[str, object]:
    src, out, row_group = job
    label = CLASS_ORDER[src.stem.rsplit("_", 1)[0]]
    if out.exists():
        # A skipped shard must still report its multiplicity, or a RESUMED run would compute
        # max_n_const over the new shards only and the pad guard below would pass vacuously.
        pf = pq.ParquetFile(out)
        meta = (pf.schema_arrow.metadata or {}).get(b"max_n_const")
        mx = int(meta) if meta else int(ak.max(ak.num(ak.from_arrow(pf.read(["pt"])["pt"]))))
        return {"file": out.name, "rows": pf.metadata.num_rows, "max_n_const": mx, "skipped": True}

    a = uproot.open(src)["tree"].arrays(
        list(PART_BRANCHES) + list(JET_BRANCHES) + list(LABEL_BRANCHES)
    )
    n = len(a)

    pt = np.hypot(a.part_px, a.part_py)
    eta = np.arcsinh(a.part_pz / pt)
    phi = _wrap(a.part_dphi)

    pf_type = ak.where(
        a.part_isChargedHadron == 1,
        0,
        ak.where(
            a.part_isNeutralHadron == 1,
            1,
            ak.where(
                a.part_isPhoton == 1,
                2,
                ak.where(a.part_isElectron == 1, 3, ak.where(a.part_isMuon == 1, 4, 7)),
            ),
        ),
    )
    n_other = int(ak.sum(pf_type == 7))
    if n_other:
        raise ValueError(f"{src.name}: {n_other} constituents match none of the species flags")

    neutral = a.part_charge == 0
    d0 = ak.where(neutral, NO_TRACK, _clip(a.part_d0val / 10.0, D0_CLIP))
    dz = ak.where(neutral, NO_TRACK, _clip(a.part_dzval / 10.0, DZ_CLIP))

    onehot = np.stack([np.asarray(a[k], dtype=np.int8) for k in LABEL_BRANCHES], axis=1)
    if not np.array_equal(onehot.argmax(1), np.full(n, label)):
        raise ValueError(f"{src.name}: label branches disagree with the filename class {label}")

    def _rag(x, dt=np.float32):
        return ak.to_arrow(ak.values_astype(x, dt))

    tbl = pa.table(
        {
            # the 8 body axes, byte-compatible with jets.yaml
            "pt": _rag(pt),
            "eta": _rag(eta),
            "phi": _rag(phi),
            "jet_idx": _rag(ak.ones_like(pt), np.int8),
            "charge": _rag(a.part_charge, np.int8),
            "pf_type": _rag(pf_type, np.int8),
            "d0": _rag(d0),
            "dz": _rag(dz),
            # globals_raw
            "jc_label": pa.array(np.full(n, label, np.int8)),
            "n_jets": pa.array(np.ones(n, np.int8)),
            # carried so the source ROOT is deletable
            "part_energy": _rag(a.part_energy),
            "d0err": _rag(a.part_d0err / 10.0),
            "dzerr": _rag(a.part_dzerr / 10.0),
            **{f"src_{k}": pa.array(np.asarray(a[k]), pa.float32()) for k in JET_BRANCHES},
        }
    )
    max_n_const = int(ak.max(ak.num(pt)))
    # In the footer so a resumed run reads it in O(1) instead of re-reading the pt column.
    tbl = tbl.replace_schema_metadata({b"max_n_const": str(max_n_const).encode()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3, row_group_size=row_group)
    tmp.rename(out)
    return {"file": out.name, "rows": n, "max_n_const": max_n_const, "skipped": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="JetClass .../Pythia dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--row-group", type=int, default=20_000)
    ap.add_argument(
        "--pad-length",
        type=int,
        default=192,
        help="asserted upper bound on constituents per jet (measured max 183 over all 125M)",
    )
    ap.add_argument("--max-files", type=int, default=None, help="cap per split (subset testing)")
    args = ap.parse_args()

    jobs: List[Tuple[Path, Path, int]] = []
    for split in args.splits:
        files = sorted((args.src / split).glob("*.root"))[: args.max_files]
        if not files:
            raise SystemExit(f"no ROOT files under {args.src / split}")
        jobs += [(f, args.out / split / f"{f.stem}.parquet", args.row_group) for f in files]

    print(f"{len(jobs)} files -> {args.out}  ({args.jobs} workers)", flush=True)
    t0 = time.time()
    with Pool(args.jobs) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(convert_one, jobs), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:6.0f}s", flush=True)

    rows = sum(int(r["rows"]) for r in results)
    max_nc = max(int(r.get("max_n_const", 0)) for r in results)
    if max_nc > args.pad_length:
        raise SystemExit(
            f"pad_length {args.pad_length} is too small: measured max {max_nc} constituents. "
            "Raise dataset.pad_length (and this flag) — a pad below the max silently drops hits."
        )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.src),
                "splits": args.splits,
                "files": len(jobs),
                "rows": rows,
                "max_n_const": max_nc,
                "pad_length": args.pad_length,
                "class_order": CLASS_ORDER,
                "d0_clip": D0_CLIP,
                "dz_clip": DZ_CLIP,
                "no_track": NO_TRACK,
                "sentinel_rule": "charge == 0",
                "d0_units": "mm/10 -> cm",
                "phi": "part_dphi (lead-jet frame)",
                "eta": "arcsinh(pz/pt), absolute",
                "git_commit": commit,
            },
            indent=2,
        )
    )
    print(f"done: {rows:,} jets, max {max_nc} constituents, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
