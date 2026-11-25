#!/usr/bin/env python3
"""Mix the jet types of the converted JetClass files.

convert_jetclass.py writes one file per source file, and a JetClass source file holds one jet
type. The loader reads one file after the other, so batches from those files would hold a
single jet type. This script reads, for every index of a split, the ten files
`<class>_<idx>.parquet`, shuffles their jets with a permutation fixed by the index, and writes
`mixed_<idx>.parquet` (all columns, row groups of 20k jets). The splits are unchanged.

    python scripts/data/interleave_jetclass.py --src <converted> --out <converted_interleaved> --jobs 16
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_CLASSES = 10
SPLITS = ("train_100M", "val_5M", "test_20M")
_STEM = re.compile(r"^(?P<cls>.+)_(?P<idx>\d+)$")


def interleave_block(job: Tuple[int, List[Path], Path, int]) -> Dict[str, object]:
    idx, files, out, row_group = job
    if out.exists():
        return {"file": out.name, "rows": pq.ParquetFile(out).metadata.num_rows, "skipped": True}

    tables = [pq.read_table(f) for f in sorted(files)]
    tbl = pa.concat_tables(tables)
    # Deterministic: the permutation is a pure function of the block index, so a re-run
    # (or a resume after a kill) reproduces byte-identical shards.
    perm = np.random.default_rng(seed=idx).permutation(tbl.num_rows)
    tbl = tbl.take(pa.array(perm))

    labels = tbl["jc_label"].to_numpy(zero_copy_only=False)
    counts = np.bincount(labels, minlength=N_CLASSES)
    if int(counts.min()) == 0:
        raise ValueError(f"block {idx}: a class is missing entirely ({counts.tolist()})")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3, row_group_size=row_group)
    tmp.rename(out)
    return {
        "file": out.name,
        "rows": tbl.num_rows,
        "class_counts": counts.tolist(),
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="class-pure converted/ tree")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--row-group", type=int, default=20_000)
    args = ap.parse_args()

    jobs: List[Tuple[int, List[Path], Path, int]] = []
    for split in args.splits:
        by_idx: Dict[int, List[Path]] = defaultdict(list)
        for f in sorted((args.src / split).glob("*.parquet")):
            m = _STEM.match(f.stem)
            if m is None:
                raise SystemExit(f"unparsable shard name: {f}")
            by_idx[int(m.group("idx"))].append(f)
        if not by_idx:
            raise SystemExit(f"no shards under {args.src / split}")
        for idx, files in sorted(by_idx.items()):
            if len(files) != N_CLASSES:
                raise SystemExit(
                    f"{split} block {idx}: {len(files)} class files, expected {N_CLASSES} "
                    f"({[f.name for f in files]})"
                )
            jobs.append(
                (idx, files, args.out / split / f"mixed_{idx:04d}.parquet", args.row_group)
            )

    print(f"{len(jobs)} blocks -> {args.out}  ({args.jobs} workers)", flush=True)
    t0 = time.time()
    with Pool(args.jobs) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(interleave_block, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:6.0f}s", flush=True)

    rows = sum(int(r["rows"]) for r in results)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.src),
                "splits": args.splits,
                "blocks": len(jobs),
                "rows": rows,
                "row_group": args.row_group,
                "seed_scheme": "numpy default_rng(seed=block_index).permutation",
                "note": (
                    "class-interleaved re-shard of convert_jetclass.py output; "
                    "all columns preserved; every block holds 100k jets of each class"
                ),
            },
            indent=2,
        )
    )
    print(f"done: {rows:,} jets in {len(jobs)} mixed blocks, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
