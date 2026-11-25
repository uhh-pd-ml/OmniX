#!/usr/bin/env python3
"""Generate jets or events with a released model, with the sampling settings of the paper.

    python scripts/generate.py checkpoints/omnix jets --n 10000 --out jets.npz
    python scripts/generate.py checkpoints/omnix jets --condition TTBar --n 10000 --out top_jets.npz
    python scripts/generate.py checkpoints/omnix events --condition QCD_HT50toInf --n 10000 --out qcd.npz

Settings: temperature 1, no top-k / top-p. Events use the out-of-jet exclusion (R = 0.4): a
particle outside every jet is not placed within R of a planned jet axis.

The output file holds `particles` [n, max particles, 8] (zero-padded, lab frame), `mask`,
`columns` (the names of the 8 particle columns) and the planned `jet_pt`, `jet_rapidity`,
`jet_dphi`, `jet_m`, `n_const` of each of the 20 jet slots.
"""

import argparse

import numpy as np
import torch

from humite.data.readers import PARTICLE_COLUMNS, condition_codes
from humite.data.tokenizer import EventTokenizer
from humite.models.generation import generate
from humite.pretrained import load_pretrained

SOURCE = {"jets": 1, "events": 2}
EXCLUSION_RADIUS = {"jets": None, "events": 0.4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="folder with config.yaml and model.ckpt")
    parser.add_argument("kind", choices=list(SOURCE), help="JetClass jets or COLLIDE-2V events")
    parser.add_argument("--condition", help="jet type or process to generate (default: not given)")
    parser.add_argument("--kinematics-only", action="store_true", help="no pid, d0, dz")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output .npz file")
    args = parser.parse_args()

    module, data = load_pretrained(args.model)
    codes = condition_codes(data)
    if args.condition is not None and args.condition not in codes:
        parser.error(f"--condition must be one of {sorted(codes)}")
    tokenizer = EventTokenizer(data, pad_length=data.max_particles)
    torch.manual_seed(args.seed)

    batches = []
    for start in range(0, args.n, args.batch_size):
        batches.append(
            generate(
                module.model,
                tokenizer,
                min(args.batch_size, args.n - start),
                source=SOURCE[args.kind],
                condition=codes.get(args.condition, 0),
                all_features=not args.kinematics_only,
                out_of_jet_exclusion=EXCLUSION_RADIUS[args.kind],
            )
        )
        print(f"{min(start + args.batch_size, args.n)} / {args.n}", flush=True)

    width = max(batch["mask"].shape[1] for batch in batches)

    def padded(array: torch.Tensor) -> np.ndarray:
        """Pad the particle dimension of a batch to the widest batch."""
        out = np.zeros((array.shape[0], width, *array.shape[2:]), dtype=array.cpu().numpy().dtype)
        out[:, : array.shape[1]] = array.cpu().numpy()
        return out

    out = {
        "particles": np.concatenate([padded(batch["particles"]) for batch in batches]),
        "mask": np.concatenate([padded(batch["mask"]) for batch in batches]),
        "columns": np.array(PARTICLE_COLUMNS),
    }
    for name in batches[0]["jets"]:
        out[name] = np.concatenate([b["jets"][name].cpu().numpy() for b in batches])
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
