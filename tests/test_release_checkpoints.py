"""The released checkpoints give exactly the outputs recorded at release time.

Needs the checkpoints, the data and a GPU:

    HUMITE_RELEASE_DIR=<folder with omnix/ jets_only/ events_only/> pytest tests/test_release_checkpoints.py

Compared with tests/fixtures/release_reference.json, per model:

* the loss of the first validation batches of every member (both passes), and a hash over
  every tensor the model returns for them, in float32 and in the mixed precision of the
  trainings (bfloat16 autocast);
* fixed-seed samples: without and with a class given, kinematics only, and with the
  out-of-jet exclusion.

Re-record with HUMITE_RELEASE_RECORD=1 on the same GPU type (the hashes are bitwise).
"""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from humite.data.datamodule import BatchBuilder, member_batches
from humite.models.generation import generate
from humite.pretrained import load_pretrained

RELEASE = os.environ.get("HUMITE_RELEASE_DIR", "")
REFERENCE = Path(__file__).parent / "fixtures" / "release_reference.json"
C2V = ["c2v_qcd_inc", "c2v_qcd_bb", "c2v_tt_lep", "c2v_tt_semi", "c2v_tt_had"]
MEMBERS = {"omnix": ["jetclass", *C2V], "jets_only": ["jetclass"], "events_only": C2V}
N_BATCH, BATCH_SIZE, N_GENERATED, MAX_LEN = 4, 8, 16, 704

# (member whose tokenizer decodes, source, condition, further sampling options)
KINEMATICS_ONLY = {"all_features": False}
EXCLUSION = {"out_of_jet_exclusion": 0.4}
JETS = [("jetclass", 1, 0, {}), ("jetclass", 1, 9, {}), ("jetclass", 1, 0, KINEMATICS_ONLY)]
EVENTS = [
    ("c2v_tt_semi", 2, 0, {}),
    ("c2v_tt_semi", 2, 14, {}),
    ("c2v_tt_semi", 2, 0, EXCLUSION),
    ("c2v_qcd_inc", 2, 11, EXCLUSION),
]
GENERATION = {"omnix": JETS + EVENTS, "jets_only": JETS, "events_only": EVENTS}

pytestmark = pytest.mark.skipif(
    not RELEASE or not torch.cuda.is_available(),
    reason="set HUMITE_RELEASE_DIR (released checkpoints) and run on a GPU",
)


def seed(value: int = 1234) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, dict):
        return {key: to_cuda(value) for key, value in obj.items()}
    return obj


def flatten(obj, prefix, out):
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            flatten(value, f"{prefix}.{key}", out)


def check(name: str, got: dict, exact: tuple) -> None:
    reference = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    if os.environ.get("HUMITE_RELEASE_RECORD"):
        reference[name] = got
        REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"recorded {name}")
    assert set(got) == set(reference[name])
    for case, want in reference[name].items():
        if "loss" in want:
            assert got[case]["loss"] == pytest.approx(want["loss"], rel=1e-6), case
        for key in exact:
            assert got[case][key] == want[key], (case, key)


def forward_outputs(name: str, mixed_precision: bool) -> dict:
    module, data = load_pretrained(Path(RELEASE) / name)
    got = {}
    for member in MEMBERS[name]:
        seed()
        batches = iter(member_batches(data, member, "val", BATCH_SIZE))
        losses, hashes = [], []
        for _ in range(N_BATCH):
            seed()
            batch = to_cuda(next(batches))
            seed()
            autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision)
            with torch.no_grad(), autocast:
                loss, _ = module.loss(batch, "val")
                outputs = {}
                flatten(module.model(batch=batch, tagging=False), "generative", outputs)
                flatten(module.model(batch=batch, tagging=True), "tagging", outputs)
            digest = hashlib.sha256()
            for key in sorted(outputs):
                digest.update(key.encode())
                digest.update(outputs[key].float().cpu().numpy().tobytes())
            losses.append(float(loss))
            hashes.append(digest.hexdigest())
        got[member] = {"loss": losses, "sha256": hashes}
    return got


@pytest.mark.parametrize("name", list(MEMBERS))
def test_forward(name):
    check(name, forward_outputs(name, mixed_precision=False), exact=("sha256",))


@pytest.mark.parametrize("name", list(MEMBERS))
def test_forward_mixed_precision(name):
    got = forward_outputs(name, mixed_precision=True)
    check(f"mixed_precision/{name}", got, exact=("sha256",))


@pytest.mark.parametrize("name", list(MEMBERS))
def test_generation(name):
    module, data = load_pretrained(Path(RELEASE) / name)
    tokenizers = BatchBuilder(data).tokenizers
    got = {}
    for member, source, condition, options in GENERATION[name]:
        seed(7)
        out = generate(
            module.model, tokenizers[member], N_GENERATED, source, condition=condition, **options
        )
        particles, mask = out["particles"].float(), out["mask"].bool()
        assert bool(torch.isfinite(particles[mask]).all())
        # the reference hashes cover the full generation length
        padding = MAX_LEN - mask.shape[1]
        particles = torch.nn.functional.pad(particles, (0, 0, 0, padding))
        mask = torch.nn.functional.pad(mask, (0, padding))
        digest = hashlib.sha256()
        digest.update(mask.cpu().numpy().tobytes())
        digest.update(torch.where(mask[..., None], particles, 0.0).cpu().numpy().tobytes())
        case = "/".join([member, str(condition), *options])
        got[case] = {"n_particles": mask.sum(1).tolist(), "sha256": digest.hexdigest()}
    check(f"generation/{name}", got, exact=("n_particles", "sha256"))
