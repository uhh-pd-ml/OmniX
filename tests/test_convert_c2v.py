"""Reference test for the COLLIDE-2V converter `scripts/data/convert_c2v_fullevent.py`.

The converter writes a `conversion` block into every output file, built from its own constants,
and the reader refuses files whose definitions differ from its own. A change in the converter's
logic that leaves those constants as they are passes that check. This test catches it: it
converts a small committed input file and compares the result with a stored reference output,
namely every column of every written file, the `conversion` block apart from its provenance,
and the counts the converter reports.

When a change of the output is intended, regenerate the reference and commit it with the change:

    HUMITE_REGEN_C2V_REFERENCE=1 pytest -q tests/test_convert_c2v.py

The input file holds 103 events of the raw COLLIDE-2V file
`tt0123j_5f_ckm_LO_MLM_leptonic-NEVENT10000-RS29000001.parquet` (collide2v_v1.0): its first
100 events and three later events with weight not 0 and an AK8 jet (source rows 401, 426 and
870), with only the columns the converter reads and only the `dataset_version` metadata entry.
To stay under the repository's 500 kB file limit, each event keeps every PUPPI row with
PuppiW > 0 (all jet constituents among them) but only its first 5 rows with PuppiW = 0, which the
converter drops anyway. The file keeps the source file's name, because the converter reads the
process from it and derives the split and the dequantizing draws from the name and the row
number; the event count in the name (10000) is therefore not this file's.

Run inside the container: pytest -q tests/test_convert_c2v.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "convert_c2v_fullevent.py"
_spec = importlib.util.spec_from_file_location("convert_c2v_fullevent", SCRIPT)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

FIXTURES = Path(__file__).parent / "fixtures" / "c2v_convert"
INPUT = FIXTURES / "input" / "tt0123j_5f_ckm_LO_MLM_leptonic-NEVENT10000-RS29000001.parquet"
REFERENCE = FIXTURES / "reference"
REFERENCE_COUNTS = REFERENCE / "counts.json"
REGEN = os.environ.get("HUMITE_REGEN_C2V_REFERENCE") == "1"

# Floats may differ in their last digits between machines (numpy's sinh, arcsinh and arctan2
# come from the system's math library); a change of logic moves them by far more.
RTOL = 1e-6
# Read 7 events at a time, so the 103 events span several batches, as a real file's events do.
READ_BATCH = 7
ROW_GROUP = 8192


def _written(folder: Path) -> list[Path]:
    """The converter's output files under `folder` (`<split>/<stem>_<split>.parquet`)."""
    return sorted(folder.glob("*/*.parquet"))


@pytest.fixture(scope="module")
def converted(tmp_path_factory) -> tuple[Path, dict]:
    """The input file converted into a temporary folder, and the converter's counts; with
    HUMITE_REGEN_C2V_REFERENCE=1 the result also replaces the stored reference."""
    out = tmp_path_factory.mktemp("c2v_convert")
    block = conv.conversion_block(conv._git_commit())
    counts = conv.convert_file(INPUT, out, block, READ_BATCH, ROW_GROUP)
    counts = json.loads(json.dumps(counts))  # as a manifest stores it
    counts.pop("conversion")  # compared file by file, from the files themselves
    if REGEN:
        shutil.rmtree(REFERENCE, ignore_errors=True)
        for f in _written(out):
            target = REFERENCE / f.relative_to(out)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
        REFERENCE.mkdir(parents=True, exist_ok=True)
        REFERENCE_COUNTS.write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n")
    if not REFERENCE_COUNTS.exists():
        pytest.fail(
            f"no reference in {REFERENCE}: generate it with "
            "HUMITE_REGEN_C2V_REFERENCE=1 pytest -q tests/test_convert_c2v.py"
        )
    return out, counts


def _assert_same_column(got: ak.Array, ref: ak.Array, where: str) -> None:
    """List lengths at every depth equal; strings and integers equal; floats equal within RTOL."""
    if ref.layout.parameter("__array__") == "string":
        assert ak.to_list(got) == ak.to_list(ref), f"{where}: values differ"
        return
    while ref.ndim > 1:
        np.testing.assert_array_equal(
            ak.to_numpy(ak.num(got, axis=1)),
            ak.to_numpy(ak.num(ref, axis=1)),
            err_msg=f"{where}: list lengths differ",
        )
        got, ref = ak.flatten(got, axis=1), ak.flatten(ref, axis=1)
    g, r = ak.to_numpy(got), ak.to_numpy(ref)
    assert g.dtype == r.dtype, f"{where}: dtype {g.dtype}, reference {r.dtype}"
    if np.issubdtype(r.dtype, np.floating):
        np.testing.assert_allclose(g, r, rtol=RTOL, atol=0, err_msg=f"{where}: values differ")
    else:
        np.testing.assert_array_equal(g, r, err_msg=f"{where}: values differ")


def test_writes_the_reference_files(converted):
    out, _ = converted
    got = [f.relative_to(out) for f in _written(out)]
    assert got == [f.relative_to(REFERENCE) for f in _written(REFERENCE)]


def test_columns_match_the_reference(converted):
    out, _ = converted
    for ref_file in _written(REFERENCE):
        where = ref_file.relative_to(REFERENCE)
        got = ak.from_parquet(out / where)
        ref = ak.from_parquet(ref_file)
        assert str(got.type) == str(ref.type), f"{where}: events, column names or types differ"
        for field in ref.fields:
            _assert_same_column(got[field], ref[field], f"{where}: {field}")


def test_conversion_block_matches_the_reference(converted):
    """Everything but the provenance (git commit, converter checksum), which changes with every
    commit."""
    out, _ = converted
    for ref_file in _written(REFERENCE):
        where = ref_file.relative_to(REFERENCE)
        got = conv.read_conversion(out / where)
        ref = conv.read_conversion(ref_file)
        got.pop("provenance")
        ref.pop("provenance")
        assert got == ref, f"{where}: conversion block differs"


def test_counts_match_the_reference(converted):
    _, counts = converted
    assert counts == json.loads(REFERENCE_COUNTS.read_text())
