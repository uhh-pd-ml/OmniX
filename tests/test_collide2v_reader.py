"""The COLLIDE-2V reader on the converter's reference file (tests/fixtures/c2v_convert)."""

from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from humite.data.readers import Collide2VReader, row_count

FILE = str(
    Path(__file__).parent
    / "fixtures"
    / "c2v_convert"
    / "reference"
    / "train"
    / "tt0123j_5f_ckm_LO_MLM_leptonic-NEVENT10000-RS29000001_train.parquet"
)


def read(**options) -> ak.Array:
    options.setdefault("particle_abs_eta_max", 1e9)
    reader = Collide2VReader(
        n_jet_slots=20, d0_range=(-25.0, 25.0), dz_range=(-60.0, 60.0), split="train", **options
    )
    return reader.read(FILE, 0, row_count(FILE)).particles


def stored(*columns) -> ak.Array:
    return ak.from_arrow(pq.read_table(FILE, columns=list(columns)))


def test_particle_cut_keeps_exactly_the_particles_inside():
    inside = np.abs(stored("eta")["eta"]) <= 4.0
    assert int(ak.sum(~inside)) > 0, "the reference file has no particle beyond |eta| 4"
    cut, full = read(particle_abs_eta_max=4.0), read()
    assert float(ak.max(np.abs(cut["eta"]))) <= 4.0
    for column in ("pt", "eta", "phi", "jet_idx", "pf_type", "charge", "d0", "dz"):
        np.testing.assert_array_equal(
            ak.to_numpy(ak.flatten(cut[column])), ak.to_numpy(ak.flatten(full[column][inside]))
        )


def test_the_leading_jet_sits_at_phi_zero():
    table = stored("ak4_pt", "ak4_eta", "ak4_phi", "phi")
    selected = (table["ak4_pt"] > 15.0) & (np.abs(table["ak4_eta"]) < 4.0)
    leading_phi = ak.firsts(table["ak4_phi"][selected])
    has_jet = ~ak.to_numpy(ak.is_none(leading_phi))
    assert has_jet.any()
    shift = ak.to_numpy(ak.fill_none(leading_phi, 0.0)).astype(np.float64)
    expected = (table["phi"] - shift + np.pi) % (2.0 * np.pi) - np.pi
    particles = read()
    np.testing.assert_allclose(
        ak.to_numpy(ak.flatten(particles["phi"][has_jet])),
        ak.to_numpy(ak.flatten(expected[has_jet])),
        atol=1e-5,
    )


def test_events_without_a_jet_rotate_their_leading_particle():
    particles = read(jet_pt_min=1e5)
    assert float(ak.max(particles["jet_idx"])) == 0.0
    leading = ak.firsts(particles["phi"][ak.argmax(particles["pt"], axis=1, keepdims=True)])
    np.testing.assert_allclose(ak.to_numpy(leading), 0.0, atol=1e-6)


def test_events_emptied_by_the_cut_are_dropped(monkeypatch):
    table = stored("eta", "charge")
    cut = 0.05
    inside = np.abs(table["eta"]) <= cut
    n_inside = ak.to_numpy(ak.sum(inside, axis=1))
    assert (n_inside == 0).any() and (n_inside > 0).any(), "pick another cut for this file"

    rows_seen = []
    original = Collide2VReader.fixed_normals

    def spy(self, name, row, particle):
        rows_seen.append(np.asarray(row))
        return original(name, row, particle)

    monkeypatch.setattr(Collide2VReader, "fixed_normals", spy)
    particles = read(particle_abs_eta_max=cut)
    np.testing.assert_array_equal(ak.to_numpy(ak.num(particles["eta"])), n_inside[n_inside > 0])
    # the smearing of a kept event is keyed by its row in the file
    charged_rows = np.flatnonzero(ak.to_numpy(ak.sum(inside & (table["charge"] != 0), axis=1)))
    np.testing.assert_array_equal(np.unique(np.concatenate(rows_seen)), charged_rows)


def test_smearing_does_not_depend_on_the_particle_cut():
    inside = np.abs(stored("eta")["eta"]) <= 2.5
    full, cut = read(), read(particle_abs_eta_max=2.5)
    assert len(full) == len(cut)
    for column in ("d0", "dz"):
        np.testing.assert_array_equal(
            ak.to_numpy(ak.flatten(cut[column])), ak.to_numpy(ak.flatten(full[column][inside]))
        )
