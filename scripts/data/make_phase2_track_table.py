"""Build the d0 / dz resolution table that the C2V reader uses to smear track impact parameters.

Input: the d0 and dz resolution points of the CMS Phase-2 Tracker TDR (CERN-LHCC-2017-009,
CMS-TDR-014, Fig. 12.12), read from the figure's vector graphics into a CSV with the columns
axis (d0 or dz), pt_gev (1, 10 or 100), eta (bin centre, steps of 0.1), sigma_cm (the RMS of the
residuals) and sigma_up_cm / sigma_down_cm (the ends of the drawn error bar; equal to sigma_cm
where the bar is hidden behind the marker).

Output: a JSON table of Gaussian widths per bin of track pT and |eta|, in the layout the reader
looks up (lower pT edges, upper |eta| edges, one row per |eta| bin).

The CSV is not part of the repository (the figure's values are not republished here); it is
available from the authors on request.

    python scripts/data/make_phase2_track_table.py --points <path to the CSV>
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "humite" / "data" / "phase2_track_resolution.json"

AXES = ("d0", "dz")
TDR_PT = (1.0, 10.0, 100.0)  # GeV, the transverse momenta of the figure's three curves
CM_TO_UM = 1e4
# Uncertainty (in decades) of a point whose error bar is hidden behind its marker: half the
# marker height on the figure's log axis (2.73 page units; 57.89 and 48.64 page units per decade
# in the d0 and dz panels). It bounds the hidden bar from above.
HALF_MARKER = {"d0": 1.365 / 57.89, "dz": 1.365 / 48.64}
ETA_SPLIT = 2.5  # the formula's k is fitted separately below and above this |eta|
ETA_BINS = np.round(np.arange(0.05, 4.0, 0.1), 2)  # |eta| bin centres; the figure's points

# The table's grid
BINS_PER_DECADE = 40
PT_MIN, PT_MAX = 0.1, 200.0  # GeV
ETA_STEP, ETA_MAX = 0.1, 4.0


def read_points(path):
    """points[axis][pT] = rows of (eta, width, upper end, lower end of the error bar), in um."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            raw[r["axis"]][float(r["pt_gev"])].append(
                [float(r["eta"])]
                + [float(r[c]) * CM_TO_UM for c in ("sigma_cm", "sigma_up_cm", "sigma_down_cm")]
            )
    return {ax: {q: np.array(sorted(v)) for q, v in d.items()} for ax, d in raw.items()}


def bin_points(points, axis, e):
    """pT, width and log10 uncertainty of the points of |eta| bin e (both eta signs)."""
    x, y, s = [], [], []
    for q in TDR_PT:
        t = points[axis][q]
        sel = np.isclose(np.abs(t[:, 0]), e)
        half_bar = 0.5 * (np.log10(t[sel, 2]) - np.log10(t[sel, 3]))
        x += [q] * int(sel.sum())
        y += list(t[sel, 1])
        s += list(np.maximum(half_bar, HALF_MARKER[axis]))
    return np.array(x), np.array(y), np.array(s)


def formula(par, p_t, k):
    """sigma(pT) = (a^k + (b / pT)^k)^(1/k): a constant term (hit resolution) and a 1 / pT term
    (multiple scattering); k sets how sharply the two join (2: quadrature sum, 1: plain sum)."""
    a, b = par
    return (a**k + (b / p_t) ** k) ** (1 / k)


def fit_bin(x, y, s, k):
    """a and b of one |eta| bin at fixed k, minimising the squared log10 deviations divided by
    the points' uncertainties; returns them and that sum."""
    start = [y.min(), y.max() * x.min()]
    r = least_squares(lambda p: np.log10(formula(p, x, k) / y) / s, start, bounds=(0, np.inf))
    return r.x, float(np.sum(r.fun**2))


def fit_region(points, axis, bins):
    """One k for all given |eta| bins, a and b per bin: for each trial k every bin gets its best
    a and b, and k minimises the sum over the bins."""
    data = {e: bin_points(points, axis, e) for e in bins}
    k = minimize_scalar(
        lambda q: sum(fit_bin(*data[e], q)[1] for e in bins), bounds=(0.4, 4.0), method="bounded"
    ).x
    return {e: (k, fit_bin(*data[e], k)[0]) for e in bins}


def straight_lines(x3, y3, p_t):
    """Straight lines in log width against log pT through three points; slope -1 below the
    first, constant above the last."""
    lx, lx3, ly3 = np.log10(p_t), np.log10(x3), np.log10(y3)
    out = np.interp(lx, lx3, ly3)
    low = lx < lx3[0]
    out[low] = ly3[0] - (lx[low] - lx3[0])
    return 10**out


def folded_values(points, axis, e):
    """The widths of |eta| bin e at the three pT values: the mean of log width over +eta, -eta."""
    out = []
    for q in TDR_PT:
        t = points[axis][q]
        out.append(10 ** np.mean(np.log10(t[np.isclose(np.abs(t[:, 0]), e), 1])))
    return np.array(out)


def build_widths(points, pt_centre):
    """Width (um) per |eta| bin and pT, and the fits behind it. d0 at all |eta| and dz below
    |eta| 2.5: the formula, k per axis and region. dz beyond 2.5: straight lines."""
    inner, outer = ETA_BINS[ETA_BINS < ETA_SPLIT], ETA_BINS[ETA_BINS > ETA_SPLIT]
    fits = {ax: {**fit_region(points, ax, inner), **fit_region(points, ax, outer)} for ax in AXES}
    widths = {}
    for ax in AXES:
        rows = []
        for e in ETA_BINS:
            if ax == "dz" and e > ETA_SPLIT:
                three = folded_values(points, ax, e)
                rows.append(straight_lines(np.array(TDR_PT), three, pt_centre))
            else:
                k, par = fits[ax][e]
                rows.append(formula(par, pt_centre, k))
        widths[ax] = np.array(rows)
    return widths, fits


def sig4(values):
    return [float(f"{v:.4g}") for v in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--points", type=Path, required=True, help="CSV of the extracted points")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    points = read_points(args.points)
    n_pt = int(round(BINS_PER_DECADE * np.log10(PT_MAX / PT_MIN)))
    pt_lower = PT_MIN * 10 ** (np.arange(n_pt) / BINS_PER_DECADE)
    pt_centre = PT_MIN * 10 ** ((np.arange(n_pt) + 0.5) / BINS_PER_DECADE)  # geometric centres
    n_eta = int(round(ETA_MAX / ETA_STEP))
    eta_upper = np.round(ETA_STEP * np.arange(1, n_eta), 2)
    assert np.allclose(ETA_BINS, ETA_STEP * (np.arange(n_eta) + 0.5)), "eta grid != figure bins"

    widths_um, fits = build_widths(points, pt_centre)

    table = {
        "format": "humite track resolution table, version 1",
        "description": (
            "Gaussian widths for smearing the transverse (d0) and longitudinal (dz) impact "
            "parameters of charged particles, per bin of track pT and |eta|, for the CMS Phase-2 "
            "tracker."
        ),
        "source": (
            "Derived from CMS Collaboration, The Phase-2 Upgrade of the CMS Tracker, Technical "
            "Design Report, CERN-LHCC-2017-009, CMS-TDR-014 (2017), Fig. 12.12: resolution of d0 "
            "and dz (RMS of the residuals) against eta for single isolated muons with pT = 1, 10 "
            "and 100 GeV. The points were read from the figure's vector graphics."
        ),
        "method": (
            "Points at +eta and -eta are pooled into bins of 0.1 in |eta|. Smearing for d0 at all "
            "|eta| and dz below |eta| 2.5 is calculated in the following way: "
            "sigma(pT) = (a^k + (b / pT)^k)^(1/k), with a and b fitted per |eta| "
            "bin, and with one k per axis and region (|eta| below or above 2.5), each point "
            "weighted by its error bar. Smearing for dz above |eta| 2.5 is approximated using "
            "straight lines in log sigma against log pT through the three values per bin. Each "
            "table entry is that width at the bin's geometric pT centre and |eta| centre. Below "
            "1 GeV and above 100 GeV the widths are extrapolations: the formula continued "
            "(approaching b / pT at low and a at high pT), and for dz beyond |eta| 2.5 slope -1 "
            "below 1 GeV and constant above 100 GeV."
        ),
        "units": {"pt": "GeV", "width": "cm"},
        "binning": (
            "pt_lower_edges: lower edge of each pT bin; the first bin also holds all lower pT, "
            "the last all higher pT. abs_eta_upper_edges: upper edge of each |eta| bin except the "
            "last, which holds all higher |eta|. A value equal to an edge belongs to the lower "
            "bin. Widths: one row per |eta| bin, one column per pT bin."
        ),
        "pt_lower_edges": sig4(pt_lower),
        "abs_eta_upper_edges": [float(v) for v in eta_upper],
        "d0": [sig4(row / CM_TO_UM) for row in widths_um["d0"]],
        "dz": [sig4(row / CM_TO_UM) for row in widths_um["dz"]],
    }
    text = json.dumps(table, indent=1)
    args.out.write_text(text)

    # ---- summary: the fitted k, how well the table matches the points, the file
    for ax in AXES:
        k_in, k_out = fits[ax][2.45][0], fits[ax][2.55][0]
        out_note = "not used, straight lines" if ax == "dz" else f"{k_out:.3f}"
        print(f"{ax}: k = {k_in:.3f} below |eta| 2.5, above: {out_note}")
    edges_pt = np.array(table["pt_lower_edges"])
    edges_eta = np.array(table["abs_eta_upper_edges"])
    for ax in AXES:
        lut = np.array(table[ax]) * CM_TO_UM
        for region, keep in (
            ("|eta| < 2.5", lambda a: a < ETA_SPLIT),
            ("|eta| > 2.5", lambda a: a > ETA_SPLIT),
        ):
            ratios = []
            for q in TDR_PT:
                t = points[ax][q]
                a_eta = np.abs(t[:, 0])
                sel = keep(a_eta)
                i_pt = np.searchsorted(edges_pt[1:], np.full(sel.sum(), q), side="left")
                i_eta = np.searchsorted(edges_eta, a_eta[sel], side="left")
                ratios.append(t[sel, 1] / lut[i_eta, i_pt])
            dev = np.concatenate(ratios) - 1
            rms, largest = 100 * np.sqrt(np.mean(dev**2)), 100 * np.max(np.abs(dev))
            print(
                f"{ax} {region}: TDR / table - 1 at the figure's points: rms {rms:.1f} %, "
                f"largest {largest:.1f} %"
            )
    print(
        f"wrote {args.out}: {n_eta} |eta| bins x {n_pt} pT bins, "
        f"sha256 {hashlib.sha256(text.encode()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
