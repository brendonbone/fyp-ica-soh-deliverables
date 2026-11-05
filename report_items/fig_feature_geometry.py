"""
Fig. 4. The nine features of the dominant IC peak, drawn on a measured curve
The published |dQ/dV| of cell b1c20, cycle 1 (a fresh cycle), with the
landmarks the extractor (src.features) measured on it: the apex (height,
position), the half-prominence width (fwhm), the two 25 mV flank chords
(slope_left, slope_right) and the three nested integration windows (area25,
area35, area). Reads results/feature_geometry.npz (stage 3). The chord ends
are re-interpolated here from the stored curve with the extractor's own call
(np.interp on the ascending voltage axis). Prominence is measured from the
peak's base, 0.002 Ah/V on this curve, so it coincides with height here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from report_items.common import COL_WIDTH, apply_style, save_figure
from src.config import RESULTS_DIR
from src.features import AREA_WINDOWS

LABEL = "Fig. 4"
# make() completes the caption from the data; run_report.py reads TITLE after make().
TITLE = "The nine features of the dominant IC peak on a measured curve"
TITLE_FULL = (TITLE + ": the published |dQ/dV| of cell {cell}, cycle {cycle} (SoH {soh:.2f}). Height "
              "{h:.2f} Ah/V at position {v:.3f} V, prominence {p:.2f} Ah/V (base {base:.3f} Ah/V), fwhm "
              "{fwhm:.0f} mV, area25 / area35 / area {a25:.3f} / {a35:.3f} / {a:.3f} Ah, slope_left / "
              "slope_right {sl:.0f} / {sr:.0f} Ah V⁻²")
SOURCES = ["feature_geometry.npz"]

NPZ = RESULTS_DIR / "feature_geometry.npz"
SLOPE_DV = 0.025          # the flank chord length (V), as in src.features
# Okabe-Ito, as the other figures: vermillion for the slopes, blue for the areas;
# the apex and width dimensions in black.
C_SLOPE, C_AREA, C_RULE = "#D55E00", "#0072B2", "0.4"
AREA_FILLS = {"area": "#D9E7F5", "area35": "#A8C9E9", "area25": "#6FA6D8"}   # widest palest
FS = 7                    # annotation text (the style's tick size)
DIM = dict(arrowstyle="<|-|>", lw=0.7, mutation_scale=6)                   # dimension arrows
LEADER = dict(arrowstyle="-|>", lw=0.6, mutation_scale=6, shrinkA=2, shrinkB=2)


def load() -> dict:
    """The arrays stage 3 stored for the example cycle."""
    if not NPZ.exists():
        raise FileNotFoundError(f"{NPZ} is missing: run stage 3 (python -m stages.s03_dqdv_check), "
                                "which needs the raw Severson pickles")
    return dict(np.load(NPZ, allow_pickle=False))


def draw(d: dict):
    """One panel: the curve, the nested area windows and every landmark, labelled."""
    v, y = d["v_roi"].astype(float), d["abs_dqdv"].astype(float)
    pv, ph = float(d["peak_v"][0]), float(d["peak_h"][0])
    fwv, fwy = d["fwhm_v"].astype(float), float(d["fwhm_y"][0])
    top = ph * 1.21
    x0, x1 = pv - 0.190, pv + 0.205

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.145, top=0.975)

    # the nested area windows, widest first so the narrowest reads darkest
    for name, hw in sorted(AREA_WINDOWS.items(), key=lambda kv: -kv[1]):
        m = (v >= pv - hw) & (v <= pv + hw)
        ax.fill_between(v[m], 0, y[m], color=AREA_FILLS[name], lw=0, zorder=1)
    ax.plot(v, y, color="k", lw=1.1, solid_capstyle="round", zorder=4)

    # position: a dropped line, labelled from the free upper left
    ax.plot([pv, pv], [0, ph], color=C_RULE, lw=0.7, ls=(0, (2.5, 2.5)), zorder=3)
    ax.annotate("position", xy=(pv, ph * 0.995), xytext=(pv - 0.183, top * 0.985), fontsize=FS,
                ha="left", va="top", arrowprops=dict(color=C_RULE, connectionstyle="arc3,rad=-0.18", **LEADER))

    # height: a dimension line out in the tail, clear of the curve and of the slope_right label
    x_h = pv + 0.160
    ax.plot([pv, x_h + 0.005], [ph, ph], color=C_RULE, lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.annotate("", xy=(x_h, 0), xytext=(x_h, ph), arrowprops=dict(color="k", **DIM))
    for yy in (0.0, ph):
        ax.plot([x_h - 0.006, x_h + 0.006], [yy, yy], color="k", lw=0.7)
    ax.text(x1 - 0.008, top * 0.985, "height ≈ prominence", fontsize=FS, ha="right", va="top")

    # fwhm: the width at half prominence, labelled at its left end
    ax.annotate("", xy=(fwv[0], fwy), xytext=(fwv[1], fwy), arrowprops=dict(color="k", **DIM))
    ax.text(fwv.min() - 0.006, fwy, "fwhm", fontsize=FS, ha="right", va="center")

    # flank slopes: the two 25 mV chords (left = lower voltage, as in the extractor)
    for sign, name, xt, yt in ((-1, "slope_left", pv - 0.183, ph * 0.80),
                               (1, "slope_right", pv + 0.052, ph * 0.30)):
        xf = pv + sign * SLOPE_DV
        yf = float(np.interp(xf, v[::-1], y[::-1]))     # v is descending: interpolate on the ascending axis
        ax.plot([pv, xf], [ph, yf], color=C_SLOPE, lw=1.1, zorder=5)
        ax.plot([xf], [yf], "o", ms=2.6, color=C_SLOPE, zorder=6)
        ax.annotate(name, xy=(xf, yf), xytext=(xt, yt), fontsize=FS, color=C_SLOPE, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_SLOPE, lw=0.6, shrinkA=2, shrinkB=3,
                                    connectionstyle=f"arc3,rad={0.22 * sign}"))

    # the three nested area windows
    ax.annotate("area25 / area35 / area\n(±25 / ±35 / ±50 mV)", xy=(pv - 0.044, 0.95),
                xytext=(pv - 0.185, 0.95), fontsize=FS, color=C_AREA, ha="left", va="center", linespacing=1.2,
                arrowprops=dict(color=C_AREA, connectionstyle="arc3,rad=-0.2", **LEADER))

    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.30, top)
    ax.set_xticks(np.arange(np.ceil(x0 * 10) / 10, x1, 0.1))
    ax.set_yticks(range(0, int(ph) + 1, 2))
    ax.spines["left"].set_bounds(0, top)
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel(r"$|\mathrm{d}Q/\mathrm{d}V|$ (Ah V$^{-1}$)")
    return fig


def make(stem: Path) -> list[Path]:
    global TITLE
    d = load()
    TITLE = TITLE_FULL.format(cell=str(d["cell_id"][0]), cycle=int(d["cycle"][0]), soh=float(d["soh"][0]),
                              h=float(d["peak_h"][0]), v=float(d["peak_v"][0]), p=float(d["prominence"][0]),
                              base=float(d["prom_base"][0]), fwhm=1000 * float(d["fwhm"][0]),
                              a25=float(d["area25"][0]), a35=float(d["area35"][0]), a=float(d["area"][0]),
                              sl=float(d["slope_left"][0]), sr=float(d["slope_right"][0]))
    apply_style()
    # STIX: serif mathtext to match the Times text; scoped so later items keep the default.
    with plt.rc_context({"mathtext.fontset": "stix"}):
        return save_figure(stem, draw(d))
