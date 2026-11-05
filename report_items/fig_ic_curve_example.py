"""
Fig. 1. Discharge curves and their incremental-capacity curves over one cell's life
(a) cell voltage against discharged capacity on the constant-current discharge
leg, (b) dQ/dV against voltage, for 12 cycles of cell b1c20 spread evenly over
its life, coloured by cycle number. dQ/dV keeps its discharge sign (negative),
so the IC peak is a trough, and it is the curve recomputed from the raw V and Qd
samples (src.dqdv.recompute_dqdv), not the published one. Reads
results/ic_curve_example.npz (stage 3); the peak height and position in the
caption are the extremes of the plotted curves inside panel (b)'s window.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from report_items.common import FULL_WIDTH, MINUS, apply_style, save_figure
from src.config import RESULTS_DIR
from src.peak_extraction import V_GRID

LABEL = "Fig. 1"
# make() completes the caption from the data; run_report.py reads TITLE after make().
TITLE = "Discharge curves (a) and incremental-capacity curves (b) of one cell over its life"
TITLE_FULL = (TITLE + ": {n} of the {n_total} cycles of cell {cell}, SoH {soh0:.2f} to {soh1:.2f}, "
              "coloured by cycle number. Over this life the IC peak loses {dh:.0f} % of its height "
              "and moves {dv:.0f} mV to lower voltage")
SOURCES = ["ic_curve_example.npz"]

NPZ = RESULTS_DIR / "ic_curve_example.npz"
V_WINDOW = (2.86, 3.40)   # panel (b): the whole peak and its feet, so the shift is visible
# Trimmed viridis: its pale-yellow top vanishes on white. Lines and colour bar use the same ramp.
CMAP = LinearSegmentedColormap.from_list("viridis_trim", plt.cm.viridis(np.linspace(0.02, 0.88, 256)))


def load() -> dict:
    """The arrays stage 3 stored for the example cell."""
    if not NPZ.exists():
        raise FileNotFoundError(f"{NPZ} is missing: run stage 3 (python -m stages.s03_dqdv_check), "
                                "which needs the raw Severson pickles")
    return dict(np.load(NPZ, allow_pickle=False))


def peak_of(dqdv: np.ndarray) -> tuple[float, float]:
    """(height as a magnitude, voltage) of the deepest point of the signed dQ/dV inside V_WINDOW."""
    inside = (V_GRID >= V_WINDOW[0]) & (V_GRID <= V_WINDOW[1])
    curve, grid = -dqdv[inside], V_GRID[inside]
    ok = np.isfinite(curve)
    k = int(np.argmax(curve[ok]))
    return float(curve[ok][k]), float(grid[ok][k])


def draw(d: dict):
    """Two panels side by side and a vertical colour bar of cycle number on the right."""
    cycles = d["cycles"]
    n = len(cycles)
    norm = Normalize(vmin=cycles.min(), vmax=cycles.max())
    colours = CMAP(norm(cycles))

    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6), gridspec_kw={"wspace": 0.24})

    for i in range(n):
        ax_v.plot(d[f"q{i}"], d[f"v{i}"], color=colours[i], lw=1.0, solid_capstyle="round")
        ax_d.plot(V_GRID, d[f"dqdv{i}"], color=colours[i], lw=1.0, solid_capstyle="round")

    ax_v.set_xlabel(r"Discharged capacity $Q_\mathrm{d}$ (Ah)")
    ax_v.set_ylabel("Cell voltage (V)")
    ax_v.set_xlim(-0.03, 1.19)
    ax_v.set_ylim(1.95, 3.63)
    ax_v.set_xticks([0.0, 0.5, 1.0])
    ax_v.set_yticks([2.0, 2.5, 3.0, 3.5])

    ax_d.set_xlabel("Cell voltage (V)")
    ax_d.set_ylabel(r"$\mathrm{d}Q/\mathrm{d}V$ (Ah V$^{-1}$)")
    ax_d.set_xlim(*V_WINDOW)
    ax_d.set_ylim(-7.35, 0.35)
    ax_d.set_xticks([2.9, 3.0, 3.1, 3.2, 3.3])
    ax_d.set_yticks([0, -2, -4, -6], ["0"] + [f"{MINUS}{v}" for v in (2, 4, 6)])

    for ax, tag in ((ax_v, "(a)"), (ax_d, "(b)")):
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=[ax_v, ax_d],
                        fraction=0.025, pad=0.02, aspect=30)
    lo, hi = int(cycles.min()), int(cycles.max())
    # every 100 cycles, skipping any within 50 of the first/last tick so the labels do not collide
    cbar.set_ticks([lo] + [t for t in range(100, hi, 100) if lo + 50 < t < hi - 50] + [hi])
    cbar.set_label("Cycle number")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2, width=0.5)
    return fig


def make(stem: Path) -> list[Path]:
    global TITLE
    d = load()
    heights, positions = np.array([peak_of(d[f"dqdv{i}"]) for i in range(len(d["cycles"]))]).T
    TITLE = TITLE_FULL.format(n=len(d["cycles"]), n_total=int(d["n_total"][0]), cell=str(d["cell_id"][0]),
                              soh0=d["soh"][0], soh1=d["soh"][-1],
                              dh=100.0 * (heights[0] - heights[-1]) / heights[0],
                              dv=1000.0 * abs(positions[-1] - positions[0]))
    apply_style()
    # STIX: serif mathtext to match the Times text (Q_d, dQ/dV); scoped so later items keep the default.
    with plt.rc_context({"mathtext.fontset": "stix"}):
        return save_figure(stem, draw(d))
