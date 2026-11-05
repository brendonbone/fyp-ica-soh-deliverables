"""
Fig. B1. Published and recomputed dQ/dV of the mid-life cycle of six cells, with the per-cycle Pearson r
Two cells per batch, the cycle ENG-02 overlaid (k = number of cycles // 2). The curves are
results/dqdv_example_curves.csv (stage 3 reads them once from the raw pickles); the r in each
panel title is that cycle's rc_pearson_r in features_expanded.parquet. Both curves are stored with
the published sign and drawn negated, hence the '−dQ/dV' axis. Every panel shares the y-axis up to
7 Ah/V; a curve that rises above it (b3c20 at the 2.0 V edge) is clipped and annotated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report_items.common import FULL_WIDTH, apply_style, load_csv, result_path, save_figure

LABEL = "Fig. B1"
TITLE = ("Published and recomputed dQ/dV of the mid-life cycle of six cells (two per batch), with the "
         "per-cycle Pearson r")
SOURCES = ["dqdv_example_curves.csv", "features_expanded.parquet"]

CELLS = ["b1c1", "b1c30", "b2c1", "b2c30", "b3c0", "b3c20"]   # the six cells ENG-02 overlaid
Y_MAX = 7.0              # Ah/V, common to every panel (one published curve spikes at 2.0 V)
PUBLISHED = dict(color="0.35", lw=1.1, ls="-")
RECOMPUTED = dict(color="#D55E00", lw=0.9, ls=(0, (3, 1.6)))


def pearson_r() -> pd.Series:
    """rc_pearson_r of every (cell_id, cycle) row of the feature table."""
    feat = pd.read_parquet(result_path("features_expanded.parquet"),
                           columns=["cell_id", "cycle", "rc_pearson_r"])
    return feat.set_index(["cell_id", "cycle"])["rc_pearson_r"]


def panel(ax, c: pd.DataFrame, cid: str, r: float) -> None:
    """One cell: both curves (negated), the title, and a clip mark if needed."""
    k = int(c["cycle"].iloc[0])
    v, pub, rec = c["voltage"].to_numpy(), -c["published"].to_numpy(), -c["recomputed"].to_numpy()
    top = float(np.nanmax(np.r_[pub, rec]))
    ax.plot(v, pub, zorder=2, **PUBLISHED)
    ax.plot(v, rec, zorder=3, **RECOMPUTED)
    ax.set_title(f"{cid}, cycle {k} (r = {r:.3f})", fontsize=8, pad=3)
    ax.set_xlim(2.0, 3.5)
    ax.set_xticks([2.0, 2.5, 3.0, 3.5])
    ax.set_xticklabels(["2.0", "2.5", "3.0", "3.5"])
    ax.set_ylim(-0.8, Y_MAX)
    if top > Y_MAX:
        ax.annotate(f"clipped (rises to {top:.0f} Ah/V)", xy=(2.0, Y_MAX), xytext=(2.1, Y_MAX - 1.0),
                    fontsize=7, color="0.25", va="center",
                    arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "0.25", "shrinkA": 1, "shrinkB": 1})


def draw(curves: pd.DataFrame, r: pd.Series):
    """2 x 3 panels in CELLS order, shared axes, legend below."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 3.5), sharex=True, sharey=True)
    for ax, cid in zip(axes.ravel(), CELLS):
        c = curves[curves["cell_id"] == cid]
        panel(ax, c, cid, float(r.loc[(cid, int(c["cycle"].iloc[0]))]))
    for ax in axes[1]:
        ax.set_xlabel("Voltage (V)")
    for ax in axes[:, 0]:
        ax.set_ylabel("−dQ/dV (Ah/V)")
    handles = [plt.Line2D([], [], label="published (Severson et al.)", **PUBLISHED),
               plt.Line2D([], [], label="recomputed from raw V and Q", **RECOMPUTED)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=7, handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=1.6)
    return fig


def make(stem: Path) -> list[Path]:
    apply_style()
    return save_figure(stem, draw(load_csv("dqdv_example_curves.csv"), pearson_r()))
