"""
Fig. 6. Correlation between the nine features on the development rows
Pearson r above the diagonal, Spearman ρ below. Rows follow Table V's order
with the near-duplicate block (the three areas and height, whose six pairs have
|r| > 0.85) moved to the front so its Pearson half can be outlined in one piece.
The matrices are read from eng04_feature_pool.json (correlation_matrix_full,
spearman_matrix) when stage 4 stored them; otherwise they are computed from the
development rows of features_expanded.parquet exactly as stage 4 does.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle

from report_items import tab_feature_soh_association as table_vi
from report_items.common import COL_WIDTH, FEATURES, MINUS, apply_style, load_json, save_figure

LABEL = "Fig. 6"
# make() completes the caption with the Spearman range of prominence against the block;
# run_report.py reads TITLE after make().
TITLE = ("Correlation between the nine features on the development rows (Pearson r above the "
         "diagonal, Spearman ρ below)")
TITLE_FULL = (TITLE + ". Outlined: the six pairs with |r| > {high}, the near-duplicate block of the "
              "three areas and height; on ranks prominence joins it (ρ = {lo:.2f}–{hi:.2f} with "
              "each of the four)")
SOURCES = ["eng04_feature_pool.json", "features_expanded.parquet", "eng16_pearson_vs_spearman.csv",
           "eng08_full.json"]

BLOCK = ["area", "area35", "area25", "height"]   # every pair has |r| > HIGH_CORR
HIGH_CORR = 0.85                                 # stage 4's threshold for high_correlation_pairs


def matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson and Spearman matrices of the nine features on the development rows."""
    e04 = load_json("eng04_feature_pool.json")
    if "correlation_matrix_full" in e04 and "spearman_matrix" in e04:
        return pd.DataFrame(e04["correlation_matrix_full"]), pd.DataFrame(e04["spearman_matrix"])
    dev = table_vi.dev_rows()   # eng04 written before the matrices were stored
    return dev[FEATURES].corr(), dev[FEATURES].corr(method="spearman")


def row_order() -> list[str]:
    """Table V's order (by |r|) with the block moved to the front, keeping relative order."""
    t_order = table_vi.build()[0]["feature"].tolist()
    return [f for f in t_order if f in BLOCK] + [f for f in t_order if f not in BLOCK]


def _ink(rgba) -> str:
    """Black or white text, whichever contrasts more with the cell colour (WCAG)."""
    c = np.asarray(rgba[:3], dtype=float)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lum = float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
    return "black" if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else "white"


def _cell(v: float) -> str:
    """Two decimals, no leading zero, typographic minus; a value that rounds to 0 gets no sign."""
    s = f"{abs(v):.2f}"
    if s == "0.00":
        return ".00"
    return (MINUS if v < 0 else "") + (s[1:] if s.startswith("0") else s)


def draw(P: pd.DataFrame, Sp: pd.DataFrame, order: list[str]):
    """The half-Pearson, half-Spearman matrix with the block outlined and a colour bar below."""
    n = len(order)
    P, Sp = P.loc[order, order].values, Sp.loc[order, order].values
    grid = np.full((n, n), np.nan)
    iu, il = np.triu_indices(n, 1), np.tril_indices(n, -1)
    grid[iu], grid[il] = P[iu], Sp[il]

    cmap, norm = plt.get_cmap("RdBu_r"), Normalize(-1, 1)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.45))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="0.88", edgecolor="none"))
                continue
            ax.text(j, i, _cell(grid[i, j]), ha="center", va="center", fontsize=7,
                    color=_ink(cmap(norm(grid[i, j]))))
    for x in np.arange(0.5, n - 1, 1.0):      # white cell separators, inner edges only
        ax.axhline(x, color="white", lw=0.6)
        ax.axvline(x, color="white", lw=0.6)
    # Outline the block's Pearson half only (upper triangle): the block is defined on r, and on
    # ranks prominence is as close to it (see the caption), so the outline makes no claim about rho.
    k = len(BLOCK)
    stair = [(0.5, -0.5), (k - 0.5, -0.5), (k - 0.5, k - 1.5)]
    for i in range(k - 2, -1, -1):
        stair += [(i + 0.5, i + 0.5), (i + 0.5, i - 0.5)]
    ax.add_patch(Polygon(stair[:-1], closed=True, fill=False, edgecolor="black", lw=1.4, zorder=5,
                         joinstyle="miter", clip_on=False))

    ax.set_xticks(range(n), order, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), order)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-.5, n - .5)
    ax.set_ylim(n - .5, -.5)

    # Colour bar as wide as the matrix, below the rotated column labels.
    cax = ax.inset_axes([0.0, -0.215, 1.0, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels([f"{MINUS}1", f"{MINUS}0.5", "0", "0.5", "1"])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5, labelsize=7)
    cb.set_label("Correlation: Pearson r above the diagonal, Spearman ρ below", fontsize=7.5)
    return fig


def make(stem: Path) -> list[Path]:
    global TITLE
    P, Sp = matrices()
    prom = [Sp.loc["prominence", b] for b in BLOCK]
    TITLE = TITLE_FULL.format(high=HIGH_CORR, lo=min(prom), hi=max(prom))
    apply_style()
    return save_figure(stem, draw(P, Sp, row_order()))
