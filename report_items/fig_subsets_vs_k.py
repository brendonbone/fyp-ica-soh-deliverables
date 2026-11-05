"""
Fig. 8. Dev-CV RMSE of all 511 subsets of the nine level features against the number of features
Every subset as a grey dot (jittered in k with a seeded draw over all 511 rows
in (k, key) order, so the dots sit where they did in the thesis), with the
best subset of each size, the BE, Pearson and PI paths of Table IX and the
untuned RF on all nine features. Reads eng19_exhaustive_subsets.csv and the
ENG-08 rankings. The y-axis is capped 10 % above the highest drawn line;
the subsets above it are counted on the top edge.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from report_items.common import (COL_WIDTH, FEATURES, OFF_ARROW, PATH_STYLE, TEXT_PT, Y_CAP, apply_style,
                                 off_scale_labels, save_figure)
from report_items.tab_best_subsets_paths import best_per_k, key, paths, subsets
from src.data_prep import RF_SEED

LABEL = "Fig. 8"
TITLE = ("Dev-CV RMSE of all 511 subsets of the nine level features against the number of features, "
         "with the best subset of each size, the backward-elimination path, the Pearson and tuned-RF PI "
         "forward paths and the untuned RF on all nine features (every subset scored with the untuned RF)")
SOURCES = ["eng19_exhaustive_subsets.csv", "eng08_full.json", "eng08_full_tuned.json"]


def draw(t: pd.DataFrame, best: dict, path: dict):
    """The single panel: dots, off-scale counts, the four paths and the all-nine line."""
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.3))
    ks = list(range(1, len(FEATURES) + 1))
    cv = dict(zip(t["key"], t["cv"]))
    lines = {"best": [best[k]["cv"] for k in ks],
             **{tag: [cv[path[tag][k][0]] for k in ks] for tag in ("be", "pearson", "pi")}}
    ytop = Y_CAP * max(max(v) for v in lines.values())
    lo = float(t["cv"].min())

    # every subset as a grey dot; one draw for all rows before masking the off-scale ones
    y = t["cv"].to_numpy()
    kk = t["k"].to_numpy()
    off = y > ytop
    rng = np.random.default_rng(RF_SEED)
    x = kk + rng.uniform(-0.18, 0.18, len(t))
    st = PATH_STYLE["subsets"]
    ax.scatter(x[~off], y[~off], zorder=1, **st)
    for kv, label in off_scale_labels(kk, y, ytop):
        ax.annotate("", xy=(kv, ytop), xytext=(kv, ytop - 0.07 * (ytop - lo)), annotation_clip=False,
                    arrowprops=OFF_ARROW)
        ax.text(kv + 0.12, ytop - 0.012 * (ytop - lo), label, ha="left", va="top", fontsize=TEXT_PT,
                color="0.3")

    for name, z in (("best", 5), ("be", 4), ("pearson", 3), ("pi", 3)):
        ax.plot(ks, lines[name], zorder=z, **PATH_STYLE[name])
    ax.axhline(cv[key(FEATURES)], zorder=0, label="all nine features", **PATH_STYLE["ref"])

    ax.set_xlim(0.5, len(FEATURES) + 0.5)
    ax.set_ylim(lo - 0.04 * (ytop - lo), ytop)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of features retained, k")
    ax.set_ylabel("Dev-CV RMSE (SoH, p.u.)")

    dots = Line2D([], [], ls="none", marker="o", ms=2.2, mfc=st["color"], mec="none",
                  label=f"all {len(t)} subsets")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(handles=[dots] + h, labels=[dots.get_label()] + lab, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=TEXT_PT, handlelength=2.6,
              columnspacing=1.0, handletextpad=0.5, labelspacing=0.35)
    fig.tight_layout()
    return fig


def make(stem: Path) -> list[Path]:
    apply_style()
    t = subsets()
    fig = draw(t, best_per_k(t), paths(t))
    return save_figure(stem, fig)
