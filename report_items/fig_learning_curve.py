"""
Fig. C1. Learning curve: dev-CV R² of the untuned RF on the nine features against the number of training cells
Mean ± sd over the five validation folds × five repeats for each number of training cells
(results/eng10_learning_curve.json, stage 20), with the plateau as a dashed line. The top grid
point appears twice (66 and 67 cells) because two of the five folds have only 66 other-fold cells.
The same curve as the original ENG-10 figure, without its in-figure title.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from report_items.common import COL_WIDTH, apply_style, load_json, save_figure

LABEL = "Fig. C1"
TITLE = ("Learning curve: dev-CV pooled R² of the untuned RF on the nine level features against the "
         "number of training cells (mean ± sd over 5 validation folds × 5 repeats; the top grid point "
         "is 66 cells for the two folds with only 66 other-fold cells, 10 fits, and 67 for the other "
         "three, 15 fits); the dashed line marks the plateau, the fewest cells whose mean R² is within "
         "one sd of the largest")
SOURCES = ["eng10_learning_curve.json"]


def make(stem: Path) -> list[Path]:
    d = load_json("eng10_learning_curve.json")
    agg = pd.DataFrame(d["aggregate"])
    plateau_n = d["plateau_n_cells"]

    apply_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    ax.errorbar(agg["n_cells"], agg["r2_mean"], yerr=agg["r2_std"],
                marker="o", capsize=2, color="C0")
    ax.axvline(plateau_n, color="0.4", lw=0.8, ls="--")
    ax.text(plateau_n + 1, ax.get_ylim()[0] + 0.01, f"plateau n={plateau_n}",
            fontsize=7, color="0.3")
    ax.set_xlabel("training cells")
    ax.set_ylabel("dev-CV pooled $R^2$")
    return save_figure(stem, fig)
