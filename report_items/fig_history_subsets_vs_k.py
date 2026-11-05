"""
Fig. 9. Dev-CV RMSE of the 511 subsets with history against the number of features
===================================================================================
The pair of Fig. 8 for the ENG-23 search (each feature kept or removed with its
six history columns): every subset (jittered), the best subset of each size,
backward elimination (BE) and the other branch of its near-tie, the Pearson and
PI forward paths, the all-nine-with-history line and the best level-only model
on the same rows (ENG-24). The other BE branch: the BE paths that at each step
may take any drop within the forest-seed sd of the all-63 dev-CV (ENG-22 seed
check) of the best drop; the first that ends with area is drawn from k = 4 down
(it drops height instead of position at k = 5).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from report_items.common import (COL_WIDTH, FEATURES, OFF_ARROW, PATH_STYLE, TEXT_PT, Y_CAP, apply_style, load_json,
                                 off_scale_labels, save_figure)
from report_items.tab_best_subsets_level_vs_history import HIST, LVL, search_table
from src.data_prep import RF_SEED
from src.subset_search import subset_key

LABEL = "Fig. 9"
TITLE = ("Dev-CV RMSE of all 511 subsets of the nine features with history (each feature with its six history "
         "columns) against the number of features, with the best subset of each size, the backward-elimination "
         "path and the other branch of its near-tie, the Pearson and tuned-RF PI forward paths, the "
         "all-nine-with-history reference and the best level-only model on the same rows (untuned RF)")
SOURCES = [f"{HIST}.json", f"{HIST}_analysis.json", f"{LVL}.json", "eng22_history_components_seed_check.json"]

C_BEST_LEVEL = "#CC79A7"   # Okabe-Ito reddish purple, distinct from every path colour
ALT_DX = 0.22              # horizontal offset of the other BE branch
DODGE = 0.15               # offset of a ranking-path marker that another path's marker would hide


def be_branches(cv: dict, names: list[str], tol: float) -> list[list[dict]]:
    """Every backward-elimination path that takes, at each step, any drop within
    `tol` of the best drop (the stored path always takes the best one)."""
    out = []

    def walk(cur, path):
        if len(cur) == 1:
            out.append(path)
            return
        cands = sorted((cv[subset_key([c for c in cur if c != d])], names.index(d), d) for d in cur)
        for v, _, d in cands:
            if v - cands[0][0] <= tol:
                nxt = [c for c in cur if c != d]
                walk(nxt, path + [{"k": len(nxt), "dropped": d, "blocks": nxt, "cv": float(v)}])

    walk(list(names), [])
    return out


def other_be_branch(t: pd.DataFrame) -> list[dict]:
    """k = 4..1 of the first near-tie BE branch that ends with area last standing."""
    seed_sd = float(load_json("eng22_history_components_seed_check.json")["summary"]["all 63"]["cv_sd"])
    branches = be_branches(dict(zip(t["key"], t["cv_pooled_rmse_mean"])), list(FEATURES), seed_sd)
    area_end = [b for b in branches if b[-1]["blocks"][0] == "area"][0]
    return [s for s in area_end if s["k"] <= 4]


def draw(t: pd.DataFrame, A: dict, best_level: dict, be_alt: list[dict]):
    """Fig. 9 in Fig. 8's styles (y cap, off-scale marks, path styles)."""
    apply_style()
    ks = list(range(1, len(FEATURES) + 1))
    paths = {"best": A["best_per_k"], "be": A["backward_elimination"],
             "pearson": A["ranking_paths"]["Pearson"], "pi": A["ranking_paths"]["PI (tuned RF)"]}
    at = {name: {r["k"]: r for r in p} for name, p in paths.items()}
    alt = [at["be"][be_alt[0]["k"] + 1]] + be_alt   # from the shared k = 5 subset down
    # y capped Y_CAP x the highest drawn line; poorer subsets marked on the top edge.
    ytop = Y_CAP * max(r["cv"] for p in list(paths.values()) + [alt] for r in p)
    lo = float(t["cv_pooled_rmse_mean"].min())

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.3))
    y = t["cv_pooled_rmse_mean"].to_numpy()
    kk = t["k"].to_numpy()
    off = y > ytop
    # The jitter is drawn in the table's row order (dev-CV, then key).
    x = kk + np.random.default_rng(RF_SEED).uniform(-0.18, 0.18, len(t))
    ax.scatter(x[~off], y[~off], zorder=1, **PATH_STYLE["subsets"])
    for kv, label in off_scale_labels(kk, y, ytop):
        ax.annotate("", xy=(kv, ytop), xytext=(kv, ytop - 0.07 * (ytop - lo)), annotation_clip=False,
                    arrowprops=OFF_ARROW)
        ax.text(kv + 0.12, ytop - 0.012 * (ytop - lo), label, ha="left", va="top", fontsize=TEXT_PT, color="0.3")

    # A ranking-path marker on the same subset as another path's marker of that size
    # is moved DODGE left (Pearson) or right (PI) and drawn above the hollow BE square.
    def xs(name: str) -> list[float]:
        sign = {"pearson": -1, "pi": 1}.get(name, 0)
        out = []
        for k in ks:
            here = subset_key(at[name][k]["blocks"])
            hidden = any(subset_key(at[o][k]["blocks"]) == here for o in paths if o != name)
            out.append(k + sign * DODGE if sign and hidden else float(k))
        return out

    x_path = {name: xs(name) for name in paths}
    for name in ("best", "be"):
        ax.plot(x_path[name], [at[name][k]["cv"] for k in ks], zorder={"best": 5, "be": 4}[name], **PATH_STYLE[name])
    ax.plot([alt[0]["k"]] + [r["k"] + ALT_DX for r in be_alt], [r["cv"] for r in alt],
            color=PATH_STYLE["be"]["color"], ls=(0, (1, 1.3)), lw=0.8, marker="s", ms=2.8, mfc="white", mew=0.6,
            zorder=3.5, label="BE, other branch (near-tie)")
    for name in ("pearson", "pi"):
        ax.plot(x_path[name], [at[name][k]["cv"] for k in ks], zorder=3, **PATH_STYLE[name])
        moved = [(xv, at[name][k]["cv"]) for k, xv in zip(ks, x_path[name]) if xv != k]
        if moved:
            ax.plot(*zip(*moved), ls="none", zorder=4.5,
                    **{a: v for a, v in PATH_STYLE[name].items() if a not in ("ls", "lw", "label")})
    ax.axhline(A["all_blocks"]["cv"], zorder=0, label="all nine with history", **PATH_STYLE["ref"])
    ax.axhline(best_level["cv"], color=C_BEST_LEVEL, lw=1.3, ls=(0, (5, 2.5)), zorder=0,
               label=f"best without history ({best_level['k']} features)")

    ax.set_xlim(0.5, len(FEATURES) + 0.5)
    ax.set_ylim(lo - 0.04 * (ytop - lo), ytop)
    ax.set_xticks(ks)
    ax.set_xlabel("Number of features retained (each with its six history columns), k")
    ax.set_ylabel("Dev-CV RMSE (SoH, p.u.)")

    dots = Line2D([], [], ls="none", marker="o", ms=2.2, mfc=PATH_STYLE["subsets"]["color"], mec="none",
                  label=f"all {len(t)} subsets")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(handles=[dots] + h, labels=[dots.get_label()] + lab, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=TEXT_PT, handlelength=2.6,
              columnspacing=1.0, handletextpad=0.5, labelspacing=0.35)
    fig.tight_layout()
    return fig


def make(stem: Path) -> list[Path]:
    t_h = search_table(HIST)
    best_level = search_table(LVL).iloc[0]
    fig = draw(t_h, load_json(f"{HIST}_analysis.json"), {"cv": float(best_level["cv_pooled_rmse_mean"]),
                                                         "k": int(best_level["k"])}, other_be_branch(t_h))
    return save_figure(stem, fig)
