"""
Fig. S1. Backward elimination on six train/test partitions
==========================================================
Two grids of the ENG-25 backward-elimination order, (a) the nine level features and (b) each
feature with its six history columns removed as one block, in the six partitions of Table S1
(leave one batch out B1-B3, B3 = the study split; batch-mixed thirds M1-M3). Cell = size of the
subset the feature is removed from (9 = removed first, 1 = last standing). Shaded = in the
partition's five-feature set: five is the smallest size within 2% of the full set's dev-CV in every
partition and both panels (the palette and the caption are fixed at five; the module verifies that
claim from the steps and refuses to draw if it no longer holds). Orange outline = in the partition's set
of the size D picks (four, level) or F picks (two, history), read from the two one-SE picks. Right
column: in how many partitions the feature is in the five-set; bottom row: dev-CV change of the
five-set against the full set (exact ratios). Reads eng25_partition_robustness.json,
eng19b_analysis.json and eng23_grouped_history_subsets_analysis.json. The figure width is iterated
until the saved tight bounding box is exactly FULL_WIDTH, so 7 pt text prints at 7 pt.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from report_items.common import FEATURES, FULL_WIDTH, PARTITIONS, apply_style, fmt_pct, load_json, save_figure

LABEL = "Fig. S1"
TITLE = ("Backward elimination on six train/test partitions: (a) the nine level features, (b) each feature "
         "with its six history columns removed as one block. Columns: leave one batch out (B1–B3; B3, outlined "
         "in black, is the study split) and the three batch-mixed thirds (M1–M3). Cell = size of the subset "
         "from which the feature is removed (9 = removed first, 1 = last standing). Shaded = in that "
         "partition's five-feature set; five is the smallest size within 2% of the full set's dev-CV RMSE in "
         "every partition and both panels. Orange outline = in that partition's four-feature set in (a) and "
         "two-feature set in (b), the sizes that D and F pick (in B3 these are D's and F's picks). Right "
         "column: in how many of the six partitions the feature is in the five-feature set. Bottom row: "
         "change in dev-CV RMSE of the five-feature set against the full set in that partition. Untuned RF, "
         "inner five-fold CV on each partition's development cells (ENG-25)")
E25 = "eng25_partition_robustness.json"
A19 = "eng19b_analysis.json"
A23 = "eng23_grouped_history_subsets_analysis.json"
SOURCES = [E25, A19, A23]

SUFFICE = 0.02          # a subset size suffices if within 2% of the full set's dev-CV in every partition
SAVE_DPI = 600
COL_LABELS = {"lobo_b1": "B1", "lobo_b2": "B2", "lobo_b3": "B3", "cv3_f0": "M1", "cv3_f1": "M2", "cv3_f2": "M3"}
K_SHOW = 5              # the shaded set size; the palette below and TITLE are written for this value
SHADE = [plt.cm.Blues(x) for x in (0.92, 0.80, 0.68, 0.55, 0.42)]   # rank 1 (darkest) .. K_SHOW
OUT_FILL = "#f0f0f0"
PICK_EDGE = "#D55E00"   # Okabe-Ito vermillion: the outline of the D / F pick size


# ------------------------ the elimination paths ------------------------

def steps_by_partition(e25: dict, mode: str) -> dict[str, list[dict]]:
    """partition -> the BE steps of that mode ('level' | 'grouped'), k = 9 first, k = 1 last."""
    return {p: e25["backward_elimination"][mode][p]["steps"] for p in PARTITIONS}


def removal_rank(steps: list[dict]) -> dict[str, int]:
    """feature -> size of the subset it was removed from (1 = last standing, 9 = removed first)."""
    ranking = [steps[-1]["blocks"][0]] + [s["dropped"] for s in reversed(steps[1:])]
    return {f: i + 1 for i, f in enumerate(ranking)}


def at_k(steps: list[dict], k: int) -> dict:
    """The one step of size k."""
    hits = [s for s in steps if s["k"] == k]
    if len(hits) != 1:
        raise KeyError(f"k = {k}: {len(hits)} steps")
    return hits[0]


def change(steps: list[dict], k: int) -> float:
    """Relative dev-CV change of the size-k subset against the full set (steps[0])."""
    full = steps[0]["cv"]
    return (at_k(steps, k)["cv"] - full) / full


def k_suffice(be: dict[str, dict[str, list[dict]]]) -> int:
    """The smallest size within SUFFICE of the full set in every partition of every mode."""
    return min(k for k in range(1, len(FEATURES) + 1)
               if all(change(steps, k) <= SUFFICE for mode in be.values() for steps in mode.values()))


# ------------------------ drawing ------------------------

def draw_grid(ax, be: dict[str, list[dict]], title: str, ref_label: str, k_pick: int) -> None:
    """One panel: the rank grid, the B3 outline, the in-K_SHOW-set count and the bottom change row."""
    n_r, n_c = len(FEATURES), len(PARTITIONS)
    rank = {p: removal_rank(be[p]) for p in PARTITIONS}
    for j, p in enumerate(PARTITIONS):
        for i, f in enumerate(FEATURES):
            r = rank[p][f]
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=SHADE[r - 1] if r <= K_SHOW else OUT_FILL,
                                   edgecolor="white", lw=1.2))
            if r <= k_pick:   # in the partition's k_pick-feature set, the size D (k = 4) or F (k = 2) picks
                ax.add_patch(Rectangle((j + 0.07, i + 0.07), 0.86, 0.86, fill=False, edgecolor=PICK_EDGE,
                                       lw=1.1, zorder=2))
            ax.text(j + 0.5, i + 0.52, str(r), ha="center", va="center", fontsize=7,
                    color="white" if r <= 3 else ("black" if r <= K_SHOW else "0.45"),
                    fontweight="bold" if r == 1 else "normal")
    js = PARTITIONS.index("lobo_b3")
    ax.add_patch(Rectangle((js, 0), 1, n_r, fill=False, edgecolor="black", lw=1.0, zorder=3))
    for i, f in enumerate(FEATURES):
        ax.text(-0.12, i + 0.52, f, ha="right", va="center", fontsize=7)
    for j, p in enumerate(PARTITIONS):
        ax.text(j + 0.5, -0.15, COL_LABELS[p], ha="center", va="bottom", fontsize=7)
    for x0, x1, lab in ((0, 3, "batch held out"), (3, 6, "mixed third held out")):
        ax.plot([x0 + 0.08, x1 - 0.08], [-0.95, -0.95], color="0.3", lw=0.6, clip_on=False)
        ax.text((x0 + x1) / 2, -1.05, lab, ha="center", va="bottom", fontsize=7)
    # how often each feature is in the K_SHOW-feature set
    ax.text(n_c + 0.55, -0.15, f"in {K_SHOW}-set", ha="center", va="bottom", fontsize=7)
    for i, f in enumerate(FEATURES):
        n = sum(f in at_k(be[p], K_SHOW)["blocks"] for p in PARTITIONS)
        ax.text(n_c + 0.55, i + 0.52, f"{n}/{n_c}", ha="center", va="center", fontsize=7,
                fontweight="bold" if n == n_c else "normal")
    # dev-CV of the K_SHOW-feature set against the full set, per partition
    ax.text(-0.12, n_r + 0.6, f"ΔRMSE, k = {K_SHOW}\nvs {ref_label}", ha="right", va="center", fontsize=7,
            linespacing=1.0)
    for j, p in enumerate(PARTITIONS):
        ax.text(j + 0.5, n_r + 0.6, fmt_pct(change(be[p], K_SHOW), 1, signed=True), ha="center", va="center",
                fontsize=7)
    ax.text(n_c / 2, -2.0, title, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, n_c + 1.1)
    ax.set_ylim(n_r + 1.15, -2.05)
    ax.axis("off")


def fit_to_width(fig, width: float = FULL_WIDTH, dpi: int = SAVE_DPI) -> float:
    """Resize the figure so that the tight bounding box save_figure writes (bbox_inches='tight', plus
    the pad on each side) is `width` inches wide. The grids are in data coordinates and scale with the
    figure; the text does not, so a few fixed-point steps converge. Returns the final saved width (in)."""
    pad = float(plt.rcParams["savefig.pad_inches"])
    h = fig.get_size_inches()[1]
    fig.set_dpi(dpi)
    for _ in range(12):
        fig.canvas.draw()
        saved = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad
        if abs(saved - width) < 1e-5:
            break
        fig.set_size_inches(fig.get_size_inches()[0] + (width - saved), h)
    fig.canvas.draw()
    return float(fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad)


def draw(be: dict[str, dict[str, list[dict]]], k_level: int, k_hist: int):
    """The two panels at FULL_WIDTH."""
    # the palette, the right column, the bottom row and TITLE all say "five": refuse to draw a figure
    # whose caption the steps no longer support (OLD checked the same thing)
    if k_suffice(be) != K_SHOW:
        raise ValueError(f"the smallest size within {100 * SUFFICE:.0f}% of the full set is {k_suffice(be)}, "
                         f"not {K_SHOW}: the shading and the caption of Fig. S1 assume {K_SHOW}")
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    draw_grid(axes[0], be["level"], "(a) Level features", "all nine", k_level)
    draw_grid(axes[1], be["grouped"], "(b) Features with their 20-cycle history", "all 63", k_hist)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.99, bottom=0.01, wspace=0.42)
    fit_to_width(fig)
    return fig


def make(stem: Path) -> list[Path]:
    e25 = load_json(E25)
    be = {mode: steps_by_partition(e25, mode) for mode in ("level", "grouped")}
    k_level = len(load_json(A19)["selection_rules"]["one_standard_error"]["pick"]["blocks"])
    k_hist = len(load_json(A23)["selection_rules"]["one_standard_error"]["pick"]["blocks"])
    return save_figure(stem, draw(be, k_level, k_hist), dpi=SAVE_DPI)
