"""
Fig. 7. Median SoC window of each feature in fresh and aged cycles
==================================================================
The rows of Table VI drawn as bars on the SoC axis (100 % on the left: the
discharge runs left to right), once for fresh (SoH >= 0.99) and once for aged
(SoH < 0.90) cycles, with the centre of the apex detection window of each stage
and the fresh / aged spans beside each row. Reads eng17_soc_windows_summary.csv;
the cycle counts in the legend are the clean rows of features_expanded.parquet
in each SoH band.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from report_items.common import COL_WIDTH, apply_style, load_parquet, save_figure
from report_items.tab_feature_soc_windows import (AGED, FRESH, SUMMARY, VARIANT_LABEL, load_summary, pct,
                                                  row, table_rows)
from src.data_prep import clean_feature_table

LABEL = "Fig. 7"
TITLE = ("Median SoC window of the discharge that each feature needs, in fresh (SoH ≥ 0.99) and aged "
         "(SoH < 0.90) cycles. The discharge runs from 100% to 0% SoC (left to right). Dashed and dotted "
         "lines: centre of the apex detection window (approximately the apex). Right: span of each "
         f"window, fresh / aged. Last row, '{VARIANT_LABEL}': fwhm measured only between its "
         "half-prominence crossings, roughly what a fixed-baseline fwhm would need (not used by any model).")
PARQUET = "features_expanded.parquet"
SOURCES = [SUMMARY, PARQUET]

VARIANT_TICK = "fwhm, crossings\nonly (variant)"
# light solid blue for fresh, dark vermillion hatched for aged (Okabe-Ito; the two
# differ in lightness and texture, so they survive greyscale)
STYLE = {FRESH: dict(label="fresh, SoH ≥ 0.99", color="#56B4E9", hatch=None, off=-0.19, apex_ls="--"),
         AGED: dict(label="aged, SoH < 0.90", color="#D55E00", hatch="//////", off=0.19, apex_ls=":")}
FIG_SIZE = (COL_WIDTH, 3.0)     # the whole saved figure fits one IEEE column
FS = 7                          # annotation and legend text (the style's tick size)


def stage_counts() -> dict[str, int]:
    """Clean cycles in the fresh and aged SoH bands (the legend counts)."""
    clean, _ = clean_feature_table(load_parquet(PARQUET), verbose=False)
    soh = clean["SOH"]
    return {FRESH: int((soh >= 0.99).sum()), AGED: int((soh < 0.90).sum())}


def make(stem: Path) -> list[Path]:
    summ = load_summary()
    counts = stage_counts()
    apply_style()
    plot_rows = [(VARIANT_TICK if name == VARIANT_LABEL else name, f, basis)
                 for name, f, basis in table_rows(summ)]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # axes placed so the tick labels, the span column and the legend all fit inside the canvas
    fig.subplots_adjust(left=0.265, right=0.815, bottom=0.235, top=0.875)
    for i, (_, f, basis) in enumerate(plot_rows):
        for st, sty in STYLE.items():
            r = row(summ, f, st, basis)
            s, e = 100 * r.soc_start_median, 100 * r.soc_end_median
            ax.barh(i + sty["off"], s - e, left=e, height=0.36, color=sty["color"], hatch=sty["hatch"],
                    edgecolor="0.15" if sty["hatch"] is None else "white", lw=0.4 if sty["hatch"] is None else 0,
                    zorder=2)
            if sty["hatch"]:                                  # outline drawn separately (hatch uses edgecolor)
                ax.barh(i + sty["off"], s - e, left=e, height=0.36, fill=False, edgecolor="0.15", lw=0.4, zorder=3)
    # apex: the centre of the apex detection window of each stage
    for st, sty in STYLE.items():
        h = row(summ, "height", st)
        ax.axvline(50 * (h.soc_start_median + h.soc_end_median), color="0.1", lw=0.7, ls=sty["apex_ls"], zorder=4)
    ax.axhline(len(plot_rows) - 1.5, color="0.5", lw=0.5)    # the variant below the line
    ax.set_yticks(range(len(plot_rows)), [r[0] for r in plot_rows])
    ax.tick_params(axis="y", length=2, pad=2)
    ax.set_ylim(len(plot_rows) - 0.5, -0.5)
    ax.set_xlim(100, 0)                                       # the discharge runs from 100 % to 0 % SoC
    ax.set_xticks(range(100, -1, -20))
    ax.set_xlabel("State of charge (%)", labelpad=2)
    ax.grid(axis="x", lw=0.3, alpha=0.5, zorder=0)
    ax.annotate("", xy=(0.99, 1.04), xytext=(0.01, 1.04), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.2", mutation_scale=6))
    ax.text(0.5, 1.055, "discharge", transform=ax.transAxes, ha="center", va="bottom", fontsize=FS,
            color="0.2", bbox=dict(fc="white", ec="none", pad=0.6))
    # spans, fresh / aged, beside each row
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.04, -0.62, "span (%)\nfresh / aged", transform=tr, ha="left", va="bottom", fontsize=FS,
            linespacing=1.1)
    for i, (_, f, basis) in enumerate(plot_rows):
        ax.text(1.04, i, f"{pct(row(summ, f, FRESH, basis).soc_span_median)} / "
                         f"{pct(row(summ, f, AGED, basis).soc_span_median)}",
                transform=tr, ha="left", va="center", fontsize=FS)
    # the aged key is a white-hatched patch under a separate outline, as on the bars
    handles = [Patch(facecolor=STYLE[FRESH]["color"], edgecolor="0.15", lw=0.4),
               (Patch(facecolor=STYLE[AGED]["color"], edgecolor="white", lw=0, hatch=STYLE[AGED]["hatch"]),
                Patch(fill=False, edgecolor="0.15", lw=0.4)),
               Line2D([], [], color="0.1", lw=0.7, ls=STYLE[FRESH]["apex_ls"]),
               Line2D([], [], color="0.1", lw=0.7, ls=STYLE[AGED]["apex_ls"])]
    names = [f"{STYLE[FRESH]['label']} ({counts[FRESH]:,} cycles)",
             f"{STYLE[AGED]['label']} ({counts[AGED]:,} cycles)",
             "apex (approx.), fresh", "apex (approx.), aged"]
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=FS, handlelength=1.6, handletextpad=0.5, columnspacing=0.9, borderaxespad=0.1)
    return save_figure(stem, fig)
