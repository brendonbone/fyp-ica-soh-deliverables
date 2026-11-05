"""
Table XI. The seven history components: grouped PI in the tuned 63-column RF, and dev-CV alone / without
Grouped permutation importance of each component (its nine columns shuffled together, ENG-21) next to
the dev-CV RMSE of the untuned RF on that component alone and on all 63 columns without it (ENG-22,
all 127 component subsets), and the batch-3 RMSE of the same two forests (ENG-22's listed designs).
Rows are ordered by descending grouped PI (recomputed, not stored). The 'without it' row of a
component is the ENG-22 subset of the other six, keyed in the stored component order
(src.subset_search.COMPONENTS).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import (COMPONENT_LABELS, fmt_pct, fmt_pm, fmt_rmse, load_json,
                                 write_table)
from src.features_temporal import LEVEL_COLS, TEMPORAL_COLS
from src.subset_search import COMPONENTS

LABEL = "Table XI"
TITLE = ("The seven history components: grouped permutation importance (PI) in the tuned "
         "63-column RF, and dev-CV and batch-3 RMSE of each component alone and of all 63 columns "
         "without it")
SOURCES = ["eng21_grouped_permutation.json", "eng22_history_components.json",
           "eng22_history_components_analysis.json"]


def by_key(table_json: dict) -> dict:
    """Subset-search table {'table': [rows]} -> {'+'.join(blocks): row}."""
    return {"+".join(r["blocks"]): r for r in table_json["table"]}


def batch3_of(a22: dict) -> dict[tuple, float]:
    """Batch-3 RMSE of each ENG-22 design scored with the untuned RF, by its components."""
    return {tuple(r["blocks"]): r["b3_rmse"] for r in a22["heldout"] if r["config"] == "untuned"}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table (formatted strings) and the CSV (one row per component + 'all')."""
    e21 = load_json(SOURCES[0])
    t22, a22 = by_key(load_json(SOURCES[1])), load_json(SOURCES[2])
    pi = e21["history_summary"]["by_component"]
    design = {d["component"]: d for d in a22["designs"]}
    kall = "+".join(COMPONENTS)
    full = t22[kall]["cv_pooled_rmse_mean"]
    b3 = batch3_of(a22)
    b3_full = b3[tuple(COMPONENTS)]
    # the 63 per-column PIs summed in the combined-pool order
    hist_pi = e21["models"]["history_tuned"]
    total_one_col = sum(hist_pi[c]["mean"] for c in list(LEVEL_COLS) + TEMPORAL_COLS)

    rows, csv_rows = [], []
    for k in sorted(COMPONENTS, key=lambda k: -pi[k]["grouped"]):
        d, p = design[k], pi[k]
        rest = "+".join(c for c in COMPONENTS if c != k)
        b3_alone, b3_minus = b3[(k,)], b3[tuple(c for c in COMPONENTS if c != k)]
        rows.append({"Component": COMPONENT_LABELS[k],
                     "Grouped PI (± fold sd)": fmt_pm(p["grouped"], p["grouped_fold_sd"]),
                     "Sum of one-column PIs": fmt_rmse(p["sum_of_columns"]),
                     "Alone: dev-CV": fmt_rmse(d["cv_alone"]),
                     "Alone vs all 63": fmt_pct(d["alone_vs_all_rel"], signed=True),
                     "Alone: batch 3": fmt_rmse(b3_alone),
                     "All 63 without it: dev-CV": fmt_rmse(d["cv_all_minus"]),
                     "Cost of dropping": fmt_pct(d["cost_of_dropping_rel"], signed=True),
                     "All 63 without it: batch 3": fmt_rmse(b3_minus),
                     "Batch-3 cost of dropping": fmt_pct(b3_minus / b3_full - 1, signed=True)})
        csv_rows.append({"component": k, "label": COMPONENT_LABELS[k],
                         "grouped_pi": p["grouped"], "grouped_pi_fold_sd": p["grouped_fold_sd"],
                         "sum_one_column_pi": p["sum_of_columns"],
                         "grouped_over_sum": p["grouped"] / p["sum_of_columns"],
                         "cv_alone": d["cv_alone"], "cv_alone_fold_sd": t22[k]["cv_pooled_rmse_std"],
                         "alone_vs_all63_rel": d["alone_vs_all_rel"],
                         "cv_all_minus": d["cv_all_minus"],
                         "cv_all_minus_fold_sd": t22[rest]["cv_pooled_rmse_std"],
                         "cost_of_dropping_rel": d["cost_of_dropping_rel"],
                         "shapley": d["shapley"],
                         "b3_alone": b3_alone, "b3_all_minus": b3_minus,
                         "b3_cost_of_dropping_rel": b3_minus / b3_full - 1})
    # Reference row: the dev-CV of all 63 goes in the label, not under 'Alone'.
    rows.append({"Component": f"All seven together (63 columns): dev-CV {fmt_rmse(full)}, "
                              f"batch 3 {fmt_rmse(b3_full)}",
                 "Grouped PI (± fold sd)": "", "Sum of one-column PIs": fmt_rmse(total_one_col),
                 "Alone: dev-CV": "", "Alone vs all 63": "", "Alone: batch 3": "",
                 "All 63 without it: dev-CV": "", "Cost of dropping": "",
                 "All 63 without it: batch 3": "", "Batch-3 cost of dropping": ""})
    # this row adds the two cv_all63 columns at the end of the CSV
    csv_rows.append({"component": "all", "label": "All seven together (63 columns)",
                     "sum_one_column_pi": total_one_col, "cv_all63": full,
                     "cv_all63_fold_sd": t22[kall]["cv_pooled_rmse_std"],
                     "shapley": sum(a22["shapley"].values()), "b3_all63": b3_full})
    return pd.DataFrame(rows), pd.DataFrame(csv_rows)


def notes(full: str) -> list[str]:
    """How the PI and the two dev-CV columns are defined."""
    return [
        "Grouped PI: rise in validation RMSE when the component's nine columns (one per feature) "
        "are shuffled together; tuned RF on all 63 columns, ten repeats, mean over the five "
        "development folds ± sd over folds (ddof 0). Sum of one-column PIs: the same nine columns "
        "shuffled one at a time; the last row sums all 63.",
        "Alone / all 63 without it: dev-CV of the untuned RF on the component's nine columns / on "
        f"the other 54; percentages relative to all 63 together (dev-CV {full}, the untuned "
        "63-column model of Table X). PI and RMSE in SoH units (fraction of nominal capacity).",
        "Batch 3: the same untuned forests refitted on all development rows and scored once on the "
        "batch-3 history rows; the batch-3 cost of dropping is relative to all 63 on batch 3. Nothing "
        "was chosen on these scores.",
    ]


def make(stem: Path) -> list[Path]:
    md, csv = build()
    full = csv.loc[csv["component"] == "all", "cv_all63"].iloc[0]
    return write_table(stem, md, label=LABEL, title=TITLE, csv_df=csv, notes=notes(fmt_rmse(full)),
                       sources=SOURCES)
