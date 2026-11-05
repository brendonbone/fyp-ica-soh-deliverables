"""
Table S5. Held-out RMSE of the untuned random forest in the six train/test partitions
Per partition (ENG-15): the Pearson x untuned-RF model on the nine level features (all rows and
history rows) and on the ENG-14 combined pool (level + history, history rows), scored on the
partition's test cells. Change with history = (combined - level) / level, both on the history rows.
The two mean rows average the three partitions of each kind; their change is the mean of the three
changes, not the change of the means. The CSV adds the temporal (history-only) pool.
Reads eng15_cells.json and eng15_split_robustness.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from report_items.common import (PARTITION_NAMES, PARTITIONS, fmt_int, fmt_pct, fmt_rmse, load_json,
                                 write_table)

LABEL = "Table S5"
TITLE = "Held-out RMSE of the untuned random forest in the six train/test partitions"
SOURCES = ["eng15_cells.json", "eng15_split_robustness.json"]

INT_COLS = ("n_test_cells", "n_test_rows_all", "n_test_rows_history", "combined_k", "temporal_k")
MEANS = [("Mean, batch held out", "lobo", "mean_lobo"), ("Mean, mixed thirds", "cv3", "mean_cv3")]


def rmse(c: dict) -> float:
    """Held-out RMSE pooled over the partition's test rows."""
    return c["test"]["pooled"]["rmse"]


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, int]]]:
    """The markdown table, the CSV (full precision) and the partitions whose combined k is not 63."""
    cells = load_json("eng15_cells.json")["cells"]
    parts = load_json("eng15_split_robustness.json")["partitions"]

    def cell(exp: str, p: str, pool: str) -> dict:
        return cells[f"{exp}|{p}|{pool}|pearson|rf"]

    rows, csv = [], []
    for p in PARTITIONS:
        lv, lh, cb, tp = (cell("e08", p, "level"), cell("e14", p, "level"), cell("e14", p, "combined"),
                          cell("e14", p, "temporal"))
        gain = (rmse(cb) - rmse(lh)) / rmse(lh)
        rows.append({"Partition": PARTITION_NAMES[p], "Test cells": parts[p]["n_test_cells"],
                     "Nine level features, all rows": rmse(lv), "Nine level features, history rows": rmse(lh),
                     "Level + history": rmse(cb), "k (level + history)": cb["k"], "Change with history": gain})
        csv.append({"partition": p, "name": PARTITION_NAMES[p], "kind": parts[p]["kind"],
                    "n_test_cells": parts[p]["n_test_cells"],
                    "test_cells_by_batch": "; ".join(f"{b} {n}" for b, n in parts[p]["test_cells_by_batch"].items()),
                    "n_test_rows_all": lv["n_test_rows"], "n_test_rows_history": lh["n_test_rows"],
                    "level_all_rmse": rmse(lv), "level_all_r2": lv["test"]["pooled"]["r2"],
                    "level_hist_rmse": rmse(lh), "combined_rmse": rmse(cb), "combined_r2": cb["test"]["pooled"]["r2"],
                    "combined_k": cb["k"], "change_with_history": gain,
                    "temporal_rmse": rmse(tp), "temporal_k": tp["k"], "temporal_change": (rmse(tp) - rmse(lh)) / rmse(lh)})
    csv = pd.DataFrame(csv)

    mean_rows = []
    for name, kind, pid in MEANS:
        g = csv[csv["kind"] == kind]
        means = {"level_all_rmse": g["level_all_rmse"].mean(), "level_hist_rmse": g["level_hist_rmse"].mean(),
                 "combined_rmse": g["combined_rmse"].mean(), "change_with_history": g["change_with_history"].mean()}
        rows.append({"Partition": name, "Test cells": np.nan,
                     "Nine level features, all rows": means["level_all_rmse"],
                     "Nine level features, history rows": means["level_hist_rmse"],
                     "Level + history": means["combined_rmse"], "k (level + history)": np.nan,
                     "Change with history": means["change_with_history"]})
        mean_rows.append({"partition": pid, "name": name, **means})
    csv_out = pd.concat([csv, pd.DataFrame(mean_rows)], ignore_index=True)
    csv_out = csv_out.astype({c: "Int64" for c in INT_COLS})   # integers stay integers, blank in the mean rows

    not63 = [(p, int(k)) for p, k in zip(csv["partition"], csv["combined_k"]) if k != 63]
    return pd.DataFrame(rows), csv_out, not63


def notes(not63: list[tuple[str, int]]) -> list[str]:
    """How the models were chosen and scored, and what the change and the mean rows are."""
    not63_text = "; ".join(f"{PARTITION_NAMES[p].lower()} (k = {k})" for p, k in not63) or "none"
    return [
        "Held-out RMSE pooled over the partition's test rows; untuned RF (300 trees, seed 0), features "
        "ranked by Pearson |r| on the partition's development cells and k chosen by their own five-fold "
        "CV (ENG-15). Batch 3 out is the study split (imported from ENG-08 / ENG-14, Pearson column "
        "order; Table XVI scores the same models in the pool order, which differs by the random-forest "
        "column-order effect only). Mixed thirds: a "
        "seeded three-fold split of all 124 cells stratified on batch and cycle-life quartile.",
        "History rows drop each cell's first 20 cycles. Level + history = the ENG-14 combined pool (the "
        "nine level features and their 54 history columns, Pearson order), with the Pearson-selected k: "
        f"all 63 except for {not63_text}; the held-out score of all 63 is not stored there. Change with "
        "history = (level + history − nine level features) / nine level features, both on the history "
        "rows; negative = history lowers the error.",
        "Untuned models only: the tuned configuration (ENG-09) was chosen on batches 1 + 2, so it would "
        "leak in the partitions that test on them. Mean rows: plain means of the three partitions "
        "(change: mean of the three changes).",
    ]


def make(stem: Path) -> list[Path]:
    df, csv, not63 = build()
    formats = {"Test cells": fmt_int, "Nine level features, all rows": fmt_rmse,
               "Nine level features, history rows": fmt_rmse, "Level + history": fmt_rmse,
               "k (level + history)": fmt_int, "Change with history": lambda x: fmt_pct(x, 1, signed=True)}
    return write_table(stem, df, label=LABEL, title=TITLE, formats=formats, csv_df=csv,
                       notes=notes(not63), sources=SOURCES)
