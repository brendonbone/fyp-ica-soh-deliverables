"""
Table D1. The best subset of each size without and with history, development CV and batch 3
For k = 1..9, the best subset of the nine level features (ENG-24) and the best subset of
the features with their six history columns (ENG-23), both on the history rows, with
their dev-CV RMSE and their batch-3 RMSE (untuned RF). Reads best_per_k and the stored
batch-3 records (config untuned, reason best_of_k) of the two analysis files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_rmse, load_json, write_table
from report_items.tab_best_subsets_level_vs_history import HIST, LVL
from report_items.tab_best_subsets_paths import LABEL as ALL_ROWS_TABLE

LABEL = "Table D1"
TITLE = ("Direct comparison between the best subset with and without history: the best subset of each "
         "size, level features only and each feature with its six history columns, on the history rows "
         "(untuned RF; development-CV and batch-3 RMSE)")
SOURCES = [f"{LVL}_analysis.json", f"{HIST}_analysis.json", "eng14_full.json"]
COLUMNS = {"level_subset": "Best level subset (no history)", "level_dev_cv": "Dev-CV, no history",
           "level_b3": "Batch 3, no history", "history_subset": "Best subset with history",
           "history_dev_cv": "Dev-CV, with history", "history_b3": "Batch 3, with history"}


def subset_text(blocks, history: bool) -> str:
    """'height + position + fwhm' in pool order; 'all nine' (with '(63 columns)' for history)."""
    if len(blocks) == len(FEATURES):
        return "all nine (63 columns)" if history else "all nine"
    return " + ".join(f for f in FEATURES if f in blocks)


def best_of_k(stem: str) -> dict[int, dict]:
    """k -> {blocks, dev-CV, batch-3 RMSE} of the best subset of each size."""
    A = load_json(f"{stem}_analysis.json")
    b3 = {tuple(r["blocks"]): r["b3_rmse"] for r in A["heldout"]
          if r["config"] == "untuned" and "best_of_k" in r["reasons"]}
    return {r["k"]: {"blocks": r["blocks"], "cv": r["cv"], "b3": b3[tuple(r["blocks"])]}
            for r in A["best_per_k"]}


def build() -> pd.DataFrame:
    """One row per k, full precision."""
    lvl, hist = best_of_k(LVL), best_of_k(HIST)
    return pd.DataFrame([{
        "k": k,
        "level_subset": subset_text(lvl[k]["blocks"], history=False),
        "level_dev_cv": lvl[k]["cv"], "level_b3": lvl[k]["b3"],
        "history_subset": subset_text(hist[k]["blocks"], history=True),
        "history_dev_cv": hist[k]["cv"], "history_b3": hist[k]["b3"],
    } for k in sorted(lvl)])


def make(stem: Path) -> list[Path]:
    df = build()
    shown = df.rename(columns=COLUMNS)
    formats = {COLUMNS[c]: fmt_rmse for c in ("level_dev_cv", "level_b3", "history_dev_cv", "history_b3")}
    rows = load_json("eng14_full.json")["history_filter"]
    notes = [f"Both searches use the same rows: the {rows['dev']['rows_with_history']:,} development and "
             f"{rows['test']['rows_with_history']:,} batch-3 cycles with at least 20 earlier clean cycles, "
             f"so the level-only values differ slightly from {ALL_ROWS_TABLE} (all {rows['dev']['rows']:,} "
             f"development rows)."]
    return write_table(stem, shown, label=LABEL, title=TITLE, formats=formats, csv_df=df, notes=notes,
                       sources=SOURCES)
