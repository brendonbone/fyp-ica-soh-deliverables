"""
Table XII. Best subset of each size, level features alone and with history
===========================================================================
For k = 1..9, the best subset of the nine features scored without history
(ENG-24) and with each feature's six history columns (ENG-23), on the same rows
(untuned RF, dev-CV). Reads the best_per_k of the two analysis files, the two
511-subset tables (lead of each best subset over the runner-up of its size) and
the paired table, where the gain from history of a subset is -rel_change =
(level - history) / level.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from report_items.common import FEATURES, fmt_pct, fmt_rmse, join_features, load_csv, load_json, write_table
from src.subset_search import subset_key, table_from_records

LABEL = "Table XII"
TITLE = ("Best subset of each size: the nine level features alone (ENG-24) and each feature with its six "
         "20-cycle history columns (ENG-23), on the same rows (dev-CV RMSE, untuned RF)")
HIST = "eng23_grouped_history_subsets"     # each feature with its six history columns (7k columns)
LVL = "eng24_level_subsets_history_rows"   # the same subsets without history, on the same rows
PAIRED = "eng23_paired_level_history.csv"
SOURCES = [f"{LVL}_analysis.json", f"{HIST}_analysis.json", PAIRED, f"{LVL}.json", f"{HIST}.json",
           "eng14_full.json"]
FORMATS = {"Level only: dev-CV": fmt_rmse, "With history: dev-CV": fmt_rmse, "Gain from history": fmt_pct}


def search_table(stem: str) -> pd.DataFrame:
    """The 511-subset table of a search, sorted by dev-CV then key (cv_rank 1..511)."""
    d = load_json(f"{stem}.json")
    return table_from_records({subset_key(r["blocks"]): r for r in d["table"]})


def short_subset(blocks) -> str:
    """Up to six features listed; seven or eight as 'all but ...'."""
    blocks = list(blocks)
    if len(blocks) == len(FEATURES):
        return "all nine"
    if len(blocks) >= 7:
        return "all but " + join_features([f for f in FEATURES if f not in blocks])
    return join_features(blocks)


def margins_to_runner_up(t: pd.DataFrame) -> dict[int, float]:
    """Lead of the best subset of each size over the runner-up of that size (none at k = 9)."""
    out = {}
    for k, g in t.groupby("k"):
        v = np.sort(g["cv_pooled_rmse_mean"].to_numpy())
        if len(v) > 1:
            out[int(k)] = float(v[1] - v[0])
    return out


def gain_range(r) -> str:
    """'24.1% (10.8–42.2%)': median and range of the paired gain over the subsets of size k."""
    if r.n_subsets_of_size_k > 1:
        return f"{100 * r.paired_gain_median:.1f}% ({100 * r.paired_gain_min:.1f}–{100 * r.paired_gain_max:.1f}%)"
    return f"{100 * r.paired_gain_median:.1f}% (one subset)"


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The markdown table, the full-precision CSV table (one row per k) and the notes."""
    bk_l = {r["k"]: r for r in load_json(f"{LVL}_analysis.json")["best_per_k"]}
    bk_h = {r["k"]: r for r in load_json(f"{HIST}_analysis.json")["best_per_k"]}
    mg_l, mg_h = margins_to_runner_up(search_table(LVL)), margins_to_runner_up(search_table(HIST))
    paired = load_csv(PAIRED)
    paired["gain"] = -paired["rel_change"]
    by_k = paired.groupby("k")["gain"].agg(["median", "min", "max", "count"])

    rows = []
    for k in range(1, len(FEATURES) + 1):
        lv, hs = bk_l[k], bk_h[k]
        rows.append({
            "k": k, "n_subsets_of_size_k": int(by_k.loc[k, "count"]),
            "level_best_subset": join_features(lv["blocks"]), "level_n_columns": k,
            "level_dev_cv": lv["cv"], "level_dev_cv_fold_sd": lv["cv_std"], "level_cv_rank": lv["cv_rank"],
            "level_margin_to_runner_up": mg_l.get(k, math.nan),
            "history_best_subset": join_features(hs["blocks"]), "history_n_columns": 7 * k,
            "history_dev_cv": hs["cv"], "history_dev_cv_fold_sd": hs["cv_std"],
            "history_cv_rank": hs["cv_rank"], "history_margin_to_runner_up": mg_h.get(k, math.nan),
            "gain_best_vs_best": (lv["cv"] - hs["cv"]) / lv["cv"],
            "paired_gain_median": float(by_k.loc[k, "median"]),
            "paired_gain_min": float(by_k.loc[k, "min"]),
            "paired_gain_max": float(by_k.loc[k, "max"])})
    t = pd.DataFrame(rows)

    md = pd.DataFrame({
        "k": t["k"],
        "Level only: best subset": [short_subset(bk_l[k]["blocks"]) for k in t["k"]],
        "Level only: dev-CV": t["level_dev_cv"],
        "With history: best subset": [short_subset(bk_h[k]["blocks"]) for k in t["k"]],
        "With history: dev-CV": t["history_dev_cv"],
        "Gain from history": t["gain_best_vs_best"],
        "Gain over all subsets of size k: median (range)": [gain_range(r) for r in t.itertuples()],
    })

    hf = load_json("eng14_full.json")["history_filter"]["dev"]
    n_better = int((paired["gain"] > 0).sum())
    notes = [
        f"Both searches use the same {int(hf['rows_with_history']):,} development rows (of "
        f"{int(hf['rows']):,}) that have 20 cycles of history, so the level-only values are slightly higher "
        "than in Table IX (every development row). With history, a subset of k features has 7k columns "
        "(the level and its d10, d20, slope10, slope20, drift and rm20).",
        "Gain from history = (level - history) / level for the two best subsets of that size; the last "
        "column pairs every subset of size k with itself without history "
        f"({n_better} of the {len(paired)} pairs improved).",
    ]
    return md, t, notes


def make(stem: Path) -> list[Path]:
    md, t, notes = build()
    return write_table(stem, md, label=LABEL, title=TITLE, formats=FORMATS, csv_df=t, notes=notes,
                       sources=SOURCES)
