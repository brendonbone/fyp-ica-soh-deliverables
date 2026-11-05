"""
Table X. Dev-CV RMSE of the nine level features and of the nine features with their 20-cycle history
Untuned and tuned RF, nine level columns against all 63 (level + history), all scored on the
development rows that have 20 cycles of history. The four cells come from the subset searches
(ENG-24 all nine, ENG-23 all nine groups and its tuned re-score) and from ENG-28 (tuned nine level);
ENG-14 scored the same models in Pearson column order (CSV column eng14_pearson_order_dev_cv_rmse).
Changes are relative: history against the nine level features, tuning against the untuned RF.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_int, fmt_pct, fmt_pm, load_json, write_table

LABEL = "Table X"
TITLE = ("Dev-CV RMSE of the nine level features and of the nine features with their 20-cycle "
         "history (63 columns), RF untuned and tuned")
SOURCES = ["eng24_level_subsets_history_rows.json", "eng23_grouped_history_subsets.json",
           "eng23_grouped_history_subsets_analysis.json", "eng28_feature_sets_models.json",
           "eng14_full.json", "eng14_full_tuned.json", "eng09_tuned_params.json"]

MODEL_LABELS = {"RF": "RF, untuned", "RF tuned": "RF, tuned (reference configuration)"}
SET_LABELS = {"level": "nine level features", "history": "nine features + history"}
COL_LEVEL = "Nine level features (9 columns)"
COL_HISTORY = "Nine features + history (63 columns)"


def by_key(table_json: dict) -> dict:
    """Subset-search table {'table': [rows]} -> {'+'.join(blocks): row}."""
    return {"+".join(r["blocks"]): r for r in table_json["table"]}


def cells() -> dict:
    """(model, set) -> (mean, fold sd, n columns, source) for the four cells."""
    k9 = "+".join(FEATURES)    # the all-nine subset, blocks in pool order
    lvl_rf = by_key(load_json(SOURCES[0]))[k9]
    h_rf = by_key(load_json(SOURCES[1]))[k9]
    h_tu = load_json(SOURCES[2])["tuned"][k9]
    lvl_tu = {(c["set"], c["model"]): c for c in load_json(SOURCES[3])["cells"]}[("all9", "RF tuned")]
    return {
        ("RF", "level"): (lvl_rf["cv_pooled_rmse_mean"], lvl_rf["cv_pooled_rmse_std"], 9,
                          "results/eng24_level_subsets_history_rows.json table, all nine"),
        ("RF", "history"): (h_rf["cv_pooled_rmse_mean"], h_rf["cv_pooled_rmse_std"], 63,
                            "results/eng23_grouped_history_subsets.json table, all nine groups"),
        ("RF tuned", "level"): (lvl_tu["cv_pooled_rmse_mean"], lvl_tu["cv_pooled_rmse_std"], 9,
                                "results/eng28_feature_sets_models.json cells, all9 x RF tuned"),
        ("RF tuned", "history"): (h_tu["cv"], h_tu["cv_std"], 63,
                                  "results/eng23_grouped_history_subsets_analysis.json tuned, "
                                  "all nine groups"),
    }


def eng14_pearson_order() -> dict:
    """(model, set) -> ENG-14's dev-CV of the same model with the columns in Pearson order."""
    def all_columns(grid_json: dict, pool: str) -> float:
        """Last point of the Pearson-order k curve = every column of the pool."""
        return grid_json["grid"][pool]["pearson|rf"]["k_curve"][-1]["cv_pooled_rmse_mean"]

    e14, e14t = load_json("eng14_full.json"), load_json("eng14_full_tuned.json")
    return {("RF", "level"): all_columns(e14, "level"),
            ("RF", "history"): all_columns(e14, "combined"),
            ("RF tuned", "level"): all_columns(e14t, "level"),
            ("RF tuned", "history"): all_columns(e14t, "combined")}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table (formatted strings) and the CSV (one row per cell, full precision)."""
    c = cells()
    gain = {m: (c[(m, "history")][0] - c[(m, "level")][0]) / c[(m, "level")][0]
            for m in ("RF", "RF tuned")}
    tune = {s: (c[("RF tuned", s)][0] - c[("RF", s)][0]) / c[("RF", s)][0]
            for s in ("level", "history")}

    rows = [{"Model": MODEL_LABELS[m],
             COL_LEVEL: fmt_pm(*c[(m, "level")][:2]),
             COL_HISTORY: fmt_pm(*c[(m, "history")][:2]),
             "Change with history": fmt_pct(gain[m], signed=True)} for m in ("RF", "RF tuned")]
    rows.append({"Model": "Change with tuning",
                 COL_LEVEL: fmt_pct(tune["level"], signed=True),
                 COL_HISTORY: fmt_pct(tune["history"], signed=True),
                 "Change with history": ""})

    e14 = eng14_pearson_order()
    csv = pd.DataFrame([
        {"model": m, "features": SET_LABELS[s], "n_columns": v[2],
         "dev_cv_rmse": v[0], "dev_cv_fold_sd": v[1],
         "change_with_history_rel": gain[m] if s == "history" else float("nan"),
         "change_with_tuning_rel": tune[s] if m == "RF tuned" else float("nan"),
         "eng14_pearson_order_dev_cv_rmse": e14[(m, s)],
         "source": v[3]}
        for (m, s), v in c.items()])
    return pd.DataFrame(rows), csv


def notes() -> list[str]:
    """What the dev-CV is measured on, and what 'tuned' means."""
    n_rows = load_json("eng14_full.json")["history_filter"]["dev"]["rows_with_history"]
    rf = load_json("eng09_tuned_params.json")["rf"]
    return [
        "Dev-CV: pooled RMSE averaged over the five development folds, ± the sd over the folds "
        f"(ddof 0), on the {fmt_int(n_rows)} development rows that have 20 cycles of history, so "
        "the nine-level values differ slightly from Table VII (every clean row). Change with "
        "history is relative to the nine level features, change with tuning to the untuned RF.",
        "Tuned: the reference configuration of Table VII, fixed by the ENG-09 search "
        f"(max_features {rf['max_features']}, min_samples_leaf {rf['min_samples_leaf']}, no depth "
        "limit, 300 trees, seed 0); it was chosen on the nine level features and is not re-tuned "
        "for the history columns.",
    ]


def make(stem: Path) -> list[Path]:
    md, csv = build()
    return write_table(stem, md, label=LABEL, title=TITLE, csv_df=csv, notes=notes(),
                       sources=SOURCES)
