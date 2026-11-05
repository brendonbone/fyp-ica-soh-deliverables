"""
Table V. Association of each feature with SoH on the development rows
Pearson r and Spearman ρ of each feature with SoH (stage 4, level-pool rows of
eng16_pearson_vs_spearman.csv) and its mutual information with SoH (the scores
of the mutual-information selector, eng08_full.json grid['mutual_info|rf']).
The three ranks are recomputed from the values (1 = largest |r|, |ρ| or MI;
ties by order of appearance) and the rows are ordered by |r|. The row and cell
counts in the notes come from the cleaned feature table.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from report_items.common import (FEATURES, fmt_float, fmt_int, fmt_r, load_csv, load_json,
                                 load_parquet, write_table)
from src.data_prep import clean_feature_table, split_dev_test

LABEL = "Table V"
TITLE = "Association of each feature with SoH on the development rows"
SOURCES = ["eng16_pearson_vs_spearman.csv", "eng08_full.json", "features_expanded.parquet"]

COLUMNS = {"feature": "Feature", "pearson_r": "Pearson r", "pearson_rank": "r rank",
           "spearman_rho": "Spearman ρ", "spearman_rank": "ρ rank",
           "mi_nats": "MI (nats)", "mi_rank": "MI rank"}


def dev_rows() -> pd.DataFrame:
    """The development rows (batches 1 and 2) of the cleaned feature table."""
    clean, _ = clean_feature_table(load_parquet("features_expanded.parquet"), verbose=False)
    return split_dev_test(clean)[0]


def build() -> tuple[pd.DataFrame, dict]:
    """The table (one row per feature, full precision, ordered by |r|) and the markdown formats."""
    e16 = load_csv("eng16_pearson_vs_spearman.csv")
    level = e16[e16["pool"] == "level"].set_index("feature")
    ranking = load_json("eng08_full.json")["grid"]["mutual_info|rf"]["ranking"]
    mi = {r["feature"]: r["score"] for r in ranking}

    df = pd.DataFrame({"feature": FEATURES,
                       "pearson_r": [level.loc[f, "pearson_r"] for f in FEATURES],
                       "spearman_rho": [level.loc[f, "spearman_rho"] for f in FEATURES],
                       "mi_nats": [mi[f] for f in FEATURES]})
    df["pearson_rank"] = df["pearson_r"].abs().rank(ascending=False, method="first").astype(int)
    df["spearman_rank"] = df["spearman_rho"].abs().rank(ascending=False, method="first").astype(int)
    df["mi_rank"] = df["mi_nats"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("pearson_rank").reset_index(drop=True)
    df = df[["feature", "pearson_r", "pearson_rank", "spearman_rho", "spearman_rank", "mi_nats", "mi_rank"]]

    formats = {"Pearson r": fmt_r, "Spearman ρ": fmt_r, "MI (nats)": fmt_float(3),
               "r rank": fmt_int, "ρ rank": fmt_int, "MI rank": fmt_int}
    return df, formats


def notes(df: pd.DataFrame) -> list[str]:
    """What the rows are, what MI is, how far the three rankings agree, and why there are no p-values."""
    dev = dev_rows()
    n_rows, n_cells = len(dev), dev["cell_id"].nunique()
    ranks = {"pearson": df["pearson_rank"], "spearman": df["spearman_rank"], "mutual_info": df["mi_rank"]}
    tau = {(a, b): float(kendalltau(ranks[a], ranks[b])[0]) for a, b in itertools.combinations(ranks, 2)}
    return [f"Development rows only: {n_rows:,} clean cycles of the {n_cells} cells in batches 1 and 2. "
            "Rows are ordered by |r|; each rank is by |r|, |ρ| or MI (1 = strongest association).",
            "MI: k-nearest-neighbour estimate (scikit-learn mutual_info_regression, 3 neighbours, "
            "random_state 0) on all development rows, in nats; it is the score of the "
            "mutual-information selector in ENG-08.",
            f"Kendall τ between the rankings: Pearson–Spearman {tau[('pearson', 'spearman')]:.2f}, "
            f"Pearson–MI {tau[('pearson', 'mutual_info')]:.2f}, "
            f"Spearman–MI {tau[('spearman', 'mutual_info')]:.2f}.",
            f"The rows are about {n_rows / n_cells:.0f} cycles from each cell and are not independent, "
            "so the coefficients are ranking scores, not test statistics; no p-values are given."]


def make(stem: Path) -> list[Path]:
    df, formats = build()
    return write_table(stem, df.rename(columns=COLUMNS), label=LABEL, title=TITLE, formats=formats,
                       csv_df=df, notes=notes(df), sources=SOURCES)
