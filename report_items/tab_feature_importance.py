"""
Table VIII. Permutation importance of the reference model (tuned RF) and impurity importance of the untuned RF, beside the ranks of Table V
Permutation importance (PI) of the tuned RF (eng08_full_tuned.json permutation|rf; fold sd from
eng21_grouped_permutation.json, which reproduces it) and impurity of the untuned RF beside the Pearson,
Spearman and MI ranks (eng08_full.json). Each rank is the feature's 1-based position in that ENG-08
selector's ranking; rows in PI order. The CSV adds r and ρ (eng16 level rows), the untuned PI and the
Shapley values of the 511-subset search (eng19b_analysis.json).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_float, fmt_pm, load_csv, load_json, write_table

LABEL = "Table VIII"
TITLE = ("Permutation importance of the reference model (tuned RF) and impurity importance of "
         "the untuned RF, beside the ranks of Table V")
SOURCES = ["eng16_pearson_vs_spearman.csv", "eng08_full.json", "eng08_full_tuned.json",
           "eng21_grouped_permutation.json", "eng19b_analysis.json"]


def ranks_of(ranking: list[dict]) -> dict[str, int]:
    """1-based position of each feature in a selector's ranking list."""
    return {r["feature"]: i + 1 for i, r in enumerate(ranking)}


def scores_of(ranking: list[dict]) -> dict[str, float]:
    """The selector's score of each feature."""
    return {r["feature"]: r["score"] for r in ranking}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The CSV table (one row per feature, PI order, full precision) and its formatted markdown view."""
    grid = load_json("eng08_full.json")["grid"]                      # untuned RF
    pi_cell = load_json("eng08_full_tuned.json")["grid"]["permutation|rf"]   # tuned RF
    e16 = load_csv("eng16_pearson_vs_spearman.csv")
    level = e16[e16["pool"] == "level"].set_index("feature")
    pi_fold_sd = {f: v["fold_sd"] for f, v in load_json("eng21_grouped_permutation.json")
                  ["models"]["level_tuned"].items() if f in FEATURES}
    shapley = load_json("eng19b_analysis.json")["shapley"]
    shapley_rank = {f: i + 1 for i, f in enumerate(sorted(FEATURES, key=lambda x: -shapley[x]))}

    rank = {"pearson": ranks_of(grid["pearson|rf"]["ranking"]),
            "spearman": ranks_of(grid["spearman|rf"]["ranking"]),
            "mi": ranks_of(grid["mutual_info|rf"]["ranking"]),
            "imp": ranks_of(grid["rf_importance|rf"]["ranking"]),
            "pi_u": ranks_of(grid["permutation|rf"]["ranking"]),
            "pi": ranks_of(pi_cell["ranking"])}
    mi = scores_of(grid["mutual_info|rf"]["ranking"])
    imp = scores_of(grid["rf_importance|rf"]["ranking"])
    pi_u = scores_of(grid["permutation|rf"]["ranking"])
    pi = scores_of(pi_cell["ranking"])

    order = [r["feature"] for r in pi_cell["ranking"]]
    csv = pd.DataFrame([{
        "feature": f,
        "pearson_r": level.loc[f, "pearson_r"], "pearson_rank": rank["pearson"][f],
        "spearman_rho": level.loc[f, "spearman_rho"], "spearman_rank": rank["spearman"][f],
        "mi_nats": mi[f], "mi_rank": rank["mi"][f],
        "pi_tuned_rf": pi[f], "pi_tuned_rf_fold_sd": pi_fold_sd[f], "pi_rank": rank["pi"][f],
        "impurity_untuned_rf": imp[f], "impurity_rank": rank["imp"][f],
        "pi_untuned_rf": pi_u[f], "pi_untuned_rank": rank["pi_u"][f],
        "shapley_untuned_rf": shapley[f], "shapley_rank": shapley_rank[f]} for f in order])

    disp = pd.DataFrame({
        "Feature": csv["feature"], "Pearson rank": csv["pearson_rank"],
        "Spearman rank": csv["spearman_rank"], "MI rank": csv["mi_rank"],
        "PI, tuned RF (± fold sd)": [fmt_pm(a, b, nd=4)
                                     for a, b in zip(csv["pi_tuned_rf"], csv["pi_tuned_rf_fold_sd"])],
        "PI rank": csv["pi_rank"],
        "Impurity, untuned RF": [fmt_float(3)(x) for x in csv["impurity_untuned_rf"]],
        "Impurity rank": csv["impurity_rank"]})
    return csv, disp


def notes() -> list[str]:
    """How each column was measured, and how far the five rankings agree."""
    e08t = load_json("eng08_full_tuned.json")
    # Stored by ENG-08 over the five selectors of this table (four untuned rankings, tuned PI).
    tau_mean = e08t["agreement_summary"]["rf"]["all_methods"]["mean"]
    jaccard = e08t["top3_jaccard_mean"]["rf"]
    return [
        "Rows in order of permutation importance. Pearson and Spearman ranks: |r| and |ρ| with SoH "
        "on all development rows (Table V). MI: Kraskov estimate on all development rows "
        "(mutual_info_regression, random_state 0).",
        "PI: rise in validation RMSE (SoH units) when one feature's column is shuffled; tuned RF "
        "(ENG-09 configuration) on all nine features, pool column order, seed 0; ten repeats per "
        "fold, mean over the five development folds; ± is the sd over the folds (ENG-08 tuned "
        "grid; fold values from ENG-21, which reproduces it exactly).",
        "Impurity: mean decrease in impurity of the untuned 300-tree RF (pool order, seed 0), "
        "averaged over the five fold-training fits; it sums to 1.",
        f"Agreement of the five rankings: mean pairwise Kendall τ {tau_mean:.2f}, mean top-three "
        f"Jaccard overlap {jaccard:.2f}.",
        "The CSV also holds the untuned RF's PI and the exact Shapley values of Section V-D "
        "(511-subset search, untuned RF).",
    ]


def make(stem: Path) -> list[Path]:
    csv, disp = build()
    return write_table(stem, disp, label=LABEL, title=TITLE, csv_df=csv, notes=notes(), sources=SOURCES)
