"""
Table S2. Batch-3 RMSE, R² and MAE of every ENG-28 feature set of Subsection G under each model
The 44 ENG-28 cells (11 feature sets x LR, untuned GPR, untuned RF, tuned RF) in the file's set and
model order, each with its dev-CV and batch-3 scores and their ranks over all 44 cells (1 = lowest
RMSE; ties share the lower rank). Reads eng28_feature_sets_models.json; the set names come from
feature_sets.describe, so they match Tables XIV, XV and S8. The note's Kendall τ is between the
dev-CV and batch-3 RMSE columns of this table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from report_items.common import fmt_pct, fmt_r, fmt_rmse, load_json, write_table
from report_items.feature_sets import describe

LABEL = "Table S2"
TITLE = ("Batch-3 RMSE, R² and MAE of every ENG-28 feature set of Subsection G under each model "
         "(ENG-28), beside their dev-CV RMSE")
E28 = "eng28_feature_sets_models.json"
SOURCES = [E28]

MODEL_LABEL = {"LR": "LR", "GPR": "GPR, untuned", "RF": "RF, untuned", "RF tuned": "RF, tuned"}
MODEL_WORDS = {"LR": "LR", "GPR, untuned": "untuned GPR", "RF, untuned": "untuned RF", "RF, tuned": "tuned RF"}


def build() -> pd.DataFrame:
    """One row per ENG-28 cell, full precision, with the two ranks (the CSV)."""
    e28 = load_json(E28)
    sets = {fs["id"]: fs for fs in e28["feature_sets"]}
    order = [fs["id"] for fs in e28["feature_sets"]]
    cells = sorted(e28["cells"], key=lambda c: (order.index(c["set"]), e28["models"].index(c["model"])))
    df = pd.DataFrame([{
        "set": c["set"], "feature_set": describe(sets[c["set"]]), "model_key": c["model"],
        "model": MODEL_LABEL[c["model"]],
        "n_columns": c["n_columns"], "soc_span": sets[c["set"]]["soc_window"]["soc_span"],
        "dev_cv_rmse": c["cv_pooled_rmse_mean"], "b3_rmse": c["b3_rmse"], "b3_r2": c["b3_r2"],
        "b3_mae": c["b3_mae"], "b3_reason": c["b3_reason"]} for c in cells])
    df["dev_cv_rank"] = df["dev_cv_rmse"].rank(method="min").astype("Int64")
    df["b3_rank"] = df["b3_rmse"].rank(method="min").astype("Int64")
    return df


def display(df: pd.DataFrame) -> pd.DataFrame:
    """The markdown columns (every cell pre-formatted)."""
    rank = [("" if pd.isna(x) else str(int(x))) for x in df["dev_cv_rank"]]
    return pd.DataFrame({
        "Feature set": df["feature_set"], "Model": df["model"],
        "Cols": df["n_columns"].astype(int).astype(str),
        "Discharge needed": [fmt_pct(x, 1) for x in df["soc_span"]],
        "Dev-CV RMSE": [fmt_rmse(x) for x in df["dev_cv_rmse"]],
        "Dev-CV rank": rank,
        "Batch-3 RMSE": [fmt_rmse(x) for x in df["b3_rmse"]],
        "Batch-3 rank": [("" if pd.isna(x) else str(int(x))) for x in df["b3_rank"]],
        "Batch-3 R²": [fmt_r(x) for x in df["b3_r2"]],
        "Batch-3 MAE": [fmt_rmse(x) for x in df["b3_mae"]],
    })


def notes(df: pd.DataFrame) -> list[str]:
    """What the cells are scored on, and how far dev-CV and batch 3 agree on their order."""
    tau_all = kendalltau(df["dev_cv_rmse"], df["b3_rmse"])[0]
    tau_model = {m: kendalltau(g["dev_cv_rmse"], g["b3_rmse"])[0] for m, g in df.groupby("model", sort=False)}
    n_sets = df["set"].nunique()
    return [
        "ENG-28: every set on the history rows (cycles with at least 20 earlier clean cycles), ENG-08 "
        "cell-to-fold map, columns in the ENG-14 combined-pool order restricted to the set, seed 0; batch 3 "
        "was scored once per cell, after every dev-CV cell of that model was finished, and no choice in G is "
        "made from it. Discharge needed = the SoC span the set needs (ENG-17), as in Tables XIV and XV.",
        f"Ranks: 1 = lowest RMSE of the {len(df)} cells. Kendall τ between the dev-CV and batch-3 RMSE of "
        f"the {len(df)} cells = {fmt_r(tau_all, 2)}; within each model ({n_sets} sets): "
        + "; ".join(f"{MODEL_WORDS[m]} {fmt_r(t, 2)}" for m, t in tau_model.items()) + ".",
    ]


def make(stem: Path) -> list[Path]:
    df = build()
    return write_table(stem, display(df), label=LABEL, title=TITLE, csv_df=df, notes=notes(df), sources=SOURCES)
