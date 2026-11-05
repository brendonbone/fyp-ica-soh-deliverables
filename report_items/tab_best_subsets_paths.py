"""
Table IX. Best subset of each size against the backward-elimination path and the Pearson and PI forward paths
For each k = 1..9: the best of the 511 subsets of the nine level features and
the subsets that backward elimination (BE) and the Pearson and tuned-RF PI
top-k rankings reach at that size (stage 11, eng19_exhaustive_subsets.csv;
rankings from eng08_full.json and eng08_full_tuned.json). Every dev-CV is the
ENG-19 pool-order score of the subset, never ENG-08's own k-curve; a method's
gap is measured against the best subset of the same size. The CSV also carries
sequential forward selection (sfs_*). Fig. 8 reuses the helpers below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import (FEATURES, MINUS, fmt_pct, fmt_rmse, join_features, load_csv,
                                 load_json, load_parquet, write_table)
from src.data_prep import clean_feature_table, split_dev_test

LABEL = "Table IX"
TITLE = ("Best subset of each size against the backward-elimination path and the Pearson and PI "
         "forward paths (dev-CV RMSE, untuned RF, nine level features)")
SOURCES = ["eng19_exhaustive_subsets.csv", "eng08_full.json", "eng08_full_tuned.json",
           "eng20_reference_seeds.json", "features_expanded.parquet"]

# (column tag, markdown label, step verb) in column order; SFS goes to the CSV only
METHODS = [("be", "BE", "dropped"), ("pearson", "Pearson", "added"),
           ("pi", "PI (tuned RF)", "added"), ("sfs", None, "added")]


# ------------------------ the 511 scores and the paths through them ------------------------

def key(feats) -> str:
    """Subset key as ENG-19 writes it: features joined by '+' in pool order."""
    s = set(feats)
    return "+".join(f for f in FEATURES if f in s)


def subsets() -> pd.DataFrame:
    """The 511 ENG-19 rows with columns key, k, cv, cv_std, cv_rank, features; sorted by (k, key)."""
    # pandas' default float parser, not round-trip: the reference table was built from
    # these values (up to one ulp off the written digits) and the CSV writes them back
    t = load_csv("eng19_exhaustive_subsets.csv", float_precision=None)
    t = t.rename(columns={"features": "key", "cv_pooled_rmse_mean": "cv", "cv_pooled_rmse_std": "cv_std"})
    t["features"] = t["key"].str.split("+")
    return t.sort_values(["k", "key"]).reset_index(drop=True)


def best_per_k(t: pd.DataFrame) -> dict[int, dict]:
    """k -> the row of the lowest dev-CV of that size (ties by key)."""
    return {int(k): g.sort_values(["cv", "key"]).iloc[0].to_dict() for k, g in t.groupby("k")}


def backward_elimination(cv: dict) -> dict[int, tuple[str, str | None]]:
    """k -> (subset key, feature dropped to reach it); ties go to pool order."""
    cur = list(FEATURES)
    path = {len(cur): (key(cur), None)}
    while len(cur) > 1:
        drop = min(cur, key=lambda d: (cv[key(c for c in cur if c != d)], FEATURES.index(d)))
        cur.remove(drop)
        path[len(cur)] = (key(cur), drop)
    return path


def forward_selection(cv: dict) -> dict[int, tuple[str, str]]:
    """k -> (subset key, feature added to reach it); ties go to pool order."""
    cur: list[str] = []
    path = {}
    while len(cur) < len(FEATURES):
        add = min((f for f in FEATURES if f not in cur),
                  key=lambda a: (cv[key(cur + [a])], FEATURES.index(a)))
        cur.append(add)
        path[len(cur)] = (key(cur), add)
    return path


def ranking_path(ranking: list[str]) -> dict[int, tuple[str, str]]:
    """k -> (key of the top-k prefix of a fixed ranking, feature at rank k)."""
    return {k: (key(ranking[:k]), ranking[k - 1]) for k in range(1, len(ranking) + 1)}


def paths(t: pd.DataFrame) -> dict[str, dict]:
    """tag -> path for BE, the Pearson ranking (untuned grid), the tuned-RF PI ranking and SFS."""
    cv = dict(zip(t["key"], t["cv"]))
    pearson = load_json("eng08_full.json")["grid"]["pearson|rf"]["ranking"]
    pi = load_json("eng08_full_tuned.json")["grid"]["permutation|rf"]["ranking"]
    return {"be": backward_elimination(cv),
            "pearson": ranking_path([r["feature"] for r in pearson]),
            "pi": ranking_path([r["feature"] for r in pi]),
            "sfs": forward_selection(cv)}


# ------------------------ the table ------------------------

def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full-precision table (one row per k; the CSV) and its markdown view."""
    t = subsets()
    cv = dict(zip(t["key"], t["cv"]))
    best, path = best_per_k(t), paths(t)
    rows = []
    for k in range(1, len(FEATURES) + 1):
        bk = best[k]
        row = {"k": k, "best_subset": join_features(bk["features"]), "best_cv": bk["cv"],
               "best_cv_fold_sd": bk["cv_std"], "best_cv_rank_of_511": int(bk["cv_rank"])}
        for tag, _, verb in METHODS:
            sk, feat = path[tag][k]
            v = cv[sk]
            row.update({f"{tag}_{verb}": feat or "", f"{tag}_subset": join_features(sk.split("+")),
                        f"{tag}_cv": v, f"{tag}_same_as_best": sk == bk["key"],
                        f"{tag}_rel_gap_to_best": v / bk["cv"] - 1})
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, markdown_view(df)


def subset_text(feats: list[str]) -> str:
    """'all nine'; 'all but ...' for k >= 7 (shorter); else the features in pool order."""
    if len(feats) == len(FEATURES):
        return "all nine"
    if len(feats) >= 7:
        return "all but " + join_features(f for f in FEATURES if f not in feats)
    return join_features(feats)


def cv_cell(v: float, gap: float, same: bool) -> str:
    """Bold when the method's subset is the best of its size, else the gap to it."""
    return f"**{fmt_rmse(v)}**" if same else f"{fmt_rmse(v)} ({fmt_pct(gap, 1, signed=True)})"


def markdown_view(df: pd.DataFrame) -> pd.DataFrame:
    """The printed table: best subset, then the step and dev-CV of BE, Pearson and PI."""
    md = []
    for _, r in df.iterrows():
        row = {"k": str(r["k"]), "Best subset of size k": subset_text(r["best_subset"].split(", ")),
               "Dev-CV": fmt_rmse(r["best_cv"])}
        for tag, label, verb in METHODS:
            if label is None:
                continue
            feat, same = r[f"{tag}_{verb}"], r[f"{tag}_same_as_best"]
            if not feat:
                step = "(start)"
            elif tag == "be":
                step = f"{MINUS}{feat}" if same else f"{MINUS}{feat} → {r[f'{tag}_subset']}"
            else:
                step = f"+{feat}"
            row[f"{label}: {verb}"] = step
            row[f"{label}: dev-CV"] = cv_cell(r[f"{tag}_cv"], r[f"{tag}_rel_gap_to_best"], same)
        md.append(row)
    return pd.DataFrame(md)


def notes(df: pd.DataFrame) -> list[str]:
    """What dev-CV is, how to read the method columns and the k = 9 row, and the CSV-only columns."""
    clean, _ = clean_feature_table(load_parquet("features_expanded.parquet"), verbose=False)
    n_dev_cells = split_dev_test(clean)[0]["cell_id"].nunique()
    # ENG-08 scored the Pearson top-k subsets with columns in ranking order, ENG-19 in pool order
    # (the same order only at k = 1); the RF is not invariant to column order
    pearson_cv = {key(s.split(", ")): v for s, v in zip(df["pearson_subset"], df["pearson_cv"])}
    k_curve = load_json("eng08_full.json")["grid"]["pearson|rf"]["k_curve"]
    col_order = max(abs(r["cv_pooled_rmse_mean"] - pearson_cv[key(r["features"])]) for r in k_curve
                    if r["features"] != [f for f in FEATURES if f in r["features"]])
    e20 = load_json("eng20_reference_seeds.json")["summary"]
    rf10, ref10 = e20["untuned|both_orders"]["cv_rmse"], e20["tuned|both_orders"]["cv_rmse"]
    all9 = df.loc[df["k"] == len(FEATURES), "best_cv"].iloc[0]
    return [f"Dev-CV: pooled RMSE averaged over the five development folds (ENG-08 folds, {n_dev_cells} "
            "cells of batches 1 and 2); untuned RF, seed 0, every value read off the ENG-19 table of all "
            "511 subsets (columns in pool order). ENG-08's Pearson k-curve (columns in ranking order) "
            f"differs from the Pearson column by up to {col_order:.1e}.",
            "BE starts from all nine and at each step drops the feature whose removal gives the lowest "
            "dev-CV; its column gives the feature dropped to reach size k (and the subset where it differs "
            "from the best). The Pearson and PI columns give the feature added at rank k, so the subset at k "
            "is the top k of the ranking; PI is the tuned-RF permutation-importance ranking of ENG-08, its "
            "subsets scored with the untuned RF like every other column.",
            "Bold: the method's subset is the best subset of that size. Percentages: dev-CV above the best "
            "subset of the same size.",
            f"Row k = 9 is the search's own all-nine run ({fmt_rmse(all9)}, seed 0, pool order); Table VII "
            f"gives the mean over ten seeds × two column orders ({fmt_rmse(rf10['mean'])}, seed sd "
            f"{fmt_rmse(rf10['sd'])}) and the reference model, the tuned RF ({fmt_rmse(ref10['mean'])}).",
            "The CSV also carries sequential forward selection (sfs_* columns), read off the same table."]


def make(stem: Path) -> list[Path]:
    df, md = build()
    return write_table(stem, md, label=LABEL, title=TITLE, csv_df=df, notes=notes(df), sources=SOURCES)
