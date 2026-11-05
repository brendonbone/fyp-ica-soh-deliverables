"""
Table S6. Stability of each feature ranking across the six train/test partitions
For each ranking method, the ranking of every partition (made on its development cells only) is
compared with every other: mean and minimum Kendall tau of the rank positions and mean top-3 Jaccard
over the 15 pairs, recomputed here from the rankings; plus the top feature in each partition.
Rankings from eng15_cells.json (Pearson, mutual information, RF impurity, permutation; level pool and
the 63-column combined pool) and eng25_partition_robustness.json (Spearman, level pool only).
"""
from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from report_items.common import PARTITIONS, fmt_float, fmt_r, load_json, write_table

LABEL = "Table S6"
TITLE = "Stability of each feature ranking across the six train/test partitions"
SOURCES = ["eng15_cells.json", "eng25_partition_robustness.json"]

METHOD_NAME = {"pearson": "Pearson |r|", "spearman": "Spearman |ρ|", "mutual_info": "Mutual information",
               "rf_importance": "RF impurity", "permutation": "Permutation importance"}
# (row label, ENG-15 experiment, pool, methods); Spearman was run on the level pool only (ENG-25)
GROUPS = [("Nine level features", "e08", "level",
           ["pearson", "spearman", "mutual_info", "rf_importance", "permutation"]),
          ("Level + history (63 columns)", "e14", "combined",
           ["pearson", "mutual_info", "rf_importance", "permutation"])]
# the CSV's provenance strings, as written by the Section V script
SRC_ENG15 = "results/eng15_cells.json (rankings); results/eng15_split_robustness.json: summary"
SRC_ENG25 = "results/eng25_partition_robustness.json: spearman"


def stability(ranks: dict[str, list[str]]) -> tuple[float, float, float]:
    """Mean / min pairwise Kendall tau and mean top-3 Jaccard of rankings (ENG-15/25 method)."""
    names = sorted(next(iter(ranks.values())))
    taus, jac = [], []
    for a, b in itertools.combinations(list(ranks), 2):
        taus.append(kendalltau([ranks[a].index(f) for f in names], [ranks[b].index(f) for f in names])[0])
        ta, tb = set(ranks[a][:3]), set(ranks[b][:3])
        jac.append(len(ta & tb) / len(ta | tb))
    return float(np.mean(taus)), float(np.min(taus)), float(np.mean(jac))


def count_text(seq: list[str]) -> str:
    """['height', 'height', 'area25'] -> 'height 2, area25 1' (most frequent first, ties by first seen)."""
    c = Counter(seq)
    order = sorted(c, key=lambda f: (-c[f], seq.index(f)))
    return "all six: " + order[0] if len(order) == 1 else ", ".join(f"{f} {c[f]}" for f in order)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table and the CSV (one row per pool and method, full precision)."""
    cells = load_json("eng15_cells.json")["cells"]
    spearman = load_json("eng25_partition_robustness.json")["spearman"]["by_partition"]
    rows, csv = [], []
    for pool_label, exp, pool, methods in GROUPS:
        for m in methods:
            if m == "spearman":
                ranks = {p: spearman[p]["ranking"] for p in PARTITIONS}
                src = SRC_ENG25
            else:
                ranks = {p: [x["feature"] for x in cells[f"{exp}|{p}|{pool}|{m}|rf"]["ranking"]]
                         for p in PARTITIONS}
                src = SRC_ENG15
            tau_m, tau_min, jac = stability(ranks)
            top1 = [ranks[p][0] for p in PARTITIONS]
            rows.append({"Features": pool_label, "Method": METHOD_NAME[m], "Kendall τ, mean": tau_m,
                         "Kendall τ, min": tau_min, "Top-3 Jaccard, mean": jac,
                         "Top feature (B1, B2, B3, M1, M2, M3)": ", ".join(top1),
                         "Top feature, count": count_text(top1)})
            csv.append({"pool": pool, "method": m, "kendall_tau_mean": tau_m, "kendall_tau_min": tau_min,
                        "top3_jaccard_mean": jac, **{f"top1_{p}": t for p, t in zip(PARTITIONS, top1)},
                        "source": src})
    return pd.DataFrame(rows), pd.DataFrame(csv)


NOTES = [
    "Kendall τ between the rankings of each pair of partitions (15 pairs), mean and minimum; top-3 "
    "Jaccard likewise. Each ranking uses only the partition's development cells. B1, B2, B3 = batch "
    "1, 2, 3 held out (B3 = the study split); M1–M3 = the mixed thirds.",
    "Pearson, mutual information, RF impurity and permutation importance (untuned RF, mean over the "
    "five inner folds) from ENG-15; the first three do not depend on the regressor. Spearman from "
    "ENG-25.",
    "With history: the 63-column combined pool on the history rows (Spearman was not run there).",
]


def make(stem: Path) -> list[Path]:
    df, csv = build()
    formats = {"Kendall τ, mean": lambda x: fmt_r(x, 2), "Kendall τ, min": lambda x: fmt_r(x, 2),
               "Top-3 Jaccard, mean": fmt_float(2)}
    return write_table(stem, df, label=LABEL, title=TITLE, formats=formats, csv_df=csv,
                       notes=NOTES, sources=SOURCES)
