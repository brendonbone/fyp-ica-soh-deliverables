"""
The six train/test partitions of the split-design check (stages 9 and 17).
===========================================================================
Every result of the study trains on Severson batches 1+2 and tests on
batch 3. Stages 9 (ENG-15) and 17 (ENG-25) repeat the core comparisons
under six partitions of the 124 cells; every cell not listed as a test
cell is a development cell:

    lobo_b1, lobo_b2, lobo_b3   leave one batch out (lobo_b3 is the study split)
    cv3_f0, cv3_f1, cv3_f2      an outer 3-fold over all cells, stratified on
                                batch x within-batch cycle-life quartile
                                (StratifiedKFold on cells, shuffled, seed 0):
                                ~83 dev / ~41 test cells, batch-mixed

`relabel` makes the unchanged harness (split_dev_test, assign_folds, the
selectors, cv_evaluate, heldout_evaluate) see a partition: its test cells
get batch "b3" and every other cell "b1" (the original batch is kept in
`batch_orig`), so each partition is a full nested evaluation. A partition's
cell -> fold map is assign_folds on its full clean dev rows, reused
unchanged on the history rows (the ENG-14 rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data_prep import DEV_BATCHES, RF_SEED, TEST_BATCH

PARTITIONS = ["lobo_b1", "lobo_b2", "lobo_b3", "cv3_f0", "cv3_f1", "cv3_f2"]
N_OUTER = 3
ID_COLS = ["cell_id", "batch", "cycle", "SOH", "cycle_life"]


def make_partitions(clean: pd.DataFrame) -> dict[str, dict]:
    """Test-cell lists for the six partitions (deterministic, seed 0)."""
    cells = (clean.groupby("cell_id", sort=True)
             .agg(batch=("batch", "first"), cycle_life=("cycle_life", "first"))
             .reset_index())
    parts: dict[str, dict] = {}
    for b in sorted(cells["batch"].unique()):
        parts[f"lobo_{b}"] = {
            "kind": "lobo",
            "test_cells": sorted(cells.loc[cells["batch"] == b, "cell_id"]),
        }
    q = cells.groupby("batch")["cycle_life"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
    label = cells["batch"].astype(str) + "_q" + q.astype(int).astype(str)
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RF_SEED)
    for i, (_, te) in enumerate(skf.split(cells, label)):
        parts[f"cv3_f{i}"] = {
            "kind": "cv3",
            "test_cells": sorted(cells["cell_id"].iloc[te]),
        }
    batch_of = dict(zip(cells["cell_id"], cells["batch"]))
    for p in parts.values():
        comp = pd.Series([batch_of[c] for c in p["test_cells"]]).value_counts()
        p["n_test_cells"] = len(p["test_cells"])
        p["n_dev_cells"] = len(cells) - len(p["test_cells"])
        p["test_cells_by_batch"] = {k: int(v) for k, v in sorted(comp.items())}
    return parts


def relabel(table: pd.DataFrame, test_cells: list[str]) -> pd.DataFrame:
    """Make the unchanged harness see this partition: test -> b3, rest -> b1."""
    out = table.copy()
    out["batch_orig"] = out["batch"].astype(str)
    is_test = out["cell_id"].isin(set(test_cells)).to_numpy()
    out["batch"] = np.where(is_test, TEST_BATCH, DEV_BATCHES[0])
    return out
