"""
Table S1. The six development/test partitions of the split-design check
Each partition of the 124 cells (stage 9, eng15_split_robustness.json
`partitions`): its kind, the held-out test cells per batch, and the number of
test and development cells. Under every partition the whole evaluation is
nested: ranking, the choice of k and the refit use the development cells only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import PARTITION_NAMES, PARTITIONS, fmt_int, load_json, write_table

LABEL = "Table S1"
TITLE = "The six development/test partitions used to check the split design"
SOURCES = ["eng15_split_robustness.json"]

BATCHES = ["b1", "b2", "b3"]
KIND = {"lobo": "leave one batch out", "cv3": "mixed third (3-fold)"}
COLUMNS = {"name": "Partition", "kind": "Kind", "test_b1": "Test cells, batch 1",
           "test_b2": "Test cells, batch 2", "test_b3": "Test cells, batch 3",
           "n_test_cells": "Test cells", "n_dev_cells": "Development cells"}
STUDY_SPLIT = "lobo_b3"


def build() -> tuple[pd.DataFrame, dict]:
    """The table (one row per partition, in the report's partition order) and the markdown formats."""
    parts = load_json("eng15_split_robustness.json")["partitions"]
    rows = []
    for p in PARTITIONS:
        d = parts[p]
        by_batch = d["test_cells_by_batch"]
        rows.append({"partition": p, "name": PARTITION_NAMES[p], "kind": KIND[d["kind"]],
                     **{f"test_{b}": by_batch.get(b, 0) for b in BATCHES},
                     "n_test_cells": d["n_test_cells"], "n_dev_cells": d["n_dev_cells"]})
    df = pd.DataFrame(rows)
    formats = {COLUMNS[c]: fmt_int for c in ("test_b1", "test_b2", "test_b3", "n_test_cells",
                                             "n_dev_cells")}
    return df, formats


def make(stem: Path) -> list[Path]:
    df, formats = build()
    notes = ["Bold: the study split (batches 1 and 2 develop, batch 3 tests). Mixed thirds: an outer "
             "3-fold split of all cells, stratified on batch and cycle-life quartile (scikit-learn "
             "StratifiedKFold, shuffled, seed 0)."]
    return write_table(stem, df.drop(columns="partition").rename(columns=COLUMNS), label=LABEL,
                       title=TITLE, formats=formats, csv_df=df,
                       bold_rows=[PARTITIONS.index(STUDY_SPLIT)], notes=notes, sources=SOURCES)
