"""
Table II. The Severson LFP dataset by batch
One row per batch and a total: cells, recorded cycles (rows of the extracted
feature table, cycle 0 included), cycles kept by the row cleaning
(src.data_prep.clean_feature_table), distinct charge policies, the mean cycle
life of the cells, and the batch's role in the study. The total counts distinct
policies over all batches, so it is not the sum of the batch rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import fmt_int, load_parquet, write_table
from src.data_prep import DEV_BATCHES, SOH_MAX, SOH_MIN, TEST_BATCH, clean_feature_table
from src.severson_loader import BATCH1_INCOMPLETE_CELLS, BATCH3_NOISY_CELLS, CONTINUATION_MAP

LABEL = "Table II"
TITLE = ("The Severson LFP dataset by batch: cells, cycles, charge policies, mean cycle life and role in "
         "the study")
SOURCES = ["features_expanded.parquet"]

COLUMNS = {"batch": "Batch", "role": "Role", "cells": "Cells", "recorded_cycles": "Recorded cycles",
           "clean_cycles": "Cycles kept after cleaning", "charge_policies": "Charge policies",
           "mean_cycle_life": "Mean cycle life"}
ROLE = {**{b: "development" for b in DEV_BATCHES}, TEST_BATCH: "held-out test"}


def counts(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Cells, recorded and kept cycles, distinct charge policies and the mean cycle life of the
    cells (each cell's cycle_life counted once) of one group of rows."""
    return {"cells": raw["cell_id"].nunique(), "recorded_cycles": len(raw),
            "clean_cycles": len(clean), "charge_policies": raw["charge_policy"].nunique(),
            "mean_cycle_life": float(raw.groupby("cell_id")["cycle_life"].first().mean())}


def build() -> tuple[pd.DataFrame, dict]:
    """The table (one row per batch, then the total) and the markdown formats."""
    raw = load_parquet("features_expanded.parquet")
    clean, _ = clean_feature_table(raw, verbose=False)
    rows = [{"batch": b, "role": ROLE[b], **counts(raw[raw["batch"] == b], clean[clean["batch"] == b])}
            for b in sorted(raw["batch"].unique())]
    rows.append({"batch": "Total", "role": "", **counts(raw, clean)})
    df = pd.DataFrame(rows)
    formats = {c: fmt_int for c in ("Cells", "Recorded cycles", "Cycles kept after cleaning",
                                    "Charge policies", "Mean cycle life")}
    return df, formats


def make(stem: Path) -> list[Path]:
    df, formats = build()
    notes = [f"Cells after the loader's exclusions: {len(BATCH1_INCOMPLETE_CELLS)} incomplete batch-1 "
             f"cells and {len(BATCH3_NOISY_CELLS)} noisy batch-3 cells dropped, and the "
             f"{len(CONTINUATION_MAP)} batch-2 cells that continue batch-1 cells stitched onto them.",
             f"Cleaning keeps a cycle when its peak extraction succeeded (status ok) and "
             f"{SOH_MIN} < SoH ≤ {SOH_MAX}. Development batches are used for ranking, tuning and "
             f"cross-validation; the held-out batch is scored once. The total counts each charge "
             f"policy once, so it is less than the sum when batches share a policy.",
             "Mean cycle life: the dataset's cycle life of each cell (after the continuation merges), "
             "averaged over the cells."]
    return write_table(stem, df.rename(columns=COLUMNS), label=LABEL, title=TITLE, formats=formats,
                       csv_df=df, bold_rows=[len(df) - 1], notes=notes, sources=SOURCES)
