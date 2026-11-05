"""
Shared data-prep utilities for the Sem-2 feature-selection study.
=================================================================
Single source of truth for the conventions established by the full-dataset
baseline (notebooks/260531 Full dataset baseline/full_dataset_baseline.ipynb):

- row cleaning: status == "ok" -> dropna(SOH) -> SOH > 0.5 -> SOH <= 1.05,
  applied in that order (the order affects the logged drop counts);
- dev/test design: dev = batches 1+2, held-out OOD test = batch 3;
- CV folds: StratifiedGroupKFold(n_splits=5, shuffle=False), grouped by
  cell_id, stratified on per-cell cycle-life quartiles (pd.qcut, q=4);
- metrics: RMSE / MAE / R2 reported both pooled-over-rows and
  averaged-over-cells (per-cell R2 is NaN for cells with <2 rows or
  constant SoH; the average uses np.nanmean).

Every stage in stages/ imports from here rather than re-implementing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import REPO_ROOT, RESULTS_DIR

# ------------------------ conventions (fixed) ------------------------

SOH_MIN: float = 0.5    # drops formation-cycle artifacts
SOH_MAX: float = 1.05   # drops single-cycle QD measurement spikes
N_SPLITS: int = 5
N_QUARTILES: int = 4
RF_SEED: int = 0        # random_state used everywhere a seed is needed

DEV_BATCHES: tuple[str, ...] = ("b1", "b2")
TEST_BATCH: str = "b3"

BASELINE_FEATURES: list[str] = ["height", "position", "area"]


# ------------------------ repo root / imports ------------------------

def chdir_repo_root() -> Path:
    """chdir to the repo root and put it on sys.path (for the notebooks and
    for running a stage on its own). Returns the root."""
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


# ------------------------ row cleaning ------------------------

def clean_feature_table(
    df: pd.DataFrame,
    *,
    soh_min: float = SOH_MIN,
    soh_max: float = SOH_MAX,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the baseline's row-cleaning pipeline, in order, and log counts.

    Returns (clean_df, log) where log records the row count after each step.
    """
    log: dict = {"start": len(df)}
    out = df[df["status"] == "ok"]
    log["after_status_ok"] = len(out)
    out = out.dropna(subset=["SOH"])
    log["after_dropna_soh"] = len(out)
    out = out[out["SOH"] > soh_min]
    log["after_soh_min"] = len(out)
    out = out[out["SOH"] <= soh_max]
    log["after_soh_max"] = len(out)
    log["soh_clean_band"] = [soh_min, soh_max]
    if verbose:
        print(
            f"  clean: {log['start']} -> ok {log['after_status_ok']} "
            f"-> dropna {log['after_dropna_soh']} -> SoH>{soh_min} "
            f"{log['after_soh_min']} -> SoH<={soh_max} {log['after_soh_max']}"
        )
    return out.reset_index(drop=True), log


def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dev = batches 1+2; held-out OOD test = batch 3. Split at cell level."""
    dev = df[df["batch"].isin(DEV_BATCHES)].reset_index(drop=True)
    test = df[df["batch"] == TEST_BATCH].reset_index(drop=True)
    return dev, test


# ------------------------ fold assignment ------------------------

def assign_folds(
    dev: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    n_quartiles: int = N_QUARTILES,
) -> tuple[pd.Series, dict[str, int]]:
    """
    StratifiedGroupKFold fold assignment on the dev set.

    Groups = cell_id (no cell's cycles span two folds); stratification label
    = cycle_life binned into quartiles per cell. shuffle=False so the
    assignment is deterministic. Returns (row_fold: Series aligned to dev,
    cell_to_fold: dict).
    """
    cell_meta = dev.groupby("cell_id")["cycle_life"].first()
    cl_bin = pd.qcut(cell_meta, q=n_quartiles, labels=False)
    row_bin = dev["cell_id"].map(cl_bin).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    row_fold = np.full(len(dev), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(dev, row_bin, dev["cell_id"].values)
    ):
        row_fold[val_idx] = fold
    assert (row_fold >= 0).all(), "every dev row must land in exactly one fold"

    fold_series = pd.Series(row_fold, index=dev.index, name="fold")
    cell_to_fold = (
        pd.DataFrame({"cell_id": dev["cell_id"], "fold": row_fold})
        .groupby("cell_id")["fold"]
        .first()
        .to_dict()
    )
    return fold_series, cell_to_fold


# ------------------------ metrics ------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pooled_metrics(y_true, y_pred) -> dict:
    """RMSE / MAE / R2 pooled over all rows."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def per_cell_metrics(cell_ids, y_true, y_pred) -> pd.DataFrame:
    """Per-cell RMSE / MAE / R2. R2 is NaN for <2 rows or constant SoH."""
    df = pd.DataFrame({"cell_id": cell_ids, "y": y_true, "yhat": y_pred})
    rows = []
    for cid, g in df.groupby("cell_id"):
        r2 = (
            float(r2_score(g["y"], g["yhat"]))
            if len(g) >= 2 and g["y"].nunique() > 1
            else float("nan")
        )
        rows.append({
            "cell_id": cid,
            "n": len(g),
            "rmse": rmse(g["y"], g["yhat"]),
            "mae": float(mean_absolute_error(g["y"], g["yhat"])),
            "r2": r2,
        })
    return pd.DataFrame(rows)


def cell_averaged_metrics(cell_ids, y_true, y_pred) -> dict:
    """Metrics averaged over cells (np.nanmean for R2)."""
    pc = per_cell_metrics(cell_ids, y_true, y_pred)
    return {
        "rmse": float(pc["rmse"].mean()),
        "mae": float(pc["mae"].mean()),
        "r2": float(np.nanmean(pc["r2"])),
    }


def both_metrics(cell_ids, y_true, y_pred) -> dict:
    """{'pooled': {...}, 'cell_avg': {...}} — the standard reporting pair."""
    return {
        "pooled": pooled_metrics(y_true, y_pred),
        "cell_avg": cell_averaged_metrics(cell_ids, y_true, y_pred),
    }


# ------------------------ manifest logging ------------------------

MANIFEST_PATH = RESULTS_DIR / "manifest.json"


def log_manifest(entry_name: str, payload: dict) -> None:
    """Append/overwrite one experiment entry in results/manifest.json."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    manifest[entry_name] = payload
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
