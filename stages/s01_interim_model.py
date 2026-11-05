"""
1. Interim model: three peak features and a random forest (ENG-00)
===================================================================
Does the interim (semester-1) model reproduce from the raw data? Three
features of the dominant discharge dQ/dV peak (height, position, area), a
300-tree random forest, 5-fold grouped CV on the development batches
(b1+b2) and one refit scored on the held-out batch 3.
Inputs:  the three Severson pickles in PKL_DIR (~8 GB RAM to load), or with
         --cached the stored results/features_3feat.parquet.
Outputs: results/features_3feat.parquet (one row per recorded cycle of the
         124 cells: cell_id, batch, charge_policy, cycle, height, position,
         area, SOH, status, cycle_life) and results/eng00_baseline.json
         (verdict, features, model, clean_log, cv{per_fold, pooled_summary,
         importances_cv_mean}, batch3_ood, cell_to_fold: the cell -> fold
         map; later stages recompute the same map with assign_folds).
Acceptance: batch-3 pooled R2 within 0.83 +/- 0.01 and RMSE within
0.0168 +/- 0.0005 (the interim report's numbers); exits 1 on FAIL.
Runtime: minutes, mostly loading the pickles; not resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import PKL_DIR, RESULTS_DIR
from src.data_prep import (
    BASELINE_FEATURES,
    RF_SEED,
    assign_folds,
    both_metrics,
    clean_feature_table,
    log_manifest,
    split_dev_test,
)
from src.peak_extraction import extract_features
from src.severson_loader import load_clean_severson, summarise_cell

FEATURES_PARQUET = RESULTS_DIR / "features_3feat.parquet"
OUT_JSON = RESULTS_DIR / "eng00_baseline.json"

COLUMNS = ["cell_id", "batch", "charge_policy", "cycle",
           "height", "position", "area", "SOH", "status", "cycle_life"]

# Acceptance targets: the interim report's batch-3 numbers.
TARGET_R2, R2_TOL = 0.83, 0.01
TARGET_RMSE, RMSE_TOL = 0.0168, 0.0005


def build_feature_table() -> pd.DataFrame:
    """Load the 124 cleaned cells and extract the three features of every cycle."""
    print("Loading the cleaned Severson dataset (all 3 batches, ~8 GB RAM) ...")
    t0 = time.time()
    cells = load_clean_severson(PKL_DIR, verbose=True)
    counts = pd.Series([cid[:2] for cid in cells]).value_counts().to_dict()
    assert (counts["b1"], counts["b2"], counts["b3"]) == (41, 43, 40), counts
    print(f"  loaded in {time.time() - t0:.0f} s")

    print("Extracting height/position/area for every cycle ...")
    t0 = time.time()
    parts = []
    for cid, cell in cells.items():
        df = extract_features(cell, cell_id=cid, verbose=False)
        meta = summarise_cell(cell)
        df.insert(0, "cell_id", cid)
        df.insert(1, "batch", cid[:2])
        df.insert(2, "charge_policy", meta["policy"])
        df["cycle_life"] = meta["cycle_life"]
        parts.append(df)
    feat = pd.concat(parts, ignore_index=True)[COLUMNS]
    print(f"  extracted {len(feat)} cycle-rows from {len(cells)} cells "
          f"in {time.time() - t0:.0f} s")
    return feat


def load_feature_table(cached: bool) -> pd.DataFrame:
    """The feature table: the stored parquet with --cached, else re-extracted and stored."""
    if cached:
        if not FEATURES_PARQUET.exists():
            sys.exit(f"--cached: {FEATURES_PARQUET} not found; run without --cached first")
        print(f"Using the stored feature table: {FEATURES_PARQUET}")
        return pd.read_parquet(FEATURES_PARQUET)
    feat = build_feature_table()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(FEATURES_PARQUET, index=False)
    print(f"  wrote {FEATURES_PARQUET}")
    return feat


def make_rf() -> RandomForestRegressor:
    """The interim model: 300 trees, the shared seed, scikit-learn defaults otherwise."""
    return RandomForestRegressor(n_estimators=300, random_state=RF_SEED)


def cross_validate(dev: pd.DataFrame, fold: pd.Series, x_cols: list[str]) -> list[dict]:
    """One RF per fold on the development rows; pooled and cell-averaged metrics per fold."""
    print("\n5-fold StratifiedGroupKFold CV on dev (b1+b2):")
    rows = []
    for f in sorted(fold.unique()):
        tr, va = dev[fold != f], dev[fold == f]
        rf = make_rf()
        rf.fit(tr[x_cols], tr["SOH"])
        m = both_metrics(va["cell_id"], va["SOH"], rf.predict(va[x_cols]))
        rows.append({"fold": int(f), **m,
                     "importances": dict(zip(x_cols, rf.feature_importances_))})
        print(f"  fold {f}: pooled R2={m['pooled']['r2']:.4f} "
              f"RMSE={m['pooled']['rmse']:.5f} MAE={m['pooled']['mae']:.5f}")
    return rows


def summarise_cv(fold_rows: list[dict], x_cols: list[str]) -> tuple[dict, dict]:
    """Mean and std (ddof=0) of the pooled metrics over the folds; mean importances."""
    pooled = {
        k: {"mean": float(np.mean([r["pooled"][k] for r in fold_rows])),
            "std": float(np.std([r["pooled"][k] for r in fold_rows]))}
        for k in ("rmse", "mae", "r2")
    }
    importances = {c: float(np.mean([r["importances"][c] for r in fold_rows]))
                   for c in x_cols}
    print(f"  CV pooled mean: R2={pooled['r2']['mean']:.4f} "
          f"+/- {pooled['r2']['std']:.4f}, RMSE={pooled['rmse']['mean']:.5f}")
    print("  importances (CV mean): "
          + ", ".join(f"{c}={importances[c]:.3f}" for c in x_cols))
    return pooled, importances


def score_held_out(dev: pd.DataFrame, test: pd.DataFrame, x_cols: list[str]) -> dict:
    """One refit on all development rows, scored once on batch 3."""
    rf = make_rf()
    rf.fit(dev[x_cols], dev["SOH"])
    ood = both_metrics(test["cell_id"], test["SOH"], rf.predict(test[x_cols]))
    print("\nBatch-3 held-out (single touch):")
    for kind in ("pooled", "cell_avg"):
        m = ood[kind]
        print(f"  {kind + ':':9s} R2={m['r2']:.4f} RMSE={m['rmse']:.5f} MAE={m['mae']:.5f}")
    return ood


def acceptance(ood: dict) -> str:
    """PASS when the batch-3 pooled R2 and RMSE are within the interim report's bands."""
    r2, rmse = ood["pooled"]["r2"], ood["pooled"]["rmse"]
    r2_ok = abs(r2 - TARGET_R2) <= R2_TOL
    rmse_ok = abs(rmse - TARGET_RMSE) <= RMSE_TOL
    verdict = "PASS" if (r2_ok and rmse_ok) else "FAIL"
    print(f"\nACCEPTANCE [{verdict}]: "
          f"R2 {r2:.4f} (target {TARGET_R2}+/-{R2_TOL}) {'OK' if r2_ok else 'OUT OF BAND'}; "
          f"RMSE {rmse:.5f} (target ~{TARGET_RMSE}) {'OK' if rmse_ok else 'OUT OF BAND'}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cached", action="store_true",
                    help="reuse results/features_3feat.parquet instead of re-extracting")
    args = ap.parse_args()

    feat = load_feature_table(args.cached)
    clean, clean_log = clean_feature_table(feat)
    dev, test = split_dev_test(clean)
    print(f"  dev  (b1+b2): {len(dev):6d} rows, {dev['cell_id'].nunique()} cells")
    print(f"  test (b3):    {len(test):6d} rows, {test['cell_id'].nunique()} cells")
    fold, cell_to_fold = assign_folds(dev)

    x_cols = BASELINE_FEATURES  # column order matters to the forest: keep it
    fold_rows = cross_validate(dev, fold, x_cols)
    cv_pooled, imp_mean = summarise_cv(fold_rows, x_cols)
    ood = score_held_out(dev, test, x_cols)
    verdict = acceptance(ood)

    payload = {
        "verdict": verdict,
        "features": x_cols,
        "model": {"type": "RandomForestRegressor", "n_estimators": 300,
                  "random_state": RF_SEED},
        "clean_log": clean_log,
        "cv": {"per_fold": fold_rows, "pooled_summary": cv_pooled,
               "importances_cv_mean": imp_mean},
        "batch3_ood": ood,
        "cell_to_fold": cell_to_fold,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    log_manifest("eng00_baseline", {
        "script": "stages/s01_interim_model.py",
        "seed": RF_SEED,
        "features": x_cols,
        "cv_pooled_r2_mean": cv_pooled["r2"]["mean"],
        "batch3_pooled": ood["pooled"],
        "verdict": verdict,
    })

    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
