"""
5. Linear, ridge, elastic-net, GPR and RF on all nine features (ENG-06/07)
==========================================================================
Does a simpler regressor (linear, ridge, elastic net) or a Gaussian process
beat the untuned random forest when every model sees the same nine-feature
pool and the same cell-level folds? No feature selection here, so the
regressor comparison is isolated from the selection comparison. Each
regressor is scored by the 5-fold grouped dev CV and once on held-out
batch 3.

Input:   results/features_expanded.parquet (stage 2)
Outputs: results/eng06_07_model_table.csv   one row per regressor in ORDER:
             regressor, cv_rmse, cv_rmse_std, cv_r2, b3_rmse, b3_mae, b3_r2,
             b3_rmse_cellavg, b3_r2_cellavg, fit_seconds
         results/eng06_simple_baselines.json  pool, table (linear, ridge,
             enet, rf rows), details[name] = {cv, heldout}
         results/eng07_gpr.json  pool, kernel (documentation string), table
             (rf, gpr rows), details[name] = {cv, heldout}
Runtime: about 3 min, most of it the GPR. Not resumable: every run refits
all five regressors and overwrites the three outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import assign_folds, clean_feature_table, log_manifest, split_dev_test
from src.features import FEATURE_COLS
from src.harness import REGRESSORS, cv_evaluate, heldout_evaluate

PARQUET = RESULTS_DIR / "features_expanded.parquet"
OUT_CSV = RESULTS_DIR / "eng06_07_model_table.csv"
OUT_ENG06 = RESULTS_DIR / "eng06_simple_baselines.json"
OUT_ENG07 = RESULTS_DIR / "eng07_gpr.json"

ORDER = ["linear", "ridge", "enet", "rf", "gpr"]

# What the GPR is, for the record (the kernel bounds live in src.harness).
GPR_KERNEL_DOC = (
    "ConstantKernel(1.0) * RBF(length_scale=1.0, isotropic) + "
    "WhiteKernel(1e-3); normalize_y=True; StandardScaler inputs; "
    "n_restarts_optimizer=2; random_state=0; train subsampled to 2000 "
    "rows per fit (per-cell stratified, seed 0)"
)


def evaluate(name: str, dev: pd.DataFrame, test: pd.DataFrame,
             fold: pd.Series, pool: list[str]) -> tuple[dict, dict, float]:
    """5-fold grouped CV and the single batch-3 fit of one untuned regressor."""
    spec = REGRESSORS[name]

    def make():  # a fresh untuned estimator for every fit
        return spec["factory"](None)

    t0 = time.time()
    cv = cv_evaluate(dev, pool, fold, make, spec["max_train_rows"])
    ho = heldout_evaluate(dev, test, pool, make, spec["max_train_rows"])
    return cv, ho, time.time() - t0


def table_row(name: str, cv: dict, ho: dict, seconds: float) -> dict:
    """One row of eng06_07_model_table.csv."""
    return {
        "regressor": name,
        "cv_rmse": cv["pooled_rmse_mean"],
        "cv_rmse_std": cv["pooled_rmse_std"],
        "cv_r2": cv["pooled_r2_mean"],
        "b3_rmse": ho["pooled"]["rmse"],
        "b3_mae": ho["pooled"]["mae"],
        "b3_r2": ho["pooled"]["r2"],
        "b3_rmse_cellavg": ho["cell_avg"]["rmse"],
        "b3_r2_cellavg": ho["cell_avg"]["r2"],
        "fit_seconds": round(seconds, 1),
    }


def pick(rows: list[dict], details: dict, names: tuple[str, ...]) -> tuple[list, dict]:
    """The table rows and details of the named regressors, in ORDER."""
    return ([r for r in rows if r["regressor"] in names],
            {k: v for k, v in details.items() if k in names})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()
    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")

    feat = pd.read_parquet(PARQUET)
    clean, _ = clean_feature_table(feat)
    dev, test = split_dev_test(clean)
    fold, _ = assign_folds(dev)
    pool = FEATURE_COLS
    print(f"Pool ({len(pool)}): {pool}\n")

    rows, details = [], {}
    for name in ORDER:
        cv, ho, seconds = evaluate(name, dev, test, fold, pool)
        rows.append(table_row(name, cv, ho, seconds))
        details[name] = {"cv": cv, "heldout": ho}
        print(f"{name:>7}: CV RMSE={cv['pooled_rmse_mean']:.5f} | "
              f"b3 RMSE={ho['pooled']['rmse']:.5f} MAE={ho['pooled']['mae']:.5f} "
              f"R2={ho['pooled']['r2']:.4f} | {seconds:.0f}s")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote results/{OUT_CSV.name}")

    table, det = pick(rows, details, ("linear", "ridge", "enet", "rf"))
    OUT_ENG06.write_text(json.dumps({
        "pool": pool, "table": table, "details": det,
    }, indent=2), encoding="utf-8")
    table, det = pick(rows, details, ("gpr", "rf"))
    OUT_ENG07.write_text(json.dumps({
        "pool": pool, "kernel": GPR_KERNEL_DOC, "table": table, "details": det,
    }, indent=2), encoding="utf-8")
    print(f"Wrote results/{OUT_ENG06.name}, results/{OUT_ENG07.name}")

    log_manifest("eng06_07_model_baselines", {
        "script": "stages/s05_simple_models.py",
        "seed": 0,
        "pool_size": len(pool),
        "b3_rmse": {r["regressor"]: r["b3_rmse"] for r in rows},
    })


if __name__ == "__main__":
    main()
