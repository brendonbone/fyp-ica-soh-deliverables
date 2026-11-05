"""
Nested-CV tuning of the RF and the GPR (ENG-09)
===============================================
Which hyperparameters should the random forest and the GPR use for the rest
of the study?  One fixed configuration per regressor is chosen on the
development set (batches 1+2) by nested cross-validation, so the choice never
sees batch 3 and is not entangled with any ranking method.  Outer loop: the
study's 5-fold StratifiedGroupKFold; inner loop: GroupKFold(4) by cell on the
outer-train rows.  The best inner config of each outer fold is scored on the
outer validation fold (the unbiased "nested" estimate of the tuning
procedure); the single FIXED config is the one with the best mean inner RMSE
across the five outer folds.  Grids: RF (300 trees) max_depth {None, 15} x
min_samples_leaf {1, 10, 30} x max_features {1.0, 0.5}; GPR kernel family
{rbf, matern15, matern25, rq}.  Both are then refit on the dev rows (the
GPR, here and in every fit above, on a 2000-row per-cell subsample, seed 0)
and scored once on batch 3, untuned and tuned, on the full nine-feature pool.

Input:  results/features_expanded.parquet (stage 2).
Output: results/eng09_tuning.json - pool, protocol, rf / gpr {grid,
        inner_rmse_matrix (5 outer x n configs), nested_outer_estimates,
        nested_rmse_mean, nested_r2_mean, fixed_config}, before_after_heldout
        {rf_untuned, rf_tuned, gpr_untuned, gpr_tuned: pooled + cell_avg}.
        results/eng09_tuned_params.json - {rf: {...}, gpr: {kernel_name}};
        every later stage reads its tuned parameters from this file.
Runtime: about 1.5 h on a laptop (the GPR configs dominate).  Not resumable:
        both grids and the before/after block are recomputed in one run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import RESULTS_DIR
from src.data_prep import (
    assign_folds,
    clean_feature_table,
    log_manifest,
    pooled_metrics,
    split_dev_test,
)
from src.features import FEATURE_COLS
from src.harness import GPR_KERNEL_NAMES, REGRESSORS, _subsample_train, heldout_evaluate

PARQUET = RESULTS_DIR / "features_expanded.parquet"
OUT_JSON = RESULTS_DIR / "eng09_tuning.json"
OUT_PARAMS = RESULTS_DIR / "eng09_tuned_params.json"

RF_GRID = [
    {"max_depth": d, "min_samples_leaf": l, "max_features": f}
    for d, l, f in itertools.product([None, 15], [1, 10, 30], [1.0, 0.5])
]
GPR_GRID = [{"kernel_name": k} for k in GPR_KERNEL_NAMES]

N_INNER = 4


def inner_cv_rmse(train: pd.DataFrame, params: dict, reg: str,
                  pool: list[str]) -> float:
    """Mean pooled RMSE of one config over an inner GroupKFold on `train`."""
    spec = REGRESSORS[reg]
    gkf = GroupKFold(n_splits=N_INNER)
    scores = []
    for tr_idx, va_idx in gkf.split(train, groups=train["cell_id"].values):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        tr = _subsample_train(tr, spec["max_train_rows"])
        est = spec["factory"](params)
        est.fit(tr[pool], tr["SOH"])
        scores.append(pooled_metrics(va["SOH"], est.predict(va[pool]))["rmse"])
    return float(np.mean(scores))


def tune(reg: str, grid: list[dict], dev: pd.DataFrame, fold: pd.Series,
         pool: list[str]) -> dict:
    """Nested CV of one regressor over its grid: the stored block incl. fixed_config."""
    print(f"\n===== tuning {reg} ({len(grid)} configs) =====", flush=True)
    spec = REGRESSORS[reg]
    # Rows are indexed by the fold label directly (labels are 0..4).
    inner_scores = np.zeros((len(fold.unique()), len(grid)))
    nested = []
    for f in sorted(fold.unique()):
        outer_tr, outer_va = dev[fold != f], dev[fold == f]
        for gi, params in enumerate(grid):
            t0 = time.time()
            s = inner_cv_rmse(outer_tr, params, reg, pool)
            inner_scores[f, gi] = s
            print(f"  outer {f} cfg {gi} {params}: inner RMSE={s:.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        best_gi = int(np.argmin(inner_scores[f]))
        est = spec["factory"](grid[best_gi])
        tr = _subsample_train(outer_tr, spec["max_train_rows"])
        est.fit(tr[pool], tr["SOH"])
        m = pooled_metrics(outer_va["SOH"], est.predict(outer_va[pool]))
        nested.append({"outer_fold": int(f), "chosen": grid[best_gi], **m})
        print(f"  outer {f}: chose {grid[best_gi]} -> "
              f"outer RMSE={m['rmse']:.5f}", flush=True)

    # The fixed config is the most stable winner (best mean inner RMSE), not
    # the config of the best outer fold.
    mean_inner = inner_scores.mean(axis=0)
    fixed = grid[int(np.argmin(mean_inner))]
    block = {
        "grid": grid,
        "inner_rmse_matrix": inner_scores.round(6).tolist(),
        "nested_outer_estimates": nested,
        "nested_rmse_mean": float(np.mean([n["rmse"] for n in nested])),
        "nested_r2_mean": float(np.mean([n["r2"] for n in nested])),
        "fixed_config": fixed,
    }
    print(f"FIXED {reg} config: {fixed} "
          f"(nested CV RMSE {block['nested_rmse_mean']:.5f})", flush=True)
    return block


def before_after(dev: pd.DataFrame, test: pd.DataFrame, pool: list[str],
                 tuned_params: dict) -> dict:
    """Held-out batch-3 metrics of each regressor, untuned and tuned, on the pool."""
    print("\n===== before/after =====", flush=True)
    ba = {}
    for reg in ("rf", "gpr"):
        spec = REGRESSORS[reg]
        for label, params in (("untuned", None), ("tuned", tuned_params[reg])):
            ho = heldout_evaluate(dev, test, pool, partial(spec["factory"], params),
                                  spec["max_train_rows"])
            ba[f"{reg}_{label}"] = ho
            print(f"  {reg} {label}: b3 RMSE={ho['pooled']['rmse']:.5f} "
                  f"R2={ho['pooled']['r2']:.4f}", flush=True)
    return ba


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()

    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")
    clean, _ = clean_feature_table(pd.read_parquet(PARQUET))
    dev, test = split_dev_test(clean)
    fold, _ = assign_folds(dev)
    pool = FEATURE_COLS

    out: dict = {"pool": pool, "protocol": {
        "outer": "study 5-fold StratifiedGroupKFold",
        "inner": f"GroupKFold({N_INNER}) by cell",
        "fixed_config_rule": "best mean inner RMSE across outer folds",
    }}
    tuned_params: dict = {}
    for reg, grid in (("rf", RF_GRID), ("gpr", GPR_GRID)):
        out[reg] = tune(reg, grid, dev, fold, pool)
        tuned_params[reg] = out[reg]["fixed_config"]
    out["before_after_heldout"] = before_after(dev, test, pool, tuned_params)

    # The params file is written last: run_results.py treats it as the
    # marker that the whole stage finished.
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_PARAMS.write_text(json.dumps(tuned_params, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON.name}, {OUT_PARAMS.name}")

    log_manifest("eng09_tuning", {
        "script": "stages/s06_tune_models.py",
        "seed": 0,
        "tuned_params": tuned_params,
        "nested_rmse": {r: out[r]["nested_rmse_mean"] for r in ("rf", "gpr")},
    })


if __name__ == "__main__":
    main()
