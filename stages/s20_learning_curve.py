"""
20. Learning curve: dev-CV R2 against the number of training cells (ENG-10)
===========================================================================
Is the development set big enough? For each of the five development folds
(validation fold fixed), n training cells are drawn at random from the other
four folds, n in 10, 20, ..., 60, 67, five seeded repeats each; the untuned RF
on the nine level features is fitted on those cells and scored on the
validation fold. Batch 3 is never used. The plateau is the smallest n whose
mean R2 is within one std of the largest mean R2.

Folds 0-2 have 68 other-fold cells and folds 3-4 have 66, so the top grid
point gives two rows: n = 66 (folds 3-4, 10 fits) and n = 67 (folds 0-2, 15 fits).

Inputs   results/features_expanded.parquet (stage 2).
Outputs  results/eng10_learning_curve.csv: one row per fit (fold, n_cells,
         repeat, rmse, mae, r2; 175 rows).
         results/eng10_learning_curve.json: grid, n_repeats, aggregate (per
         n_cells: r2_mean, r2_std, rmse_mean, rmse_std, std with ddof 1,
         rounded to 6 decimals), plateau_n_cells.
Runtime  about 9 min on a 24-core laptop (175 forests); not resumable.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import log_manifest, pooled_metrics
from src.features_temporal import LEVEL_COLS
from src.subset_search import load_level_rows, rf_factory

OUT_CSV = RESULTS_DIR / "eng10_learning_curve.csv"
OUT_JSON = RESULTS_DIR / "eng10_learning_curve.json"
N_CELLS_GRID = [10, 20, 30, 40, 50, 60, 67]
N_REPEATS = 5


def fit_all(dev: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """One record per (validation fold, number of training cells, repeat)."""
    pool = list(LEVEL_COLS)
    make = rf_factory(None)
    records = []
    t0 = time.time()
    for f in sorted(fold.unique()):
        va = dev[fold == f]
        train_pool = dev[fold != f]
        pool_cells = np.array(sorted(train_pool["cell_id"].unique()))
        for n in N_CELLS_GRID:
            n_eff = min(n, len(pool_cells))
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(1000 * f + 10 * n + rep)
                chosen = rng.choice(pool_cells, size=n_eff, replace=False)
                tr = train_pool[train_pool["cell_id"].isin(chosen)]
                est = make()
                est.fit(tr[pool], tr["SOH"])
                m = pooled_metrics(va["SOH"], est.predict(va[pool]))
                records.append({"fold": int(f), "n_cells": n_eff, "repeat": rep, **m})
        print(f"  fold {f} done ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Mean and std (ddof 1) of R2 and RMSE per number of cells, and the plateau."""
    agg = df.groupby("n_cells").agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    ).reset_index()
    best = agg["r2_mean"].max()
    band = agg.loc[agg["r2_mean"].idxmax(), "r2_std"]
    plateau_n = int(agg.loc[agg["r2_mean"] >= best - band, "n_cells"].min())
    return agg, plateau_n


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    dev, _, fold = load_level_rows()
    df = fit_all(dev, fold)
    df.to_csv(OUT_CSV, index=False)

    agg, plateau_n = aggregate(df)
    print(agg.to_string(index=False))
    print(f"plateau: n = {plateau_n} cells", flush=True)

    OUT_JSON.write_text(json.dumps({
        "grid": N_CELLS_GRID, "n_repeats": N_REPEATS,
        "aggregate": agg.round(6).to_dict(orient="records"),
        "plateau_n_cells": plateau_n,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT_CSV.name} and {OUT_JSON.name}")

    log_manifest("eng10_learning_curve", {
        "script": "stages/s20_learning_curve.py",
        "plateau_n_cells": plateau_n,
        "max_r2_mean": float(agg["r2_mean"].max()),
    })


if __name__ == "__main__":
    main()
