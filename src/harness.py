"""
The study engine: regressors, feature-ranking methods, grouped CV and the
single held-out evaluation.
========================================================================
One entry point for a selection-method x regressor cell:

    from src.harness import compare

    out = compare(feature_table, regressor="rf",
                  selection_method="rf_importance")
    # out = {ranking, k_curve, selected_features, cv_scores,
    #        heldout_scores, config}

Design rules (fixed by the study protocol):
- All selection decisions use the DEV set only (batches 1+2). Batch 3 is
  touched exactly once per (selection_method x regressor) cell, with the
  final selected feature set.
- Fold structure is the baseline's 5-fold StratifiedGroupKFold (grouped by
  cell, stratified on cycle-life quartiles) — shared across every method so
  comparisons are paired.
- Selection methods return a full ranking (best feature first). The number
  of features k is chosen by grouped-CV pooled RMSE over k = 1..p with the
  same regressor ("auto"), or can be forced with k=<int>.
- Seeds: every stochastic component uses random_state = RF_SEED (0).

Regressors (registry `REGRESSORS`):
- "rf":     RandomForestRegressor(n_estimators=300, random_state=0), untuned.
- "linear": LinearRegression in a StandardScaler pipeline.
- "ridge":  RidgeCV over alphas 10^-3..10^3 (inner CV row-wise on train).
- "enet":   ElasticNetCV (l1_ratio in {.1,.5,.7,.9,.95,1}, inner CV row-wise).
- "gpr":    StandardScaler -> GaussianProcessRegressor with kernel
            C(1.0) * RBF(length_scale=1.0, isotropic) + WhiteKernel(1e-3),
            normalize_y=True, n_restarts_optimizer=2, random_state=0.
            GPR is O(n^3), so training rows are subsampled to
            MAX_GPR_TRAIN_ROWS (2000) stratified per cell with a fixed seed.
            Prediction is on all rows (O(n^2 m) is fine).

Selection methods (registry `SELECTORS`):
- "pearson":     |Pearson r| between each feature and SoH on all dev rows.
- "spearman":    |Spearman rho| (Pearson on ranks) on all dev rows: monotonic
                 rather than strictly linear association (ENG-16).
- "mutual_info": mutual_info_regression on all dev rows (seeded).
- "rf_importance": mean impurity importance of RF(300, seed 0) across the
                   5 CV folds (fit on fold-train only) — this is the
                   convention that produced the interim height ~= 0.72.
- "permutation": mean permutation importance across the 5 folds; the given
                 regressor is fit on fold-train and importances are
                 computed on the fold-validation rows (n_repeats=10).

Tuned hyperparameters (ENG-09) can be injected via `regressor_params` —
they are then held fixed across all selection methods.

`compare_k_grid` (last section) is the ENG-14 replica of `compare` used on
the history rows: an explicit k grid and a fixed cell -> fold map.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import parallel_config
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_prep import (
    RF_SEED,
    assign_folds,
    both_metrics,
    split_dev_test,
)
from src.features_temporal import LEVEL_COLS, is_temporal

MAX_GPR_TRAIN_ROWS = 2000


# ------------------------ regressor registry ------------------------

def _make_rf(params: dict | None = None):
    # n_jobs only parallelises the trees, which are seeded from random_state, so the
    # fitted forest is the same for any thread count; predict adds the trees' outputs
    # in thread order, so a prediction can differ in its last binary digit between runs.
    p = {"n_estimators": 300, "random_state": RF_SEED, "n_jobs": -1}
    p.update(params or {})
    return RandomForestRegressor(**p)


def _make_linear(params: dict | None = None):
    return make_pipeline(StandardScaler(), LinearRegression(**(params or {})))


def _make_ridge(params: dict | None = None):
    p = {"alphas": np.logspace(-3, 3, 13)}
    p.update(params or {})
    return make_pipeline(StandardScaler(), RidgeCV(**p))


def _make_enet(params: dict | None = None):
    p = {
        "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        "alphas": 100,
        "cv": 5,
        "random_state": RF_SEED,
        "max_iter": 5000,
    }
    p.update(params or {})
    return make_pipeline(StandardScaler(), ElasticNetCV(**p))


# Named kernel families so tuned configs stay JSON-serialisable (ENG-09).
# Each is amplitude * stationary-kernel + white noise on standardised inputs.
def _kernel(name: str):
    core = {
        "rbf": lambda: RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)),
        "matern15": lambda: Matern(length_scale=1.0, nu=1.5,
                                   length_scale_bounds=(1e-2, 1e3)),
        "matern25": lambda: Matern(length_scale=1.0, nu=2.5,
                                   length_scale_bounds=(1e-2, 1e3)),
        "rq": lambda: RationalQuadratic(length_scale=1.0, alpha=1.0),
    }[name]()
    return (
        ConstantKernel(1.0, (1e-3, 1e3)) * core
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-7, 1e1))
    )


GPR_KERNEL_NAMES = ["rbf", "matern15", "matern25", "rq"]


def _make_gpr(params: dict | None = None):
    # Kernel: amplitude * isotropic stationary kernel + white noise, on
    # standardised inputs. Hyperparameters (length scale, amplitude, noise)
    # are set by marginal-likelihood optimisation at fit time; "kernel_name"
    # selects the family (JSON-friendly for tuned configs).
    params = dict(params or {})
    kernel_name = params.pop("kernel_name", "rbf")
    p = {
        "kernel": _kernel(kernel_name),
        "normalize_y": True,
        "n_restarts_optimizer": 2,
        "random_state": RF_SEED,
    }
    p.update(params)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(**p))



REGRESSORS: dict[str, dict] = {
    "rf": {"factory": _make_rf, "max_train_rows": None},
    "linear": {"factory": _make_linear, "max_train_rows": None},
    "ridge": {"factory": _make_ridge, "max_train_rows": None},
    "enet": {"factory": _make_enet, "max_train_rows": None},
    "gpr": {"factory": _make_gpr, "max_train_rows": MAX_GPR_TRAIN_ROWS},
}



def _subsample_train(train: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Deterministic per-cell stratified subsample to cap training size."""
    if max_rows is None or len(train) <= max_rows:
        return train
    frac = max_rows / len(train)
    parts = [
        g.sample(n=max(1, int(round(len(g) * frac))), random_state=RF_SEED)
        for _, g in train.groupby("cell_id")
    ]
    return pd.concat(parts).reset_index(drop=True)


# ------------------------ selection registry ------------------------

def _rank_from_scores(features: list[str], scores: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"feature": features, "score": scores})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# Cap on the number of validation rows scored during permutation importance.
# Permutation importance re-predicts n_features * n_repeats times per fold;
# for an O(n^2) predictor like GPR the full ~12k-row fold is far too slow, so
# the scoring set is subsampled (per-cell stratified, seed 0) when the
# regressor declares a max_train_rows.
MAX_PERM_EVAL_ROWS = 3000


def _sel_pearson(dev, features, fold, make_regressor, max_train_rows=None):
    scores = np.array(
        [abs(pearsonr(dev[f], dev["SOH"])[0]) for f in features]
    )
    return _rank_from_scores(features, scores)


def _sel_spearman(dev, features, fold, make_regressor, max_train_rows=None):
    # Pearson on ranks: scores a monotonic feature-SoH relationship even when
    # it is curved, and is insensitive to outliers. Still univariate.
    scores = np.array(
        [abs(spearmanr(dev[f], dev["SOH"])[0]) for f in features]
    )
    return _rank_from_scores(features, scores)


def _sel_mutual_info(dev, features, fold, make_regressor, max_train_rows=None):
    scores = mutual_info_regression(
        dev[features], dev["SOH"], random_state=RF_SEED
    )
    return _rank_from_scores(features, list(scores))


def _sel_rf_importance(dev, features, fold, make_regressor, max_train_rows=None):
    # Mean impurity importance across CV folds (fit on fold-train only).
    imps = []
    for f in sorted(fold.unique()):
        tr = dev[fold != f]
        rf = _make_rf()
        rf.fit(tr[features], tr["SOH"])
        imps.append(rf.feature_importances_)
    return _rank_from_scores(features, np.mean(imps, axis=0))


def _sel_permutation(dev, features, fold, make_regressor, max_train_rows=None):
    # Mean permutation importance across folds, evaluated on fold-val rows
    # with the SAME regressor family being compared. Training rows are
    # subsampled to the regressor's cap (e.g. GPR's 2000) before fitting, and
    # the scoring set is capped too so an O(n^2) predictor stays tractable.
    imps = []
    for f in sorted(fold.unique()):
        tr, va = dev[fold != f], dev[fold == f]
        tr = _subsample_train(tr, max_train_rows)
        if max_train_rows is not None:
            va = _subsample_train(va, MAX_PERM_EVAL_ROWS)
        est = make_regressor()
        est.fit(tr[features], tr["SOH"])
        pi = permutation_importance(
            est, va[features], va["SOH"],
            n_repeats=10, random_state=RF_SEED,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        imps.append(pi.importances_mean)
    return _rank_from_scores(features, np.mean(imps, axis=0))


SELECTORS = {
    "pearson": _sel_pearson,
    "spearman": _sel_spearman,
    "mutual_info": _sel_mutual_info,
    "rf_importance": _sel_rf_importance,
    "permutation": _sel_permutation,
}


# ------------------------ evaluation ------------------------

def cv_evaluate(
    dev: pd.DataFrame,
    features: list[str],
    fold: pd.Series,
    make_regressor,
    max_train_rows: int | None = None,
) -> dict:
    """Grouped-CV evaluation of one feature subset. Returns mean/std metrics."""
    per_fold = []
    for f in sorted(fold.unique()):
        tr, va = dev[fold != f], dev[fold == f]
        tr = _subsample_train(tr, max_train_rows)
        est = make_regressor()
        est.fit(tr[features], tr["SOH"])
        pred = est.predict(va[features])
        per_fold.append(both_metrics(va["cell_id"], va["SOH"], pred))
    out = {"per_fold": per_fold}
    for scope in ("pooled", "cell_avg"):
        for k in ("rmse", "mae", "r2"):
            vals = [m[scope][k] for m in per_fold]
            out[f"{scope}_{k}_mean"] = float(np.mean(vals))
            out[f"{scope}_{k}_std"] = float(np.std(vals))
    return out


def heldout_evaluate(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    make_regressor,
    max_train_rows: int | None = None,
) -> dict:
    """Refit on all dev rows, single predict on held-out batch 3."""
    tr = _subsample_train(dev, max_train_rows)
    est = make_regressor()
    est.fit(tr[features], tr["SOH"])
    pred = est.predict(test[features])
    return both_metrics(test["cell_id"], test["SOH"], pred)


# ------------------------ main entry point ------------------------

def compare(
    feature_table: pd.DataFrame,
    regressor: str = "rf",
    selection_method: str = "rf_importance",
    *,
    features: list[str] | None = None,
    k: int | str = "auto",
    regressor_params: dict | None = None,
    evaluate_heldout: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run one (selection_method x regressor) comparison cell.

    Parameters
    ----------
    feature_table : cleaned feature table (already passed through
        `clean_feature_table`) with cell_id/batch/SOH/cycle_life columns.
    regressor : key into REGRESSORS.
    selection_method : key into SELECTORS.
    features : candidate pool; defaults to every numeric column that is not
        an id/target column.
    k : "auto" (choose by CV pooled RMSE) or an int to force the subset size.
    regressor_params : optional tuned hyperparameters (ENG-09), held fixed.
    evaluate_heldout : set False to keep batch 3 untouched (e.g. during
        tuning experiments).

    Returns
    -------
    dict with keys: ranking (DataFrame -> list of dicts), k_curve,
    selected_features, cv_scores, heldout_scores, config.
    """
    if features is None:
        exclude = {"cycle", "SOH", "cycle_life", "fold"}
        features = [
            c for c in feature_table.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(feature_table[c])
        ]
    spec = REGRESSORS[regressor]
    make_regressor = lambda: spec["factory"](regressor_params)  # noqa: E731
    max_rows = spec["max_train_rows"]

    dev, test = split_dev_test(feature_table)
    fold, _ = assign_folds(dev)

    # ---- 1. rank features on dev only ----
    ranking = SELECTORS[selection_method](
        dev, features, fold, make_regressor, max_rows
    )
    ranked_feats = ranking["feature"].tolist()
    if verbose:
        print(f"[{selection_method} x {regressor}] ranking: "
              + ", ".join(f"{r.feature}={r.score:.4f}"
                          for r in ranking.itertuples()))

    # ---- 2. choose k by grouped-CV RMSE over nested subsets ----
    k_curve = []
    for kk in range(1, len(ranked_feats) + 1):
        sub = ranked_feats[:kk]
        cv = cv_evaluate(dev, sub, fold, make_regressor, max_rows)
        k_curve.append({
            "k": kk,
            "features": sub,
            "cv_pooled_rmse_mean": cv["pooled_rmse_mean"],
            "cv_pooled_rmse_std": cv["pooled_rmse_std"],
            "cv_pooled_r2_mean": cv["pooled_r2_mean"],
            "cv_cell_avg_rmse_mean": cv["cell_avg_rmse_mean"],
            "cv_cell_avg_r2_mean": cv["cell_avg_r2_mean"],
        })
        if verbose:
            print(f"  k={kk}: CV RMSE={cv['pooled_rmse_mean']:.5f} "
                  f"+/- {cv['pooled_rmse_std']:.5f}")

    if k == "auto":
        best = min(k_curve, key=lambda r: r["cv_pooled_rmse_mean"])
    else:
        best = k_curve[int(k) - 1]
    selected = best["features"]

    # ---- 3. full CV metrics at the chosen k ----
    cv_scores = cv_evaluate(dev, selected, fold, make_regressor, max_rows)

    # ---- 4. held-out batch 3, touched once ----
    heldout_scores = (
        heldout_evaluate(dev, test, selected, make_regressor, max_rows)
        if evaluate_heldout
        else None
    )
    if verbose and heldout_scores:
        print(f"  selected k={len(selected)} {selected}; "
              f"batch-3 pooled RMSE={heldout_scores['pooled']['rmse']:.5f} "
              f"R2={heldout_scores['pooled']['r2']:.4f}")

    return {
        "ranking": ranking.to_dict(orient="records"),
        "k_curve": k_curve,
        "selected_features": selected,
        "cv_scores": cv_scores,
        "heldout_scores": heldout_scores,
        "config": {
            "regressor": regressor,
            "selection_method": selection_method,
            "regressor_params": regressor_params or {},
            "k_mode": k,
            "pool": features,
            "seed": RF_SEED,
            "max_train_rows": max_rows,
        },
    }


# ------------- k-grid replica of compare with a fixed cell->fold map (ENG-14) -------------
#
# The history-row comparisons (stages 8, 9 and 17) run compare()'s protocol
# with two documented differences:
#
# 1. Folds. compare() calls assign_folds on whatever dev rows it gets.
#    StratifiedGroupKFold's greedy assignment depends on per-cell row counts,
#    so recomputing it on the history-filtered rows moves 62 of the 84 dev
#    cells to a different fold. To keep the SAME folds as ENG-08, the
#    cell -> fold map is computed once on the full clean dev set (exactly
#    ENG-08's call) and passed in as `fold_by_cell`.
# 2. The k curve. compare() scores every k = 1..p; for p = 54 / 63 that is
#    ~30 h of GPR fits, so the k curve is scored on
#        k_grid(p) = {1..9} u {10, 12, 14, 16, 20, 27, 36, 48} u {p}  (p > 9)
#        k_grid(p) = {1..p}                                            (p <= 9)
#    so the level pool follows exactly the ENG-08 k rule. Rankings do not
#    depend on the k grid; only the chosen k does.
#
# Execution backend (no effect on numbers): `_run_cell` runs a cell under
# joblib's threading backend. The permutation selector calls
# permutation_importance(n_jobs=-1), and joblib's default process backend
# (loky) pickles the fitted regressor to every worker; with p = 54 the
# 300-tree forest is ~2 GB and the worker pool hung with no progress for 9 h.
# RF already fits and predicts with threads, GPR does not use joblib, and
# permutation importance is seeded per column, so the results are
# backend-independent (stage 8, check 9).

# FULL_K_MAX = 9 keeps the level pool on exactly the ENG-08 rule.
FULL_K_MAX = 9
K_COARSE = [10, 12, 14, 16, 20, 27, 36, 48]


def k_grid(p: int, mode: str = "coarse") -> list[int]:
    """The k values the CV curve is scored on for a pool of p columns (see above)."""
    if mode == "full" or p <= FULL_K_MAX:
        return list(range(1, p + 1))
    ks = set(range(1, FULL_K_MAX + 1)) | {k for k in K_COARSE if k <= p} | {p}
    return sorted(ks)


def compare_k_grid(
    feature_table: pd.DataFrame,
    regressor: str,
    selection_method: str,
    *,
    features: list[str],
    k_grid: list[int],
    regressor_params: dict | None = None,
    fold_by_cell: dict[str, int] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Line-for-line replica of compare(k="auto") that scores the CV k-curve on
    an explicit `k_grid` instead of every k = 1..p, and can take a fixed
    cell -> fold map (`fold_by_cell`) instead of recomputing assign_folds on
    the given rows. With k_grid == range(1, p+1) and fold_by_cell == None it
    returns the same numbers as compare() (stages/s08_history_features.py,
    check 8). Selection, subsampling and the single held-out evaluation are
    the harness's own.
    """
    spec = REGRESSORS[regressor]
    make_regressor = lambda: spec["factory"](regressor_params)  # noqa: E731
    max_rows = spec["max_train_rows"]

    dev, test = split_dev_test(feature_table)
    if fold_by_cell is None:
        fold, _ = assign_folds(dev)
        fold_source = "assign_folds(dev rows given)"
    else:
        mapped = dev["cell_id"].map(fold_by_cell)
        if mapped.isna().any():
            raise KeyError("fold_by_cell lacks some dev cells")
        fold = pd.Series(mapped.to_numpy(dtype=int), index=dev.index, name="fold")
        fold_source = "fold_by_cell (ENG-08 full clean dev set)"

    # ---- 1. rank features on dev only ----
    ranking = SELECTORS[selection_method](
        dev, features, fold, make_regressor, max_rows
    )
    ranked_feats = ranking["feature"].tolist()
    if verbose:
        print(f"[{selection_method} x {regressor}] ranking: "
              + ", ".join(f"{r.feature}={r.score:.4f}"
                          for r in ranking.itertuples()), flush=True)

    # ---- 2. choose k by grouped-CV RMSE over nested subsets (k grid) ----
    k_curve = []
    for kk in k_grid:
        sub = ranked_feats[:kk]
        cv = cv_evaluate(dev, sub, fold, make_regressor, max_rows)
        k_curve.append({
            "k": kk,
            "features": sub,
            "cv_pooled_rmse_mean": cv["pooled_rmse_mean"],
            "cv_pooled_rmse_std": cv["pooled_rmse_std"],
            "cv_pooled_r2_mean": cv["pooled_r2_mean"],
            "cv_cell_avg_rmse_mean": cv["cell_avg_rmse_mean"],
            "cv_cell_avg_r2_mean": cv["cell_avg_r2_mean"],
        })
        if verbose:
            print(f"  k={kk}: CV RMSE={cv['pooled_rmse_mean']:.5f} "
                  f"+/- {cv['pooled_rmse_std']:.5f}", flush=True)

    best = min(k_curve, key=lambda r: r["cv_pooled_rmse_mean"])
    selected = best["features"]

    # ---- 3. full CV metrics at the chosen k ----
    cv_scores = cv_evaluate(dev, selected, fold, make_regressor, max_rows)

    # ---- 4. held-out batch 3, touched once ----
    heldout_scores = heldout_evaluate(dev, test, selected, make_regressor, max_rows)
    if verbose:
        print(f"  selected k={len(selected)} {selected}; "
              f"batch-3 pooled RMSE={heldout_scores['pooled']['rmse']:.5f} "
              f"R2={heldout_scores['pooled']['r2']:.4f}", flush=True)

    return {
        "ranking": ranking.to_dict(orient="records"),
        "k_curve": k_curve,
        "selected_features": selected,
        "cv_scores": cv_scores,
        "heldout_scores": heldout_scores,
        "config": {
            "regressor": regressor,
            "selection_method": selection_method,
            "regressor_params": regressor_params or {},
            "k_mode": "auto",
            "k_grid": list(k_grid),
            "fold_source": fold_source,
            "pool": features,
            "seed": RF_SEED,
            "max_train_rows": max_rows,
        },
    }


def _run_cell(table, reg, method, cols, grid, params, folds_full, verbose):
    """One grid cell (module-level so a process pool can pickle it)."""
    t0 = time.time()
    with parallel_config(backend="threading"):   # see the section comment
        out = compare_k_grid(
            table, regressor=reg, selection_method=method,
            features=cols, k_grid=grid, regressor_params=params,
            fold_by_cell=folds_full, verbose=verbose,
        )
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def build_pools(temporal_cols: list[str]) -> dict[str, list[str]]:
    """The three ENG-14 pools on the history rows: level (9), temporal (54), combined (63)."""
    return {
        "level": list(LEVEL_COLS),
        "temporal": list(temporal_cols),
        "combined": list(LEVEL_COLS) + list(temporal_cols),
    }


def temporal_share(ranking: list[dict], selected: list[str]) -> dict:
    """Temporal-vs-level composition of the selected set + best temporal."""
    order = [r["feature"] for r in ranking]
    n_temp = sum(is_temporal(f) for f in selected)
    best_t = next((f for f in order if is_temporal(f)), None)
    best_l = next((f for f in order if not is_temporal(f)), None)
    return {
        "n_selected": len(selected),
        "n_temporal_selected": int(n_temp),
        "frac_temporal_selected": (n_temp / len(selected)) if selected else float("nan"),
        "best_temporal_feature": best_t,
        "best_temporal_rank": (order.index(best_t) + 1) if best_t else None,
        "best_level_feature": best_l,
        "best_level_rank": (order.index(best_l) + 1) if best_l else None,
        "top1_is_temporal": bool(is_temporal(order[0])) if order else None,
    }
