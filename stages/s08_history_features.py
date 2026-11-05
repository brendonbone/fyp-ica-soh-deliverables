"""
8. 20-cycle history of each feature: table, checks, pools (ENG-14)
==================================================================
Does a feature's trajectory over the preceding cycles outrank its current level and lower
the held-out error? From features_expanded.parquet: (1) nine checks of the history columns
-> eng14_validation.json, exit 1 if one fails; (2) the rows with >= 20 cycles of history ->
features_history.parquet; (3) ENG-08's selector x regressor grid on the level, temporal and
combined pools of those rows -> eng14_full{,_tuned}.json, eng14_ranking_table{,_tuned}.csv.
Default cells: untuned, four selectors x RF; tuned (needs eng09_tuned_params.json), pearson
x RF + permutation x RF on combined; --regressors rf gpr gives all 24. Runtime ~4 h (a GPR
cell 0.5-2 h); resumable; a run that omits a default cell writes *_smoke files instead.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_config
from scipy.stats import kendalltau

from src.config import RESULTS_DIR
from src.data_prep import (
    BASELINE_FEATURES,
    RF_SEED,
    assign_folds,
    clean_feature_table,
    log_manifest,
    split_dev_test,
)
from src.features_temporal import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_WINDOWS,
    HISTORY_COL,
    LEVEL_COLS,
    TEMPORAL_COLS,
    add_temporal_features,
    base_feature_of,
    history_filter_report,
    temporal_col_names,
)
from src.harness import (
    REGRESSORS,
    SELECTORS,
    _run_cell,
    build_pools,
    compare,
    compare_k_grid,
    heldout_evaluate,
    k_grid,
    temporal_share,
)
from src.subset_search import write_history_table, write_json_atomic

PARQUET = RESULTS_DIR / "features_expanded.parquet"
TUNED_PARAMS = RESULTS_DIR / "eng09_tuned_params.json"
VALIDATION_JSON = RESULTS_DIR / "eng14_validation.json"
SCRIPT = "stages/s08_history_features.py"

WINDOWS = DEFAULT_WINDOWS
MIN_HISTORY = DEFAULT_MIN_HISTORY
LEAK_TOL = 0.02          # |r| units, see check 7
ATOL = 1e-9

METHODS = ["pearson", "mutual_info", "rf_importance", "permutation"]
REGS = ["rf", "gpr"]
POOLS = ["level", "temporal", "combined"]
K_GRID_MODE = "coarse"   # the k-curve rule for pools with p > 9 (src.harness.k_grid)

# The cells the report and the later stages use, as (regressor, pool, selector).
# Untuned: every selector with the RF (stage 9 copies the batch-3 cells of these).
# Tuned: pearson with the RF (Tables X, XVI) and permutation with the RF on the
# combined pool (stage 13 checks its one-column permutation importance against it).
DEFAULT_CELLS = {
    "": list(itertools.product(["rf"], POOLS, METHODS)),
    "_tuned": [("rf", pool, "pearson") for pool in POOLS] + [("rf", "combined", "permutation")],
}

# Regressors that already use every core inside one fit (RF: n_jobs=-1);
# their cells are never run concurrently with --jobs.
INTERNALLY_PARALLEL = {"rf"}


def full_json(out_tag: str) -> Path:
    """results/eng14_full{out_tag}.json ('' untuned, '_tuned', or *_smoke; see file_tag)."""
    return RESULTS_DIR / f"eng14_full{out_tag}.json"


def ranking_csv(out_tag: str) -> Path:
    """results/eng14_ranking_table{out_tag}.csv."""
    return RESULTS_DIR / f"eng14_ranking_table{out_tag}.csv"


def progress_json(out_tag: str) -> Path:
    """results/eng14_progress{out_tag}.json: the finished cells of this grid."""
    return RESULTS_DIR / f"eng14_progress{out_tag}.json"


def file_tag(tag: str, cells: list[tuple[str, str, str]]) -> str:
    """The tag of this run's files: '{tag}' when the run covers every default cell
    of the grid (stages 9 and 13 read fixed cells from those files), '{tag}_smoke'
    when it omits one, so a reduced --selectors / --regressors run never replaces
    a finished grid."""
    return tag if set(DEFAULT_CELLS[tag]) <= set(cells) else f"{tag}_smoke"


# ------------------------ 1. checks of the history columns ------------------------

def record(checks: dict, name: str, ok: bool, **info) -> None:
    """Store one check's pass flag and details, and print it."""
    checks[name] = {"pass": bool(ok), **info}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f"  {info}" if info and not ok else ""))


def synthetic_cell(cell_id: str, batch: str, cycles: np.ndarray,
                   values: dict[str, np.ndarray]) -> pd.DataFrame:
    """A cleaned-table-shaped cell with the given feature values and a linear SoH."""
    df = pd.DataFrame({"cell_id": cell_id, "batch": batch,
                       "cycle": cycles.astype(int)})
    for f in LEVEL_COLS:
        df[f] = values[f]
    df["SOH"] = 1.0 - 1e-4 * (df["cycle"] - df["cycle"].min())
    df["status"] = "ok"
    df["cycle_life"] = float(cycles.max() + 100)
    return df


def check_synthetic(checks: dict) -> bool:
    """Checks 1-3 on synthetic cells; returns the synthetic half of check 4."""
    n_max = max(WINDOWS)
    kinds_d = [f"d{n}" for n in WINDOWS]
    kinds_s = [f"slope{n}" for n in WINDOWS]

    # 1. constant cell: every change, slope and drift is exactly zero.
    cyc = np.arange(3, 3 + 80)
    const = {f: np.full(len(cyc), 1.0 + i) for i, f in enumerate(LEVEL_COLS)}
    df_c = synthetic_cell("syn_const", "b1", cyc, const)
    out_c, cols = add_temporal_features(df_c, WINDOWS, MIN_HISTORY)
    hist = out_c[out_c[HISTORY_COL]]
    zero_cols = [f"{f}_{k}" for f in LEVEL_COLS for k in (*kinds_d, *kinds_s, "drift")]
    ok = bool(np.all(np.abs(hist[zero_cols].to_numpy()) == 0.0))
    ok &= all(np.allclose(hist[f"{f}_rm{n_max}"], 1.0 + i, atol=ATOL)
              for i, f in enumerate(LEVEL_COLS))
    record(checks, "1_constant_cell_zero_change", ok,
           max_abs=float(np.abs(hist[zero_cols].to_numpy()).max()))

    # 2. linear cell: slope, N-cycle change, drift and running mean are exact.
    cyc = np.arange(1, 1 + 120)
    slopes = {f: 0.01 * (i + 1) * (-1) ** i for i, f in enumerate(LEVEL_COLS)}
    lin = {f: 2.0 + i + slopes[f] * cyc for i, f in enumerate(LEVEL_COLS)}
    df_l = synthetic_cell("syn_lin", "b2", cyc, lin)
    out_l, _ = add_temporal_features(df_l, WINDOWS, MIN_HISTORY)
    hist = out_l[out_l[HISTORY_COL]]
    errs = {}
    for f in LEVEL_COLS:
        b = slopes[f]
        for n in WINDOWS:
            errs[f"{f}_slope{n}"] = float(np.abs(hist[f"{f}_slope{n}"] - b).max())
            errs[f"{f}_d{n}"] = float(np.abs(hist[f"{f}_d{n}"] - b * n).max())
        errs[f"{f}_drift"] = float(np.abs(
            hist[f"{f}_drift"] - b * (hist["cycle"] - cyc[0])).max())
        centre = hist["cycle"] - (n_max - 1) / 2
        expect_rm = 2.0 + LEVEL_COLS.index(f) + b * centre
        errs[f"{f}_rm{n_max}"] = float(np.abs(hist[f"{f}_rm{n_max}"] - expect_rm).max())
    ok = max(errs.values()) < ATOL
    record(checks, "2_linear_cell_recovers_slope", ok, max_err=max(errs.values()))

    # 3. cycle gaps: lags by cycle number with the last value carried forward,
    # never by row position.
    cyc_gap = np.array([c for c in range(1, 1 + 80) if c not in (30, 31, 32)])
    lin_gap = {f: 2.0 + i + slopes[f] * cyc_gap for i, f in enumerate(LEVEL_COLS)}
    df_g = synthetic_cell("syn_gap", "b3", cyc_gap, lin_gap)
    out_g, _ = add_temporal_features(df_g, WINDOWS, MIN_HISTORY)
    g = out_g.set_index("cycle")
    f, b = LEVEL_COLS[0], slopes[LEVEL_COLS[0]]
    # d10 at cycle 40: value at cycle 30 is missing -> filled from cycle 29.
    exp_d10 = lin_gap[f][cyc_gap == 40][0] - lin_gap[f][cyc_gap == 29][0]
    # slope10 at cycle 40 = OLS over cycles 31..40 with 31, 32 filled from 29.
    grid_vals = np.array([2.0 + b * (29 if c in (30, 31, 32) else c)
                          for c in range(31, 41)])
    exp_s10 = np.polyfit(np.arange(10), grid_vals, 1)[0]
    # A row-position lag would instead give f(40) - f(row-10) = f(27).
    rowpos_d10 = lin_gap[f][cyc_gap == 40][0] - lin_gap[f][cyc_gap == 27][0]
    ok = (abs(g.loc[40, f"{f}_d10"] - exp_d10) < ATOL
          and abs(g.loc[40, f"{f}_slope10"] - exp_s10) < ATOL
          and abs(g.loc[40, f"{f}_d10"] - rowpos_d10) > 1e-6
          and not g.loc[g[HISTORY_COL], cols].isna().any().any()
          and bool(g.loc[21, HISTORY_COL]) and not bool(g.loc[20, HISTORY_COL]))
    record(checks, "3_gap_lag_by_cycle_number_backward_fill", ok,
           d10_at_40=float(g.loc[40, f"{f}_d10"]), expected=float(exp_d10),
           row_position_alternative=float(rowpos_d10))

    # 4 (synthetic half): NaN exactly on the first MIN_HISTORY cycles.
    ok_syn = True
    for out in (out_c, out_l, out_g):
        first = out["cycle"].min()
        expect_nan = (out["cycle"] - first) < MIN_HISTORY
        ok_syn &= bool((out[cols].isna().all(axis=1) == expect_nan).all())
        ok_syn &= bool((out[cols].isna().any(axis=1) == expect_nan).all())
        ok_syn &= bool((out[HISTORY_COL] == ~expect_nan).all())
        ok_syn &= int(expect_nan.sum()) == MIN_HISTORY
    return ok_syn


def check_real_rows(checks: dict, clean: pd.DataFrame, with_t: pd.DataFrame,
                    real_cols: list[str], ok_syn: bool) -> None:
    """Checks 4 (NaN pattern), 5 (no look-ahead) and 6 (slope == np.polyfit) on the real table."""
    n_max = max(WINDOWS)
    assert real_cols == TEMPORAL_COLS == temporal_col_names(WINDOWS)
    first = with_t.groupby("cell_id")["cycle"].transform("min")
    expect_nan = (with_t["cycle"] - first) < MIN_HISTORY
    nan_all = with_t[real_cols].isna().all(axis=1)
    nan_any = with_t[real_cols].isna().any(axis=1)
    ok_real = bool((nan_all == expect_nan).all() and (nan_any == expect_nan).all()
                   and (with_t[HISTORY_COL] == ~expect_nan).all())
    per_cell = expect_nan.groupby(with_t["cell_id"]).sum()
    record(checks, "4_nan_exactly_first_min_history_cycles", ok_syn and ok_real,
           real_rows_without_history=int(expect_nan.sum()),
           per_cell_dropped_counts=per_cell.value_counts().to_dict())

    # 5. Perturbing every base feature of one cell per batch after cycle k0
    # must leave its history columns at cycles <= k0 bit-identical. The same
    # generator then draws check 6's rows, so the order of the draws matters.
    rng = np.random.default_rng(RF_SEED)
    ok = True
    detail = {}
    for batch in ("b1", "b2", "b3"):
        cid = sorted(clean.loc[clean["batch"] == batch, "cell_id"].unique())[0]
        cell_first = int(clean.loc[clean["cell_id"] == cid, "cycle"].min())
        k0 = cell_first + 150
        pert = clean.copy()
        mask = (pert["cell_id"] == cid) & (pert["cycle"] > k0)
        pert.loc[mask, LEVEL_COLS] = rng.normal(50, 20, size=(int(mask.sum()), len(LEVEL_COLS)))
        out_p, _ = add_temporal_features(pert, WINDOWS, MIN_HISTORY)
        before = (with_t["cell_id"] == cid) & (with_t["cycle"] <= k0)
        same = np.array_equal(with_t.loc[before, real_cols].to_numpy(),
                              out_p.loc[before, real_cols].to_numpy(), equal_nan=True)
        after = (with_t["cell_id"] == cid) & (with_t["cycle"] > k0 + n_max)
        changed = not np.array_equal(with_t.loc[after, real_cols].to_numpy(),
                                     out_p.loc[after, real_cols].to_numpy(), equal_nan=True)
        others = with_t["cell_id"] != cid
        others_same = np.array_equal(with_t.loc[others, real_cols].to_numpy(),
                                     out_p.loc[others, real_cols].to_numpy(), equal_nan=True)
        detail[cid] = {"k0": k0, "rows_before_unchanged": bool(same),
                       "rows_after_changed": bool(changed),
                       "other_cells_unchanged": bool(others_same)}
        ok &= same and changed and others_same
    record(checks, "5_no_look_ahead_perturbation", ok, **detail)

    # 6. OLS slopes against np.polyfit on 300 random real rows.
    hist_rows = with_t[with_t[HISTORY_COL]]
    idx = rng.choice(len(hist_rows), 300, replace=False)
    max_err = 0.0
    n_checked = 0
    for i in idx:
        r = hist_rows.iloc[i]
        cell = clean[clean["cell_id"] == r["cell_id"]].set_index("cycle")
        for n in WINDOWS:
            win = cell.reindex(range(int(r["cycle"]) - n + 1, int(r["cycle"]) + 1))
            if win[LEVEL_COLS].isna().any().any():
                continue  # window spans a gap; covered by check 3
            for f in ("height", "area25", "fwhm"):
                pf = np.polyfit(np.arange(n), win[f].to_numpy(), 1)[0]
                max_err = max(max_err, abs(pf - r[f"{f}_slope{n}"]))
                n_checked += 1
    record(checks, "6_slope_matches_polyfit", max_err < 1e-8, max_abs_err=float(max_err),
           n_checked=n_checked)


def check_leak(checks: dict, clean: pd.DataFrame, with_t: pd.DataFrame,
               real_cols: list[str]) -> pd.DataFrame:
    """Check 7 (dev rows): no history column is more correlated with the SoH N
    cycles ahead than with the current SoH beyond what its base feature is."""
    dev_full, _ = split_dev_test(clean)
    dev_t, _ = split_dev_test(with_t[with_t[HISTORY_COL]])
    soh_lookup = dev_full.set_index(["cell_id", "cycle"])["SOH"]
    leak_rows = []
    worst = 0.0
    for n in WINDOWS:
        fut_index = pd.MultiIndex.from_arrays([dev_t["cell_id"], dev_t["cycle"] + n])
        soh_future = soh_lookup.reindex(fut_index).to_numpy()
        have = np.isfinite(soh_future)
        y_now = dev_t["SOH"].to_numpy()[have]
        y_fut = soh_future[have]
        r_now, r_fut = {}, {}
        for c in LEVEL_COLS + real_cols:
            x = dev_t[c].to_numpy()[have]
            r_now[c] = abs(np.corrcoef(x, y_now)[0, 1])
            r_fut[c] = abs(np.corrcoef(x, y_fut)[0, 1])
        for c in real_cols:
            base = base_feature_of(c)
            excess_t = r_fut[c] - r_now[c]
            excess_b = r_fut[base] - r_now[base]
            margin = excess_t - max(excess_b, 0.0)
            worst = max(worst, margin)
            leak_rows.append({
                "shift_N": n, "feature": c, "base": base,
                "r_now": round(r_now[c], 4), "r_future": round(r_fut[c], 4),
                "excess_future": round(excess_t, 4),
                "base_excess_future": round(excess_b, 4),
                "margin_over_base": round(margin, 4),
            })
    leak_df = pd.DataFrame(leak_rows).sort_values("margin_over_base", ascending=False)
    record(checks, "7_correlation_leak_check", worst <= LEAK_TOL, worst_margin=float(worst),
           tol=LEAK_TOL)
    print("    largest future-vs-current |r| excess over the base feature:")
    print(leak_df.head(8).to_string(index=False))
    return leak_df


def check_replica(checks: dict, table: pd.DataFrame, real_cols: list[str]) -> None:
    """Checks 8 (compare_k_grid == compare on the level pool, RF with 20 trees)
    and 9 (the permutation selector is joblib-backend independent)."""
    params = {"n_estimators": 20}
    ref = compare(table, regressor="rf", selection_method="pearson",
                  features=LEVEL_COLS, k="auto", regressor_params=params,
                  verbose=False)
    # The map compare() recomputes on these rows (not ENG-08's), so the two
    # replica runs must both match it.
    _, same_fold_map = assign_folds(split_dev_test(table)[0])
    reps = {
        "folds_recomputed": compare_k_grid(
            table, regressor="rf", selection_method="pearson",
            features=LEVEL_COLS, k_grid=k_grid(len(LEVEL_COLS)),
            regressor_params=params, verbose=False),
        "folds_by_cell": compare_k_grid(
            table, regressor="rf", selection_method="pearson",
            features=LEVEL_COLS, k_grid=k_grid(len(LEVEL_COLS)),
            regressor_params=params, fold_by_cell=same_fold_map, verbose=False),
    }
    ok = True
    detail = {}
    for name, rep in reps.items():
        same = {
            "ranking": [r["feature"] for r in ref["ranking"]]
                       == [r["feature"] for r in rep["ranking"]],
            "scores": bool(np.allclose([r["score"] for r in ref["ranking"]],
                                       [r["score"] for r in rep["ranking"]], rtol=1e-12)),
            "k_curve": [r["k"] for r in ref["k_curve"]] == [r["k"] for r in rep["k_curve"]]
                       and bool(np.allclose(
                           [r["cv_pooled_rmse_mean"] for r in ref["k_curve"]],
                           [r["cv_pooled_rmse_mean"] for r in rep["k_curve"]], rtol=1e-9)),
            "selected": ref["selected_features"] == rep["selected_features"],
            "heldout": bool(np.isclose(ref["heldout_scores"]["pooled"]["rmse"],
                                       rep["heldout_scores"]["pooled"]["rmse"], rtol=1e-9)),
            "cv": bool(np.isclose(ref["cv_scores"]["pooled_rmse_mean"],
                                  rep["cv_scores"]["pooled_rmse_mean"], rtol=1e-9)),
        }
        detail[name] = same
        ok &= all(same.values())
    record(checks, "8_replica_matches_harness_compare", ok, **detail,
           k_grid_level=k_grid(len(LEVEL_COLS)),
           k_grid_temporal=k_grid(len(real_cols)),
           k_grid_combined=k_grid(len(LEVEL_COLS) + len(real_cols)))

    dev_h, _ = split_dev_test(table)
    fold_h = pd.Series(dev_h["cell_id"].map(same_fold_map).to_numpy(dtype=int),
                       index=dev_h.index, name="fold")
    make_rf20 = lambda: REGRESSORS["rf"]["factory"](params)  # noqa: E731
    rank_default = SELECTORS["permutation"](dev_h, LEVEL_COLS, fold_h, make_rf20, None)
    with parallel_config(backend="threading"):
        rank_thread = SELECTORS["permutation"](dev_h, LEVEL_COLS, fold_h, make_rf20, None)
    same_order = rank_default["feature"].tolist() == rank_thread["feature"].tolist()
    same_score = bool(np.allclose(rank_default["score"], rank_thread["score"], rtol=1e-9))
    record(checks, "9_permutation_backend_independent", same_order and same_score,
           order=same_order, scores=same_score,
           max_abs_diff=float(np.abs(rank_default["score"].to_numpy()
                                     - rank_thread["score"].to_numpy()).max()))


def run_checks(clean: pd.DataFrame, with_t: pd.DataFrame, real_cols: list[str],
               table: pd.DataFrame) -> bool:
    """All nine checks -> eng14_validation.json (verdict, windows, min_history,
    temporal_cols, checks{name: {pass, details}}, leak_table) and the manifest;
    True if all pass."""
    checks: dict[str, dict] = {}
    ok_syn = check_synthetic(checks)
    check_real_rows(checks, clean, with_t, real_cols, ok_syn)
    leak_df = check_leak(checks, clean, with_t, real_cols)
    check_replica(checks, table, real_cols)

    all_ok = all(c["pass"] for c in checks.values())
    verdict = "PASS" if all_ok else "FAIL"
    print(f"\nACCEPTANCE [{verdict}]: {sum(c['pass'] for c in checks.values())}/"
          f"{len(checks)} checks passed")
    payload = {
        "verdict": verdict,
        "windows": list(WINDOWS),
        "min_history": MIN_HISTORY,
        "temporal_cols": real_cols,
        "checks": checks,
        "leak_table": leak_df.to_dict(orient="records"),
    }
    VALIDATION_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {VALIDATION_JSON.name}")
    log_manifest("eng14_validation", {
        "script": SCRIPT,
        "seed": RF_SEED,
        "verdict": verdict,
        "checks": {k: v["pass"] for k, v in checks.items()},
    })
    return all_ok


# ------------------------ 3. the grid on the three pools ------------------------

def eng08_folds(clean: pd.DataFrame, table: pd.DataFrame) -> tuple[dict, dict]:
    """ENG-08's cell -> fold map (assign_folds on the FULL clean dev set), reused on
    the history rows, and a report of the cells a recomputed map would move."""
    dev_full, _ = split_dev_test(clean)
    dev_hist, _ = split_dev_test(table)
    _, folds_full = assign_folds(dev_full)
    _, folds_hist = assign_folds(dev_hist)
    moved = sorted(c for c in folds_full if folds_full[c] != folds_hist[c])
    fold_report = {
        "folds_used": "ENG-08 cell->fold map (assign_folds on full clean dev set)",
        "n_dev_cells": len(folds_full),
        "n_cells_that_would_change_if_recomputed": len(moved),
        "cells_that_would_change": moved,
        "fold_sizes_used": {str(f): int(sum(v == f for v in folds_full.values()))
                            for f in range(5)},
        "fold_sizes_if_recomputed": {str(f): int(sum(v == f for v in folds_hist.values()))
                                     for f in range(5)},
        "cell_to_fold": {c: int(f) for c, f in sorted(folds_full.items())},
    }
    print(f"  folds: reusing ENG-08's cell->fold map; recomputing on the reduced "
          f"rows would move {len(moved)} of {len(folds_full)} dev cells")
    return folds_full, fold_report


def baseline_same_rows(table: pd.DataFrame) -> dict:
    """Reference: the interim three features with the untuned RF on the history rows."""
    dev_hist, test_hist = split_dev_test(table)
    rf_spec = REGRESSORS["rf"]
    return heldout_evaluate(dev_hist, test_hist, BASELINE_FEATURES,
                            lambda: rf_spec["factory"](None), None)


def load_eng08_reference(tag: str) -> dict | None:
    """Best ENG-08 cell (full row set) for context; None if stage 7 has not run."""
    p = RESULTS_DIR / f"eng08_ranking_table{tag}.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p).sort_values("b3_rmse").iloc[0]
    return {
        "source": f"results/{p.name}",
        "row_set": "ENG-08 full clean rows (no history filter)",
        "selection_method": t["selection_method"],
        "regressor": t["regressor"],
        "k": int(t["k"]),
        "b3_rmse": float(t["b3_rmse"]),
        "b3_r2": float(t["b3_r2"]),
    }


def grid_cells(tag: str, selectors: list[str] | None,
               regressors: list[str] | None) -> list[tuple[str, str, str]]:
    """(regressor, pool, selector) cells, regressor-major like ENG-08 so the RF
    cells run first: DEFAULT_CELLS when neither flag is given, else the product."""
    if selectors is None and regressors is None:
        return list(DEFAULT_CELLS[tag])
    if selectors is None or "all" in selectors:
        methods = METHODS
    else:
        methods = [m for m in METHODS if m in selectors]
    regs = [r for r in REGS if r in (regressors or REGS)]
    return list(itertools.product(regs, POOLS, methods))


def resumed_cells(out_tag: str, pools: dict, tuned: dict) -> dict:
    """Finished cells of an interrupted run whose pool, k grid and regressor
    parameters match this run."""
    results: dict[str, dict] = {p: {} for p in POOLS}
    path = progress_json(out_tag)
    if not path.exists():
        return results
    prev = json.loads(path.read_text(encoding="utf-8"))
    for pool, cells in prev.get("results", {}).items():
        for key, out in cells.items():
            cfg = out["config"]
            if (cfg["pool"] == pools[pool]
                    and cfg["k_grid"] == k_grid(len(pools[pool]), K_GRID_MODE)
                    and cfg["regressor_params"] == (tuned.get(cfg["regressor"]) or {})):
                results[pool][key] = out
    print(f"  resumed {sum(len(v) for v in results.values())} finished cells from {path.name}")
    return results


def run_grid(table: pd.DataFrame, pools: dict, cells: list, tuned: dict, folds_full: dict,
             out_tag: str, jobs: int, resume: bool) -> dict:
    """Every cell through compare_k_grid (src.harness._run_cell); the progress file is
    rewritten after each cell. Returns grid[pool]["<selector>|<regressor>"]."""
    results = resumed_cells(out_tag, pools, tuned) if resume else {p: {} for p in POOLS}

    def store(pool: str, key: str, out: dict) -> None:
        out["pool"] = pool
        out["temporal_share"] = temporal_share(out["ranking"], out["selected_features"])
        results[pool][key] = out
        write_json_atomic(progress_json(out_tag), {"tag": out_tag, "results": results})
        ho = out["heldout_scores"]["pooled"]
        print(f"  [{pool} | {key}] done in {out['runtime_s']} s: "
              f"k={len(out['selected_features'])} b3 RMSE={ho['rmse']:.5f} "
              f"R2={ho['r2']:.4f}; top-5 "
              + ", ".join(r["feature"] for r in out["ranking"][:5]), flush=True)

    pending = [c for c in cells if f"{c[2]}|{c[0]}" not in results[c[1]]]
    for reg, pool, method in cells:
        if (reg, pool, method) not in pending:
            print(f"===== {pool} | {method}|{reg} ===== (resumed)", flush=True)

    # RF cells (n_jobs=-1 inside every fit) run one at a time; GPR cells (a
    # serial optimiser using ~2 cores) share the machine when --jobs > 1. Each
    # cell is independent and seeded, so the numbers do not depend on --jobs.
    sequential = [c for c in pending if jobs == 1 or c[0] in INTERNALLY_PARALLEL]
    pooled = [c for c in pending if c not in sequential]

    for reg, pool, method in sequential:
        cols = pools[pool]
        grid = k_grid(len(cols), K_GRID_MODE)
        print(f"\n===== {pool} | {method}|{reg} =====  p={len(cols)} k-grid={grid}", flush=True)
        store(pool, f"{method}|{reg}",
              _run_cell(table, reg, method, cols, grid, tuned.get(reg), folds_full, True))

    if pooled:
        print(f"\n===== {len(pooled)} cells in a pool of {jobs} processes "
              f"(per-k output suppressed): "
              + ", ".join(f"{p}|{m}|{r}" for r, p, m in pooled) + " =====", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {}
            for reg, pool, method in pooled:
                cols = pools[pool]
                fut = ex.submit(_run_cell, table, reg, method, cols,
                                k_grid(len(cols), K_GRID_MODE), tuned.get(reg), folds_full, False)
                futures[fut] = (pool, f"{method}|{reg}")
            for fut in as_completed(futures):
                pool, key = futures[fut]
                store(pool, key, fut.result())

    # Only this run's cells, in grid order (resumed cells come back first otherwise).
    return {pool: {f"{m}|{r}": results[pool][f"{m}|{r}"] for r, p, m in cells if p == pool}
            for pool in POOLS}


def ranking_table(results: dict, cells: list, pools: dict) -> pd.DataFrame:
    """One row per cell, sorted by pool (level, temporal, combined), then held-out RMSE."""
    rows = []
    for reg, pool, method in cells:
        out = results[pool][f"{method}|{reg}"]
        ho = out["heldout_scores"]
        ts = out["temporal_share"]
        rows.append({
            "pool": pool,
            "selection_method": method,
            "regressor": reg,
            "p": len(pools[pool]),
            "k": len(out["selected_features"]),
            "selected_features": "|".join(out["selected_features"]),
            "cv_rmse": out["cv_scores"]["pooled_rmse_mean"],
            "cv_rmse_std": out["cv_scores"]["pooled_rmse_std"],
            "cv_r2": out["cv_scores"]["pooled_r2_mean"],
            "b3_rmse": ho["pooled"]["rmse"],
            "b3_mae": ho["pooled"]["mae"],
            "b3_r2": ho["pooled"]["r2"],
            "b3_rmse_cellavg": ho["cell_avg"]["rmse"],
            "b3_r2_cellavg": ho["cell_avg"]["r2"],
            "n_temporal_selected": ts["n_temporal_selected"],
            "frac_temporal_selected": ts["frac_temporal_selected"],
            "best_temporal_feature": ts["best_temporal_feature"],
            "best_temporal_rank": ts["best_temporal_rank"],
            "top1_feature": out["ranking"][0]["feature"],
            "runtime_s": out["runtime_s"],
        })
    table_out = pd.DataFrame(rows)
    table_out["pool_order"] = table_out["pool"].map({p: i for i, p in enumerate(POOLS)})
    return (table_out.sort_values(["pool_order", "b3_rmse"])
            .drop(columns="pool_order").reset_index(drop=True))


def rank_vector(ranking: list[dict], pool_cols: list[str]) -> np.ndarray:
    """Position of each pool column in one method's ranking."""
    order = [r["feature"] for r in ranking]
    return np.array([order.index(f) for f in pool_cols])


def agreement(results: dict, pools: dict, methods: list[str], regs: list[str]) -> tuple:
    """Per pool and regressor: Kendall tau between the methods' rankings (4 dp)
    and the mean pairwise top-3 Jaccard; per pool the top-1 / top-3 features."""
    tau_all, top1, top3, jaccard3 = {}, {}, {}, {}
    for pool in POOLS:
        cells = results[pool]
        tau_all[pool], jaccard3[pool] = {}, {}
        for reg in regs:
            ms = [m for m in methods if f"{m}|{reg}" in cells]
            mat = np.full((len(ms), len(ms)), np.nan)
            for i, m1 in enumerate(ms):
                for j, m2 in enumerate(ms):
                    tau, _ = kendalltau(
                        rank_vector(cells[f"{m1}|{reg}"]["ranking"], pools[pool]),
                        rank_vector(cells[f"{m2}|{reg}"]["ranking"], pools[pool]),
                    )
                    mat[i, j] = tau
            tau_all[pool][reg] = {"methods": ms, "tau": mat.round(4).tolist()}
            vals = []
            for m1, m2 in itertools.combinations(ms, 2):
                a = {x["feature"] for x in cells[f"{m1}|{reg}"]["ranking"][:3]}
                b = {x["feature"] for x in cells[f"{m2}|{reg}"]["ranking"][:3]}
                vals.append(len(a & b) / len(a | b))
            jaccard3[pool][reg] = float(np.mean(vals)) if vals else float("nan")
        top1[pool] = {k: v["ranking"][0]["feature"] for k, v in cells.items()}
        top3[pool] = {k: [x["feature"] for x in v["ranking"][:3]] for k, v in cells.items()}
        print(f"[{pool}] top-1 features: {top1[pool]}; mean pairwise top-3 Jaccard: "
              f"{jaccard3[pool]}")
    return tau_all, top1, top3, jaccard3


def best_cell(results: dict, pool: str, by: str) -> dict | None:
    """The pool's cell with the lowest held-out RMSE (by='b3_rmse') or dev-CV RMSE."""
    cells = results.get(pool, {})
    if not cells:
        return None
    if by == "b3_rmse":
        key = min(cells, key=lambda k: cells[k]["heldout_scores"]["pooled"]["rmse"])
    else:  # dev-CV chosen (no test-set peeking)
        key = min(cells, key=lambda k: cells[k]["cv_scores"]["pooled_rmse_mean"])
    c = cells[key]
    return {
        "cell": key, "k": len(c["selected_features"]),
        "selected_features": c["selected_features"],
        "cv_rmse": c["cv_scores"]["pooled_rmse_mean"],
        "b3_rmse": c["heldout_scores"]["pooled"]["rmse"],
        "b3_r2": c["heldout_scores"]["pooled"]["r2"],
    }


def comparison(results: dict, methods: list[str], regs: list[str]) -> dict:
    """Best level / temporal / combined cell (by held-out and by dev-CV RMSE) and
    the relative held-out change of each history pool against level, per cell."""
    out = {}
    for by in ("b3_rmse", "cv_rmse"):
        lv, cb, tp = (best_cell(results, "level", by), best_cell(results, "combined", by),
                      best_cell(results, "temporal", by))
        entry = {"level": lv, "combined": cb, "temporal": tp}
        if lv and cb:
            entry["combined_vs_level_rel_rmse"] = (cb["b3_rmse"] - lv["b3_rmse"]) / lv["b3_rmse"]
        if lv and tp:
            entry["temporal_vs_level_rel_rmse"] = (tp["b3_rmse"] - lv["b3_rmse"]) / lv["b3_rmse"]
        out[f"best_by_{by}"] = entry
    paired = []
    for reg, method in itertools.product(regs, methods):
        key = f"{method}|{reg}"
        if key not in results["level"]:
            continue
        lv = results["level"][key]["heldout_scores"]["pooled"]["rmse"]
        rec = {"cell": key, "level_b3_rmse": lv}
        for pool in ("temporal", "combined"):
            if key in results.get(pool, {}):
                v = results[pool][key]["heldout_scores"]["pooled"]["rmse"]
                rec[f"{pool}_b3_rmse"] = v
                rec[f"{pool}_vs_level_rel"] = (v - lv) / lv
        paired.append(rec)
    out["paired_per_cell"] = paired
    for by in ("b3_rmse", "cv_rmse"):
        e = out[f"best_by_{by}"]
        for pool in ("level", "temporal", "combined"):
            if e[pool]:
                rel = e.get(f"{pool}_vs_level_rel_rmse")
                print(f"best by {by}: {pool:9s} {e[pool]['cell']} k={e[pool]['k']} "
                      f"b3 RMSE={e[pool]['b3_rmse']:.5f}"
                      + (f" ({rel:+.2%} vs level)" if rel is not None else ""))
    return out


def run_pass(tag: str, out_tag: str, tuned: dict, cells: list, table: pd.DataFrame,
             pools: dict, folds_full: dict, setup: dict, jobs: int, resume: bool) -> None:
    """One grid (tag '' untuned, '_tuned'): run or resume its cells, then write
    eng14_ranking_table{out_tag}.csv (one row per cell: pool, selection_method,
    regressor, p, k, selected_features, cv_rmse, cv_rmse_std, cv_r2, b3_rmse, b3_mae,
    b3_r2, b3_rmse_cellavg, b3_r2_cellavg, n/frac_temporal_selected,
    best_temporal_feature/rank, top1_feature, runtime_s; by pool then b3_rmse) and
    eng14_full{out_tag}.json (pools, windows, min_history, k_grid_mode, k_grids,
    tuned_params, clean_log, history_filter, fold_report,
    baseline_3feature_rf_same_rows, eng08_reference,
    grid[pool]["<selector>|<regressor>"], agreement_kendall_tau, top1_features,
    top3_features, top3_jaccard_mean, comparison). A smoke run (out_tag != tag)
    gets no manifest entry."""
    eng08_ref = load_eng08_reference(tag)
    if eng08_ref:
        print(f"ENG-08{tag} best (full rows): {eng08_ref['selection_method']} x "
              f"{eng08_ref['regressor']} k={eng08_ref['k']} b3 RMSE={eng08_ref['b3_rmse']:.5f}")
    results = run_grid(table, pools, cells, tuned, folds_full, out_tag, jobs, resume)

    table_out = ranking_table(results, cells, pools)
    table_out.to_csv(ranking_csv(out_tag), index=False)
    print("\n=== RANKING TABLE (by pool, then held-out RMSE) ===")
    print(table_out.drop(columns=["selected_features"]).to_string(index=False))

    methods = [m for m in METHODS if any(c[2] == m for c in cells)]
    regs = [r for r in REGS if any(c[0] == r for c in cells)]
    tau, top1, top3, jaccard3 = agreement(results, pools, methods, regs)
    comp = comparison(results, methods, regs)

    full = {
        "pools": pools,
        "windows": list(WINDOWS),
        "min_history": MIN_HISTORY,
        "k_grid_mode": K_GRID_MODE,
        "k_grids": {p: k_grid(len(c), K_GRID_MODE) for p, c in pools.items()},
        "tuned_params": tuned or None,
        "clean_log": setup["clean_log"],
        "history_filter": setup["history_filter"],
        "fold_report": setup["fold_report"],
        "baseline_3feature_rf_same_rows": setup["baseline"],
        "eng08_reference": eng08_ref,
        "grid": results,
        "agreement_kendall_tau": tau,
        "top1_features": top1,
        "top3_features": top3,
        "top3_jaccard_mean": jaccard3,
        "comparison": comp,
    }
    full_json(out_tag).write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"\nWrote {full_json(out_tag).name} and {ranking_csv(out_tag).name}")

    if out_tag != tag:
        return
    log_manifest(f"eng14_temporal{tag}", {
        "script": SCRIPT,
        "seed": RF_SEED,
        "tuned": bool(tuned),
        "k_grid_mode": K_GRID_MODE,
        "windows": list(WINDOWS),
        "min_history": MIN_HISTORY,
        "rows_with_history": setup["history_filter"]["total"]["rows_with_history"],
        "folds": setup["fold_report"]["folds_used"],
        "best_level": comp["best_by_b3_rmse"].get("level"),
        "best_combined": comp["best_by_b3_rmse"].get("combined"),
        "combined_vs_level_rel_rmse": comp["best_by_b3_rmse"].get("combined_vs_level_rel_rmse"),
        "baseline_b3_rmse_same_rows": setup["baseline"]["pooled"]["rmse"],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selectors", nargs="+", choices=METHODS + ["all"],
                    help="ranking methods (default: DEFAULT_CELLS, the report's cells)")
    ap.add_argument("--regressors", nargs="+", choices=REGS,
                    help="regressors crossed with the selectors and the three pools (default "
                         "when only --selectors is given: both); '--regressors rf gpr' is the "
                         "full 24-cell grid. A grid that omits a default cell is a smoke run: "
                         "it writes eng14_*{,_tuned}_smoke files and no manifest entry")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run GPR cells N at a time in separate processes (RF cells stay "
                         "sequential); the numbers are identical to --jobs 1")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="take finished cells from results/eng14_progress*.json "
                         "(default); --no-resume starts the grids over")
    args = ap.parse_args()
    if args.jobs < 1:
        sys.exit("--jobs must be >= 1")
    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")

    clean, clean_log = clean_feature_table(pd.read_parquet(PARQUET))
    with_t, temporal_cols = add_temporal_features(clean, windows=WINDOWS, min_history=MIN_HISTORY)
    table = with_t[with_t[HISTORY_COL]].reset_index(drop=True)

    print("\n---- 1. checks of the history columns ----", flush=True)
    if not run_checks(clean, with_t, temporal_cols, table):
        sys.exit("a history-column check failed; see results/eng14_validation.json")

    print("\n---- 2. history table ----", flush=True)
    write_history_table()

    print("\n---- 3. level / temporal / combined pools ----", flush=True)
    hist_report = history_filter_report(with_t)
    print(f"  history filter (min_history={MIN_HISTORY}): "
          f"{hist_report['total']['rows']} -> {hist_report['total']['rows_with_history']} rows "
          f"(dev {hist_report['dev']['rows']} -> {hist_report['dev']['rows_with_history']}, "
          f"test {hist_report['test']['rows']} -> {hist_report['test']['rows_with_history']}; "
          f"{hist_report['dropped_per_cell']['min']}-"
          f"{hist_report['dropped_per_cell']['max']} rows per cell)")
    pools = build_pools(temporal_cols)
    for name, cols in pools.items():
        n_nan = int(table[cols].isna().sum().sum())
        assert n_nan == 0, f"{name} pool has {n_nan} NaNs after the history filter"
    folds_full, fold_report = eng08_folds(clean, table)
    baseline = baseline_same_rows(table)
    print(f"3-feature RF baseline on history rows: b3 RMSE={baseline['pooled']['rmse']:.5f} "
          f"R2={baseline['pooled']['r2']:.4f}")
    setup = {"clean_log": clean_log, "history_filter": hist_report,
             "fold_report": fold_report, "baseline": baseline}

    # The untuned grid runs last: run_results.py treats the stage as done once
    # eng14_full.json exists, so that file must be the last one written.
    passes = []
    if TUNED_PARAMS.exists():
        passes.append(("_tuned", json.loads(TUNED_PARAMS.read_text(encoding="utf-8"))))
    else:
        print(f"\n{TUNED_PARAMS.name} missing - tuned grid skipped; run stage 6 "
              "(stages/s06_tune_models.py) and rerun this stage.")
    passes.append(("", {}))
    out_tags = []
    for tag, tuned in passes:
        cells = grid_cells(tag, args.selectors, args.regressors)
        out_tag = file_tag(tag, cells)
        out_tags.append(out_tag)
        print(f"\n---- {'tuned' if tuned else 'untuned'} grid"
              f"{' (smoke run: a default cell is missing)' if out_tag != tag else ''} ----",
              flush=True)
        run_pass(tag, out_tag, tuned, cells, table, pools, folds_full, setup,
                 args.jobs, args.resume)

    # Progress files go only now, so an interruption in the second grid does
    # not recompute the first one.
    for out_tag in out_tags:
        progress_json(out_tag).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
