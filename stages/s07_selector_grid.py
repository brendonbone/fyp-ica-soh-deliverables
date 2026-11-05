"""
7. Ranking methods x regressors, untuned and tuned (ENG-08)
===========================================================
Which ranking method, with which regressor, gives the lowest held-out error
on the nine-feature pool? Each cell ranks the features on the development
rows, picks k by grouped-CV pooled RMSE over the nested subsets (columns in
ranking order) and touches batch 3 once with that subset. The rankings are
then compared across methods (Kendall tau, top-1, top-3 Jaccard) for every
regressor that ran all five selectors (only the RF in the default grid). The
grid runs untuned, then again with the ENG-09 hyperparameters.

Inputs   results/features_expanded.parquet; results/eng09_tuned_params.json
         (without it the tuned grid is skipped with a message).
Outputs  results/eng08_full.json, eng08_full_tuned.json: pool, tuned_params,
         baseline_3feature_rf, grid["<selector>|<regressor>"] = {ranking,
         k_curve, selected_features, cv_scores, heldout_scores, config,
         runtime_s}, agreement_kendall_tau, agreement_summary, top1_features,
         top3_jaccard_mean, top3_jaccard_mean_original_four.
         results/eng08_ranking_table.csv, eng08_ranking_table_tuned.csv: one
         row per cell sorted by b3_rmse (selection_method, regressor, k,
         selected_features, cv_rmse, cv_rmse_std, cv_r2, b3_rmse, b3_mae,
         b3_r2, b3_rmse_cellavg, b3_r2_cellavg, runtime_s).
Runtime  the default grid (five selectors x RF, plus pearson x GPR) takes
         about 40 min untuned and 45 min tuned; the full grid (--regressors
         rf gpr --selectors all) about 2 h 20 min and 3 h 15 min. Every cell
         is deterministic, so the stored JSON is the progress file: a pass
         whose JSON already holds every cell of the grid is skipped (delete
         the file to recompute), --reuse takes the stored cells of a partial
         JSON instead of recomputing them, and a run never drops a stored
         cell: its cells are merged into the stored grid and the ranking
         table and agreement statistics are recomputed over the union.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.config import RESULTS_DIR
from src.data_prep import BASELINE_FEATURES, clean_feature_table, log_manifest, split_dev_test
from src.features import FEATURE_COLS
from src.harness import REGRESSORS, compare, heldout_evaluate

PARQUET = RESULTS_DIR / "features_expanded.parquet"
TUNED_PARAMS = RESULTS_DIR / "eng09_tuned_params.json"

METHODS = ["pearson", "spearman", "mutual_info", "rf_importance", "permutation"]
ORIGINAL_METHODS = ["pearson", "mutual_info", "rf_importance", "permutation"]  # before Spearman (ENG-16)
REGS = ["rf", "gpr"]
# The cells the report uses: every selector with the RF (Tables V, VIII, IX and
# the rankings the later subset searches start from) and pearson with the GPR
# (the tuned-GPR row of Table VII).
DEFAULT_GRID = [(m, "rf") for m in METHODS] + [("pearson", "gpr")]


def full_json(tag: str) -> Path:
    """results/eng08_full{tag}.json (tag '' untuned, '_tuned')."""
    return RESULTS_DIR / f"eng08_full{tag}.json"


def ranking_csv(tag: str) -> Path:
    """results/eng08_ranking_table{tag}.csv."""
    return RESULTS_DIR / f"eng08_ranking_table{tag}.csv"


def grid_from_args(selectors: list[str] | None, regressors: list[str] | None) -> list[tuple[str, str]]:
    """The (selector, regressor) cells: the report's grid when neither flag is
    given, otherwise the product (regressor outer, selectors in METHODS order)."""
    if selectors is None and regressors is None:
        return list(DEFAULT_GRID)
    if selectors is None or "all" in selectors:
        methods = METHODS
    else:
        methods = [m for m in METHODS if m in selectors]
    return [(m, r) for r in (regressors or REGS) for m in methods]


def baseline_3feature_rf(clean: pd.DataFrame) -> dict:
    """Reference: the interim three features with the untuned RF, scored on batch 3."""
    dev, test = split_dev_test(clean)
    rf_spec = REGRESSORS["rf"]
    return heldout_evaluate(dev, test, BASELINE_FEATURES, lambda: rf_spec["factory"](None), None)


def stored_grid(path: Path) -> dict:
    """The cells of an existing eng08_full JSON ({} when there is none)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("grid", {})


def matching_cells(stored: dict, pool: list[str], tuned: dict) -> dict:
    """The stored cells whose configuration matches this run (same pool, k
    chosen by CV, seed 0, same regressor parameters); every cell is
    deterministic, so they equal what a recomputation would give."""
    cached = {}
    for key, out in stored.items():
        cfg = out.get("config", {})
        if (cfg.get("pool") == pool and cfg.get("k_mode") == "auto"
                and cfg.get("seed") == 0
                and cfg.get("regressor_params") == (tuned.get(cfg.get("regressor")) or {})):
            cached[key] = out
    return cached


def in_grid_order(cells: dict) -> dict:
    """The cells in the full grid's order (regressor outer, METHODS inner), so
    the ranking table breaks ties the same way whatever was merged when."""
    return {f"{m}|{r}": cells[f"{m}|{r}"] for r in REGS for m in METHODS if f"{m}|{r}" in cells}


def run_grid(clean: pd.DataFrame, grid: list[tuple[str, str]], pool: list[str],
             tuned: dict, cached: dict) -> dict:
    """One compare() call per cell (or the cached cell), keyed '<selector>|<regressor>'."""
    results = {}
    for method, reg in grid:
        key = f"{method}|{reg}"
        if key in cached:
            print(f"\n===== {key} ===== (reused)", flush=True)
            out = cached[key]
        else:
            print(f"\n===== {key} =====", flush=True)
            t0 = time.time()
            out = compare(
                clean, regressor=reg, selection_method=method,
                features=pool, k="auto",
                regressor_params=tuned.get(reg), verbose=True,
            )
            out["runtime_s"] = round(time.time() - t0, 1)
        results[key] = out
    return results


def ranking_table(results: dict) -> pd.DataFrame:
    """One row per cell, sorted by held-out (batch 3) RMSE."""
    rows = []
    for key, out in results.items():
        method, reg = key.split("|")
        ho = out["heldout_scores"]
        rows.append({
            "selection_method": method,
            "regressor": reg,
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
            "runtime_s": out["runtime_s"],
        })
    return pd.DataFrame(rows).sort_values("b3_rmse").reset_index(drop=True)


# ---- cross-method agreement of the rankings ----

def rank_vector(ranking: list[dict], pool: list[str]) -> np.ndarray:
    """Position of each pool feature in one method's ranking."""
    order = [r["feature"] for r in ranking]
    return np.array([order.index(f) for f in pool])


def kendall_matrix(results: dict, methods: list[str], reg: str, pool: list[str]) -> list[list[float]]:
    """Kendall tau between the rankings of every pair of methods, rounded to 4 dp."""
    mat = np.zeros((len(methods), len(methods)))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            tau, _ = kendalltau(
                rank_vector(results[f"{m1}|{reg}"]["ranking"], pool),
                rank_vector(results[f"{m2}|{reg}"]["ranking"], pool),
            )
            mat[i, j] = tau
    return mat.round(4).tolist()


def offdiag(matrix: list[list[float]], methods: list[str], subset: list[str]) -> dict | None:
    """Mean and min of the off-diagonal entries of the (rounded) matrix over
    `subset`; None when fewer than two methods are in it."""
    idx = [methods.index(m) for m in subset]
    if len(idx) < 2:
        return None
    sub = np.array(matrix)[np.ix_(idx, idx)]
    off = sub[~np.eye(len(idx), dtype=bool)]
    return {"mean": float(off.mean()), "min": float(off.min())}


def top3_jaccard(results: dict, methods: list[str], reg: str) -> float | None:
    """Mean pairwise Jaccard similarity of the methods' top-3 sets (unrounded);
    None when fewer than two methods."""
    if len(methods) < 2:
        return None
    vals = []
    for m1, m2 in itertools.combinations(methods, 2):
        a = {x["feature"] for x in results[f"{m1}|{reg}"]["ranking"][:3]}
        b = {x["feature"] for x in results[f"{m2}|{reg}"]["ranking"][:3]}
        vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals))


def agreement_statistics(results: dict, methods: list[str], regs: list[str], pool: list[str]) -> tuple:
    """Per regressor that ran every one of `methods`: the Kendall tau matrix, its
    off-diagonal mean/min over all methods and over the original four, and the
    mean top-3 Jaccard; plus the top-1 feature of every cell."""
    complete = [r for r in regs if all(f"{m}|{r}" in results for m in methods)]
    original = [m for m in ORIGINAL_METHODS if m in methods]
    tau = {r: kendall_matrix(results, methods, r, pool) for r in complete}
    summary = {r: {"all_methods": offdiag(tau[r], methods, methods),
                   "original_four": offdiag(tau[r], methods, original)}
               for r in complete}
    top1 = {f"{m}|{r}": results[f"{m}|{r}"]["ranking"][0]["feature"]
            for m, r in itertools.product(methods, regs) if f"{m}|{r}" in results}
    jaccard = {r: top3_jaccard(results, methods, r) for r in complete}
    jaccard_original = {r: top3_jaccard(results, original, r) for r in complete}
    return tau, summary, top1, jaccard, jaccard_original


# ---- one pass of the grid (untuned or tuned) ----

def run_pass(clean: pd.DataFrame, grid: list[tuple[str, str]], tuned: dict, tag: str,
             reuse: bool, baseline: dict) -> None:
    """Run (or reuse) every cell of the grid, merge the cells into the stored
    grid, write the ranking table and the full JSON for this tag, and log the
    best cell to the manifest. Skipped when the stored JSON already holds
    every cell of the grid."""
    pool = FEATURE_COLS
    stored = stored_grid(full_json(tag))
    matching = matching_cells(stored, pool, tuned)
    if all(f"{m}|{r}" in matching for m, r in grid):
        print(f"{full_json(tag).name} already holds every cell of the grid - pass skipped")
        return
    cached = matching if reuse else {}
    if reuse:
        print(f"--reuse: {len(cached)} matching cells found in {full_json(tag).name}\n")
    results = in_grid_order({**stored, **run_grid(clean, grid, pool, tuned, cached)})

    table = ranking_table(results)
    table.to_csv(ranking_csv(tag), index=False)
    print("\n=== RANKING TABLE (by held-out RMSE) ===")
    print(table.to_string(index=False))

    methods = [m for m in METHODS if any(key.startswith(m + "|") for key in results)]
    regs = [r for r in REGS if any(key.endswith("|" + r) for key in results)]
    tau, summary, top1, jaccard, jaccard_original = agreement_statistics(results, methods, regs, pool)
    print(f"\ntop-1 features: {top1}")
    print(f"mean pairwise top-3 Jaccard: {jaccard} "
          f"(original four selectors: {jaccard_original})")
    print(f"mean / min pairwise Kendall tau: {summary}")

    full = {
        "pool": pool,
        "tuned_params": tuned or None,
        "baseline_3feature_rf": baseline,
        "grid": results,
        "agreement_kendall_tau": {"methods": methods, **tau},
        "agreement_summary": summary,
        "top1_features": top1,
        "top3_jaccard_mean": jaccard,
        "top3_jaccard_mean_original_four": jaccard_original,
    }
    full_json(tag).write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"\nWrote {full_json(tag).name} and {ranking_csv(tag).name}")

    best = table.iloc[0]
    log_manifest(f"eng08_comparison{tag}", {
        "script": "stages/s07_selector_grid.py",
        "seed": 0,
        "tuned": bool(tuned),
        "best": {
            "selection_method": best["selection_method"],
            "regressor": best["regressor"],
            "k": int(best["k"]),
            "features": best["selected_features"],
            "b3_rmse": float(best["b3_rmse"]),
            "b3_r2": float(best["b3_r2"]),
        },
        "baseline_b3_rmse": baseline["pooled"]["rmse"],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selectors", nargs="+", choices=METHODS + ["all"],
                    help="ranking methods to cross with the regressors (default: all five). "
                         "With neither --selectors nor --regressors the grid is the report's: "
                         "all five selectors with the RF, plus pearson with the GPR")
    ap.add_argument("--regressors", nargs="+", choices=REGS,
                    help="regressors to cross with the selectors (default: both); "
                         "'--regressors rf gpr --selectors all' is the full grid")
    ap.add_argument("--reuse", action="store_true",
                    help="take the stored cells with an identical configuration from a partial "
                         "results/eng08_full.json / eng08_full_tuned.json instead of recomputing "
                         "them (a pass whose file already holds every cell of the grid is "
                         "skipped anyway)")
    args = ap.parse_args()
    grid = grid_from_args(args.selectors, args.regressors)

    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")
    clean, _ = clean_feature_table(pd.read_parquet(PARQUET))
    baseline = baseline_3feature_rf(clean)
    print(f"3-feature RF baseline: b3 RMSE={baseline['pooled']['rmse']:.5f} "
          f"R2={baseline['pooled']['r2']:.4f}\n")

    print("---- untuned grid ----", flush=True)
    run_pass(clean, grid, {}, "", args.reuse, baseline)

    if not TUNED_PARAMS.exists():
        print(f"\n{TUNED_PARAMS.name} missing - tuned grid skipped; run stage 6 "
              "(stages/s06_tune_models.py) and rerun this stage.")
        return
    tuned = json.loads(TUNED_PARAMS.read_text(encoding="utf-8"))
    print(f"\n---- tuned grid ({TUNED_PARAMS.name}) ----", flush=True)
    run_pass(clean, grid, tuned, "_tuned", args.reuse, baseline)


if __name__ == "__main__":
    main()
