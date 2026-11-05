"""
15. The 511 level subsets on the history rows (ENG-24)
======================================================
Is stage 16's history gain partly its change of rows? The 511 subsets of the
nine level features are scored on exactly stage 16's rows and folds (ENG-14
history rows, ENG-08 cell -> fold map, pool column order, untuned RF, seed 0)
and analysed as in stage 11, with batch 3 scored for the listed subsets only.
Inputs: features_expanded.parquet, eng08_full{,_tuned}.json, eng14_full.json,
eng17_soc_windows_summary.csv and, when present, features_soc_windows.parquet
(without it the windows fall back to the union of the ENG-17 medians, which
changes the trade-off edge and the batch-3 set).
Outputs: eng24_level_subsets_history_rows.{csv,json} (one row per subset: key,
k, cv_pooled_rmse_mean/std, cv_fold_rmse, ...), _analysis.json (best_per_k,
backward_elimination, ranking_paths, shapley, selection_rules, tradeoff_edge,
heldout, checks) and _windows.csv (the SoC window of every subset); stage 16
reads the table and the analysis file, so this stage runs first.
Runtime: the search ~5.6 h on a laptop, resumable through the progress file
(--shard/--n-shards for a cluster array, then --merge), the analysis minutes;
the no-argument run does all of it, skipping the search when the table exists.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.features_temporal import LEVEL_COLS
from src.subset_search import (
    add_search_args,
    analyse_search,
    eng08_rankings,
    env_info,
    level_blocks,
    load_history_rows,
    load_search_table,
    merge_search,
    read_json,
    run_search,
    save_analysis,
)

STEM = "eng24_level_subsets_history_rows"
SCRIPT = "stages/s15_level_subsets_history_rows.py"
ENG14_JSON = RESULTS_DIR / "eng14_full.json"
NAMES = list(LEVEL_COLS)   # one block per feature; columns in this (pool) order


# ------------------------ search and merge ------------------------

def search(args) -> None:
    """Score this shard's subsets, skipping those already in its progress file."""
    dev, _, fold, _, _ = load_history_rows()
    run_search(args, STEM + args.tag, NAMES, level_blocks(), NAMES, dev, fold)


def merge(tag: str = "") -> pd.DataFrame | None:
    """Union of the progress files -> results/<stem>.{csv,json} once all 511 are scored.
    Files ending in _laptop.json are read first, so a subset scored on two machines
    keeps the laptop's record (the other is logged in merge_conflicts)."""
    t = merge_search(STEM + tag, NAMES, {
        "experiment": "ENG-24", "rows": "ENG-14 history rows (>= 20 cycles of history)",
        "folds": "ENG-08 cell->fold map", "columns": "pool order (FEATURE_COLS)",
        "regressor": "rf", "regressor_params": {}, "seed": RF_SEED, "env_merge": env_info()})
    if t is not None and not tag:
        log_manifest(STEM, {
            "script": SCRIPT, "seed": RF_SEED,
            "n_subsets": int(len(t)), "hosts": sorted(t["host"].unique().tolist()),
            "best": {"blocks": list(t.iloc[0]["blocks"]),
                     "cv_rmse": float(t.iloc[0]["cv_pooled_rmse_mean"])}})
    return t


# ------------------------ analysis ------------------------

def eng14_all_nine() -> float:
    """Stage 8's level pool at k = 9: the same nine columns, in Pearson-ranking order."""
    e14 = read_json(ENG14_JSON)
    if e14 is None:
        raise FileNotFoundError(f"results/{ENG14_JSON.name} missing - run stage 8 first "
                                "(the analysis checks the all-nine score against it)")
    return e14["grid"]["level"]["pearson|rf"]["k_curve"][-1]["cv_pooled_rmse_mean"]


def analyse(stem: str = STEM) -> dict:
    """Derivations read off the table, then batch 3 for the listed subsets. Untuned
    only: the tuned scores on these rows are stage 16's and stage 19's."""
    t = load_search_table(stem)
    ref = eng14_all_nine()
    dev, test, fold, _, _ = load_history_rows()
    A, tw = analyse_search(t, NAMES, level_blocks(), NAMES, dev, test, fold,
                           rankings=eng08_rankings(), windows=True, tuned=False)
    # The forest's draws go by column index, so the column order alone moves
    # the all-nine score (by 1.7e-5 in the stored run); 5e-5 allows for that.
    got = A["all_blocks"]["cv"]
    A["checks"] = {"all9_vs_eng14_level_k9": {
        "eng24_all9_pool_order": got, "eng14_level_pearson_k9": ref,
        "abs_diff": abs(got - ref), "pass_within_5e-5": abs(got - ref) <= 5e-5}}
    A["env"] = env_info()
    save_analysis(stem, A, tw)
    log_manifest(f"{stem}_analysis", {
        "script": f"{SCRIPT} --analyse", "seed": RF_SEED,
        "best_per_k_cv": {r["k"]: r["cv"] for r in A["best_per_k"]},
        "be_drop_order": A["be_drop_order"], "checks": A["checks"],
        "n_batch3_subsets": A["heldout_audit"]["n_subsets_scored_on_batch3"]})
    return A


# ------------------------ entry point ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    add_search_args(ap)
    args = ap.parse_args()
    stem = STEM + args.tag
    if args.figures:
        print("nothing to draw here: run_report.py makes the figures from the result files")
        return
    if args.merge:
        merge(args.tag)
        return
    if args.analyse:
        analyse(stem)
        return
    if args.n_shards > 1 or args.limit:   # a cluster shard or a smoke run: progress file only
        search(args)
        return

    table = RESULTS_DIR / f"{stem}.json"
    if table.exists():
        print(f"[{stem}] results/{table.name} exists: search skipped (delete it to rescore)",
              flush=True)
    else:
        search(args)
        merge(args.tag)
    analyse(stem)


if __name__ == "__main__":
    main()
