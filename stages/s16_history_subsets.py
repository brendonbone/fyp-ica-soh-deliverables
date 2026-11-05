"""
16. All 511 subsets of the features with their history (ENG-23)
================================================================
How few features are needed once each brings its 20-cycle history, and at
what discharge cost? Stage 11's search over the 511 subsets of the nine level
features is repeated with every feature kept or removed together with its six
history columns (d10, d20, slope10, slope20, drift, rm20): 7k columns in the
ENG-14 combined-pool order, on the ENG-14 history rows (57,452 dev rows,
40,400 batch-3 rows), the ENG-08 cell -> fold map and the untuned RF (300
trees, seed 0). The analysis derives what stage 11 derives (best per size,
backward elimination, forward selection, ranking paths, Shapley values, the
one-SE rules, SoC windows, the trade-off edge), scores batch 3 for the listed
subsets only (untuned; all nine and the recommended subsets also with the
ENG-09 tuned RF), and pairs every subset with its level-only score on the
same rows (stage 15).

Inputs   results/features_expanded.parquet; eng14_full.json (stage 8);
         eng24_level_subsets_history_rows.json and _analysis.json (stage 15);
         eng08_full.json, eng08_full_tuned.json (stage 7); eng09_tuned_params.json
         (stage 6); eng17_soc_windows_summary.csv and, when present,
         features_soc_windows.parquet (stage 10; without it the windows fall
         back to the union of the ENG-17 medians); eng22_history_components.json
         (stage 14, optional: an exact check).
Outputs  results/eng23_grouped_history_subsets.{csv,json}  one row per subset:
             cv_rank, key (blocks), k, n_columns, cv_pooled_rmse_mean/std,
             cv_pooled_r2_mean, cv_cell_avg_rmse_mean, cv_fold_rmse, runtime_s, host
         results/eng23_grouped_history_subsets_analysis.json  the derivations,
             tuned, heldout (+ audit), checks, paired, env
         results/eng23_grouped_history_subsets_windows.csv  SoC window per subset
         results/eng23_paired_level_history.csv  key, k, cv_pooled_rmse_mean/std
             _history and _level, rel_change = (history - level) / level
Runtime  the search ~12 h on one 24-thread machine (45-144 s per subset),
         resumable through results/eng23_grouped_history_subsets_progress.json;
         on a cluster array --shard I --n-shards N (one progress file per
         shard), then --merge. The analysis ~15 min. The no-argument run does
         search, merge and analysis, skipping the search when the table exists.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.features_temporal import LEVEL_COLS
from src.subset_search import (
    ENG17_SUMMARY,
    SOC_PARQUET,
    TUNED_PARAMS,
    add_search_args,
    analyse_search,
    eng08_rankings,
    env_info,
    grouped_blocks,
    load_history_rows,
    load_search_table,
    merge_search,
    read_json,
    run_search,
    save_analysis,
    subset_key,
)

STEM = "eng23_grouped_history_subsets"
SCRIPT = "stages/s16_history_subsets.py"
TABLE_JSON = RESULTS_DIR / f"{STEM}.json"
PAIRED_CSV = RESULTS_DIR / "eng23_paired_level_history.csv"
LEVEL_STEM = "eng24_level_subsets_history_rows"   # stage 15: level features only, same rows
LEVEL_JSON = RESULTS_DIR / f"{LEVEL_STEM}.json"
LEVEL_ANALYSIS = RESULTS_DIR / f"{LEVEL_STEM}_analysis.json"
ENG14_JSON = RESULTS_DIR / "eng14_full.json"
ENG22_JSON = RESULTS_DIR / "eng22_history_components.json"


# ------------------------ search and merge ------------------------

def search(args) -> None:
    """Score this shard's subsets (resumable); an unsharded full run merges itself."""
    dev, _, fold, combined, _ = load_history_rows()
    run_search(args, STEM + args.tag, list(LEVEL_COLS), grouped_blocks(combined),
               combined, dev, fold)
    if args.n_shards == 1 and not args.limit:
        merge(args.tag)


def merge(tag: str = "") -> pd.DataFrame | None:
    """Union of the progress files -> results/<stem>.{csv,json} once all 511 are scored."""
    t = merge_search(STEM + tag, list(LEVEL_COLS), {
        "experiment": "ENG-23", "rows": "ENG-14 history rows (>= 20 cycles of history)",
        "folds": "ENG-08 cell->fold map", "blocks": "feature + its 6 history columns",
        "columns": "ENG-14 combined-pool order restricted to the subset",
        "regressor": "rf", "regressor_params": {}, "seed": RF_SEED, "env_merge": env_info()})
    if t is not None and not tag:
        log_manifest(STEM, {
            "script": SCRIPT, "seed": RF_SEED,
            "n_subsets": int(len(t)), "hosts": sorted(t["host"].unique().tolist()),
            "best": {"blocks": list(t.iloc[0]["blocks"]),
                     "cv_rmse": float(t.iloc[0]["cv_pooled_rmse_mean"])}})
    return t


# ------------------------ analysis ------------------------

def require_inputs() -> None:
    """Stop, naming the stage to run, when an input of the analysis is missing."""
    needed = [(TABLE_JSON, "the search of this stage (or --merge)"),
              (LEVEL_JSON, "stage 15 (s15_level_subsets_history_rows)"),
              (LEVEL_ANALYSIS, "stage 15 (s15_level_subsets_history_rows)"),
              (ENG14_JSON, "stage 8"),
              (RESULTS_DIR / "eng08_full.json", "stage 7"),
              (RESULTS_DIR / "eng08_full_tuned.json", "stage 7"),
              (TUNED_PARAMS, "stage 6"),
              (ENG17_SUMMARY, "stage 10")]
    missing = [f"results/{p.name} (run {where} first)" for p, where in needed if not p.exists()]
    if missing:
        sys.exit("the ENG-23 analysis needs " + "; ".join(missing))
    if not SOC_PARQUET.exists():
        print(f"  results/{SOC_PARQUET.name} not found (stage 10, needs the raw pickles): "
              "the SoC windows fall back to the union of the ENG-17 medians "
              "(the stored run used the per-cycle basis)", flush=True)


def cross_checks(all_nine_cv: float) -> dict:
    """All nine features with history is the 63-column combined pool: against
    ENG-14's combined k = 63 (Pearson column order, so only within 5e-5) and,
    when stage 14's table is on disk, ENG-22's all-components subset (same
    columns in the same order, so exact)."""
    got = all_nine_cv
    ref = float(read_json(ENG14_JSON)["grid"]["combined"]["pearson|rf"]["k_curve"][-1]
                ["cv_pooled_rmse_mean"])
    checks = {"all63_vs_eng14_combined_k63": {
        "eng23_all63_combined_order": got, "eng14_combined_pearson_k63": ref,
        "abs_diff": abs(got - ref), "pass_within_5e-5": abs(got - ref) <= 5e-5}}
    comp = read_json(ENG22_JSON)
    if comp:
        allc = [r for r in comp["table"] if r["k"] == len(comp["names"])][0]["cv_pooled_rmse_mean"]
        checks["all63_vs_eng22_all_components"] = {
            "eng22": allc, "eng23": got, "abs_diff": abs(allc - got),
            "pass_exact": abs(allc - got) <= 1e-12}
    return checks


def paired_table(hist: pd.DataFrame, lvl: pd.DataFrame) -> pd.DataFrame:
    """Each subset's dev-CV with history and level-only (stage 15), and the relative change."""
    h = hist.set_index("key")[["k", "cv_pooled_rmse_mean", "cv_pooled_rmse_std"]]
    lv = lvl.set_index("key")[["cv_pooled_rmse_mean", "cv_pooled_rmse_std"]]
    p = h.join(lv, lsuffix="_history", rsuffix="_level", how="inner")
    p["rel_change"] = (p["cv_pooled_rmse_mean_history"] - p["cv_pooled_rmse_mean_level"]) \
        / p["cv_pooled_rmse_mean_level"]
    return p.reset_index()


def paired_summary(p: pd.DataFrame, level_A: dict, A: dict) -> dict:
    """How often and by how much history helps, overall and per size, plus both searches'
    best-of-k and backward-elimination orders side by side."""
    by_k = p.groupby("k")["rel_change"].agg(["median", "min", "max", "count"])
    return {
        "n_subsets": int(len(p)),
        "n_history_better": int((p["rel_change"] < 0).sum()),
        "rel_change_median": float(p["rel_change"].median()),
        "rel_change_by_k": {int(k): {c: float(v) for c, v in r.items()}
                            for k, r in by_k.iterrows()},
        "best_of_k_level": level_A["best_per_k"],
        "best_of_k_history": A["best_per_k"],
        "be_drop_order_level": level_A["be_drop_order"],
        "be_drop_order_history": A["be_drop_order"],
    }


def print_summary(A: dict) -> None:
    """The headline numbers of the analysis on the console."""
    print("\n=== best subset per size (dev-CV), each feature with its history ===")
    for r in A["best_per_k"]:
        print(f"  k={r['k']}: {r['cv']:.5f}  {subset_key(r['blocks'])}")
    print(f"  BE drop order: {A['be_drop_order']} (last standing {A['be_last_standing']})")
    p = A["paired"]
    print(f"  history better than level-only for {p['n_history_better']} of {p['n_subsets']} "
          f"subsets (median change {100 * p['rel_change_median']:.1f} %)")
    for name, c in A["checks"].items():
        print(f"  check {name}: abs diff {c['abs_diff']:.1e}")
    print(f"  batch-3 subsets scored: {A['heldout_audit']['n_subsets_scored_on_batch3']}", flush=True)


def analyse() -> None:
    """The derivations, listed batch-3 scores, checks and the pairing with stage 15."""
    require_inputs()
    t = load_search_table(STEM)
    dev, test, fold, combined, _ = load_history_rows()
    A, tw = analyse_search(t, list(LEVEL_COLS), grouped_blocks(combined), combined,
                           dev, test, fold, rankings=eng08_rankings(), windows=True, tuned=True)
    A["checks"] = cross_checks(A["all_blocks"]["cv"])
    p = paired_table(t, load_search_table(LEVEL_STEM))
    p.to_csv(PAIRED_CSV, index=False)
    print(f"wrote results/{PAIRED_CSV.name}", flush=True)
    A["paired"] = paired_summary(p, read_json(LEVEL_ANALYSIS), A)
    A["env"] = env_info()
    save_analysis(STEM, A, tw)
    print_summary(A)
    log_manifest(f"{STEM}_analysis", {
        "script": f"{SCRIPT} --analyse", "seed": RF_SEED,
        "best_per_k_cv": {r["k"]: r["cv"] for r in A["best_per_k"]},
        "be_drop_order": A["be_drop_order"], "checks": A["checks"],
        "n_batch3_subsets": A["heldout_audit"]["n_subsets_scored_on_batch3"]})


# ------------------------ entry point ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    add_search_args(ap)
    args = ap.parse_args()
    if args.figures:
        print("nothing to draw here: run_report.py makes Fig. 9 and Tables XII, XIII and D1 "
              "from the result files")
        return
    if args.merge:
        merge(args.tag)
        return
    if args.analyse:
        analyse()
        return

    if args.n_shards > 1 or args.limit:   # a cluster shard or a smoke run: progress file only
        search(args)
        return

    # The whole stage; a --tag run stops at its table (the analysis reads the
    # untagged one).
    table = RESULTS_DIR / f"{STEM}{args.tag}.json"
    if table.exists():
        print(f"[{STEM}] results/{table.name} exists: search skipped (delete it to rescore)",
              flush=True)
    else:
        search(args)
    if args.tag:
        return
    analyse()


if __name__ == "__main__":
    main()
