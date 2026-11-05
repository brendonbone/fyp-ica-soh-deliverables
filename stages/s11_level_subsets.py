"""
11. All 511 subsets of the nine features, and what they show (ENG-19, ENG-19b)
==============================================================================
Does a ranked k-curve (stage 7) miss a better subset? Every non-empty subset
of the nine level features (2^9 - 1 = 511) is scored with the ENG-08 protocol
(clean rows, dev = batches 1+2, 5-fold StratifiedGroupKFold, untuned RF, seed
0, columns in pool order). Everything else is then read off that table with
no new CV fits: the best subset per size, backward elimination, forward
selection, the ENG-08 ranking prefixes, exact Shapley values, the two one-SE
rules, the SoC window of every subset and the accuracy-discharge trade-off
edge. Batch 3 is scored only for the listed subsets (18 untuned; all nine and
the recommended ones also with the ENG-09 tuned RF).

Inputs   results/features_expanded.parquet (stage 2); eng08_full.json and
         eng08_full_tuned.json (stage 7); eng09_tuned_params.json (stage 6);
         eng17_soc_windows_summary.csv and features_soc_windows.parquet
         (stage 10; the parquet needs the raw pickles and is not committed,
         and the analysis stops without it rather than overwrite the stored
         per-cycle windows with the median-window fallback).
Outputs  results/eng19_exhaustive_subsets.csv   511 rows: features, k,
             cv_pooled_rmse_mean/std, cv_pooled_r2_mean,
             cv_cell_avg_rmse_mean, runtime_s, cv_rank
         results/eng19_exhaustive_subsets.json  pool, heldout {cv_best,
             one_se, all9}, one_se_threshold, n_within_one_se, best_per_k,
             top20_inclusion_rate, eng08_check
         results/eng19b_analysis.json            the derivations, the batch-3
             records (each with the reason it was scored) and the checks
         results/eng19b_windows.csv              the SoC window of every subset
Runtime  the search about 2 h on a laptop, the analysis a few minutes.
         Resumable: results/eng19_progress.json is rewritten after every
         subset and finished subsets are skipped; once both ENG-19 files
         exist the search is not repeated.

    python -m stages.s11_level_subsets             # search (if not done), then analysis
    python -m stages.s11_level_subsets --analyse   # analysis only
    python -m stages.s11_level_subsets --limit 5   # smoke test: 5 subsets, then stop
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.features import FEATURE_COLS
from src.harness import cv_evaluate, heldout_evaluate
from src.subset_search import (
    ENG17_SUMMARY,
    SOC_PARQUET,
    TUNED_PARAMS,
    all_subsets,
    analyse_search,
    eng08_rankings,
    env_info,
    level_blocks,
    load_level_rows,
    read_json,
    rf_factory,
    save_analysis,
    subset_key,
    table_from_records,
    write_json_atomic,
)

SCRIPT = "stages/s11_level_subsets.py"
PROGRESS = RESULTS_DIR / "eng19_progress.json"
OUT_CSV = RESULTS_DIR / "eng19_exhaustive_subsets.csv"
OUT_JSON = RESULTS_DIR / "eng19_exhaustive_subsets.json"
ENG08 = RESULTS_DIR / "eng08_full.json"
ENG08_TUNED = RESULTS_DIR / "eng08_full_tuned.json"
STEM = "eng19b"
POOL = list(FEATURE_COLS)


# ------------------------ ENG-19: the 511 subsets ------------------------

def score_subsets(dev, fold, limit: int | None = None) -> dict:
    """CV-score every subset not yet in the progress file, rewriting it after each one."""
    subsets = [list(s) for s in all_subsets(POOL)]   # every subset in pool order
    done = read_json(PROGRESS, {})
    todo = [s for s in subsets if subset_key(s) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(subsets)} subsets, {len(done)} already done", flush=True)
    make = rf_factory()
    t_start = time.time()
    for n_new, sub in enumerate(todo, 1):
        t0 = time.time()
        cv = cv_evaluate(dev, sub, fold, make, None)
        done[subset_key(sub)] = {
            "features": sub,
            "k": len(sub),
            "cv_pooled_rmse_mean": cv["pooled_rmse_mean"],
            "cv_pooled_rmse_std": cv["pooled_rmse_std"],
            "cv_pooled_r2_mean": cv["pooled_r2_mean"],
            "cv_cell_avg_rmse_mean": cv["cell_avg_rmse_mean"],
            "runtime_s": round(time.time() - t0, 1),
        }
        write_json_atomic(PROGRESS, done)
        eta = (time.time() - t_start) / n_new * (len(todo) - n_new)
        print(f"[{len(done):3d}/{len(subsets)}] k={len(sub)} "
              f"CV={cv['pooled_rmse_mean']:.5f}  {subset_key(sub)}  "
              f"({done[subset_key(sub)]['runtime_s']:.0f}s, ETA {eta / 60:.0f} min)", flush=True)
    return done


def finish_search(done: dict, dev, test) -> None:
    """Rank the 511 subsets, score batch 3 for three of them and write the ENG-19 files."""
    # No two of the 511 CV means tie, so the single-key sort gives a well-defined rank.
    table = (pd.DataFrame(done.values())
             .sort_values("cv_pooled_rmse_mean").reset_index(drop=True))
    table["cv_rank"] = range(1, len(table) + 1)

    # "one_se" is ENG-19's name for the smallest subset within one fold sd
    # (ddof = 0) of the best; the analysis below reports it as
    # within_one_fold_sd and adds the real one-standard-error rule.
    best = table.iloc[0]
    thresh = best["cv_pooled_rmse_mean"] + best["cv_pooled_rmse_std"]
    within = table[table["cv_pooled_rmse_mean"] <= thresh]
    one_se = within.sort_values(["k", "cv_pooled_rmse_mean"]).iloc[0]
    all9 = table[table["k"] == len(POOL)].iloc[0]

    # same rows, folds and regressor as stage 7 -> the all-nine value must match
    check = None
    e08 = read_json(ENG08)
    if e08 is not None:
        ref9 = e08["grid"]["pearson|rf"]["k_curve"][-1]["cv_pooled_rmse_mean"]
        check = {"eng08_k9_cv_rmse": ref9,
                 "eng19_all9_cv_rmse": float(all9["cv_pooled_rmse_mean"]),
                 "abs_diff": abs(ref9 - float(all9["cv_pooled_rmse_mean"]))}
        print(f"\nCheck vs ENG-08 k=9: {check}")

    make = rf_factory()
    heldout = {}
    for label, row in (("cv_best", best), ("one_se", one_se), ("all9", all9)):
        ho = heldout_evaluate(dev, test, list(row["features"]), make, None)
        heldout[label] = {
            "features": list(row["features"]),
            "k": int(row["k"]),
            "cv_rank": int(row["cv_rank"]),
            "cv_pooled_rmse_mean": float(row["cv_pooled_rmse_mean"]),
            "cv_pooled_rmse_std": float(row["cv_pooled_rmse_std"]),
            "b3_rmse": ho["pooled"]["rmse"],
            "b3_r2": ho["pooled"]["r2"],
            "b3_mae": ho["pooled"]["mae"],
        }

    best_per_k = (table.sort_values("cv_pooled_rmse_mean").groupby("k").head(1)
                  .sort_values("k"))
    top20 = table.head(20)
    inclusion = {f: float(top20["features"].apply(lambda s: f in s).mean())
                 for f in POOL}

    table.assign(features=table["features"].apply(subset_key)).to_csv(OUT_CSV, index=False)
    write_json_atomic(OUT_JSON, {
        "pool": POOL, "regressor": "rf", "regressor_params": {}, "seed": RF_SEED,
        "n_subsets": len(table),
        "total_runtime_s": round(float(table["runtime_s"].sum()), 1),
        "heldout": heldout,
        "one_se_threshold": float(thresh),
        "n_within_one_se": int(len(within)),
        "best_per_k": best_per_k[["k", "features", "cv_pooled_rmse_mean",
                                  "cv_pooled_rmse_std", "cv_rank"]].to_dict("records"),
        "top20_inclusion_rate": inclusion,
        "eng08_check": check,
    }, indent=2)

    print("\n=== best subset per size (dev-CV pooled RMSE) ===")
    for r in best_per_k.itertuples():
        print(f"  k={r.k}: {r.cv_pooled_rmse_mean:.5f} +/- {r.cv_pooled_rmse_std:.5f} "
              f"(rank {r.cv_rank})  {subset_key(r.features)}")
    print(f"\n{len(within)} subsets within one fold sd of the best ({thresh:.5f})")
    print("\n=== held-out batch 3 ===")
    for label, v in heldout.items():
        print(f"  {label:8s} k={v['k']} CV={v['cv_pooled_rmse_mean']:.5f} "
              f"b3 RMSE={v['b3_rmse']:.5f} R2={v['b3_r2']:.4f}  {subset_key(v['features'])}")
    print("\nTop-20 inclusion rate:", {f: round(v, 2) for f, v in inclusion.items()})
    print(f"wrote results/{OUT_CSV.name} and results/{OUT_JSON.name}", flush=True)

    log_manifest("eng19_exhaustive_subsets", {
        "script": SCRIPT, "seed": RF_SEED,
        "b3_rmse": {k: v["b3_rmse"] for k, v in heldout.items()},
    })


# ------------------------ ENG-19b: what the table says ------------------------

def load_search_table() -> pd.DataFrame:
    """The finished ENG-19 table, re-read from its CSV (as the analysis always was)."""
    recs = {}
    for r in pd.read_csv(OUT_CSV).to_dict("records"):
        recs[r["features"]] = {"blocks": r["features"].split("+"), "k": int(r["k"]),
                               "n_columns": int(r["k"]),
                               "cv_pooled_rmse_mean": float(r["cv_pooled_rmse_mean"]),
                               "cv_pooled_rmse_std": float(r["cv_pooled_rmse_std"]),
                               "cv_pooled_r2_mean": float(r["cv_pooled_r2_mean"]),
                               "cv_cell_avg_rmse_mean": float(r["cv_cell_avg_rmse_mean"]),
                               "runtime_s": float(r["runtime_s"])}
    return table_from_records(recs)


def cross_checks(t: pd.DataFrame, A: dict) -> dict:
    """The ENG-08 k-curve subsets re-found in the table (exact at k = 1, a
    column-order effect otherwise) and ENG-19's own batch-3 scores reproduced."""
    by_key = t.set_index("key")["cv_pooled_rmse_mean"].to_dict()
    checks = {}
    e08 = read_json(ENG08)["grid"]
    kc_rows = []
    for cell in ("pearson|rf", "spearman|rf", "mutual_info|rf", "rf_importance|rf", "permutation|rf"):
        for r in e08[cell]["k_curve"]:
            key = subset_key([f for f in POOL if f in r["features"]])
            same_order = r["features"] == [f for f in POOL if f in r["features"]]
            kc_rows.append({"cell": cell, "k": r["k"], "eng08": r["cv_pooled_rmse_mean"],
                            "eng19": by_key[key], "same_column_order": same_order,
                            "abs_diff": abs(r["cv_pooled_rmse_mean"] - by_key[key])})
    kc = pd.DataFrame(kc_rows)
    checks["eng08_k_curves"] = {
        "n_subsets_compared": int(len(kc)),
        "same_order_max_abs_diff": float(kc.loc[kc["same_column_order"], "abs_diff"].max()),
        "different_order_max_abs_diff": float(kc.loc[~kc["same_column_order"], "abs_diff"].max()),
        "different_order_median_abs_diff": float(kc.loc[~kc["same_column_order"], "abs_diff"].median()),
        "k1_max_abs_diff": float(kc.loc[kc["k"] == 1, "abs_diff"].max()),
        "rows": kc_rows}
    e19 = read_json(OUT_JSON)
    if e19:
        ho = {tuple(r["blocks"]): r for r in A["heldout"] if r["config"] == "untuned"}
        rep = {}
        for label, v in e19["heldout"].items():
            mine = ho.get(tuple(v["features"]))
            rep[label] = {"features": v["features"], "eng19_b3": v["b3_rmse"],
                          "eng19b_b3": mine["b3_rmse"] if mine else None,
                          "abs_diff": abs(mine["b3_rmse"] - v["b3_rmse"]) if mine else None}
        checks["eng19_heldout_reproduced"] = rep
        checks["eng19_all9_vs_eng08_pearson_k9"] = e19.get("eng08_check")
    return checks


def print_summary(A: dict) -> None:
    """The headline numbers of the analysis on the console."""
    print("\n=== best subset per size (dev-CV) / BE path ===")
    for b, s in zip(A["best_per_k"], reversed(A["backward_elimination"])):
        print(f"  k={b['k']}: best {b['cv']:.5f} {subset_key(b['blocks']):55s} | "
              f"BE {s['cv']:.5f} {subset_key(s['blocks'])}")
    print(f"  BE drop order: {A['be_drop_order']} (last standing {A['be_last_standing']})")
    print(f"  SFS add order: {A['sfs_add_order']}")
    print("  Shapley: " + ", ".join(f"{k} {v:.5f}" for k, v in
                                    sorted(A["shapley"].items(), key=lambda x: -x[1])))
    sr = A["selection_rules"]
    for rule in ("within_one_fold_sd", "one_standard_error", "one_standard_error_ddof1"):
        p = sr[rule]["pick"]
        print(f"  {rule}: <= {sr[rule]['threshold']:.5f} ({sr[rule]['n_within']} subsets) -> "
              f"k={p['k']} {subset_key(p['blocks'])} {p['cv']:.5f}")
    print(f"  trade-off edge ({A['window_basis']}):")
    for r in A["tradeoff_edge"]:
        print(f"    span {100 * r['soc_span']:5.1f} %: {r['cv']:.5f} {subset_key(r['blocks'])}")
    print(f"  batch-3 subsets scored: {A['heldout_audit']}")
    kc = A["checks"]["eng08_k_curves"]
    print(f"  checks: ENG-08 k-curves k=1 max diff {kc['k1_max_abs_diff']:.1e}, "
          f"other order max {kc['different_order_max_abs_diff']:.1e}")


def analyse(dev, test, fold) -> None:
    """ENG-19b: the derivations, the listed batch-3 scores and the checks -> the two eng19b files."""
    for path, where in ((OUT_CSV, "the search (this stage without --analyse)"),
                        (ENG08, "stage 7"), (ENG08_TUNED, "stage 7"),
                        (TUNED_PARAMS, "stage 6"), (ENG17_SUMMARY, "stage 10"),
                        (SOC_PARQUET, "stage 10 (needs the raw pickles)")):
        if not path.exists():
            sys.exit(f"results/{path.name} missing - run {where} first")

    t = load_search_table()
    A, tw = analyse_search(t, POOL, level_blocks(), POOL, dev, test, fold,
                           rankings=eng08_rankings(), windows=True, tuned=True)
    A["checks"] = cross_checks(t, A)
    A["eng19_one_se_label_note"] = (
        "ENG-19's 'one_se' is the smallest subset within ONE FOLD SD of the best; "
        "it is reported here as 'within_one_fold_sd'. The one-standard-error rule "
        "(SE = fold sd / sqrt(5)) is 'one_standard_error'.")
    A["env"] = env_info()
    save_analysis(STEM, A, tw)
    print_summary(A)

    log_manifest("eng19b_subset_analysis", {
        "script": SCRIPT, "seed": RF_SEED,
        "best_per_k_cv": {r["k"]: r["cv"] for r in A["best_per_k"]},
        "be_drop_order": A["be_drop_order"], "shapley": A["shapley"],
        "selection_rules": {k: A["selection_rules"][k]["pick"]["blocks"]
                            for k in ("within_one_fold_sd", "one_standard_error")},
        "per_cycle_windows": "soc_span_per_cycle" in tw,
        "n_batch3_subsets": A["heldout_audit"]["n_subsets_scored_on_batch3"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--analyse", action="store_true",
                    help="skip the search: derive the ENG-19b analysis from the finished table")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="smoke test: score N unfinished subsets, write the progress file and stop")
    args = ap.parse_args()

    dev, test, fold = load_level_rows()
    if args.limit:
        score_subsets(dev, fold, args.limit)
        return
    if not args.analyse:
        # both files: a run killed between the CSV and the JSON write redoes the finishing step
        if OUT_CSV.exists() and OUT_JSON.exists():
            print(f"[skip] results/{OUT_CSV.name} and {OUT_JSON.name} exist: "
                  "the search is finished", flush=True)
        else:
            finish_search(score_subsets(dev, fold), dev, test)
    analyse(dev, test, fold)


if __name__ == "__main__":
    main()
