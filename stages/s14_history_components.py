"""
14. All 127 subsets of the seven history components (ENG-22)
=============================================================
Which history component (level, d10, d20, slope10, slope20, drift, rm20; each
a block of nine columns, one per level feature) carries the gain of the
63-column history model? Every non-empty subset of the seven (127) is scored
by 5-fold dev-CV with the untuned RF on the ENG-14 history rows of
results/features_expanded.parquet; batch 3 is scored for each component alone,
all 63 minus each component and the best subset of each size.
Outputs: eng22_history_components.{csv,json} (one row per subset: blocks, k,
n_columns, cv_pooled_rmse_mean/std, cv_fold_rmse, ...), _analysis.json
(designs, best_per_k, be_drop_order, shapley, named_subsets, heldout, checks,
v_empty) and _seed_check.json ('all 63' vs 'all 63 minus rm20' with forest
seeds 0, 1, 2: dev-CV and batch 3 per seed, and their mean and sd).
Runtime: the search ~8 h on one 8-core machine (resumable through the progress
file; --shard I --n-shards N for a cluster array, then --merge), the analysis
minutes, the seed check ~40 min. The no-argument run does search, merge,
analysis and seed check, skipping the search when the merged table exists; it
writes _analysis.json (run_results.py's marker for this stage) last.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.harness import cv_evaluate, heldout_evaluate
from src.subset_search import (
    COMPONENTS,
    LISTED_REASONS,
    HeldoutLog,
    add_search_args,
    analyse_table,
    canonical,
    columns_for,
    component_blocks,
    env_info,
    load_history_rows,
    load_search_table,
    mean_predictor_cv,
    merge_search,
    read_json,
    rf_factory,
    run_search,
    subset_key,
    write_json_atomic,
)

STEM = "eng22_history_components"
SCRIPT = "stages/s14_history_components.py"
TABLE_JSON = RESULTS_DIR / f"{STEM}.json"
ANALYSIS_JSON = RESULTS_DIR / f"{STEM}_analysis.json"
SEED_CHECK_JSON = RESULTS_DIR / f"{STEM}_seed_check.json"

# Earlier results the analysis cross-checks against, each only when on disk.
ENG14_DIAG = RESULTS_DIR / "eng14_diagnostic_pools.json"
ENG24_JSON = RESULTS_DIR / "eng24_level_subsets_history_rows.json"
ENG23_JSON = RESULTS_DIR / "eng23_grouped_history_subsets.json"

NAMED = {  # subsets the discussion refers to by name
    "level + rm20": ["level", "rm20"],
    "level + rm20 + drift": ["level", "rm20", "drift"],
    "rm20 + drift": ["rm20", "drift"],
    "level + drift": ["level", "drift"],
    "slopes and changes only": ["d10", "d20", "slope10", "slope20"],
    "all but slopes and changes": ["level", "drift", "rm20"],
}
EXTRA_SEEDS = (1, 2)   # the seed check adds these to the stored seed-0 run


# ------------------------ search and merge ------------------------

def search(args) -> None:
    """Score this shard's subsets (resumable); an unsharded full run merges itself."""
    dev, _, fold, combined, _ = load_history_rows()
    run_search(args, STEM + args.tag, list(COMPONENTS), component_blocks(combined),
               combined, dev, fold)
    if args.n_shards == 1 and not args.limit:
        merge(args.tag)


def merge(tag: str = "") -> pd.DataFrame | None:
    """Union of the progress files -> results/<stem>.{csv,json} once all 127 are scored."""
    t = merge_search(STEM + tag, list(COMPONENTS), {
        "experiment": "ENG-22", "rows": "ENG-14 history rows (>= 20 cycles of history)",
        "folds": "ENG-08 cell->fold map", "blocks": "history component across the 9 features",
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

def designs(cv, names: list[str], full: float, shapley: dict) -> list[dict]:
    """Each component alone, and all 63 minus it, relative to the full model."""
    out = []
    for c in names:
        rest = [x for x in names if x != c]
        out.append({
            "component": c,
            "cv_alone": cv([c]),
            "cv_all_minus": cv(rest),
            "cost_of_dropping": cv(rest) - full,            # > 0: the others cannot replace it
            "cost_of_dropping_rel": (cv(rest) - full) / full,
            "alone_vs_all_rel": (cv([c]) - full) / full,
            "shapley": shapley[c],
        })
    return out


def score_batch3(dev, test, blocks, combined, names: list[str], best_per_k: list[dict]) -> dict:
    """Batch 3 for the 14 single designs and the best subset of each size (untuned RF)."""
    log = HeldoutLog(dev, test, blocks, combined,
                     LISTED_REASONS | {"single_component", "drop_one_component"})
    make = rf_factory()
    log.score(tuple(names), "reference_all", make)
    for c in names:
        log.score((c,), "single_component", make)
        log.score(canonical([x for x in names if x != c], names), "drop_one_component", make)
    for r in best_per_k:
        log.score(r["blocks"], "best_of_k", make)
    return log.records


def cross_checks(cv, full: float) -> dict:
    """{level} against ENG-24's all-nine row (exact), {rm20}, {drift}, {level, rm20} against
    the ENG-14 diagnostic pools (Pearson column order, within 5e-5) and all seven against
    ENG-23's all-features row (exact); each only when that result file is on disk."""
    checks = {}
    e24 = read_json(ENG24_JSON)
    if e24:
        all9 = [r for r in e24["table"] if r["k"] == 9][0]["cv_pooled_rmse_mean"]
        checks["level_component_vs_eng24_all9"] = {
            "eng22": cv(["level"]), "eng24": all9, "abs_diff": abs(cv(["level"]) - all9),
            "pass_exact": abs(cv(["level"]) - all9) <= 1e-12}
    diag = read_json(ENG14_DIAG)
    if diag:
        for pool, sel in [("rm20_only", ["rm20"]), ("drift_only", ["drift"]),
                          ("level_plus_rm20", ["level", "rm20"])]:
            ref = diag["results"][pool]["k_curve"][-1]["cv_pooled_rmse_mean"]
            checks[f"{pool}_vs_eng14_diagnostic"] = {
                "eng22": cv(sel), "eng14_diag_pearson_order": ref, "abs_diff": abs(cv(sel) - ref),
                "pass_within_5e-5": abs(cv(sel) - ref) <= 5e-5}
    e23 = read_json(ENG23_JSON)
    if e23:
        g = [r for r in e23["table"] if r["k"] == 9][0]["cv_pooled_rmse_mean"]
        checks["all_components_vs_eng23_all_features"] = {
            "eng22": full, "eng23": g, "abs_diff": abs(full - g), "pass_exact": abs(full - g) <= 1e-12}
    return checks


def analyse() -> dict:
    """Everything read off the merged table, then the listed batch-3 scores (not yet written)."""
    names = list(COMPONENTS)
    t = load_search_table(STEM)
    dev, test, fold, combined, _ = load_history_rows()
    blocks = component_blocks(combined)
    v0 = mean_predictor_cv(dev, fold)
    A = analyse_table(t, names, v0["cv_pooled_rmse_mean"], rankings=None)
    A["v_empty"] = v0
    by_key = {r["key"]: r for r in t.to_dict("records")}
    full = by_key[subset_key(names)]["cv_pooled_rmse_mean"]

    def cv(blocks_sel) -> float:
        return float(by_key[subset_key(canonical(blocks_sel, names))]["cv_pooled_rmse_mean"])

    A["designs"] = designs(cv, names, full, A["shapley"])
    A["named_subsets"] = {n: {"blocks": canonical(b, names), "cv": cv(b),
                              "rel_vs_all": (cv(b) - full) / full} for n, b in NAMED.items()}
    records = score_batch3(dev, test, blocks, combined, names, A["best_per_k"])
    A["heldout"] = list(records.values())
    A["heldout_audit"] = {"n_subsets_scored_on_batch3": len(records)}
    A["checks"] = cross_checks(cv, full)
    A["env"] = env_info()
    return A


def write_analysis(A: dict) -> None:
    """results/<stem>_analysis.json and its manifest entry."""
    write_json_atomic(ANALYSIS_JSON, A, indent=1)
    print(f"wrote results/{ANALYSIS_JSON.name}", flush=True)
    log_manifest(f"{STEM}_analysis", {
        "script": f"{SCRIPT} --analyse", "seed": RF_SEED,
        "designs": {d["component"]: {"cv_alone": d["cv_alone"], "cv_all_minus": d["cv_all_minus"]}
                    for d in A["designs"]},
        "best_per_k": {r["k"]: [r["blocks"], r["cv"]] for r in A["best_per_k"]},
        "checks": A["checks"]})


# ------------------------ seed check ------------------------

def seed_check(analysis: dict | None = None, seeds: tuple[int, ...] = EXTRA_SEEDS) -> None:
    """
    Batch 3 prefers 'all 63 minus rm20' to 'all 63' although dev-CV prefers the
    opposite. Re-score both with two more forest seeds (dev-CV and batch 3) to
    show that this is not seed noise; seed 0 is read from the stored table and
    from the analysis (passed in, or read from its file), so this runs after
    --merge and --analyse.
    """
    table = read_json(TABLE_JSON)
    analysis = analysis if analysis is not None else read_json(ANALYSIS_JSON)
    if table is None or analysis is None:
        raise FileNotFoundError("the seed check reads seed 0 from results/"
                                f"{TABLE_JSON.name} and {ANALYSIS_JSON.name}: "
                                "run the search (--merge) and --analyse first")
    names = list(COMPONENTS)
    dev, test, fold, combined, _ = load_history_rows()
    blocks = component_blocks(combined)
    subsets = {"all 63": tuple(names),
               "all 63 minus rm20": canonical([c for c in names if c != "rm20"], names)}
    out = {"seeds": [0, *seeds], "runs": []}
    stored = {subset_key(r["blocks"]): r for r in table["table"]}
    ho0 = {subset_key(r["blocks"]): r for r in analysis["heldout"]}
    for label, sub in subsets.items():
        k = subset_key(sub)
        out["runs"].append({"seed": 0, "subset": label, "cv": stored[k]["cv_pooled_rmse_mean"],
                            "cv_fold": stored[k]["cv_fold_rmse"], "b3_rmse": ho0[k]["b3_rmse"],
                            "b3_r2": ho0[k]["b3_r2"], "source": "stored ENG-22 table/analysis"})
    for s in seeds:
        make = rf_factory({"random_state": s})
        for label, sub in subsets.items():
            cols = columns_for(sub, blocks, combined)
            cv = cv_evaluate(dev, cols, fold, make, None)
            ho = heldout_evaluate(dev, test, cols, make, None)
            out["runs"].append({"seed": s, "subset": label, "cv": cv["pooled_rmse_mean"],
                                "cv_fold": [m["pooled"]["rmse"] for m in cv["per_fold"]],
                                "b3_rmse": ho["pooled"]["rmse"], "b3_r2": ho["pooled"]["r2"]})
            print(f"  seed {s} {label:18s} CV {cv['pooled_rmse_mean']:.5f} "
                  f"b3 {ho['pooled']['rmse']:.5f}", flush=True)
    df = pd.DataFrame(out["runs"])
    out["summary"] = {lab: {"cv_mean": float(g["cv"].mean()), "cv_sd": float(g["cv"].std(ddof=1)),
                            "b3_mean": float(g["b3_rmse"].mean()), "b3_sd": float(g["b3_rmse"].std(ddof=1)),
                            "b3_r2_mean": float(g["b3_r2"].mean())}
                      for lab, g in df.groupby("subset")}
    out["env"] = env_info()
    write_json_atomic(SEED_CHECK_JSON, out, indent=1)
    print(f"wrote results/{SEED_CHECK_JSON.name}", flush=True)
    print(out["summary"])
    log_manifest(f"{STEM}_seed_check", {"script": f"{SCRIPT} --seed-check",
                                        "seeds": [0, *seeds], "summary": out["summary"]})


# ------------------------ entry point ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    add_search_args(ap)
    ap.add_argument("--seed-check", action="store_true",
                    help="only the seed check: 'all 63' vs 'all 63 minus rm20' with forest "
                         "seeds 1 and 2 (seed 0 from the stored table and analysis)")
    args = ap.parse_args()
    if args.figures:
        print("nothing to draw here: run_report.py makes the tables from the result files")
        return
    if args.merge:
        merge(args.tag)
        return
    if args.analyse:
        write_analysis(analyse())
        return
    if args.seed_check:
        seed_check()
        return
    if args.n_shards > 1 or args.limit:   # a cluster shard or a smoke run: progress file only
        search(args)
        return

    # The whole stage; a --tag run stops at its table (analysis and seed check
    # read the untagged files).
    table = RESULTS_DIR / f"{STEM}{args.tag}.json"
    if table.exists():
        print(f"[{STEM}] results/{table.name} exists: search skipped (delete it to rescore)",
              flush=True)
    else:
        search(args)
    if args.tag:
        return
    # The analysis file is written last: run_results.py takes it as the sign that
    # this stage is done, so an interrupted seed check must not leave it behind.
    A = analyse()
    seed_check(A)
    write_analysis(A)


if __name__ == "__main__":
    main()
