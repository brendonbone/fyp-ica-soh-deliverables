"""
12. The reference model over ten forest seeds (ENG-20)
======================================================
How much does the reference model (the RF on all nine level features) move
with the forest's random_state and with the column order? Its stored batch-3
RMSE differs in the third significant figure between Pearson order (0.01557,
tuned) and pool order (0.01565), because the forest's random draws and its
tie-breaks between equally good splits go by column index. This stage scores

    {untuned, tuned (ENG-09 params)} x {pool order, Pearson order} x random_state 0..9

= 40 forests, each by the study's 5-fold dev CV and once on batch 3, and
reports mean +/- sd (ddof 1) over the ten seeds per configuration and order,
and over both orders together. Seed 0 must reproduce the stored ENG-08,
ENG-09 and ENG-19 numbers exactly (abs diff <= 1e-10); the checks are stored
in the JSON, not enforced.

Inputs   results/features_expanded.parquet (stage 2), eng08_full.json and
         eng08_full_tuned.json (stage 7; the Pearson order is read from the
         untuned grid, not recomputed), eng09_tuned_params.json and
         eng09_tuning.json (stage 6), eng19_exhaustive_subsets.csv (stage 11).
Outputs  results/eng20_reference_seeds.json: configs, orders, seeds,
         summary {config|order, config|both_orders: {cv_rmse, cv_r2, b3_rmse,
         b3_r2, b3_mae: {mean, sd, min, max}, n}}, checks_seed0, all_checks_pass,
         runs (the 40 records), env.
         results/eng20_progress.json: one record per forest, written after
         each one, so a rerun skips the forests already scored.
Runtime  about 15 min on a 24-core laptop (untuned ~30 s, tuned ~15 s per forest).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.features_temporal import LEVEL_COLS
from src.harness import cv_evaluate, heldout_evaluate
from src.subset_search import (
    env_info,
    load_level_rows,
    read_json,
    rf_factory,
    subset_key,
    tuned_rf_params,
    write_json_atomic,
)

PROGRESS = RESULTS_DIR / "eng20_progress.json"
OUT_JSON = RESULTS_DIR / "eng20_reference_seeds.json"
SEEDS = list(range(10))
METRICS = ["cv_rmse", "cv_r2", "b3_rmse", "b3_r2", "b3_mae"]

# Result files of earlier stages that this stage reads, and the stage that writes them.
NEEDED = {
    "eng08_full.json": "stage 7", "eng08_full_tuned.json": "stage 7",
    "eng09_tuned_params.json": "stage 6", "eng09_tuning.json": "stage 6",
    "eng19_exhaustive_subsets.csv": "stage 11",
}


def require_inputs() -> None:
    """Stop with a clear message when an earlier stage's result file is missing."""
    missing = [f"results/{name} ({stage})" for name, stage in NEEDED.items()
               if not (RESULTS_DIR / name).exists()]
    if missing:
        sys.exit("missing inputs, run the earlier stages first: " + ", ".join(missing))


def score(dev, test, fold, cols: list[str], params: dict) -> dict:
    """Dev-CV and batch-3 metrics of one forest on `cols` in that column order."""
    make = rf_factory(params)
    cv = cv_evaluate(dev, cols, fold, make, None)
    ho = heldout_evaluate(dev, test, cols, make, None)
    return {"cv_rmse": cv["pooled_rmse_mean"], "cv_rmse_std": cv["pooled_rmse_std"],
            "cv_r2": cv["pooled_r2_mean"],
            "b3_rmse": ho["pooled"]["rmse"], "b3_r2": ho["pooled"]["r2"],
            "b3_mae": ho["pooled"]["mae"], "b3_rmse_cellavg": ho["cell_avg"]["rmse"]}


def run_forests(dev, test, fold, configs: dict, orders: dict) -> dict:
    """Score every (config, order, seed) not yet in the progress file; return all records."""
    done = read_json(PROGRESS, {})
    units = list(itertools.product(configs, orders, SEEDS))
    t_start = time.time()
    for cfg, order, seed in units:
        key = f"{cfg}|{order}|seed{seed}"
        if key in done:
            continue
        t0 = time.time()
        r = score(dev, test, fold, orders[order], {**configs[cfg], "random_state": seed})
        done[key] = {"config": cfg, "order": order, "seed": seed, **r,
                     "runtime_s": round(time.time() - t0, 1)}
        write_json_atomic(PROGRESS, done)
        print(f"[{len(done):2d}/{len(units)}] {key}: CV {r['cv_rmse']:.5f} "
              f"b3 {r['b3_rmse']:.5f} R2 {r['b3_r2']:.4f} "
              f"({time.time() - t0:.0f} s, {(time.time() - t_start) / 60:.1f} min)", flush=True)
    return done


def summarise(done: dict) -> dict:
    """Mean, sd (ddof 1 over seeds), min, max of each metric per config|order and per config."""
    df = pd.DataFrame(done.values())

    def summ(g: pd.DataFrame) -> dict:
        return {m: {"mean": float(g[m].mean()), "sd": float(g[m].std(ddof=1)),
                    "min": float(g[m].min()), "max": float(g[m].max())} for m in METRICS} | {
            "n": int(len(g))}

    summary = {f"{c}|{o}": summ(g) for (c, o), g in df.groupby(["config", "order"])}
    summary.update({f"{c}|both_orders": summ(g) for c, g in df.groupby("config")})
    return summary


def seed0_checks(done: dict, e08: dict, e08t: dict) -> dict:
    """Seed 0 of each config|order against the stored ENG-08 / ENG-09 / ENG-19 numbers."""
    e19 = pd.read_csv(RESULTS_DIR / "eng19_exhaustive_subsets.csv").set_index("features")
    e09 = read_json(RESULTS_DIR / "eng09_tuning.json")["before_after_heldout"]
    ref = {
        "untuned|pearson": (e08["k_curve"][-1]["cv_pooled_rmse_mean"], e08["heldout_scores"]["pooled"]["rmse"]),
        "untuned|pool": (float(e19.loc[subset_key(LEVEL_COLS), "cv_pooled_rmse_mean"]),
                         e09["rf_untuned"]["pooled"]["rmse"]),
        "tuned|pearson": (e08t["k_curve"][-1]["cv_pooled_rmse_mean"], e08t["heldout_scores"]["pooled"]["rmse"]),
        "tuned|pool": (None, e09["rf_tuned"]["pooled"]["rmse"]),   # ENG-09 stored no CV for it
    }
    checks = {}
    for k, (cv_ref, b3_ref) in ref.items():
        r = done[f"{k}|seed0"]
        checks[k] = {"cv": r["cv_rmse"], "cv_stored": cv_ref,
                     "cv_abs_diff": None if cv_ref is None else abs(r["cv_rmse"] - cv_ref),
                     "b3": r["b3_rmse"], "b3_stored": b3_ref, "b3_abs_diff": abs(r["b3_rmse"] - b3_ref)}
        checks[k]["pass"] = bool(checks[k]["b3_abs_diff"] <= 1e-10 and
                                 (cv_ref is None or checks[k]["cv_abs_diff"] <= 1e-10))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()
    require_inputs()

    e08 = read_json(RESULTS_DIR / "eng08_full.json")["grid"]["pearson|rf"]
    e08t = read_json(RESULTS_DIR / "eng08_full_tuned.json")["grid"]["pearson|rf"]
    orders = {"pool": list(LEVEL_COLS),
              "pearson": [r["feature"] for r in e08["ranking"]]}
    configs = {"untuned": {}, "tuned": tuned_rf_params()}
    dev, test, fold = load_level_rows()

    done = run_forests(dev, test, fold, configs, orders)
    summary = summarise(done)
    checks = seed0_checks(done, e08, e08t)
    out = {"configs": configs, "orders": orders, "seeds": SEEDS, "summary": summary,
           "checks_seed0": checks, "all_checks_pass": all(c["pass"] for c in checks.values()),
           "runs": sorted(done.values(), key=lambda r: (r["config"], r["order"], r["seed"])),
           "env": env_info()}
    write_json_atomic(OUT_JSON, out, indent=1)

    print("\n=== all nine features: mean +/- sd over seeds 0-9 ===")
    for k, s in summary.items():
        print(f"  {k:22s} CV {s['cv_rmse']['mean']:.5f} +/- {s['cv_rmse']['sd']:.5f} | "
              f"b3 {s['b3_rmse']['mean']:.5f} +/- {s['b3_rmse']['sd']:.5f} "
              f"[{s['b3_rmse']['min']:.5f}, {s['b3_rmse']['max']:.5f}] | "
              f"R2 {s['b3_r2']['mean']:.4f} +/- {s['b3_r2']['sd']:.4f}")
    print("seed-0 checks:", {k: v["pass"] for k, v in checks.items()})
    print(f"wrote {OUT_JSON.relative_to(RESULTS_DIR.parent)}")
    log_manifest("eng20_reference_seeds", {
        "script": "stages/s12_reference_seeds.py", "seeds": SEEDS,
        "seed_note": f"random_state 0..9 for the forest; data/folds use seed {RF_SEED}",
        "b3_rmse_mean_sd": {k: [s["b3_rmse"]["mean"], s["b3_rmse"]["sd"]] for k, s in summary.items()},
        "checks_seed0_pass": out["all_checks_pass"]})


if __name__ == "__main__":
    main()
