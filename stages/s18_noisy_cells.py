"""
18. Training without the 16 cells with noisy published curves (ENG-27)
======================================================================
Some cells' published dQ/dV is so noisy that `height` jumps from one cycle to
the next. Which cells are they, and does training without them change three
thesis models? The rule uses no SoH label: per cell, the median |height(k) -
height(k-1)| over consecutive status-ok cycles of the raw parquet, divided by
the cell's median height; noisy when > 0.05. On the published data this gives
16 cells (b1c38-45, b2c40-47), all in the dev batches; the largest value of
the other 108 cells is 0.047 (b2c1), the smallest of the 16 is 0.101.

Models: tuned RF (ENG-09 params), forest seeds 0-4, on
  level9   the nine level features, ENG-08 rows, pool order (thesis reference)
  hp_hist  height + position with their history columns, history rows
  all63    the 63-column combined pool, history rows
Each (model, seed) is trained twice, on every dev cell and without the noisy
ones, with the ENG-08 cell -> fold map unchanged. dev-CV = mean over the five
folds of the pooled RMSE over the clean validation cells (the paired
comparison), and, for reference, over all and over the noisy validation
cells; batch 3 = one refit on the training cells, scored on all 40 cells. At
seed 0, for level9 and hp_hist, each batch-3 cell's RMSE change is set against
the cell's first charge rate. Checks (training on every dev cell): level9
reproduces ENG-20 tuned|pool seeds 0-4 (<= 1e-10), hp_hist and all63 at seed
0 reproduce ENG-23's tuned values (<= 1e-6). Batch 3 decides nothing here.

Inputs   results/features_expanded.parquet (stage 2), eng09_tuned_params.json
         (stage 6), eng20_reference_seeds.json (stage 12),
         eng23_grouped_history_subsets_analysis.json (stage 16).
Outputs  results/eng27_noisy_cells.json: rule, noisy_cells, cells (per-cell
         height_step table), gap, models, params, seeds, summary {model:
         {n_columns, all_dev, without_noisy, change_cv_clean_rmse,
         change_b3_rmse, change_b3_r2}}, checks, all_checks_pass,
         b3_per_cell_seed0, runs (30 records), heldout_audit, env.
         results/eng27_progress.json: one record per (model, train set, seed),
         written after each one, so a rerun skips the units already scored.
Runtime  about 25 min on a 24-core laptop (per unit: all63 ~100 s, hp_hist
         ~23 s, level9 ~15 s; each unit = 5 fold fits + 1 batch-3 fit).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import both_metrics, log_manifest
from src.features_temporal import LEVEL_COLS
from src.subset_search import (
    PARQUET,
    columns_for,
    env_info,
    grouped_blocks,
    load_clean,
    load_history_rows,
    load_level_rows,
    read_json,
    rf_factory,
    tuned_rf_params,
    write_json_atomic,
)

NOISE_THRESHOLD = 0.05
SEEDS = [0, 1, 2, 3, 4]
TRAIN_SETS = ("all_dev", "without_noisy")
PROGRESS = RESULTS_DIR / "eng27_progress.json"
OUT_JSON = RESULTS_DIR / "eng27_noisy_cells.json"

# Result files of earlier stages that this stage reads, and the stage that writes them.
NEEDED = {
    "features_expanded.parquet": "stage 2", "eng09_tuned_params.json": "stage 6",
    "eng20_reference_seeds.json": "stage 12",
    "eng23_grouped_history_subsets_analysis.json": "stage 16",
}


def require_inputs() -> None:
    """Stop with a clear message when an earlier stage's result file is missing."""
    missing = [f"results/{name} ({stage})" for name, stage in NEEDED.items()
               if not (RESULTS_DIR / name).exists()]
    if missing:
        sys.exit("missing inputs, run the earlier stages first: " + ", ".join(missing))


def natural(cid: str) -> tuple[int, int]:
    """'b2c40' -> (2, 40), so cells sort by batch then cell number."""
    return int(cid[1]), int(cid[3:])


def noisy_cells() -> tuple[list[str], pd.DataFrame]:
    """The noisy cells (natural order) and the per-cell height-step table."""
    # status-ok rows of the raw table, not the SoH-cleaned rows: the rule uses no label.
    feat = pd.read_parquet(PARQUET)
    ok = feat[feat["status"] == "ok"].sort_values(["cell_id", "cycle"])
    step = ok.groupby("cell_id")["height"].agg(
        lambda h: float(np.median(np.abs(np.diff(h.to_numpy()))) / np.median(h.to_numpy())))
    table = pd.DataFrame({
        "batch": ok.groupby("cell_id")["batch"].first(),
        "policy": ok.groupby("cell_id")["charge_policy"].first(),
        "ok_cycles": ok.groupby("cell_id").size(),
        "height_step": step,
        "rc_rms_diff_median": ok.groupby("cell_id")["rc_rms_diff"].median(),
    })
    table["noisy"] = table["height_step"] > NOISE_THRESHOLD
    return sorted(table.index[table["noisy"]], key=natural), table


def load_models(noisy: list[str]) -> dict:
    """model name -> (dev, test, fold, columns) for level9, hp_hist and all63."""
    clean_rows = load_clean()
    level_dev, level_test, level_fold = load_level_rows(clean_rows)
    hist_dev, hist_test, hist_fold, combined, _ = load_history_rows(clean_rows)
    assert not level_test["cell_id"].isin(noisy).any(), "a batch-3 cell is noisy"
    blocks = grouped_blocks(combined)
    return {
        "level9": (level_dev, level_test, level_fold, list(LEVEL_COLS)),
        # the combined-pool column order, as ENG-23 fitted it (the forest depends on it)
        "hp_hist": (hist_dev, hist_test, hist_fold,
                    columns_for(["height", "position"], blocks, combined)),
        "all63": (hist_dev, hist_test, hist_fold, list(combined)),
    }


def evaluate(dev, test, fold, cols, make, noisy, train_set) -> dict:
    """Per fold: fit on the chosen training cells, score the whole validation fold over
    all / clean / noisy cells; then refit on the training cells and score batch 3."""
    is_noisy = dev["cell_id"].isin(noisy).to_numpy()
    keep = ~is_noisy if train_set == "without_noisy" else np.ones(len(dev), bool)
    scores = {"all": [], "clean": [], "noisy": []}
    n_val = []
    for f in sorted(fold.unique()):
        in_fold = (fold == f).to_numpy()
        tr = dev[~in_fold & keep]
        va = dev[in_fold]   # every validation row, noisy cells included, in both arms
        est = make()
        est.fit(tr[cols], tr["SOH"])
        pred = est.predict(va[cols])
        noisy_va = is_noisy[in_fold]
        for scope, m in (("all", np.ones(len(va), bool)), ("clean", ~noisy_va),
                         ("noisy", noisy_va)):
            if m.any():
                scores[scope].append(both_metrics(va["cell_id"][m], va["SOH"][m], pred[m]))
        n_val.append({"fold": int(f), "train_rows": len(tr),
                      "val_cells_clean": int(va["cell_id"][~noisy_va].nunique()),
                      "val_cells_noisy": int(va["cell_id"][noisy_va].nunique())})
    out = {}
    for scope, per_fold in scores.items():
        if per_fold:
            out[f"cv_{scope}_rmse"] = float(np.mean([m["pooled"]["rmse"] for m in per_fold]))
            out[f"cv_{scope}_r2"] = float(np.mean([m["pooled"]["r2"] for m in per_fold]))
            out[f"cv_{scope}_folds"] = len(per_fold)
    tr = dev[keep]
    est = make()
    est.fit(tr[cols], tr["SOH"])
    ho = both_metrics(test["cell_id"], test["SOH"], est.predict(test[cols]))
    out.update({"b3_rmse": ho["pooled"]["rmse"], "b3_r2": ho["pooled"]["r2"],
                "b3_mae": ho["pooled"]["mae"], "b3_rmse_cellavg": ho["cell_avg"]["rmse"],
                "train_rows": int(keep.sum()), "train_cells": int(dev["cell_id"][keep].nunique()),
                "folds": n_val})
    return out


def run_units(models: dict, params: dict, noisy: list[str]) -> dict:
    """Score every (model, train set, seed) not yet in the progress file; return all records."""
    done = read_json(PROGRESS, {})
    units = [(m, t, s) for s in SEEDS for m in models for t in TRAIN_SETS]
    t_start = time.time()
    for model, train_set, seed in units:
        key = f"{model}|{train_set}|seed{seed}"
        if key in done:
            continue
        t0 = time.time()
        dev, test, fold, cols = models[model]
        make = rf_factory({**params, "random_state": seed})
        rec = evaluate(dev, test, fold, cols, make, noisy, train_set)
        rec.update({"model": model, "train_set": train_set, "seed": seed, "n_columns": len(cols),
                    "runtime_s": round(time.time() - t0, 1)})
        done[key] = rec
        write_json_atomic(PROGRESS, done)
        print(f"[{len(done):2d}/{len(units)}] {key}: CV clean {rec['cv_clean_rmse']:.5f} "
              f"b3 {rec['b3_rmse']:.5f} R2 {rec['b3_r2']:.4f} ({rec['runtime_s']:.0f} s, "
              f"{(time.time() - t_start) / 60:.1f} min)", flush=True)
    return done


def reproduction_checks(done: dict) -> list[dict]:
    """Training on every dev cell must reproduce ENG-20 (level9) and ENG-23 (hp_hist, all63)."""
    e20 = {(r["config"], r["order"], r["seed"]): r
           for r in read_json(RESULTS_DIR / "eng20_reference_seeds.json")["runs"]}
    e23 = read_json(RESULTS_DIR / "eng23_grouped_history_subsets_analysis.json")["tuned"]
    checks = []
    for seed in SEEDS:
        r = done[f"level9|all_dev|seed{seed}"]
        ref = e20[("tuned", "pool", seed)]
        checks.append({"what": f"level9 seed {seed} vs ENG-20 tuned|pool",
                       "cv_diff": abs(r["cv_all_rmse"] - ref["cv_rmse"]),
                       "b3_diff": abs(r["b3_rmse"] - ref["b3_rmse"]), "tolerance": 1e-10})
    for model, key in (("hp_hist", "height+position"), ("all63", "+".join(LEVEL_COLS))):
        r = done[f"{model}|all_dev|seed0"]
        checks.append({"what": f"{model} seed 0 vs ENG-23 tuned {key}",
                       "cv_diff": abs(r["cv_all_rmse"] - e23[key]["cv"]),
                       "b3_diff": abs(r["b3_rmse"] - e23[key]["b3_rmse"]),
                       # history columns can differ in the last bits across hosts (ENG-22..25)
                       "tolerance": 1e-6})
    for c in checks:
        c["pass"] = bool(c["cv_diff"] <= c["tolerance"] and c["b3_diff"] <= c["tolerance"])
    return checks


def summarise(done: dict, models) -> dict:
    """Per model and train set: mean +/- sd (ddof 1) over seeds; and the paired change per seed."""
    runs = pd.DataFrame(done.values())
    metrics = ["cv_clean_rmse", "cv_clean_r2", "b3_rmse", "b3_r2", "b3_mae", "b3_rmse_cellavg"]
    summary = {}
    for model in models:
        m = runs[runs["model"] == model]
        a = m[m["train_set"] == "all_dev"].set_index("seed").sort_index()
        b = m[m["train_set"] == "without_noisy"].set_index("seed").sort_index()
        s = {"n_columns": int(a["n_columns"].iloc[0])}
        for ts, g in (("all_dev", a), ("without_noisy", b)):
            s[ts] = {k: {"mean": float(g[k].mean()), "sd": float(g[k].std(ddof=1))} for k in metrics}
            s[ts]["train_rows"] = int(g["train_rows"].iloc[0])
            s[ts]["train_cells"] = int(g["train_cells"].iloc[0])
        s["all_dev"]["cv_all_rmse"] = {"mean": float(a["cv_all_rmse"].mean()),
                                       "sd": float(a["cv_all_rmse"].std(ddof=1))}
        s["all_dev"]["cv_noisy_rmse"] = {"mean": float(a["cv_noisy_rmse"].mean()),
                                         "sd": float(a["cv_noisy_rmse"].std(ddof=1))}
        for k in ("cv_clean_rmse", "b3_rmse", "b3_r2"):
            d = b[k] - a[k]
            s[f"change_{k}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                "relative_mean": float((d / a[k]).mean()),
                                "seeds_lower": int((d < 0).sum()), "n_seeds": len(d)}
        summary[model] = s
    return summary


def b3_per_cell(models: dict, params: dict, noisy: list[str],
                names=("level9", "hp_hist")) -> dict:
    """Seed 0: each batch-3 cell's RMSE with and without the noisy training cells, against the
    test cell's first-step charge rate (is the loss concentrated on fast-charged test cells?)."""
    out = {}
    for name in names:
        dev, test, _, cols = models[name]
        pred = {}
        for train_set in TRAIN_SETS:
            tr = dev if train_set == "all_dev" else dev[~dev["cell_id"].isin(noisy)]
            est = rf_factory({**params, "random_state": 0})()
            est.fit(tr[cols], tr["SOH"])
            pred[train_set] = est.predict(test[cols])
        err = pd.DataFrame({"cell_id": test["cell_id"].to_numpy(),
                            **{t: (pred[t] - test["SOH"].to_numpy()) ** 2 for t in TRAIN_SETS}})
        per = np.sqrt(err.groupby("cell_id")[list(TRAIN_SETS)].mean())
        per["change"] = per["without_noisy"] - per["all_dev"]
        policy = test.groupby("cell_id")["charge_policy"].first()
        per["first_c_rate"] = policy.str.extract(r"^([0-9.]+)C")[0].astype(float)
        rho = per[["first_c_rate", "change"]].corr(method="spearman").iloc[0, 1]
        out[name] = {"cells_worse": int((per["change"] > 0).sum()), "cells": len(per),
                     "spearman_rate_vs_change": float(rho),
                     "median_change_by_rate": per.groupby("first_c_rate")["change"].median().to_dict(),
                     "cells_table": per.reset_index().to_dict("records")}
        print(f"  batch 3 per cell, {name}: worse in {out[name]['cells_worse']}/{len(per)} cells; "
              f"Spearman(charge rate, change) {rho:+.2f}", flush=True)
    return out


def print_summary(out: dict) -> None:
    """The gap between clean and noisy cells, then each model's all-dev -> without-noisy change."""
    print(f"\nnoisy cells: {len(out['noisy_cells'])}; height-step gap "
          f"{out['gap']['largest_clean']:.4f} (largest clean) vs "
          f"{out['gap']['smallest_noisy']:.4f} (smallest noisy)")
    for model, s in out["summary"].items():
        a, b = s["all_dev"], s["without_noisy"]
        print(f"\n{model} ({s['n_columns']} columns): train rows {a['train_rows']:,} -> "
              f"{b['train_rows']:,}, cells {a['train_cells']} -> {b['train_cells']}")
        print(f"  dev-CV, all validation cells (published)  {a['cv_all_rmse']['mean']:.5f}")
        print(f"  dev-CV, noisy validation cells only        {a['cv_noisy_rmse']['mean']:.5f}")
        for k, label in (("cv_clean_rmse", "dev-CV, clean validation cells"),
                         ("b3_rmse", "batch 3 RMSE"), ("b3_r2", "batch 3 R2")):
            c = s[f"change_{k}"]
            print(f"  {label:42s} {a[k]['mean']:.5f} +/- {a[k]['sd']:.5f} -> "
                  f"{b[k]['mean']:.5f} +/- {b[k]['sd']:.5f}  (change {c['mean']:+.5f}, "
                  f"{c['relative_mean']:+.1%}, lower in {c['seeds_lower']}/{c['n_seeds']} seeds)")
    print("\nchecks:", "PASS" if out["all_checks_pass"] else "FAIL",
          [(c["what"], f"{max(c['cv_diff'], c['b3_diff']):.1e}") for c in out["checks"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()
    require_inputs()

    noisy, cell_table = noisy_cells()
    print(f"noisy cells ({len(noisy)}): {', '.join(noisy)}", flush=True)
    models = load_models(noisy)
    params = tuned_rf_params()

    done = run_units(models, params, noisy)
    checks = reproduction_checks(done)
    summary = summarise(done, models)
    per_cell = b3_per_cell(models, params, noisy)
    runs = sorted(done.values(), key=lambda r: (r["model"], r["train_set"], r["seed"]))
    audit = [{"model": r["model"], "train_set": r["train_set"], "seed": r["seed"],
              "reason": "data_quality_check (no decision)", "b3_rmse": r["b3_rmse"]} for r in runs]
    out = {
        "rule": {"measure": "median |height(k) - height(k-1)| / median height, status-ok rows",
                 "threshold": NOISE_THRESHOLD},
        "noisy_cells": noisy,
        "cells": cell_table.reset_index().rename(columns={"index": "cell_id"})
                           .sort_values("height_step", ascending=False).to_dict("records"),
        "gap": {"largest_clean": float(cell_table.loc[~cell_table["noisy"], "height_step"].max()),
                "smallest_noisy": float(cell_table.loc[cell_table["noisy"], "height_step"].min())},
        "models": {k: {"rows": "ENG-08 level rows" if k == "level9" else "ENG-14 history rows",
                       "columns": v[3]} for k, v in models.items()},
        "params": params, "seeds": SEEDS, "summary": summary,
        "checks": checks, "all_checks_pass": all(c["pass"] for c in checks),
        "b3_per_cell_seed0": per_cell,
        "runs": runs,
        "heldout_audit": audit, "env": env_info(),
    }
    write_json_atomic(OUT_JSON, out, indent=1)

    print_summary(out)
    print(f"wrote {OUT_JSON.relative_to(RESULTS_DIR.parent)}")
    log_manifest("eng27_noisy_cells", {
        "script": "stages/s18_noisy_cells.py", "noisy_cells": noisy, "seeds": SEEDS,
        "b3_rmse_all_dev_vs_without_noisy": {
            k: [s["all_dev"]["b3_rmse"]["mean"], s["without_noisy"]["b3_rmse"]["mean"]]
            for k, s in summary.items()},
        "checks_pass": out["all_checks_pass"]})


if __name__ == "__main__":
    main()
