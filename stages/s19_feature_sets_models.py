"""
19. The chosen feature sets under four models (ENG-28)
======================================================
Do the feature sets the earlier stages chose hold up under other models?
Eleven sets, read from the analyses that chose them (stages 11, 14, 16) --
interim3, all9, best4, budget7, all63, lvl_rm_drift, hist_rule, hist_budget,
interim3_hist, best4_hist, budget7_hist (the `--sets` ids) -- each scored with
LR, untuned GPR, untuned RF and the ENG-09 tuned RF: 5-fold dev-CV on the
ENG-14 history rows (ENG-08 cell -> fold map), then batch 3 once per cell.
Inputs   features_expanded.parquet (stage 2), eng09_tuned_params.json (stage 6),
         eng17_soc_windows_summary.csv (stage 10), the ENG-19b/22/23/24 result
         and analysis files (stages 11, 14, 15, 16), all under results/.
Outputs  results/eng28_feature_sets_models.json (protocol, feature_sets, cells
         in dev-CV rank order, checks_vs_stored, checks_picks, all_checks_pass,
         heldout_audit, env) and the cells as eng28_feature_sets_models.csv.
Runtime  ~73 min of fits (~49 min GPR). Resumable: one progress file per
         model, written after every cell (`--models GPR` can run in a second
         process); `--merge` assembles the four files.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import BASELINE_FEATURES, RF_SEED, log_manifest
from src.features_temporal import LEVEL_COLS
from src.harness import REGRESSORS, cv_evaluate, heldout_evaluate
from src.subset_search import (
    columns_for,
    component_blocks,
    env_info,
    feature_windows_from_medians,
    grouped_blocks,
    level_blocks,
    load_history_rows,
    per_cycle_spans,
    read_json,
    subset_window,
    tuned_rf_params,
    write_json_atomic,
)

STEM = "eng28_feature_sets_models"
SCRIPT = "stages/s19_feature_sets_models.py"
OUT_JSON = RESULTS_DIR / f"{STEM}.json"
OUT_CSV = RESULTS_DIR / f"{STEM}.csv"

A19 = RESULTS_DIR / "eng19b_analysis.json"
A22 = RESULTS_DIR / "eng22_history_components_analysis.json"
A23 = RESULTS_DIR / "eng23_grouped_history_subsets_analysis.json"
A24 = RESULTS_DIR / "eng24_level_subsets_history_rows_analysis.json"
T22 = RESULTS_DIR / "eng22_history_components.json"
T23 = RESULTS_DIR / "eng23_grouped_history_subsets.json"
T24 = RESULTS_DIR / "eng24_level_subsets_history_rows.json"

# Result files of earlier stages that this stage reads, and the stage that writes them.
NEEDED = {
    "eng09_tuned_params.json": "stage 6", "eng17_soc_windows_summary.csv": "stage 10",
    A19.name: "stage 11", T22.name: "stage 14", A22.name: "stage 14",
    T24.name: "stage 15", A24.name: "stage 15", T23.name: "stage 16", A23.name: "stage 16",
}

MODELS: list[str] = ["LR", "GPR", "RF", "RF tuned"]
MODEL_SPEC = {  # label -> (harness registry key, description)
    "LR": ("linear", "linear regression (standardised inputs)"),
    "GPR": ("gpr", "GPR, RBF + white noise, 2000-row subsample, untuned"),
    "RF": ("rf", "random forest, 300 trees, untuned"),
    "RF tuned": ("rf", "random forest, ENG-09 fixed tuned configuration"),
}
HELDOUT_REASON = "methods_models_table_XV"

# The picks the upstream analyses made on 2026-10-11; a changed pick is
# reported (all_checks_pass False), not silently absorbed.
EXPECTED = {
    "best4": ["position", "fwhm", "slope_right", "prominence"],
    "budget7": ["height", "position", "area", "area35", "area25", "slope_left", "slope_right"],
    "lvl_rm_drift": ["level", "drift", "rm20"],
    "hist_rule": ["height", "fwhm"],
    "hist_budget": ["height", "position"],
    "interim3_hist": ["height", "position", "area"],
    "best4_hist": ["position", "fwhm", "slope_right", "prominence"],
    "budget7_hist": ["height", "position", "area", "area35", "area25", "slope_left", "slope_right"],
}

# ENG-22/23/24 units ran partly on Katana; Linux and Windows builds differ by
# <= 3.4e-7 in rare near-tie splits, so 1e-6 is the bar for the cross-checks.
CHECK_TOL = 1e-6


def require_inputs() -> None:
    """Stop with a clear message when an earlier stage's result file is missing."""
    missing = [f"results/{name} ({stage})" for name, stage in NEEDED.items()
               if not (RESULTS_DIR / name).exists()]
    if missing:
        sys.exit("missing inputs, run the earlier stages first: " + ", ".join(missing))


def slug(model: str) -> str:
    """'RF tuned' -> 'rf_tuned' (progress-file suffix)."""
    return model.lower().replace(" ", "_")


def progress_path(model: str) -> Path:
    """The resume file of one model."""
    return RESULTS_DIR / f"{STEM}_progress_{slug(model)}.json"


# ------------------------ the eleven feature sets ------------------------

def feature_sets() -> list[dict]:
    """The eleven sets, with their blocks read from the analyses that chose them."""
    a19, a22, a23 = read_json(A19), read_json(A22), read_json(A23)
    rule19 = a19["selection_rules"]["one_standard_error"]["pick"]["blocks"]
    # The 42.3 % step: the largest edge step that does not need the whole
    # discharge (prominence and fwhm need ~96 %).
    partial = [e for e in a19["tradeoff_edge"] if e["soc_span"] < 0.9]
    budget7 = max(partial, key=lambda e: e["soc_span"])["blocks"]
    comp = a22["named_subsets"]["level + rm20 + drift"]["blocks"]
    rule23 = a23["selection_rules"]["one_standard_error"]["pick"]["blocks"]
    budget2 = min(a23["tradeoff_edge"], key=lambda e: e["soc_span"])["blocks"]
    return [
        {"id": "interim3", "label": "Interim three features", "kind": "level",
         "blocks": list(BASELINE_FEATURES), "section": "C",
         "method": "interim three-feature model (semester 1)"},
        {"id": "all9", "label": "All nine features", "kind": "level",
         "blocks": list(LEVEL_COLS), "section": "A, C",
         "method": "expanded nine-feature pool (reference)"},
        {"id": "best4", "label": "Best four-feature subset", "kind": "level",
         "blocks": list(rule19), "section": "D",
         "method": "exhaustive subset search, one-standard-error rule"},
        {"id": "budget7", "label": "Seven features, 42.3% of discharge", "kind": "level",
         "blocks": list(budget7), "section": "B, F",
         "method": "best level set without prominence and fwhm (discharge budget)"},
        {"id": "all63", "label": "All nine + history (63 columns)", "kind": "grouped",
         "blocks": list(LEVEL_COLS), "section": "E",
         "method": "20-cycle feature history"},
        {"id": "lvl_rm_drift", "label": "Level + rolling mean + drift (27 columns)",
         "kind": "component", "blocks": list(comp), "section": "E",
         "method": "history reduced to three components"},
        {"id": "hist_rule", "label": "height, fwhm + history", "kind": "grouped",
         "blocks": list(rule23), "section": "F",
         "method": "grouped subset search with history, one-standard-error rule"},
        {"id": "hist_budget", "label": "height, position + history", "kind": "grouped",
         "blocks": list(budget2), "section": "F",
         "method": "best history set within the 13.6% apex window (discharge budget)"},
        # the three level sets of the first methods table, with their history
        {"id": "interim3_hist", "label": "Interim three features + history", "kind": "grouped",
         "blocks": list(BASELINE_FEATURES), "section": "C, E",
         "method": "interim three features with their 20-cycle history"},
        {"id": "best4_hist", "label": "Best four-feature subset + history", "kind": "grouped",
         "blocks": list(rule19), "section": "D, E",
         "method": "level one-standard-error pick with its 20-cycle history"},
        {"id": "budget7_hist", "label": "Seven features, 42.3% of discharge + history", "kind": "grouped",
         "blocks": list(budget7), "section": "B, E",
         "method": "level discharge-budget set with its 20-cycle history"},
    ]


def columns_of(fs: dict, combined: list[str]) -> list[str]:
    """A set's columns, in the ENG-14 combined-pool order."""
    blocks = {"level": level_blocks(), "grouped": grouped_blocks(combined),
              "component": component_blocks(combined)}[fs["kind"]]
    return columns_for(fs["blocks"], blocks, combined)


def base_features_of(fs: dict) -> list[str]:
    """The level features a set needs to measure (for its SoC window)."""
    return list(LEVEL_COLS) if fs["kind"] == "component" else list(fs["blocks"])


def make_factory(model: str):
    """(estimator factory, max training rows, params) of one model."""
    key, _ = MODEL_SPEC[model]
    spec = REGRESSORS[key]
    params = tuned_rf_params() if model == "RF tuned" else None
    return (lambda: spec["factory"](params)), spec["max_train_rows"], params


# ------------------------ scoring ------------------------

def run_model(model: str, sets: list[dict], dev, test, fold, combined) -> dict:
    """Dev-CV of every set, then batch 3 of every set, for one model (resumable)."""
    path = progress_path(model)
    done = read_json(path, {})
    make, max_rows, params = make_factory(model)
    print(f"[{model}] {len(sets)} feature sets, {sum(f['id'] in done for f in sets)} done "
          f"({path.name}); params {params}, max_train_rows {max_rows}", flush=True)
    # 1) development CV for every set
    for fs in sets:
        rec = done.get(fs["id"])
        if rec and "cv_pooled_rmse_mean" in rec:
            continue
        cols = columns_of(fs, combined)
        t0 = time.time()
        cv = cv_evaluate(dev, cols, fold, make, max_rows)
        done[fs["id"]] = {
            "set": fs["id"], "model": model, "n_columns": len(cols), "columns": cols,
            "cv_pooled_rmse_mean": cv["pooled_rmse_mean"], "cv_pooled_rmse_std": cv["pooled_rmse_std"],
            "cv_pooled_r2_mean": cv["pooled_r2_mean"], "cv_pooled_mae_mean": cv["pooled_mae_mean"],
            "cv_cell_avg_rmse_mean": cv["cell_avg_rmse_mean"],
            "cv_fold_rmse": [m["pooled"]["rmse"] for m in cv["per_fold"]],
            "cv_runtime_s": round(time.time() - t0, 1), "host": socket.gethostname(),
        }
        write_json_atomic(path, done, indent=1)
        print(f"[{model}] CV {fs['id']:>12s} p={len(cols):2d} RMSE {cv['pooled_rmse_mean']:.5f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    # 2) batch 3, once per cell, after every dev-CV cell of this model
    for fs in sets:
        rec = done[fs["id"]]
        if "b3_rmse" in rec:
            continue
        cols = columns_of(fs, combined)
        t0 = time.time()
        ho = heldout_evaluate(dev, test, cols, make, max_rows)
        rec.update({"b3_rmse": ho["pooled"]["rmse"], "b3_r2": ho["pooled"]["r2"],
                    "b3_mae": ho["pooled"]["mae"], "b3_rmse_cellavg": ho["cell_avg"]["rmse"],
                    "b3_reason": HELDOUT_REASON, "b3_runtime_s": round(time.time() - t0, 1)})
        write_json_atomic(path, done, indent=1)
        print(f"[{model}] b3 {fs['id']:>12s} RMSE {ho['pooled']['rmse']:.5f} "
              f"R2 {ho['pooled']['r2']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return done


# ------------------------ merge + checks ------------------------

def soc_windows(sets: list[dict]) -> dict:
    """Median discharge window each set needs (ENG-17 per-cycle union when the
    per-cycle table is on disk, else the union of the median windows)."""
    feats = [tuple(base_features_of(fs)) for fs in sets]
    pc = per_cycle_spans(sorted(set(feats), key=lambda f: (len(f), f)))
    med = feature_windows_from_medians()
    out = {}
    for fs, f in zip(sets, feats):
        start, end, span = subset_window(f, med)
        rec = {"soc_start_median_union": start, "soc_end_median_union": end,
               "soc_span_median_union": span}
        if pc is not None:
            p = pc["+".join(f)]
            rec.update({"soc_start": p["soc_start_median"], "soc_end": p["soc_end_median"],
                        "soc_span": p["soc_span_median"], "basis": "per-cycle union, median over cycles"})
        else:
            rec.update({"soc_start": start, "soc_end": end, "soc_span": span,
                        "basis": "union of the ENG-17 median windows"})
        out[fs["id"]] = rec
    return out


def stored_values(sets: list[dict]) -> list[dict]:
    """Untuned-RF dev-CV and batch-3 values and tuned values already stored by
    ENG-22/23/24 for the same columns and rows (the cross-checks)."""
    def table(path):
        return {"+".join(r["blocks"]): r for r in read_json(path)["table"]}

    t22, t23, t24 = table(T22), table(T23), table(T24)
    a23, a24 = read_json(A23), read_json(A24)
    ho23 = {(r["config"], "+".join(r["blocks"])): r for r in a23["heldout"]}
    ho24 = {(r["config"], "+".join(r["blocks"])): r for r in a24["heldout"]}
    a22 = read_json(A22)
    ho22 = {(r["config"], "+".join(r["blocks"])): r for r in a22["heldout"]}
    out = []
    for fs in sets:
        key = "+".join(fs["blocks"])
        src = {"level": (t24, ho24, None, "ENG-24"), "grouped": (t23, ho23, a23.get("tuned", {}), "ENG-23"),
               "component": (t22, ho22, None, "ENG-22")}[fs["kind"]]
        tab, ho, tuned, name = src
        if key in tab:
            out.append({"set": fs["id"], "model": "RF", "what": "cv", "stored": tab[key]["cv_pooled_rmse_mean"],
                        "source": f"{name} table {key}"})
        if ("untuned", key) in ho:
            out.append({"set": fs["id"], "model": "RF", "what": "b3", "stored": ho[("untuned", key)]["b3_rmse"],
                        "source": f"{name} heldout {key}"})
        if tuned and key in tuned:
            out.append({"set": fs["id"], "model": "RF tuned", "what": "cv", "stored": tuned[key]["cv"],
                        "source": f"{name} tuned {key}"})
            out.append({"set": fs["id"], "model": "RF tuned", "what": "b3", "stored": tuned[key]["b3_rmse"],
                        "source": f"{name} tuned {key}"})
    return out


def merge(combined: list[str]) -> None:
    """The four progress files -> ranked cells, checks, SoC windows -> json/csv."""
    sets = feature_sets()
    cells = []
    for model in MODELS:
        done = read_json(progress_path(model), {})
        for fs in sets:
            rec = done.get(fs["id"])
            if not rec or "b3_rmse" not in rec:
                sys.exit(f"[{STEM}] {model} x {fs['id']} not finished - run the search first")
            cells.append(rec)
    by = {(c["set"], c["model"]): c for c in cells}
    base = by[("interim3", "RF")]["cv_pooled_rmse_mean"]
    for c in cells:
        c["cv_vs_interim_rf_rel"] = (c["cv_pooled_rmse_mean"] - base) / base
    ranked = sorted(cells, key=lambda c: (c["cv_pooled_rmse_mean"], c["set"], c["model"]))
    for i, c in enumerate(ranked, 1):
        c["cv_rank"] = i

    checks = []
    for s in stored_values(sets):
        mine = by[(s["set"], s["model"])]["cv_pooled_rmse_mean" if s["what"] == "cv" else "b3_rmse"]
        checks.append({**s, "eng28": mine, "abs_diff": abs(mine - s["stored"]),
                       "pass": abs(mine - s["stored"]) <= CHECK_TOL})
    picks = {fs["id"]: fs["blocks"] for fs in sets}
    pick_checks = [{"set": k, "expected": v, "derived": picks[k], "pass": picks[k] == v}
                   for k, v in EXPECTED.items()]
    for c in checks:
        print(f"  check {'PASS' if c['pass'] else 'FAIL'} {c['set']:>12s} {c['model']:>8s} "
              f"{c['what']}: {c['eng28']:.8f} vs {c['stored']:.8f} ({c['source']})", flush=True)
    for c in pick_checks:
        print(f"  pick {'PASS' if c['pass'] else 'CHANGED'} {c['set']}: {c['derived']}", flush=True)

    windows = soc_windows(sets)
    for fs in sets:
        fs["columns"] = columns_of(fs, combined)
        fs["n_columns"] = len(fs["columns"])
        fs["base_features"] = base_features_of(fs)
        fs["soc_window"] = windows[fs["id"]]
    out = {
        "protocol": {
            "rows": "ENG-14 history rows (>= 20 cycles of history); dev = batches 1+2, test = batch 3",
            "folds": "ENG-08 cell -> fold map (StratifiedGroupKFold(5), grouped by cell)",
            "column_order": "ENG-14 combined-pool order restricted to each set",
            "dev_score": "pooled RMSE averaged over the five development folds",
            "batch3": f"refit on all development rows, scored once per cell (reason {HELDOUT_REASON!r})",
            "seed": RF_SEED, "tuned_rf_params": tuned_rf_params(),
            "models": {m: MODEL_SPEC[m][1] for m in MODELS},
        },
        "feature_sets": sets,
        "models": MODELS,
        "interim_rf_cv": base,
        "cells": ranked,
        "checks_vs_stored": checks,
        "checks_picks": pick_checks,
        "all_checks_pass": all(c["pass"] for c in checks) and all(c["pass"] for c in pick_checks),
        "heldout_audit": {"n_cells_scored_on_batch3": len(cells), "reason": HELDOUT_REASON,
                          "scored_after_all_dev_cv": True},
        "total_runtime_s": round(sum(c["cv_runtime_s"] + c["b3_runtime_s"] for c in cells), 1),
        "env": env_info(),
    }
    write_json_atomic(OUT_JSON, out, indent=1)
    cols = ["cv_rank", "set", "model", "n_columns", "cv_pooled_rmse_mean", "cv_pooled_rmse_std",
            "cv_pooled_r2_mean", "cv_pooled_mae_mean", "cv_cell_avg_rmse_mean", "cv_vs_interim_rf_rel",
            "b3_rmse", "b3_r2", "b3_mae", "b3_rmse_cellavg", "cv_runtime_s", "b3_runtime_s", "host"]
    pd.DataFrame(ranked)[cols].to_csv(OUT_CSV, index=False, lineterminator="\n")
    log_manifest(STEM, {
        "script": SCRIPT,
        "outputs": [f"results/{OUT_JSON.name}", f"results/{OUT_CSV.name}"],
        "all_checks_pass": out["all_checks_pass"],
        "best_cell": {k: ranked[0][k] for k in ("set", "model", "cv_pooled_rmse_mean")},
    })
    print(f"[{STEM}] wrote results/{OUT_JSON.name} and .csv; checks pass: {out['all_checks_pass']}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                    help="models to score, in this order (default: all four)")
    ap.add_argument("--sets", nargs="+", default=None, help="feature-set ids (default: all eleven)")
    ap.add_argument("--merge", action="store_true",
                    help="only merge the four progress files (no fits) and run the checks")
    args = ap.parse_args()
    require_inputs()
    if args.merge:
        # load_history_rows is needed here only for the combined-pool column order
        merge(load_history_rows()[3])
        return
    sets = feature_sets()
    print(f"[{STEM}] env {env_info()}", flush=True)
    for fs in sets:
        print(f"  {fs['id']:>12s} ({fs['kind']}): {fs['blocks']}", flush=True)
    if args.sets:
        sets = [fs for fs in sets if fs["id"] in args.sets]
    dev, test, fold, combined, _ = load_history_rows()
    print(f"[{STEM}] history rows: dev {len(dev)}, batch 3 {len(test)}", flush=True)
    for model in args.models:
        run_model(model, sets, dev, test, fold, combined)
    if args.sets is None and set(args.models) == set(MODELS):
        merge(combined)


if __name__ == "__main__":
    main()
