"""
13. Permutation importance of feature groups (ENG-21)
=====================================================
Per-column permutation importance (PI) splits the credit among near-duplicate
columns: shuffle area25 and the forest leans on area and area35 instead.
Shuffling a whole group with ONE row permutation (src.grouped_importance)
measures what the reference RF takes from the group as a whole. Protocol as
the ENG-08 permutation selector: per development fold, fit on the fold-train
rows, PI on all fold-validation rows, ten repeats, random_state 0, neg RMSE;
reported as the mean (and sd) over the five folds. Models: the tuned and the
untuned RF on the nine level features (groups: each feature, the area block
{area, area35, area25}, the area block + height) and the tuned RF on all 63
history columns (groups: each column, each feature with its six history
columns, each history component across the nine features).
Inputs: features_expanded.parquet, eng09_tuned_params.json; eng08_full.json,
eng08_full_tuned.json and eng14_full_tuned.json hold the stored one-column PI
that the checks reproduce (and sklearn's own PI is checked on fold 0).
Outputs: eng21_grouped_permutation.json (models.<model>.<group>.{mean,
fold_sd, folds}, checks, history_grid_share_pct, history_summary,
level_tuned_summary, level_untuned_summary, env) and eng21_history_pi_grid.csv
(features x components, fold-mean one-column PI). About 10 min on a laptop;
resumable per (model, fold) through eng21_progress.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, log_manifest
from src.features_temporal import LEVEL_COLS, base_feature_of
from src.grouped_importance import grouped_permutation_importance
from src.subset_search import (
    COMPONENTS,
    component_of,
    env_info,
    load_clean,
    load_history_rows,
    load_level_rows,
    read_json,
    rf_factory,
    tuned_rf_params,
    write_json_atomic,
)

OUT_JSON = RESULTS_DIR / "eng21_grouped_permutation.json"
GRID_CSV = RESULTS_DIR / "eng21_history_pi_grid.csv"
PROGRESS = RESULTS_DIR / "eng21_progress.json"

AREA_BLOCK = ["area", "area35", "area25"]
GROUP_AREA = "GROUP area block (area, area35, area25)"
GROUP_AREA_HEIGHT = "GROUP area block + height"
N_REPEATS = 10
TOL = 1e-12

# (model, stored result file, pool, grid cell): the one-column groups must
# reproduce the permutation|rf scores these files hold.
STORED = (("level_tuned", "eng08_full_tuned.json", None, "permutation|rf"),
          ("level_untuned", "eng08_full.json", None, "permutation|rf"),
          ("history_tuned", "eng14_full_tuned.json", "combined", "permutation|rf"))


def level_groups() -> dict[str, list[str]]:
    """Each level feature alone, the area block, the area block + height."""
    g = {f: [f] for f in LEVEL_COLS}
    g[GROUP_AREA] = list(AREA_BLOCK)
    g[GROUP_AREA_HEIGHT] = ["height"] + AREA_BLOCK
    return g


def history_groups(combined: list[str]) -> dict[str, list[str]]:
    """Each of the 63 columns alone, each feature's seven, each component's nine."""
    g = {c: [c] for c in combined}
    for f in LEVEL_COLS:
        g[f"FEATURE {f}"] = [c for c in combined if base_feature_of(c) == f]
    for k in COMPONENTS:
        g[f"COMPONENT {k}"] = [c for c in combined if component_of(c) == k]
    return g


def sklearn_check(est, va, cols, means: dict) -> dict:
    """sklearn's own permutation importance must equal the one-column groups."""
    sk = permutation_importance(est, va[cols], va["SOH"], n_repeats=N_REPEATS,
                                random_state=RF_SEED,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    diffs = [abs(sk.importances_mean[i] - means[c]) for i, c in enumerate(cols)]
    return {"max_abs_diff_one_column_groups": float(max(diffs)),
            "pass": bool(max(diffs) <= TOL)}


def run_model(name, dev, fold, cols, groups, make, done, sk_check: bool) -> None:
    """Grouped PI per fold; the progress file is written after every fold."""
    for f in sorted(fold.unique()):
        key = f"{name}|fold{f}"
        if key in done:
            continue
        t0 = time.time()
        tr, va = dev[fold != f], dev[fold == f]
        est = make()
        est.fit(tr[cols], tr["SOH"])
        t_fit = time.time() - t0
        imp = grouped_permutation_importance(est, va[cols], va["SOH"], groups,
                                             n_repeats=N_REPEATS, random_state=RF_SEED)
        rec = {"mean": {g: float(v.mean()) for g, v in imp.items()},
               "repeat_sd": {g: float(v.std()) for g, v in imp.items()},
               "fit_s": round(t_fit, 1), "runtime_s": round(time.time() - t0, 1)}
        if sk_check and f == 0:
            rec["sklearn_check"] = sklearn_check(est, va, cols, rec["mean"])
            print(f"  [{name}] sklearn one-column check: max diff "
                  f"{rec['sklearn_check']['max_abs_diff_one_column_groups']:.1e}", flush=True)
        done[key] = rec
        write_json_atomic(PROGRESS, done)
        print(f"[{name}] fold {f}: fit {t_fit:.0f} s, total {rec['runtime_s']:.0f} s", flush=True)


def fold_summary(done: dict, name: str, groups, n_folds: int = 5) -> dict:
    """Per group: mean and sd (ddof 0) over the folds, and the fold values."""
    per = np.array([[done[f"{name}|fold{f}"]["mean"][g] for g in groups] for f in range(n_folds)])
    return {g: {"mean": float(per[:, i].mean()), "fold_sd": float(per[:, i].std()),
                "folds": per[:, i].tolist()} for i, g in enumerate(groups)}


def check_against_stored(models: dict) -> dict:
    """The one-column fold means reproduce the stored ENG-08 / ENG-14 permutation|rf scores.

    A file or grid cell that is not there (e.g. s08 run with a reduced tuned
    grid) is recorded as skipped (pass None) rather than raising, so the main
    output is still written.
    """
    checks = {}
    for name, file, pool, cell in STORED:
        source = f"results/{file} {pool or ''} {cell}"
        data = read_json(RESULTS_DIR / file)
        grid = (data or {}).get("grid", {})
        grid = grid.get(pool, {}) if pool else grid
        if cell not in grid:
            reason = "file missing" if data is None else "cell not in file"
            checks[f"{name}_vs_stored"] = {"source": source, "skipped": reason, "pass": None}
            print(f"  [{name}] stored check skipped: {source}: {reason}", flush=True)
            continue
        g = grid[cell]
        stored = {r["feature"]: r["score"] for r in g["ranking"]}
        diffs = {c: abs(stored[c] - models[name][c]["mean"]) for c in stored}
        checks[f"{name}_vs_stored"] = {"source": source,
                                       "max_abs_diff": float(max(diffs.values())),
                                       "pass": bool(max(diffs.values()) <= TOL)}
    return checks


def history_tables(h: dict) -> tuple[pd.DataFrame, dict, dict]:
    """The 9 x 7 grid of one-column PI, its share in percent, and grouped vs summed PI."""
    grid = pd.DataFrame([[h[f if k == "level" else f"{f}_{k}"]["mean"] for k in COMPONENTS]
                         for f in LEVEL_COLS], index=LEVEL_COLS, columns=COMPONENTS)
    total = float(grid.to_numpy().sum())
    share = (100 * grid / total).round(2).to_dict(orient="index")
    summary = {
        "by_feature": {f: {"grouped": h[f"FEATURE {f}"]["mean"],
                           "grouped_fold_sd": h[f"FEATURE {f}"]["fold_sd"],
                           "sum_of_columns": float(grid.loc[f].sum())} for f in LEVEL_COLS},
        "by_component": {k: {"grouped": h[f"COMPONENT {k}"]["mean"],
                             "grouped_fold_sd": h[f"COMPONENT {k}"]["fold_sd"],
                             "sum_of_columns": float(grid[k].sum())} for k in COMPONENTS},
        "sum_of_all_columns": total,
    }
    return grid, share, summary


def level_summary(m: dict) -> dict:
    """The area block (and + height) shuffled together against the sum of its columns' PI."""
    return {
        "area_block_grouped": m[GROUP_AREA]["mean"],
        "area_block_sum_of_columns": float(sum(m[c]["mean"] for c in AREA_BLOCK)),
        "area_block_plus_height_grouped": m[GROUP_AREA_HEIGHT]["mean"],
        "area_block_plus_height_sum_of_columns": float(sum(m[c]["mean"] for c in AREA_BLOCK + ["height"])),
        "per_feature": {f: m[f]["mean"] for f in LEVEL_COLS}}


def pass_flags(checks: dict) -> dict:
    """Just the pass flag of every check (nested checks stay as they are)."""
    return {k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
            for k, v in checks.items()}


def report(out: dict) -> None:
    """Print the headline numbers."""
    print("\n=== level, tuned RF: area block ===")
    for name in ("level_tuned", "level_untuned"):
        s = out[f"{name}_summary"]
        print(f"  {name}: area block grouped {s['area_block_grouped']:.5f} vs sum "
              f"{s['area_block_sum_of_columns']:.5f}; +height grouped "
              f"{s['area_block_plus_height_grouped']:.5f} vs sum {s['area_block_plus_height_sum_of_columns']:.5f}")
    print("=== history, tuned RF ===")
    for key in ("by_feature", "by_component"):
        for g, v in sorted(out["history_summary"][key].items(), key=lambda x: -x[1]["grouped"]):
            print(f"  {key:12s} {g:12s} grouped {v['grouped']:.5f} (+/- {v['grouped_fold_sd']:.5f}) "
                  f"sum of columns {v['sum_of_columns']:.5f}")
    print("checks:", pass_flags(out["checks"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()

    clean = load_clean()
    tuned = tuned_rf_params()
    done = read_json(PROGRESS, {})
    out = {"protocol": "per dev fold: fit on fold-train, grouped PI on fold-val, 10 repeats, "
                       "random_state 0, neg RMSE; mean over the 5 folds", "models": {}}

    # Level models: ENG-08 rows and folds, the nine features in pool order.
    dev, _, fold = load_level_rows(clean)
    lg = level_groups()
    for name, params in (("level_tuned", tuned), ("level_untuned", {})):
        run_model(name, dev, fold, list(LEVEL_COLS), lg, rf_factory(params), done, sk_check=True)
        out["models"][name] = fold_summary(done, name, lg)

    # History model: ENG-14 rows with the ENG-08 fold map, all 63 columns in
    # the combined-pool order (the same model as the all-blocks subset of ENG-22/23).
    hdev, _, hfold, combined, _ = load_history_rows(clean)
    hg = history_groups(combined)
    run_model("history_tuned", hdev, hfold, combined, hg, rf_factory(tuned), done, sk_check=False)
    out["models"]["history_tuned"] = fold_summary(done, "history_tuned", hg)

    checks = {"sklearn_one_column": {n: done[f"{n}|fold0"].get("sklearn_check")
                                     for n in ("level_tuned", "level_untuned")}}
    checks.update(check_against_stored(out["models"]))
    out["checks"] = checks

    grid, share, summary = history_tables(out["models"]["history_tuned"])
    grid.to_csv(GRID_CSV)
    out["history_grid_share_pct"] = share
    out["history_summary"] = summary
    for name in ("level_tuned", "level_untuned"):
        out[f"{name}_summary"] = level_summary(out["models"][name])
    out["env"] = env_info()
    write_json_atomic(OUT_JSON, out, indent=1)

    report(out)
    log_manifest("eng21_grouped_permutation", {
        "script": "stages/s13_grouped_importance.py", "seed": RF_SEED,
        "history_by_component_grouped": {k: v["grouped"] for k, v in
                                         out["history_summary"]["by_component"].items()},
        "level_tuned_area_block": out["level_tuned_summary"]["area_block_grouped"],
        "checks": pass_flags(checks)})


if __name__ == "__main__":
    main()
