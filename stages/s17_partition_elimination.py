"""
17. Spearman ranking and backward elimination per partition (ENG-25)
====================================================================
Do the elimination order and the chosen feature set depend on the train/test
split? Stage 9 re-ran the selection grid under six partitions of the cells
(src/partitions.py). This stage adds, per partition and on its development
cells only (inner 5-fold, assign_folds on its full clean dev rows, as in
stage 9), two things the partitions' test cells are never scored for:

  1. Spearman's ranking of the nine level features (seconds), with the
     Kendall tau and top-3 Jaccard across partitions.
  2. A direct backward elimination (BE): from all nine, drop at each step the
     feature whose removal gives the lowest dev-CV RMSE (44 untuned-RF fits,
     pool-order columns). Mode `level` uses the level rows; mode `grouped`
     the history rows with each feature and its six history columns as one
     block (stage 16's blocks; 44 fits of 7k columns). The elimination order
     is a ranking: is it the same in every partition?

lobo_b3 is the study's own split: its grouped path is read off stage 16's
table instead of being refitted, and its direct level path must reproduce
the path read off stage 11's table at every step (the stored check).

Inputs   results/features_expanded.parquet; eng15_cells.json (stage 9: the
         all-nine / all-63 start values of the Pearson x RF k-curves, reported
         but never used for a decision); eng19_exhaustive_subsets.csv (stage
         11); eng23_grouped_history_subsets.json (stage 16; optional).
Outputs  results/eng25_progress_<mode>_<partition>.json  one record per fit
             {blocks, k, n_columns, cv_pooled_rmse_mean/std, cv_pooled_r2_mean,
             cv_cell_avg_rmse_mean, cv_fold_rmse, runtime_s, host}
         results/eng25_partition_robustness.json  partitions; spearman
             {by_partition{ranking, scores}, top1, top3, stability,
             top3_jaccard_vs_lobo_b3}; be_level_stability; be_grouped_stability;
             backward_elimination{level, grouped -> partition -> {source, steps
             [{k, blocks, cv, dropped}], drop_order, elimination_ranking,
             last_standing, last_two, last_three, n_fits}};
             check_lobo_b3_vs_eng19_table; env
Runtime  eleven tasks (level x six partitions, grouped x five): 10-70 min
         (level) and 1.5-3.5 h (grouped) each on 24 threads, ~17 h in all;
         every fit is stored, so a task resumes where it stopped. The
         no-argument run does every task in turn, then the report; `--task I`
         runs one task (a PBS array index, cluster/eng2x_array.pbs) and
         `--report` (seconds) rebuilds the JSON from the progress files. A
         JSON already on disk is never replaced by one with fewer BE paths:
         a missing progress file stops the report instead.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, assign_folds, log_manifest, split_dev_test
from src.features_temporal import LEVEL_COLS
from src.harness import SELECTORS
from src.partitions import ID_COLS, PARTITIONS, make_partitions, relabel
from src.subset_search import (
    canonical,
    columns_for,
    cv_record,
    env_info,
    grouped_blocks,
    history_table,
    level_blocks,
    load_clean,
    read_json,
    rf_factory,
    subset_key,
    write_json_atomic,
)

ENG15_CELLS = RESULTS_DIR / "eng15_cells.json"
ENG19_CSV = RESULTS_DIR / "eng19_exhaustive_subsets.csv"
ENG23_JSON = RESULTS_DIR / "eng23_grouped_history_subsets.json"
OUT_JSON = RESULTS_DIR / "eng25_partition_robustness.json"
SCRIPT = "stages/s17_partition_elimination.py"
MODES = ["level", "grouped"]
TASKS = ([("level", p) for p in PARTITIONS]
         + [("grouped", p) for p in PARTITIONS if p != "lobo_b3"])


def progress_file(mode: str, partition: str) -> Path:
    return RESULTS_DIR / f"eng25_progress_{mode}_{partition}.json"


# ------------------------ the fits (one task) ------------------------

def partition_rows(clean, hist, combined, parts, name, mode):
    """(dev, fold, blocks, order) for one partition, exactly as stage 9 builds them:
    the cell -> fold map comes from the partition's full clean dev rows and is
    reused on the history rows."""
    test_cells = parts[name]["test_cells"]
    full = relabel(clean[ID_COLS + list(LEVEL_COLS)], test_cells)
    _, fmap = assign_folds(split_dev_test(full)[0])
    if mode == "level":
        dev = split_dev_test(full)[0]
        blocks, order = level_blocks(), list(LEVEL_COLS)
    else:
        hst = relabel(hist[ID_COLS + combined], test_cells)
        dev = split_dev_test(hst)[0]
        blocks, order = grouped_blocks(combined), combined
    fold = pd.Series(dev["cell_id"].map(fmap).to_numpy(dtype=int), index=dev.index, name="fold")
    return dev, fold, blocks, order


def backward_elimination_fit(dev, fold, names, blocks, order, make, path: Path, label: str) -> dict:
    """Direct BE with every fit stored in `path` (resumable); ties go to the
    feature earlier in the pool. The path itself is re-derived from the stored
    fits by `be_from_table` in the report."""
    store = read_json(path, {})
    cur = tuple(names)
    while len(cur) > 1:
        cands = []
        for d in cur:
            sub = canonical([c for c in cur if c != d], names)
            key = subset_key(sub)
            if key not in store:
                store[key] = cv_record(dev, fold, columns_for(sub, blocks, order), make, sub)
                write_json_atomic(path, store)
                print(f"{label} k={len(sub)} CV={store[key]['cv_pooled_rmse_mean']:.5f} {key} "
                      f"({store[key]['runtime_s']:.0f} s)", flush=True)
            cands.append((store[key]["cv_pooled_rmse_mean"], names.index(d), d, sub))
        _, _, _, cur = min(cands)
    return store


def run_task(mode: str, partition: str, n_jobs: int | None) -> None:
    """The 44 fits of one (mode, partition); no-op once its progress file is complete."""
    clean = load_clean()
    hist, combined, _ = history_table(clean) if mode == "grouped" else (None, None, None)
    parts = make_partitions(clean)
    dev, fold, blocks, order = partition_rows(clean, hist, combined, parts, partition, mode)
    print(f"[eng25 {mode} {partition}] dev {dev['cell_id'].nunique()} cells / {len(dev)} rows; "
          f"env {env_info()}", flush=True)
    backward_elimination_fit(dev, fold, list(LEVEL_COLS), blocks, order,
                             rf_factory(None, n_jobs), progress_file(mode, partition),
                             f"[eng25 {mode} {partition}]")


# ------------------------ the report (no fits) ------------------------

def be_from_table(rows: dict, names) -> list[dict]:
    """BE path read off a complete subset table {key: cv}."""
    cur = tuple(names)
    steps = [{"k": len(cur), "blocks": list(cur), "cv": rows[subset_key(cur)], "dropped": None}]
    while len(cur) > 1:
        cands = [(rows[subset_key(canonical([c for c in cur if c != d], names))],
                  names.index(d), d) for d in cur]
        cv, _, drop = min(cands)
        cur = canonical([c for c in cur if c != drop], names)
        steps.append({"k": len(cur), "blocks": list(cur), "cv": cv, "dropped": drop})
    return steps


def elimination_ranking(steps) -> list[str]:
    """Last standing first, first dropped last."""
    return [steps[-1]["blocks"][0]] + [s["dropped"] for s in reversed(steps[1:])]


def tau_summary(rankings: dict[str, list[str]], names) -> dict:
    """Mean and min Kendall tau and mean top-3 Jaccard over every pair of rankings."""
    taus, jac = [], []
    for a, b in itertools.combinations(rankings, 2):
        ra = [rankings[a].index(f) for f in names]
        rb = [rankings[b].index(f) for f in names]
        taus.append(kendalltau(ra, rb)[0])
        jac.append(len(set(rankings[a][:3]) & set(rankings[b][:3]))
                   / len(set(rankings[a][:3]) | set(rankings[b][:3])))
    return {"n": len(rankings), "kendall_tau_mean": float(np.mean(taus)) if taus else None,
            "kendall_tau_min": float(np.min(taus)) if taus else None,
            "top3_jaccard_mean": float(np.mean(jac)) if jac else None}


def require_inputs() -> None:
    """Stop, naming the stage to run, when an input of the report is missing."""
    needed = [(ENG15_CELLS, "stage 9 (s09_partitions)"),
              (ENG19_CSV, "stage 11 (s11_level_subsets)")]
    missing = [f"results/{p.name} (run {where} first)" for p, where in needed if not p.exists()]
    if missing:
        sys.exit("the ENG-25 report needs " + "; ".join(missing))
    if not ENG23_JSON.exists():
        print(f"  results/{ENG23_JSON.name} not found (stage 16): the grouped lobo_b3 path "
              "is left out", flush=True)


def spearman_block(clean, parts, names) -> dict:
    """Spearman's ranking on each partition's dev rows and its stability across partitions."""
    sp = {}
    for p in PARTITIONS:
        full = relabel(clean[ID_COLS + names], parts[p]["test_cells"])
        dev = split_dev_test(full)[0]
        r = SELECTORS["spearman"](dev, names, None, None)
        sp[p] = {"ranking": r["feature"].tolist(), "scores": r["score"].round(6).tolist()}
    out = {"by_partition": sp,
           "top1": {p: v["ranking"][0] for p, v in sp.items()},
           "top3": {p: v["ranking"][:3] for p, v in sp.items()},
           "stability": tau_summary({p: v["ranking"] for p, v in sp.items()}, names)}
    # top-3 Jaccard per partition against the batch-3 (study) partition
    ref3 = set(sp["lobo_b3"]["ranking"][:3])
    out["top3_jaccard_vs_lobo_b3"] = {
        p: len(set(v["ranking"][:3]) & ref3) / len(set(v["ranking"][:3]) | ref3) for p, v in sp.items()}
    return out


def be_block(names) -> dict:
    """The BE path of every (mode, partition) whose progress file is complete,
    the grouped lobo_b3 path read off the ENG-23 table; the all-nine / all-63
    start value from stage 9's Pearson x RF k-curve (Pearson column order)."""
    e15 = read_json(ENG15_CELLS)["cells"]
    e23 = read_json(ENG23_JSON)
    e23_rows = ({subset_key(r["blocks"]): r["cv_pooled_rmse_mean"] for r in e23["table"]}
                if e23 else None)
    be = {"level": {}, "grouped": {}}
    for mode in MODES:
        for p in PARTITIONS:
            store = read_json(progress_file(mode, p), {})
            rows = {k: v["cv_pooled_rmse_mean"] for k, v in store.items()}
            source = "direct BE (eng25 progress)"
            if mode == "grouped" and p == "lobo_b3" and not store:
                if e23_rows is None:
                    continue
                rows, source = e23_rows, "read off the ENG-23 table"
            try:
                rows_full = dict(rows)
                rows_full.setdefault(subset_key(names), float("nan"))
                steps = be_from_table(rows_full, names)
            except KeyError:
                print(f"  {mode} {p}: incomplete ({len(store)} fits)")
                continue
            pool = "level" if mode == "level" else "combined"
            exp = "e08" if mode == "level" else "e14"
            cell = e15.get(f"{exp}|{p}|{pool}|pearson|rf")
            start = cell["k_curve"][-1]["cv_pooled_rmse_mean"] if cell else None
            steps[0]["cv"] = start
            steps[0]["cv_source"] = "eng15_cells.json Pearson x RF k-curve (Pearson order)"
            be[mode][p] = {"source": source, "steps": steps,
                           "drop_order": [s["dropped"] for s in steps[1:]],
                           "elimination_ranking": elimination_ranking(steps),
                           "last_standing": steps[-1]["blocks"][0],
                           "last_two": steps[-2]["blocks"], "last_three": steps[-3]["blocks"],
                           "n_fits": len(store)}
    return be


def check_lobo_b3(direct: list[dict], names) -> dict:
    """The direct level BE of the study split against the path read off the ENG-19 table."""
    e19 = pd.read_csv(ENG19_CSV)
    table = be_from_table(dict(zip(e19["features"], e19["cv_pooled_rmse_mean"])), names)
    same_subsets = [d["blocks"] == t["blocks"] for d, t in zip(direct[1:], table[1:])]
    diffs = [abs(d["cv"] - t["cv"]) for d, t in zip(direct[1:], table[1:])]
    return {"same_subset_each_step": same_subsets, "max_abs_cv_diff": float(max(diffs)),
            "first_two_steps_match": bool(all(same_subsets[:2]) and max(diffs[:2]) <= 1e-12),
            "all_steps_match": bool(all(same_subsets) and max(diffs) <= 1e-12)}


def build_report() -> dict:
    """Everything eng25_partition_robustness.json holds, from the progress files."""
    require_inputs()
    names = list(LEVEL_COLS)
    clean = load_clean()
    parts = make_partitions(clean)
    out: dict = {"partitions": {p: {k: parts[p][k] for k in ("kind", "n_dev_cells", "n_test_cells",
                                                            "test_cells_by_batch")}
                                for p in PARTITIONS}}
    out["spearman"] = spearman_block(clean, parts, names)
    be = be_block(names)
    for mode in MODES:
        if be[mode]:
            out[f"be_{mode}_stability"] = tau_summary(
                {p: v["elimination_ranking"] for p, v in be[mode].items()}, names)
    out["backward_elimination"] = be
    if "lobo_b3" in be["level"]:
        out["check_lobo_b3_vs_eng19_table"] = check_lobo_b3(be["level"]["lobo_b3"]["steps"], names)
    out["env"] = env_info()
    return out


def lost_paths(be: dict) -> list[str]:
    """The (mode, partition) BE paths the JSON on disk holds and `be` does not."""
    old = read_json(OUT_JSON, {}).get("backward_elimination", {})
    return [f"{m} {p}" for m in MODES for p in old.get(m, {}) if p not in be[m]]


def report() -> None:
    """Write the JSON and the manifest entry, unless that would drop a BE path the
    JSON on disk holds: a progress file missing on the laptop (they are copied from
    the cluster, cluster/README.md) stops the report rather than degrading the
    finished analysis the report items read."""
    out = build_report()
    lost = lost_paths(out["backward_elimination"])
    if lost:
        sys.exit(f"results/{OUT_JSON.name} holds the BE path of {', '.join(lost)} and the progress "
                 "files do not: not overwriting it. Restore results/eng25_progress_<mode>_<partition>"
                 f".json (grouped lobo_b3: results/{ENG23_JSON.name}) or rerun those tasks, then "
                 "--report again.")
    write_json_atomic(OUT_JSON, out, indent=1)
    print(f"wrote results/{OUT_JSON.name}")
    be = out["backward_elimination"]
    log_manifest("eng25_partition_robustness", {
        "script": SCRIPT, "seed": RF_SEED,
        "spearman_top1": out["spearman"]["top1"],
        "be_level_last_standing": {p: v["last_standing"] for p, v in be["level"].items()},
        "be_grouped_last_standing": {p: v["last_standing"] for p, v in be["grouped"].items()},
        "check_lobo_b3_vs_eng19_table": out.get("check_lobo_b3_vs_eng19_table")})


# ------------------------ entry point ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--task", type=int, choices=range(len(TASKS)), metavar="I",
                    help="run one task only, by index (a PBS array index): " + ", ".join(
                        f"{i}={m}:{p}" for i, (m, p) in enumerate(TASKS)))
    ap.add_argument("--mode", choices=MODES, help="with --partition: run that one task only")
    ap.add_argument("--partition", choices=PARTITIONS,
                    help="with --mode (grouped lobo_b3 is allowed: a check against stage 16)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="forest threads (cluster nodes); the numbers do not depend on it")
    ap.add_argument("--report", action="store_true",
                    help="only rebuild the JSON from the stored progress files (no fit); a JSON on "
                         "disk is never replaced by one with fewer BE paths")
    args = ap.parse_args()
    if (args.mode is None) != (args.partition is None):
        sys.exit("give --mode and --partition together")
    if args.report:
        report()
    elif args.task is not None:
        run_task(*TASKS[args.task], args.n_jobs)
    elif args.mode:
        run_task(args.mode, args.partition, args.n_jobs)
    else:
        for mode, partition in TASKS:
            run_task(mode, partition, args.n_jobs)
        report()


if __name__ == "__main__":
    main()
