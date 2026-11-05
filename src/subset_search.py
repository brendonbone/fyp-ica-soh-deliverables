"""
Exhaustive subset search over feature blocks (ENG-19b, ENG-22 ... ENG-25, ENG-28).
=================================================================================
ENG-19 scored every non-empty subset of the nine level features. The later
searches repeat that idea with three kinds of *block*:

    level      one block per level feature          (9 blocks x 1 column)
    grouped    one block per feature + its history  (9 blocks x 7 columns)
    component  one block per history component      (7 blocks x 9 columns)

A subset of blocks is turned into columns by `columns_for`, which keeps the
columns in the order of a fixed reference list (`LEVEL_COLS` for the level
search, the ENG-14 combined-pool order `LEVEL_COLS + temporal_cols` for the
history searches). The random forest's draws and its tie-breaks between equal
splits go by column index, so a fixed order makes every subset reproducible
and makes the all-blocks subset of the grouped search and of the component
search the same model (identical columns in identical order).

Protocol (unchanged from ENG-08 / ENG-14): `clean_feature_table`, dev =
batches 1+2, test = batch 3, `cv_evaluate` / `heldout_evaluate` from
src/harness.py, untuned RF unless tuned params are passed, seed 0. Level
rows use `assign_folds` on the full clean dev set; history rows reuse that
cell -> fold map exactly as ENG-14 does (`load_history_rows`).

Searches are resumable and shardable: `run_units` writes the progress file
after every unit (atomically) and skips units already in it; `shard_units`
splits a unit list into balanced, deterministic shards for a PBS job array,
each with its own progress file; `merge_progress` unions the shard files.

Everything after the search is read off the table without new fits
(`analyse_table`): best subset per size, backward elimination and
sequential forward selection, prefixes of fixed rankings, top-20 inclusion,
exact Shapley values, the two "one-SE" rules, SoC windows and the
accuracy-discharge trade-off edge. `HeldoutLog` is the only place that
touches batch 3; every call records why the subset was scored.

The figures drawn from these tables live in report_items/ (nothing here plots).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.data_prep import (
    N_SPLITS,
    RF_SEED,
    assign_folds,
    both_metrics,
    clean_feature_table,
    split_dev_test,
)
from src.features_temporal import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_WINDOWS,
    HISTORY_COL,
    LEVEL_COLS,
    add_temporal_features,
    base_feature_of,
    is_temporal,
)
from src.harness import REGRESSORS, cv_evaluate, heldout_evaluate

PARQUET = RESULTS_DIR / "features_expanded.parquet"
TUNED_PARAMS = RESULTS_DIR / "eng09_tuned_params.json"
ENG17_SUMMARY = RESULTS_DIR / "eng17_soc_windows_summary.csv"
SOC_PARQUET = RESULTS_DIR / "features_soc_windows.parquet"

# History components in the column order of src/features_temporal (_kinds),
# with the raw level first.
COMPONENTS: list[str] = ["level", "d10", "d20", "slope10", "slope20", "drift", "rm20"]


# ------------------------ environment / IO ------------------------

def env_info() -> dict:
    import joblib
    import scipy
    import sklearn
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "cpu_count": os.cpu_count(),
        "history_table": os.environ.get("FYP_HISTORY_TABLE") or "computed (features_temporal)",
    }


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json_atomic(path: Path, obj, indent: int | None = None) -> None:
    """Write via a temp file + os.replace so a killed job never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def rf_factory(params: dict | None = None, n_jobs: int | None = None):
    """Harness RF factory with optional params. n_jobs never changes the fitted trees;
    predict with n_jobs > 1 adds the trees' outputs in thread order, so a prediction can
    differ in its last binary digit between runs."""
    p = dict(params or {})
    if n_jobs is not None:
        p["n_jobs"] = n_jobs
    spec = REGRESSORS["rf"]
    return lambda: spec["factory"](p)


def tuned_rf_params() -> dict:
    return read_json(TUNED_PARAMS)["rf"]


# ------------------------ data ------------------------

def load_clean() -> pd.DataFrame:
    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run the feature-extraction stage first")
    clean, _ = clean_feature_table(pd.read_parquet(PARQUET), verbose=False)
    return clean


def load_level_rows(clean: pd.DataFrame | None = None):
    """ENG-08 rows: (dev, test, fold) on the full clean table."""
    clean = load_clean() if clean is None else clean
    dev, test = split_dev_test(clean)
    fold, _ = assign_folds(dev)
    return dev, test, fold


# Cross-machine reproducibility. The OLS-slope history columns are a matrix
# product (features_temporal: `win @ weights`), whose BLAS summation order
# depends on the CPU's vector width, so Katana and the laptop differ in the
# last bits (relative ~1e-12; a position slope that is exactly 0 on one
# machine is +-1e-17 on the other). A forest occasionally splits between
# those values, which moved the 63-column dev-CV RMSE by 2e-7. Setting
# FYP_HISTORY_TABLE to a parquet written on the laptop (write_history_table)
# makes every machine use the laptop's history table, bit for bit. Unset
# (the default), the table is computed as in ENG-14.
HISTORY_TABLE_ENV = "FYP_HISTORY_TABLE"
HISTORY_PARQUET = RESULTS_DIR / "features_history.parquet"


def history_table(clean: pd.DataFrame | None = None):
    """All rows with >= 20 cycles of history (ENG-14), every batch.
    Returns (table, combined_cols, temporal_cols)."""
    cached = os.environ.get(HISTORY_TABLE_ENV)
    if cached:
        table = pd.read_parquet(cached)
        temporal_cols = [c for c in table.columns if is_temporal(c)]
        if len(table) != 97852 or len(temporal_cols) != 54:
            raise ValueError(f"{cached}: unexpected shape {table.shape}")
        print(f"  history table read from {cached} ({HISTORY_TABLE_ENV})", flush=True)
        return table, list(LEVEL_COLS) + temporal_cols, temporal_cols
    clean = load_clean() if clean is None else clean
    with_t, temporal_cols = add_temporal_features(
        clean, windows=DEFAULT_WINDOWS, min_history=DEFAULT_MIN_HISTORY)
    table = with_t[with_t[HISTORY_COL]].reset_index(drop=True)
    return table, list(LEVEL_COLS) + list(temporal_cols), list(temporal_cols)


def write_history_table(path: Path = HISTORY_PARQUET) -> None:
    """Write this machine's history table (gitignored) for FYP_HISTORY_TABLE."""
    table, _, _ = history_table()
    table.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    bad = [c for c in table.columns if not back[c].equals(table[c])]
    if bad:
        raise ValueError(f"parquet round trip changed the history table: {bad[:5]}")
    print(f"wrote {path} ({table.shape[0]} rows, {table.shape[1]} columns)")


def load_history_rows(clean: pd.DataFrame | None = None):
    """
    ENG-14 rows: cycles with >= 20 cycles of history, the ENG-08 cell -> fold
    map applied to them (not recomputed), and the combined-pool column order.
    Returns (dev, test, fold, combined_cols, temporal_cols).
    """
    clean = load_clean() if clean is None else clean
    table, _, temporal_cols = history_table(clean)
    _, folds_full = assign_folds(split_dev_test(clean)[0])
    dev, test = split_dev_test(table)
    mapped = dev["cell_id"].map(folds_full)
    if mapped.isna().any():
        raise KeyError("ENG-08 fold map lacks some history dev cells")
    fold = pd.Series(mapped.to_numpy(dtype=int), index=dev.index, name="fold")
    combined = list(LEVEL_COLS) + list(temporal_cols)
    return dev, test, fold, combined, list(temporal_cols)


def component_of(col: str) -> str:
    """'height_slope10' -> 'slope10'; a level column -> 'level'."""
    return col[len(base_feature_of(col)) + 1:] if is_temporal(col) else "level"


# ------------------------ blocks ------------------------

def level_blocks() -> dict[str, list[str]]:
    return {f: [f] for f in LEVEL_COLS}


def grouped_blocks(combined: list[str]) -> dict[str, list[str]]:
    """Each level feature with its six history columns (7 columns per block)."""
    return {f: [c for c in combined if base_feature_of(c) == f] for f in LEVEL_COLS}


def component_blocks(combined: list[str]) -> dict[str, list[str]]:
    """Each history component across the nine features (9 columns per block)."""
    return {k: [c for c in combined if component_of(c) == k] for k in COMPONENTS}


def columns_for(blocks_sel, blocks: dict[str, list[str]], order: list[str]) -> list[str]:
    cols = set().union(*(blocks[b] for b in blocks_sel))
    return [c for c in order if c in cols]


def subset_key(blocks_sel) -> str:
    return "+".join(blocks_sel)


def all_subsets(names: list[str]) -> list[tuple[str, ...]]:
    """Every non-empty subset, in ENG-19's order (size, then itertools order)."""
    return [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]


def canonical(blocks_sel, names: list[str]) -> tuple[str, ...]:
    s = set(blocks_sel)
    return tuple(n for n in names if n in s)


# ------------------------ shards ------------------------

def unit_cost(n_columns: int) -> float:
    # ENG-19 timing on the laptop: ~3.6 s + 2.6 s per column per 5-fold CV.
    return 1.4 + n_columns


def shard_units(units: list[tuple[str, ...]], costs: list[float],
                shard: int, n_shards: int) -> list[tuple[str, ...]]:
    """Deterministic longest-processing-time assignment of units to shards."""
    if n_shards <= 1:
        return list(units)
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} outside 0..{n_shards - 1}")
    order = sorted(range(len(units)), key=lambda i: (-costs[i], i))
    load = [0.0] * n_shards
    owner = {}
    for i in order:
        s = min(range(n_shards), key=lambda j: (load[j], j))
        owner[i] = s
        load[s] += costs[i]
    return [units[i] for i in range(len(units)) if owner[i] == shard]


def progress_path(stem: str, shard: int | None, n_shards: int | None) -> Path:
    if n_shards and n_shards > 1:
        return RESULTS_DIR / f"{stem}_progress_shard{shard:02d}of{n_shards:02d}.json"
    return RESULTS_DIR / f"{stem}_progress.json"


# A unit scored on two machines must agree. Katana's older nodes (Cascade
# Lake, Ice Lake) occasionally differ from the laptop by ~1e-8 in one fold
# (see results/reproduce_check_cross_host.json); a conflict up to this
# tolerance keeps the laptop's record (files ending in _laptop.json are read
# first) and is logged, a larger one stops the merge.
MERGE_TOL = 1e-6


def merge_progress(stem: str, conflicts: list | None = None) -> dict:
    """Union of the unsharded and every shard progress file of `stem`."""
    files = sorted([str(f) for f in RESULTS_DIR.glob(f"{stem}_progress.json")]
                   + [str(f) for f in RESULTS_DIR.glob(f"{stem}_progress_shard*of*.json")],
                   key=lambda f: (not f.endswith("_laptop.json"), f))
    merged: dict = {}
    for f in files:
        for k, rec in read_json(Path(f), {}).items():
            if k in merged:
                a, b = merged[k]["cv_pooled_rmse_mean"], rec["cv_pooled_rmse_mean"]
                if abs(a - b) > MERGE_TOL:
                    raise ValueError(f"{k}: conflicting CV RMSE {a} vs {b} across progress files")
                if abs(a - b) > 1e-12 and conflicts is not None:
                    conflicts.append({"key": k, "kept": a, "kept_host": merged[k].get("host"),
                                      "other": b, "other_host": rec.get("host"),
                                      "abs_diff": abs(a - b)})
                continue
            merged[k] = rec
    return merged


# ------------------------ search ------------------------

def cv_record(dev, fold, cols: list[str], make, blocks_sel) -> dict:
    t0 = time.time()
    cv = cv_evaluate(dev, cols, fold, make, None)
    return {
        "blocks": list(blocks_sel),
        "k": len(blocks_sel),
        "n_columns": len(cols),
        "cv_pooled_rmse_mean": cv["pooled_rmse_mean"],
        "cv_pooled_rmse_std": cv["pooled_rmse_std"],
        "cv_pooled_r2_mean": cv["pooled_r2_mean"],
        "cv_cell_avg_rmse_mean": cv["cell_avg_rmse_mean"],
        "cv_fold_rmse": [m["pooled"]["rmse"] for m in cv["per_fold"]],
        "runtime_s": round(time.time() - t0, 1),
        "host": socket.gethostname(),
    }


def run_units(units, blocks, order, dev, fold, make, path: Path, label: str = "") -> dict:
    """Score every unit not yet in `path`; write the file after each unit."""
    done = read_json(path, {})
    todo = [u for u in units if subset_key(u) not in done]
    print(f"{label}{len(units)} units, {len(units) - len(todo)} already in {path}", flush=True)
    t_start, n_new = time.time(), 0
    for u in todo:
        cols = columns_for(u, blocks, order)
        rec = cv_record(dev, fold, cols, make, u)
        done[subset_key(u)] = rec
        write_json_atomic(path, done)
        n_new += 1
        left = sum(subset_key(v) not in done for v in units)
        eta = (time.time() - t_start) / n_new * left
        print(f"{label}[{len(units) - left:3d}/{len(units)}] k={rec['k']} p={rec['n_columns']} "
              f"CV={rec['cv_pooled_rmse_mean']:.5f} {subset_key(u)} "
              f"({rec['runtime_s']:.0f} s, ETA {eta / 60:.0f} min)", flush=True)
    return done


def table_from_records(records: dict) -> pd.DataFrame:
    t = pd.DataFrame(records.values())
    t["key"] = t["blocks"].apply(subset_key)
    t = t.sort_values(["cv_pooled_rmse_mean", "key"]).reset_index(drop=True)
    t["cv_rank"] = range(1, len(t) + 1)
    return t


def mean_predictor_cv(dev, fold) -> dict:
    """v(empty set): dev-CV RMSE of predicting the training-fold mean SoH."""
    per_fold = []
    for f in sorted(fold.unique()):
        tr, va = dev[fold != f], dev[fold == f]
        pred = np.full(len(va), float(tr["SOH"].mean()))
        per_fold.append(both_metrics(va["cell_id"], va["SOH"], pred))
    vals = [m["pooled"]["rmse"] for m in per_fold]
    return {"cv_pooled_rmse_mean": float(np.mean(vals)),
            "cv_pooled_rmse_std": float(np.std(vals)),
            "cv_fold_rmse": vals}


def add_search_args(ap) -> None:
    ap.add_argument("--shard", type=int, default=0, help="this job's shard (0-based)")
    ap.add_argument("--n-shards", type=int, default=1, help="number of shards (PBS array size)")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="RF threads (default: all visible cores; numbers unchanged)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N units of this shard (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix for every output (smoke tests)")
    ap.add_argument("--merge", action="store_true", help="merge shard progress files into the table")
    ap.add_argument("--analyse", action="store_true",
                    help="derivations, batch-3 scores and figures from the merged table")
    ap.add_argument("--figures", action="store_true",
                    help="redraw the figures from the saved analysis (no fits)")


def run_search(args, stem: str, names, blocks, order, dev, fold) -> dict:
    units = all_subsets(list(names))
    costs = [unit_cost(len(columns_for(u, blocks, order))) for u in units]
    mine = shard_units(units, costs, args.shard, args.n_shards)
    if args.limit:
        mine = mine[:args.limit]
    path = progress_path(stem, args.shard, args.n_shards)
    label = f"[{stem} shard {args.shard}/{args.n_shards}] " if args.n_shards > 1 else f"[{stem}] "
    print(f"{label}env {env_info()}", flush=True)
    return run_units(mine, blocks, order, dev, fold, rf_factory(None, args.n_jobs), path, label)


def merge_search(stem: str, names, meta: dict) -> pd.DataFrame | None:
    """Merge the progress files; when complete write results/<stem>_*.{csv,json}."""
    conflicts: list = []
    records = merge_progress(stem, conflicts)
    if conflicts:
        print(f"[{stem}] {len(conflicts)} unit(s) scored on two machines differ by up to "
              f"{max(c['abs_diff'] for c in conflicts):.1e}; kept the first (laptop) record", flush=True)
    expected = 2 ** len(names) - 1
    print(f"[{stem}] merged {len(records)} of {expected} subsets", flush=True)
    if len(records) < expected:
        return None
    t = table_from_records(records)
    out_csv = RESULTS_DIR / f"{stem}.csv"
    cols = ["cv_rank", "key", "k", "n_columns", "cv_pooled_rmse_mean", "cv_pooled_rmse_std",
            "cv_pooled_r2_mean", "cv_cell_avg_rmse_mean", "cv_fold_rmse", "runtime_s", "host"]
    t[cols].assign(cv_fold_rmse=t["cv_fold_rmse"].apply(json.dumps)).to_csv(out_csv, index=False)
    write_json_atomic(RESULTS_DIR / f"{stem}.json", {
        **meta, "n_subsets": int(len(t)), "names": list(names),
        "total_runtime_s": round(float(t["runtime_s"].sum()), 1),
        "hosts": sorted(t["host"].unique().tolist()),
        "merge_conflicts": conflicts,
        "table": t.drop(columns=["key"]).to_dict("records"),
    }, indent=1)
    print(f"[{stem}] wrote {out_csv} and results/{stem}.json", flush=True)
    return t


def load_search_table(stem: str) -> pd.DataFrame:
    d = read_json(RESULTS_DIR / f"{stem}.json")
    if d is None:
        raise FileNotFoundError(f"results/{stem}.json missing - run the search (or --merge) first")
    return table_from_records({subset_key(r["blocks"]): r for r in d["table"]})


# ------------------------ derivations (no fits) ------------------------

def _row(t: pd.DataFrame, by_key: dict, blocks_sel) -> dict:
    r = by_key[subset_key(blocks_sel)]
    return {"k": int(r["k"]), "blocks": list(r["blocks"]),
            "cv": float(r["cv_pooled_rmse_mean"]), "cv_std": float(r["cv_pooled_rmse_std"]),
            "cv_rank": int(r["cv_rank"])}


def best_per_k(t: pd.DataFrame, by_key: dict) -> list[dict]:
    out = []
    for k, g in t.groupby("k", sort=True):
        r = g.sort_values(["cv_pooled_rmse_mean", "key"]).iloc[0]
        out.append(_row(t, by_key, r["blocks"]))
    return out


def backward_elimination(t, by_key, names) -> list[dict]:
    """From all blocks down to one: drop the block whose removal gives the lowest CV."""
    cur = tuple(names)
    path = [{**_row(t, by_key, cur), "dropped": None}]
    while len(cur) > 1:
        cands = [(by_key[subset_key(canonical([c for c in cur if c != d], names))]
                  ["cv_pooled_rmse_mean"], names.index(d), d) for d in cur]
        _, _, drop = min(cands)
        cur = canonical([c for c in cur if c != drop], names)
        path.append({**_row(t, by_key, cur), "dropped": drop})
    return path


def forward_selection(t, by_key, names) -> list[dict]:
    """From empty up to all blocks: add the block that gives the lowest CV."""
    cur: tuple[str, ...] = ()
    path = []
    while len(cur) < len(names):
        cands = [(by_key[subset_key(canonical(cur + (a,), names))]["cv_pooled_rmse_mean"],
                  names.index(a), a) for a in names if a not in cur]
        _, _, add = min(cands)
        cur = canonical(cur + (add,), names)
        path.append({**_row(t, by_key, cur), "added": add})
    return path


def ranking_path(t, by_key, ranking: list[str], names) -> list[dict]:
    """Top-k prefixes of a fixed ranking (the ENG-08 forward k-curve subsets)."""
    return [{**_row(t, by_key, canonical(ranking[:k], names)), "added": ranking[k - 1]}
            for k in range(1, len(ranking) + 1)]


def top_inclusion(t: pd.DataFrame, names, n: int = 20) -> dict:
    top = t.head(n)
    return {b: float(top["blocks"].apply(lambda s: b in s).mean()) for b in names}


def shapley(by_key: dict, names, v_empty: float) -> dict:
    """
    Exact Shapley value of each block for the game v(S) = dev-CV RMSE of S
    (v(empty) = mean predictor). phi_i is the average RMSE *reduction* from
    adding i over all orders; the phi sum to v(empty) - v(all).
    """
    n = len(names)

    def v(s) -> float:
        return v_empty if not s else by_key[subset_key(canonical(s, names))]["cv_pooled_rmse_mean"]

    phi = {}
    for i in names:
        rest = [x for x in names if x != i]
        total = 0.0
        for r in range(0, n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for s in itertools.combinations(rest, r):
                total += w * (v(s) - v(s + (i,)))
        phi[i] = total
    return phi


def selection_rules(t: pd.DataFrame, by_key: dict, n_folds: int = N_SPLITS) -> dict:
    """
    Two ways to pick a smaller subset that is 'as good as' the best:
      within_one_fold_sd   smallest k with CV <= best + fold sd of the best
                           (ENG-19 calls this "one_se"; it is not a standard error)
      one_standard_error   smallest k with CV <= best + fold sd / sqrt(n_folds)
                           (the usual one-standard-error rule)
    Ties at the smallest k go to the lower CV. The fold sd is the stored
    np.std over the five folds (ddof = 0); the ddof = 1 variant is reported
    as a sensitivity check.
    """
    best = t.iloc[0]
    sd = float(best["cv_pooled_rmse_std"])
    rules = {
        "within_one_fold_sd": sd,
        "one_standard_error": sd / math.sqrt(n_folds),
        "one_standard_error_ddof1": sd * math.sqrt(n_folds / (n_folds - 1)) / math.sqrt(n_folds),
    }
    out = {"best": _row(t, by_key, best["blocks"]), "fold_sd_of_best": sd}
    for name, margin in rules.items():
        thresh = float(best["cv_pooled_rmse_mean"]) + margin
        within = t[t["cv_pooled_rmse_mean"] <= thresh]
        pick = within.sort_values(["k", "cv_pooled_rmse_mean", "key"]).iloc[0]
        out[name] = {"margin": margin, "threshold": thresh, "n_within": int(len(within)),
                     "pick": _row(t, by_key, pick["blocks"])}
    return out


# ------------------------ SoC windows (ENG-17) ------------------------

def feature_windows_from_medians(path: Path = ENG17_SUMMARY) -> dict[str, tuple[float, float]]:
    """ENG-17 median window per feature (all cycles, as implemented): (start, end) SoC."""
    s = pd.read_csv(path)
    s = s[(s["stage"] == "all cycles") & (s["basis"] == "as_implemented")]
    return {r.feature: (float(r.soc_start_median), float(r.soc_end_median))
            for r in s.itertuples()}


def subset_window(features, windows: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
    """Union of the median windows: they all grow outwards from the apex,
    so the union is (max start, min end)."""
    start = max(windows[f][0] for f in features)
    end = min(windows[f][1] for f in features)
    return start, end, start - end


def per_cycle_spans(features_list: list[tuple[str, ...]], path: Path = SOC_PARQUET) -> dict | None:
    """
    Median over cycles of the per-cycle union window of each subset (what
    src/feature_requirements.set_requirement does, vectorised). None if the
    per-cycle table (gitignored, needs the raw pickles) is not on disk.
    """
    if not path.exists():
        return None
    win = pd.read_parquet(path)
    win = win[win["basis"] == "as_implemented"]
    st = win.pivot_table(index=["cell_id", "cycle"], columns="feature",
                         values="soc_start", aggfunc="first")
    en = win.pivot_table(index=["cell_id", "cycle"], columns="feature",
                         values="soc_end", aggfunc="first")
    out = {}
    for feats in features_list:
        a = st[list(feats)].to_numpy()
        b = en[list(feats)].to_numpy()
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        start, end = a[ok].max(axis=1), b[ok].min(axis=1)
        out[subset_key(feats)] = {"soc_start_median": float(np.median(start)),
                                  "soc_end_median": float(np.median(end)),
                                  "soc_span_median": float(np.median(start - end)),
                                  "n_cycles": int(ok.sum())}
    return out


def tradeoff_edge(t: pd.DataFrame, span_col: str = "soc_span", decimals: int = 3) -> list[dict]:
    """Lower edge of CV RMSE against discharge needed: for each amount of
    discharge (ascending), the best subset needing at most that much, kept
    only where it improves on the previous step. Spans are rounded to
    `decimals` (default 0.1 % SoC) so that subsets whose per-cycle median
    windows differ only by noise count as needing the same discharge."""
    s = t.assign(_span=t[span_col].round(decimals)).sort_values(["_span", "cv_pooled_rmse_mean", "key"])
    edge, best = [], math.inf
    for span, g in s.groupby("_span", sort=True):
        r = g.iloc[0]
        if r["cv_pooled_rmse_mean"] < best:
            best = float(r["cv_pooled_rmse_mean"])
            edge.append({"soc_span": float(span), "k": int(r["k"]), "blocks": list(r["blocks"]),
                         "cv": best, "cv_std": float(r["cv_pooled_rmse_std"]),
                         "cv_rank": int(r["cv_rank"])})
    return edge


def decision_margins(t: pd.DataFrame, by_key: dict, names) -> dict:
    """
    For every choice read off the table, the dev-CV gap between the chosen
    subset and the runner-up. A choice can only change if the scores move by
    more than this gap, so comparing the smallest gap with the largest
    cross-machine difference shows whether any conclusion depends on which
    machine scored which subset.
    """
    def cv(sel) -> float:
        return float(by_key[subset_key(canonical(sel, names))]["cv_pooled_rmse_mean"])

    out: dict = {"best_of_k": {}, "backward_elimination": [], "forward_selection": []}
    for k, g in t.groupby("k"):
        v = np.sort(g["cv_pooled_rmse_mean"].to_numpy())
        if len(v) > 1:
            out["best_of_k"][int(k)] = float(v[1] - v[0])
    cur = tuple(names)
    while len(cur) > 1:
        v = sorted(cv([c for c in cur if c != d]) for d in cur)
        out["backward_elimination"].append(float(v[1] - v[0]))
        drop = min(cur, key=lambda d: (cv([c for c in cur if c != d]), names.index(d)))
        cur = canonical([c for c in cur if c != drop], names)
    cur = ()
    while len(cur) < len(names) - 1:
        cands = [a for a in names if a not in cur]
        v = sorted(cv(cur + (a,)) for a in cands)
        out["forward_selection"].append(float(v[1] - v[0]))
        add = min(cands, key=lambda a: (cv(cur + (a,)), names.index(a)))
        cur = canonical(cur + (add,), names)
    # A rule's pick changes only if a smaller subset crosses the threshold,
    # the pick itself drops out, or the pick's runner-up at the same size
    # overtakes it.
    best = t.iloc[0]
    sd = float(best["cv_pooled_rmse_std"])
    out["rule_thresholds"] = {}
    for rule, margin in (("within_one_fold_sd", sd), ("one_standard_error", sd / math.sqrt(N_SPLITS))):
        thresh = float(best["cv_pooled_rmse_mean"]) + margin
        within = t[t["cv_pooled_rmse_mean"] <= thresh]
        kp = int(within["k"].min())
        at_k = np.sort(within.loc[within["k"] == kp, "cv_pooled_rmse_mean"].to_numpy())
        gaps = [thresh - at_k[0]]
        if len(at_k) > 1:
            gaps.append(at_k[1] - at_k[0])
        smaller = t.loc[t["k"] < kp, "cv_pooled_rmse_mean"].to_numpy()
        if len(smaller):
            gaps.append(float(smaller.min() - thresh))
        out["rule_thresholds"][rule] = float(min(gaps))
    flat = (list(out["best_of_k"].values()) + out["backward_elimination"]
            + out["forward_selection"] + list(out["rule_thresholds"].values()))
    out["smallest"] = float(min(flat))
    return out


# ------------------------ the analysis bundle ------------------------

def analyse_table(t: pd.DataFrame, names: list[str], v_empty: float,
                  rankings: dict[str, list[str]] | None = None) -> dict:
    by_key = {r["key"]: r for r in t.to_dict("records")}
    expected = 2 ** len(names) - 1
    if len(t) != expected:
        raise ValueError(f"table has {len(t)} subsets, expected {expected}")
    be = backward_elimination(t, by_key, names)
    sfs = forward_selection(t, by_key, names)
    phi = shapley(by_key, names, v_empty)
    full = by_key[subset_key(names)]["cv_pooled_rmse_mean"]
    out = {
        "names": list(names),
        "n_subsets": int(len(t)),
        "v_empty_cv_rmse": v_empty,
        "all_blocks": _row(t, by_key, names),
        "best_per_k": best_per_k(t, by_key),
        "backward_elimination": be,
        "be_drop_order": [s["dropped"] for s in be[1:]],
        "be_last_standing": be[-1]["blocks"][0],
        "forward_selection": sfs,
        "sfs_add_order": [s["added"] for s in sfs],
        "ranking_paths": {name: ranking_path(t, by_key, r, names)
                          for name, r in (rankings or {}).items()},
        "top20_inclusion_rate": top_inclusion(t, names, 20),
        "shapley": phi,
        "shapley_rank": sorted(phi, key=lambda b: -phi[b]),
        "shapley_efficiency_check": {"sum_phi": float(sum(phi.values())),
                                     "v_empty_minus_v_all": float(v_empty - full)},
        "selection_rules": selection_rules(t, by_key),
        "decision_margins": decision_margins(t, by_key, list(names)),
    }
    return out


# ------------------------ batch 3 (the only place it is touched) ------------------------

class HeldoutLog:
    """
    Every batch-3 evaluation goes through `score`, which records the reason
    (e.g. 'best_of_k', 'backward_elimination'). A subset is fitted once per
    configuration and further reasons are appended. `records` is written to
    the result file so the set of scored subsets can be audited.
    """

    def __init__(self, dev, test, blocks, order, allowed_reasons: set[str]):
        self.dev, self.test, self.blocks, self.order = dev, test, blocks, order
        self.allowed = set(allowed_reasons)
        self.records: dict[str, dict] = {}

    def score(self, blocks_sel, reason: str, make, config: str = "untuned") -> dict:
        if reason not in self.allowed:
            raise ValueError(f"batch-3 reason {reason!r} is not in the allowed list {sorted(self.allowed)}")
        key = f"{config}|{subset_key(blocks_sel)}"
        if key in self.records:
            if reason not in self.records[key]["reasons"]:
                self.records[key]["reasons"].append(reason)
            return self.records[key]
        cols = columns_for(blocks_sel, self.blocks, self.order)
        ho = heldout_evaluate(self.dev, self.test, cols, make, None)
        rec = {"config": config, "blocks": list(blocks_sel), "k": len(blocks_sel),
               "n_columns": len(cols), "reasons": [reason],
               "b3_rmse": ho["pooled"]["rmse"], "b3_r2": ho["pooled"]["r2"],
               "b3_mae": ho["pooled"]["mae"], "b3_rmse_cellavg": ho["cell_avg"]["rmse"]}
        self.records[key] = rec
        print(f"  batch 3 [{config}] {subset_key(blocks_sel)}: RMSE {rec['b3_rmse']:.5f} "
              f"R2 {rec['b3_r2']:.4f} ({reason})", flush=True)
        return rec


# ------------------------ analysis + listed batch-3 scores ------------------------

LISTED_REASONS = {"reference_all", "best_of_k", "backward_elimination",
                  "tradeoff_edge", "recommended"}


def eng08_rankings() -> dict[str, list[str]]:
    """The ENG-08 RF-cell rankings (untuned grid; PI also from the tuned grid)."""
    u = read_json(RESULTS_DIR / "eng08_full.json")["grid"]
    tu = read_json(RESULTS_DIR / "eng08_full_tuned.json")["grid"]

    def rk(g, key):
        return [r["feature"] for r in g[key]["ranking"]]

    return {"Pearson": rk(u, "pearson|rf"),
            "Spearman": rk(u, "spearman|rf"),
            "mutual information": rk(u, "mutual_info|rf"),
            "RF impurity": rk(u, "rf_importance|rf"),
            "PI (untuned RF)": rk(u, "permutation|rf"),
            "PI (tuned RF)": rk(tu, "permutation|rf")}


def add_windows(t: pd.DataFrame, features_of=lambda blocks: blocks) -> pd.DataFrame:
    """SoC window columns for every subset: from the ENG-17 medians, and the
    median per-cycle union when results/features_soc_windows.parquet exists."""
    med = feature_windows_from_medians()
    feats = [tuple(features_of(b)) for b in t["blocks"]]
    w = [subset_window(f, med) for f in feats]
    t = t.assign(soc_start=[x[0] for x in w], soc_end=[x[1] for x in w],
                 soc_span=[x[2] for x in w])
    pc = per_cycle_spans(sorted(set(feats), key=lambda f: (len(f), f)))
    if pc is not None:
        t = t.assign(soc_span_per_cycle=[pc[subset_key(f)]["soc_span_median"] for f in feats],
                     soc_start_per_cycle=[pc[subset_key(f)]["soc_start_median"] for f in feats],
                     soc_end_per_cycle=[pc[subset_key(f)]["soc_end_median"] for f in feats])
    return t


def recommended_subsets(A: dict, edge: list[dict] | None) -> list[tuple[str, ...]]:
    """The two one-SE picks and the trade-off edge, deduplicated, in that order."""
    out = []
    for rule in ("within_one_fold_sd", "one_standard_error"):
        b = tuple(A["selection_rules"][rule]["pick"]["blocks"])
        if b not in out:
            out.append(b)
    for r in edge or []:
        if tuple(r["blocks"]) not in out:
            out.append(tuple(r["blocks"]))
    return out


def analyse_search(t: pd.DataFrame, names, blocks, order, dev, test, fold, *,
                   rankings: dict | None, windows: bool, tuned: bool,
                   features_of=lambda blocks: blocks) -> tuple[dict, pd.DataFrame]:
    """
    Everything read off a finished table, then batch 3 for the listed subsets
    only: the all-blocks reference, best-of-k, the backward-elimination path,
    the trade-off edge and the recommended subsets (both one-SE picks, edge).
    With tuned=True the recommended subsets and the reference are also
    scored with the fixed ENG-09 tuned RF (dev-CV and batch 3).
    """
    v0 = mean_predictor_cv(dev, fold)
    A = analyse_table(t, list(names), v0["cv_pooled_rmse_mean"], rankings)
    A["v_empty"] = v0
    edge = None
    if windows:
        t = add_windows(t, features_of)
        # Primary: median over cycles of the per-cycle union window (matches
        # ENG-17's per-feature spans); fallback: union of the median windows.
        A["window_basis"] = ("per-cycle union, median over cycles" if "soc_span_per_cycle" in t
                             else "union of the ENG-17 median windows")
        A["tradeoff_edge_from_medians"] = tradeoff_edge(t, "soc_span")
        edge = (tradeoff_edge(t, "soc_span_per_cycle") if "soc_span_per_cycle" in t
                else A["tradeoff_edge_from_medians"])
        A["tradeoff_edge"] = edge
    rec = recommended_subsets(A, edge)
    A["recommended_subsets"] = [list(b) for b in rec]

    make_u = rf_factory()
    log = HeldoutLog(dev, test, blocks, order, LISTED_REASONS)
    log.score(tuple(names), "reference_all", make_u)
    for r in A["best_per_k"]:
        log.score(r["blocks"], "best_of_k", make_u)
    for r in A["backward_elimination"]:
        log.score(r["blocks"], "backward_elimination", make_u)
    for r in edge or []:
        log.score(r["blocks"], "tradeoff_edge", make_u)
    for b in rec:
        log.score(b, "recommended", make_u)
    if tuned:
        make_t = rf_factory(tuned_rf_params())
        A["tuned"] = {}
        for b in [tuple(names)] + rec:
            reason = "reference_all" if list(b) == list(names) else "recommended"
            cols = columns_for(b, blocks, order)
            t0 = time.time()
            cv = cv_evaluate(dev, cols, fold, make_t, None)
            ho = log.score(b, reason, make_t, config="tuned")
            A["tuned"][subset_key(b)] = {
                "blocks": list(b), "k": len(b), "cv": cv["pooled_rmse_mean"],
                "cv_std": cv["pooled_rmse_std"], "b3_rmse": ho["b3_rmse"], "b3_r2": ho["b3_r2"],
                "b3_mae": ho["b3_mae"], "runtime_s": round(time.time() - t0, 1)}
            print(f"  tuned {subset_key(b)}: CV {cv['pooled_rmse_mean']:.5f} "
                  f"b3 {ho['b3_rmse']:.5f}", flush=True)
    A["heldout"] = list(log.records.values())
    A["heldout_audit"] = {
        "n_subsets_scored_on_batch3": len({subset_key(r["blocks"]) for r in log.records.values()}),
        "reasons_allowed": sorted(LISTED_REASONS),
        "every_record_has_listed_reason": all(set(r["reasons"]) <= LISTED_REASONS
                                              for r in log.records.values()),
    }
    return A, t


def save_analysis(stem: str, A: dict, tw: pd.DataFrame) -> None:
    """results/<stem>_analysis.json and the per-subset table with windows."""
    write_json_atomic(RESULTS_DIR / f"{stem}_analysis.json", A, indent=1)
    cols = [c for c in ["cv_rank", "key", "k", "n_columns", "cv_pooled_rmse_mean",
                        "cv_pooled_rmse_std", "soc_start", "soc_end", "soc_span",
                        "soc_start_per_cycle", "soc_end_per_cycle", "soc_span_per_cycle"]
            if c in tw]
    tw[cols].to_csv(RESULTS_DIR / f"{stem}_windows.csv", index=False)
    print(f"wrote results/{stem}_analysis.json and results/{stem}_windows.csv", flush=True)
