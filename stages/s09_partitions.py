"""
9. The same comparisons under six train/test partitions (ENG-15)
================================================================
Do the selector comparison (stage 7) and the level / history pool comparison
(stage 8) depend on the batches 1+2 -> batch 3 split? Both are re-run untuned
under the six partitions of src/partitions.py (the relabel mechanism is there).
Inputs   results/features_expanded.parquet; eng08_full.json and eng14_full.json
         (the lobo_b3 cells are copied from there, one recomputed as a self-check).
Outputs  eng15_cells.json {cells["<exp>|<partition>|<pool>|<selector>|<regressor>"]
         = a slim() record, errors, self_check, partitions}; eng15_cells.csv (one row
         per cell); eng15_split_robustness.json {partitions, self_check, n_cells,
         summary} (see summarise), written once every cell of the grid is stored.
Grid     60 untuned RF cells (grid_cells); --regressors rf gpr adds 24 e08 GPR cells.
Runtime  ~10 h (RF cells 3-25 min each, GPR cells 1-1.5 h, --jobs N at a time);
         resumable from the store, --report-only rebuilds CSV and summary, no fit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.config import RESULTS_DIR
from src.data_prep import RF_SEED, assign_folds, clean_feature_table, log_manifest, split_dev_test
from src.features_temporal import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_WINDOWS,
    HISTORY_COL,
    LEVEL_COLS,
    TEMPORAL_COLS,
    add_temporal_features,
)
from src.harness import _run_cell, build_pools, k_grid, temporal_share
from src.partitions import ID_COLS, PARTITIONS, make_partitions, relabel
from src.subset_search import write_json_atomic

PARQUET = RESULTS_DIR / "features_expanded.parquet"
ENG08_JSON = RESULTS_DIR / "eng08_full.json"
ENG14_JSON = RESULTS_DIR / "eng14_full.json"
CELLS_JSON = RESULTS_DIR / "eng15_cells.json"
CELLS_CSV = RESULTS_DIR / "eng15_cells.csv"
OUT_JSON = RESULTS_DIR / "eng15_split_robustness.json"
SCRIPT = "stages/s09_partitions.py"

WINDOWS = DEFAULT_WINDOWS
MIN_HISTORY = DEFAULT_MIN_HISTORY

METHODS = ["pearson", "mutual_info", "rf_importance", "permutation"]
E14_METHOD_ORDER = ["pearson", "permutation", "rf_importance", "mutual_info"]  # run order
POOLS = ["level", "temporal", "combined"]
REGS = ["rf", "gpr"]
IMPORTED = "lobo_b3"            # the study split: stages 7 and 8 computed its cells
SOURCE_FILE = {"e08": ENG08_JSON, "e14": ENG14_JSON}
SOURCE_STAGE = {"e08": "stage 7 (stages/s07_selector_grid.py)",
                "e14": "stage 8 (stages/s08_history_features.py)"}

Cell = tuple[str, str, str, str, str]   # (experiment, partition, pool, selector, regressor)


def cell_key(exp: str, partition: str, pool: str, method: str, reg: str) -> str:
    return f"{exp}|{partition}|{pool}|{method}|{reg}"


def grid_cells(regressors: list[str]) -> list[Cell]:
    """The cells of the grid in run order: e08 on the level pool with RF and, when
    asked, GPR; then e14 (RF only) by selector, pearson on every pool and the other
    selectors on the combined pool. The RF cells are never optional: the report
    reads them, and run_results.py takes the summary JSON as the stage's done-marker."""
    cells: list[Cell] = []
    for reg in ["rf"] + (["gpr"] if "gpr" in regressors else []):
        cells += [("e08", p, "level", m, reg) for p in PARTITIONS for m in METHODS]
    for m in E14_METHOD_ORDER:
        pools = POOLS if m == "pearson" else ["combined"]
        cells += [("e14", p, pool, m, "rf") for p in PARTITIONS for pool in pools]
    return cells


# ------------------------ the data of each partition ------------------------

class Data:
    """The clean rows (e08) and the history rows (e14), and per partition their
    relabelled tables with the cell -> fold map (assign_folds on the partition's
    full clean dev rows, reused on the history rows: the ENG-14 rule)."""

    def __init__(self, clean: pd.DataFrame, hist: pd.DataFrame, temporal_cols: list[str],
                 parts: dict, pools: dict[str, list[str]]):
        self.level = clean[ID_COLS + LEVEL_COLS].copy()
        self.history = hist[ID_COLS + LEVEL_COLS + temporal_cols].copy()
        self.parts = parts
        self.pools = pools
        self._tables: dict[str, tuple] = {}

    def tables(self, partition: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """(level table, history table, cell -> fold map) of one partition."""
        if partition not in self._tables:
            test_cells = self.parts[partition]["test_cells"]
            full = relabel(self.level, test_cells)
            hst = relabel(self.history, test_cells)
            _, fmap = assign_folds(split_dev_test(full)[0])
            self._tables[partition] = (full, hst, fmap)
        return self._tables[partition]

    def table(self, exp: str, partition: str) -> pd.DataFrame:
        """The rows an experiment uses: e08 every clean row, e14 the history rows."""
        full, hst, _ = self.tables(partition)
        return full if exp == "e08" else hst

    def meta(self, exp: str, partition: str, pool: str, method: str, reg: str,
             source: str) -> dict:
        """The identifying fields of one cell record."""
        dev, test = split_dev_test(self.table(exp, partition))
        return {"experiment": exp, "partition": partition, "kind": self.parts[partition]["kind"],
                "pool": pool, "method": method, "reg": reg, "p": len(self.pools[pool]),
                "n_dev_cells": int(dev["cell_id"].nunique()),
                "n_test_cells": int(test["cell_id"].nunique()),
                "n_dev_rows": int(len(dev)), "n_test_rows": int(len(test)),
                "source": source}


# ------------------------ per-cell store ------------------------

def slim(out: dict, meta: dict) -> dict:
    """Compact, JSON-friendly record of one compare_k_grid() result."""
    return {
        **meta,
        "k": len(out["selected_features"]),
        "selected_features": out["selected_features"],
        "ranking": out["ranking"],
        "k_curve": [{k: v for k, v in r.items() if k != "features"}
                    for r in out["k_curve"]],
        "cv_rmse": out["cv_scores"]["pooled_rmse_mean"],
        "cv_rmse_std": out["cv_scores"]["pooled_rmse_std"],
        "cv_r2": out["cv_scores"]["pooled_r2_mean"],
        "test": out["heldout_scores"],
        "temporal_share": temporal_share(out["ranking"], out["selected_features"]),
        "runtime_s": out.get("runtime_s"),
    }


class Store:
    """results/eng15_cells.json: the finished cells, the errors, the self-check and
    the partitions; rewritten after every cell so an interrupted run resumes."""

    def __init__(self, path):
        self.path = path
        self.data: dict = {"cells": {}, "errors": [], "self_check": None}
        if path.exists():
            self.data.update(json.loads(path.read_text(encoding="utf-8")))

    def __contains__(self, key: str) -> bool:
        return key in self.data["cells"]

    def cell(self, key: str) -> dict:
        return self.data["cells"][key]

    def put(self, key: str, rec: dict) -> None:
        self.data["cells"][key] = rec
        self.save()

    def add_error(self, key: str, exc: Exception) -> None:
        self.data["errors"].append({"key": key, "error": repr(exc)})
        self.save()

    def save(self) -> None:
        write_json_atomic(self.path, self.data)


def cells_frame(store: Store) -> pd.DataFrame:
    """One row per stored cell (the eng15_cells.csv columns)."""
    rows = []
    for key, c in store.data["cells"].items():
        ts = c.get("temporal_share") or {}
        rows.append({
            "key": key, "experiment": c["experiment"], "partition": c["partition"],
            "kind": c["kind"], "pool": c["pool"], "selection_method": c["method"],
            "regressor": c["reg"], "p": c["p"], "k": c["k"],
            "test_rmse": c["test"]["pooled"]["rmse"], "test_mae": c["test"]["pooled"]["mae"],
            "test_r2": c["test"]["pooled"]["r2"],
            "test_rmse_cellavg": c["test"]["cell_avg"]["rmse"],
            "cv_rmse": c["cv_rmse"], "cv_rmse_std": c["cv_rmse_std"],
            "top1_feature": c["ranking"][0]["feature"],
            "frac_temporal_selected": ts.get("frac_temporal_selected"),
            "best_level_rank": ts.get("best_level_rank"),
            "n_dev_cells": c["n_dev_cells"], "n_test_cells": c["n_test_cells"],
            "n_dev_rows": c["n_dev_rows"], "n_test_rows": c["n_test_rows"],
            "source": c["source"], "runtime_s": c.get("runtime_s"),
        })
    return pd.DataFrame(rows)


# ------------------------ the cells ------------------------

def import_study_split(store: Store, cells: list[Cell], data: Data) -> None:
    """The lobo_b3 cells of the grid are the study split: copy them from stages 7
    and 8 (eng08_full.json, eng14_full.json) instead of recomputing them."""
    todo = [c for c in cells if c[1] == IMPORTED and cell_key(*c) not in store]
    if not todo:
        return
    grids: dict[str, dict] = {}
    for exp in sorted({c[0] for c in todo}):
        if not SOURCE_FILE[exp].exists():
            sys.exit(f"{SOURCE_FILE[exp]} missing - run {SOURCE_STAGE[exp]} first")
        grids[exp] = json.loads(SOURCE_FILE[exp].read_text(encoding="utf-8"))["grid"]
    for exp, partition, pool, method, reg in todo:
        stored = grids[exp] if exp == "e08" else grids[exp][pool]
        if f"{method}|{reg}" not in stored:
            sys.exit(f"{SOURCE_FILE[exp].name} has no {method}|{reg} cell"
                     f"{f' on the {pool} pool' if exp == 'e14' else ''} - rerun "
                     f"{SOURCE_STAGE[exp]} with --regressors rf gpr --selectors all")
        source = f"results/{SOURCE_FILE[exp].name}"
        store.put(cell_key(exp, partition, pool, method, reg),
                  slim(stored[f"{method}|{reg}"],
                       data.meta(exp, partition, pool, method, reg, source)))
    print(f"  copied {len(todo)} {IMPORTED} cells from "
          + " and ".join(SOURCE_FILE[e].name for e in sorted(grids)), flush=True)


def self_check(store: Store, data: Data) -> None:
    """The relabel mechanism must reproduce stage 7's Pearson x RF cell on the
    study split (held-out and CV RMSE to 1e-9, same features); exit otherwise."""
    key = cell_key("e08", IMPORTED, "level", "pearson", "rf")
    if store.data.get("self_check") is not None or key not in store:
        return
    ref = store.cell(key)
    full, _, fmap = data.tables(IMPORTED)
    print(f"\n===== self-check: {key} through the relabel mechanism =====", flush=True)
    cols = data.pools["level"]
    out = _run_cell(full, "rf", "pearson", cols, k_grid(len(cols)), None, fmap, False)
    got = out["heldout_scores"]["pooled"]["rmse"]
    ok = bool(np.isclose(got, ref["test"]["pooled"]["rmse"], rtol=1e-9)
              and np.isclose(out["cv_scores"]["pooled_rmse_mean"], ref["cv_rmse"], rtol=1e-9)
              and out["selected_features"] == ref["selected_features"])
    store.data["self_check"] = {"pass": ok, "recomputed_test_rmse": got,
                                "stored_test_rmse": ref["test"]["pooled"]["rmse"]}
    store.save()
    print(f"  self-check {'PASS' if ok else 'FAIL'}: {got:.10f} vs "
          f"{ref['test']['pooled']['rmse']:.10f}", flush=True)
    if not ok:
        sys.exit("self-check failed: the relabel mechanism does not reproduce stage 7")


def finished(store: Store, cell: Cell, out: dict, data: Data, note: str = "") -> None:
    """Store one computed cell and report it."""
    key = cell_key(*cell)
    store.put(key, slim(out, data.meta(*cell, "eng15")))
    print(f"  [{key}] done in {out['runtime_s']} s{note}: k={len(out['selected_features'])} "
          f"test RMSE={out['heldout_scores']['pooled']['rmse']:.5f}", flush=True)


def run_cell(store: Store, cell: Cell, data: Data) -> None:
    """One RF cell in the foreground; a failure is recorded and the run goes on."""
    key = cell_key(*cell)
    if key in store:
        return
    exp, partition, pool, method, reg = cell
    table = data.table(exp, partition)
    cols = data.pools[pool]
    _, _, fmap = data.tables(partition)
    print(f"\n===== {key} =====  p={len(cols)}", flush=True)
    try:
        finished(store, cell, _run_cell(table, reg, method, cols, k_grid(len(cols)),
                                        None, fmap, True), data)
    except Exception as exc:  # keep a long unattended run alive
        traceback.print_exc()
        store.add_error(key, exc)


def harvest(store: Store, futures: dict, data: Data, block: bool = False) -> None:
    """Store the background (GPR) cells that have finished; block = wait for all."""
    done = list(as_completed(futures)) if block else [f for f in futures if f.done()]
    for fut in done:
        cell = futures.pop(fut)
        try:
            finished(store, cell, fut.result(), data, note=" (background)")
        except Exception as exc:
            traceback.print_exc()
            store.add_error(cell_key(*cell), exc)


# ------------------------ summary ------------------------

def test_rmse(c: dict) -> float:
    return c["test"]["pooled"]["rmse"]


def rank_vector(ranking: list[dict], pool_cols: list[str]) -> np.ndarray:
    """Position of each pool column in one method's ranking."""
    order = [r["feature"] for r in ranking]
    return np.array([order.index(f) for f in pool_cols])


def ranking_stability(store: Store, exp: str, pool: str, reg: str,
                      pool_cols: list[str]) -> dict:
    """Mean pairwise Kendall tau / top-3 Jaccard of each selector's ranking across
    the stored partitions, and of the selectors against each other within a partition."""
    out: dict = {"across_partitions": {}, "across_methods": {}}
    for m in METHODS:
        ranks = {p: store.cell(cell_key(exp, p, pool, m, reg))["ranking"]
                 for p in PARTITIONS if cell_key(exp, p, pool, m, reg) in store}
        taus, jac = [], []
        for a, b in itertools.combinations(ranks, 2):
            taus.append(kendalltau(rank_vector(ranks[a], pool_cols),
                                   rank_vector(ranks[b], pool_cols))[0])
            ta = {x["feature"] for x in ranks[a][:3]}
            tb = {x["feature"] for x in ranks[b][:3]}
            jac.append(len(ta & tb) / len(ta | tb))
        out["across_partitions"][m] = {
            "n_partitions": len(ranks),
            "kendall_tau_mean": float(np.mean(taus)) if taus else None,
            "kendall_tau_min": float(np.min(taus)) if taus else None,
            "top3_jaccard_mean": float(np.mean(jac)) if jac else None,
            "top1_by_partition": {p: r[0]["feature"] for p, r in ranks.items()},
        }
    for p in PARTITIONS:
        ranks = {m: store.cell(cell_key(exp, p, pool, m, reg))["ranking"]
                 for m in METHODS if cell_key(exp, p, pool, m, reg) in store}
        taus = [kendalltau(rank_vector(ranks[a], pool_cols),
                           rank_vector(ranks[b], pool_cols))[0]
                for a, b in itertools.combinations(ranks, 2)]
        out["across_methods"][p] = {
            "n_methods": len(ranks),
            "kendall_tau_mean": float(np.mean(taus)) if taus else None,
        }
    return out


def relative_vs_level(store: Store) -> dict:
    """(rmse_pool - rmse_level) / rmse_level per partition, for every selector whose
    level and history-pool e14 cells are both stored."""
    out: dict = {}
    for m in E14_METHOD_ORDER:
        for pool in ["temporal", "combined"]:
            rels = {}
            for p in PARTITIONS:
                lv, tp = cell_key("e14", p, "level", m, "rf"), cell_key("e14", p, pool, m, "rf")
                if lv in store and tp in store:
                    level = test_rmse(store.cell(lv))
                    rels[p] = float((test_rmse(store.cell(tp)) - level) / level)
            if rels:
                out[f"{m}|{pool}"] = rels
    return out


def summarise(store: Store, pools: dict[str, list[str]]) -> dict:
    """Ranking stability of the e08 grid (per regressor with stored cells) and of the
    e14 combined pool, and the history pools' held-out RMSE relative to the level pool."""
    regs = [r for r in REGS if any(c["experiment"] == "e08" and c["reg"] == r
                                   for c in store.data["cells"].values())]
    return {
        "e08": {r: ranking_stability(store, "e08", "level", r, pools["level"]) for r in regs},
        "e14": {"relative_vs_level": relative_vs_level(store),
                "stability_combined_rf": ranking_stability(store, "e14", "combined", "rf",
                                                           pools["combined"])},
    }


def finish(store: Store, cells: list[Cell], pools: dict[str, list[str]]) -> None:
    """eng15_cells.csv from every stored cell; the summary JSON and the manifest entry
    once every cell of the grid is stored and no error is left (exit 1 on errors)."""
    df = cells_frame(store)
    if df.empty:
        sys.exit("no cells stored")
    df.to_csv(CELLS_CSV, index=False)
    expected = [cell_key(*c) for c in cells]
    done = sum(k in store for k in expected)
    errors = store.data["errors"]
    print(f"\nWrote {CELLS_CSV.name}: {done}/{len(expected)} cells of the grid stored, "
          f"{len(errors)} errors", flush=True)
    for e in errors:
        print(f"  error {e['key']}: {e['error']}")
    if done < len(expected) or errors:
        sys.exit(1 if errors else 0)

    summary = summarise(store, pools)
    OUT_JSON.write_text(json.dumps({
        "partitions": store.data["partitions"],
        "self_check": store.data.get("self_check"),
        "n_cells": len(expected),
        "summary": summary,
    }, indent=2), encoding="utf-8")
    e14 = df[(df.experiment == "e14") & (df.selection_method == "pearson")]
    log_manifest("eng15_split_robustness", {
        "script": SCRIPT, "seed": RF_SEED,
        "partitions": PARTITIONS, "n_cells": len(expected),
        "self_check": store.data.get("self_check"),
        "pearson_rf_test_rmse_by_pool_partition": {
            f"{r.pool}|{r.partition}": float(r.test_rmse) for r in e14.itertuples()},
    })
    print(f"Wrote {OUT_JSON.name}")


# ------------------------ main ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--regressors", nargs="+", choices=REGS, default=["rf"],
                    help="regressors of the e08 grid (default: rf); 'rf gpr' adds the 24 GPR "
                         "cells (the e14 pools are RF only). The RF cells always run: the "
                         "report reads them")
    ap.add_argument("--jobs", type=int, default=12,
                    help="GPR cells run N at a time in background processes while the RF "
                         "cells run one after another; the numbers do not depend on N")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild eng15_cells.csv and, when every cell of the grid is "
                         "stored, eng15_split_robustness.json from the stored cells; no fit")
    args = ap.parse_args()
    if args.jobs < 1:
        sys.exit("--jobs must be >= 1")
    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")

    clean, _ = clean_feature_table(pd.read_parquet(PARQUET))
    parts = make_partitions(clean)
    for name in PARTITIONS:
        p = parts[name]
        print(f"  {name:8s} dev {p['n_dev_cells']:3d} cells / test {p['n_test_cells']:3d} cells "
              f"{p['test_cells_by_batch']}")

    # The stored cells must belong to these partitions (the store is the resume point).
    store = Store(CELLS_JSON)
    stored = store.data.get("partitions") or {}
    if any(stored[k]["test_cells"] != parts[k]["test_cells"] for k in PARTITIONS if k in stored):
        sys.exit(f"{CELLS_JSON} holds cells of different partitions; move it away to start over")
    store.data["partitions"] = {k: parts[k] for k in PARTITIONS}
    cells = grid_cells(args.regressors)
    pools = build_pools(TEMPORAL_COLS)

    if args.report_only:
        finish(store, cells, pools)
        return

    with_t, temporal_cols = add_temporal_features(clean, windows=WINDOWS, min_history=MIN_HISTORY)
    assert temporal_cols == TEMPORAL_COLS
    hist = with_t[with_t[HISTORY_COL]].reset_index(drop=True)
    data = Data(clean, hist, temporal_cols, parts, pools)

    import_study_split(store, cells, data)
    self_check(store, data)

    # The GPR cells (a serial optimiser each) run in a process pool while the RF
    # cells (every core inside one fit) run one after another in the foreground.
    background = [c for c in cells if c[4] == "gpr" and cell_key(*c) not in store]
    futures: dict = {}
    executor = None
    if background:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        for cell in background:
            full, _, fmap = data.tables(cell[1])
            cols = data.pools[cell[2]]
            futures[executor.submit(_run_cell, full, cell[4], cell[3], cols,
                                    k_grid(len(cols)), None, fmap, False)] = cell
        print(f"\n  submitted {len(background)} GPR cells to a pool of {args.jobs}", flush=True)

    for cell in cells:
        if cell[4] == "rf":
            run_cell(store, cell, data)
            harvest(store, futures, data)
    if futures:
        print(f"\n  waiting for {len(futures)} background cells ...", flush=True)
        harvest(store, futures, data, block=True)
    if executor is not None:
        executor.shutdown()

    # A cell that failed in an earlier attempt but is stored now is not an error.
    store.data["errors"] = [e for e in store.data["errors"] if e["key"] not in store]
    store.save()
    finish(store, cells, pools)


if __name__ == "__main__":
    main()
