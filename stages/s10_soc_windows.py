"""
How much of the discharge each feature needs (ENG-17)
=====================================================
Which stretch of a discharge must be observed to compute each of the nine
peak features?  The part of the dQ/dV curve that each feature's computation
touches (src.feature_requirements) is turned into a state-of-charge window,
SoC(v) = 1 - Qdlin(v) / QD_total, for every clean cycle of all 124 cells.
Descriptive (nothing is fitted or selected), so all three batches are used.

Inputs : results/features_expanded.parquet (stage 2); the three Severson
         pickles (--pkl-dir, ~8 GB RAM); results/eng16_pearson_vs_spearman.csv
         (stage 4; optional, feeds the cost_benefit block).
Outputs: results/features_soc_windows.parquet  one row per clean cycle and
           (feature, basis): cycle, cell_id, batch, SOH, feature, basis, v_hi,
           v_lo, width_mv, soc_start, soc_end, soc_span, q_span_ah, minutes_at_4A
           (gitignored; without it stages 11, 15, 16 and 19 fall back to the
           union of the ENG-17 medians and Table VI leaves its dev-only
           columns empty. run_results.py skips this stage once the JSON below
           exists, so rebuild a missing parquet with
           `python run_results.py --force --only 10`)
         results/eng17_soc_windows_summary.csv  medians and the 5-95 % envelope
           per life stage x feature x basis
         results/eng17_feature_soc_requirements.json  n_cells, n_cycles, checks,
           per_feature (= the CSV), feature_sets (union window of five
           pre-declared sets), cost_benefit (span vs association with SoH)
Checks : a synthetic Gaussian peak with known windows; the located apex
         reproduces the stored height; the capacity passed in each area window
         matches the stored area feature.
Runtime: one pickle load plus one pass over the 100,331 cycles (minutes);
         single pass, not resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PKL_DIR, RESULTS_DIR
from src.data_prep import clean_feature_table, log_manifest
from src.feature_requirements import (
    LIFE_STAGES,
    cycle_windows,
    set_requirement,
    soc_windows_for_cell,
    summarise_windows,
)
from src.features import AREA_WINDOWS, FEATURE_COLS
from src.peak_extraction import DV, ROI_MASK, V_ROI
from src.severson_loader import load_clean_severson

FEATURES_PARQUET = RESULTS_DIR / "features_expanded.parquet"
ENG16_CSV = RESULTS_DIR / "eng16_pearson_vs_spearman.csv"
OUT_PARQUET = RESULTS_DIR / "features_soc_windows.parquet"
OUT_CSV = RESULTS_DIR / "eng17_soc_windows_summary.csv"
OUT_JSON = RESULTS_DIR / "eng17_feature_soc_requirements.json"

# "all cycles" plus the three life stages of src.feature_requirements.
STAGES: list[tuple[str, float, float]] = [("all cycles", -np.inf, np.inf)] + list(LIFE_STAGES)

# Pre-declared feature sets (fixed here, never chosen on the test batch), in pool order.
FEATURE_SETS: dict[str, list[str]] = {
    "height + position": ["height", "position"],
    "height, position, area25": ["height", "position", "area25"],
    "baseline 3 (height, position, area)": ["height", "position", "area"],
    "7 features (no prominence, no fwhm)": [f for f in FEATURE_COLS
                                            if f not in ("prominence", "fwhm")],
    "all 9 features": list(FEATURE_COLS),
}


def synthetic_check() -> dict:
    """A Gaussian dQ/dV peak with an exactly known Q(V): windows must come out right."""
    vp, sigma, amp = 3.1502, 0.030, 6.0
    idx = int(np.argmin(np.abs(V_ROI - vp)))
    vp = float(V_ROI[idx])                                    # put the apex on the grid
    absq = amp * np.exp(-0.5 * ((V_ROI - vp) / sigma) ** 2)
    dqdv = -absq                                              # negative on discharge
    qd = np.concatenate([[0.0], np.cumsum((absq[1:] + absq[:-1]) / 2 * DV)])   # Q(V), descending V
    q_total = float(qd[-1])
    rows = {(r["feature"], r["basis"]): r
            for r in cycle_windows(dqdv, qd, vp, q_total, pad_points=0)}
    out = {}
    for name, hw in AREA_WINDOWS.items():
        r = rows[(name, "as_implemented")]
        mask = (V_ROI >= vp - hw) & (V_ROI <= vp + hw)
        expect_q = float(np.trapezoid(absq[mask], dx=DV))
        # the window brackets +/-hw by at most one grid step per side
        out[f"{name}_width_ok"] = bool(2 * hw * 1000 - 1e-6 <= r["width_mv"] <= 2 * hw * 1000 + 2 * DV * 1000 + 1e-6)
        out[f"{name}_charge_ok"] = bool(abs(r["q_span_ah"] - expect_q) / expect_q < 0.06)
        out[f"{name}_soc_is_charge"] = bool(abs(r["soc_span"] - r["q_span_ah"] / q_total) < 1e-12)
    prom = rows[("prominence", "as_implemented")]
    out["prominence_needs_whole_roi"] = bool(prom["v_hi"] == V_ROI[0] and prom["v_lo"] == V_ROI[-1])
    det = rows[("height", "as_implemented")]
    out["detection_symmetric"] = bool(abs((det["v_hi"] - vp) - (vp - det["v_lo"])) <= DV + 1e-9)
    out["nesting"] = bool(rows[("area25", "as_implemented")]["soc_span"]
                          < rows[("area35", "as_implemented")]["soc_span"]
                          < rows[("area", "as_implemented")]["soc_span"]
                          <= prom["soc_span"])
    return out


def load_feature_rows() -> pd.DataFrame:
    """The clean feature rows with the columns the windows and the checks need."""
    if not FEATURES_PARQUET.exists():
        sys.exit(f"{FEATURES_PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")
    clean, _ = clean_feature_table(pd.read_parquet(FEATURES_PARQUET))
    return clean[["cell_id", "batch", "cycle", "SOH", "position", "height", *AREA_WINDOWS.keys()]]


def per_cycle_windows(cells: dict, feat: pd.DataFrame) -> tuple[pd.DataFrame, float, dict]:
    """
    The SoC windows of every clean cycle of every cell, plus two consistency
    checks against the stored features on ~40 cycles per cell: the located apex
    reproduces `height`, and the capacity passed in each area window matches
    the stored area feature. Returns (windows, max_height_diff, area_check).
    """
    parts, max_height_diff, area_rel = [], 0.0, {k: [] for k in AREA_WINDOWS}
    t0 = time.time()
    for i, cid in enumerate(cells):
        rows = feat[feat["cell_id"] == cid]
        if rows.empty:
            continue
        parts.append(soc_windows_for_cell(cells[cid], rows))
        for r in rows.iloc[:: max(1, len(rows) // 40)].itertuples(index=False):   # ~40 cycles per cell
            cyc = cells[cid]["cycles"][str(int(r.cycle))]
            dq, ql = np.asarray(cyc["dQdV"])[ROI_MASK], np.asarray(cyc["Qdlin"])[ROI_MASK]
            ip = int(np.argmin(np.abs(V_ROI - r.position)))
            max_height_diff = max(max_height_diff, abs(abs(dq[ip]) - r.height))
            for name, hw in AREA_WINDOWS.items():
                m = np.nonzero((V_ROI >= r.position - hw) & (V_ROI <= r.position + hw))[0]
                dq_cap = float(ql[m[-1]] - ql[m[0]])                 # capacity passed in the window
                area_rel[name].append((dq_cap - getattr(r, name)) / getattr(r, name))
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(cells)} cells, {time.time() - t0:.0f} s", flush=True)
    win = pd.concat(parts, ignore_index=True)
    area_check = {k: {"median_rel_diff": float(np.median(v)), "p95_abs_rel_diff": float(np.quantile(np.abs(v), 0.95))}
                  for k, v in area_rel.items()}
    return win, max_height_diff, area_check


def feature_set_windows(win: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle union window of each pre-declared set, summarised per life stage."""
    rows = []
    for name, feats in FEATURE_SETS.items():
        req = set_requirement(win, feats)
        for label, lo, hi in STAGES:
            g = req[(req["SOH"] >= lo) & (req["SOH"] < hi)]
            if g.empty:
                continue
            rows.append({
                "feature_set": name, "stage": label, "n_cycles": int(len(g)),
                "soc_start_median": float(g["soc_start"].median()),
                "soc_end_median": float(g["soc_end"].median()),
                "soc_span_median": float(g["soc_span"].median()),
                "soc_start_p95": float(g["soc_start"].quantile(0.95)),
                "soc_end_p05": float(g["soc_end"].quantile(0.05)),
            })
    return pd.DataFrame(rows)


def cost_benefit(impl_all: pd.DataFrame) -> pd.DataFrame | None:
    """SoC span needed per feature against its association with SoH (ENG-16, level pool)."""
    if not ENG16_CSV.exists():
        return None
    assoc = pd.read_csv(ENG16_CSV)
    assoc = assoc[assoc["pool"] == "level"][["feature", "abs_r", "abs_rho"]]
    cb = impl_all[["feature", "soc_span_median", "width_mv_median",
                   "minutes_at_4A_median"]].merge(assoc, on="feature")
    cb["abs_r_per_10pct_soc"] = cb["abs_r"] / (cb["soc_span_median"] * 10)
    return cb.sort_values("soc_span_median").reset_index(drop=True)


def pct(x: float) -> str:
    """A SoC fraction as a fixed-width percentage."""
    return f"{100 * x:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pkl-dir", type=Path, default=PKL_DIR,
                    help="folder holding batch1.pkl / batch2.pkl / batch3.pkl")
    args = ap.parse_args()

    syn = synthetic_check()
    print("synthetic cycle checks:", syn)
    assert all(syn.values()), f"synthetic check failed: {syn}"

    feat = load_feature_rows()
    if not all((args.pkl_dir / f"batch{n}.pkl").exists() for n in (1, 2, 3)):
        sys.exit(f"raw Severson pickles not found in {args.pkl_dir} (see README.md, 'Data')")
    print(f"Loading cleaned Severson dataset from {args.pkl_dir} ...", flush=True)
    t0 = time.time()
    cells = load_clean_severson(args.pkl_dir, verbose=False)
    print(f"  {len(cells)} cells loaded in {time.time() - t0:.0f} s", flush=True)

    win, max_height_diff, area_check = per_cycle_windows(cells, feat)
    OUT_PARQUET.parent.mkdir(exist_ok=True)
    win.to_parquet(OUT_PARQUET, index=False)
    print(f"\napex check: max |stored height - |dQdV| at located apex| = {max_height_diff:.2e}")
    print(f"area check (capacity from Q(V) vs stored area feature): {area_check}")
    assert max_height_diff < 1e-9, "located apex does not reproduce the stored height"
    assert all(abs(v["median_rel_diff"]) < 0.05 for v in area_check.values()), "area / capacity mismatch"

    summary = summarise_windows(win)
    summary.to_csv(OUT_CSV, index=False)
    impl_all = summary[(summary["basis"] == "as_implemented") & (summary["stage"] == "all cycles")]
    print("\nSoC window needed per feature (as implemented, all cycles; median start -> end, % SoC)")
    for r in impl_all.itertuples():
        print(f"  {r.feature:12s} {pct(r.soc_start_median)} -> {pct(r.soc_end_median)} "
              f"(span {pct(r.soc_span_median)}, {r.width_mv_median:.0f} mV)")

    sets = feature_set_windows(win)
    print("\nwindow needed by feature SETS (all cycles: median, and the 5-95 % envelope)")
    for r in sets[sets["stage"] == "all cycles"].itertuples():
        print(f"  {r.feature_set:38s} {pct(r.soc_start_median)} -> {pct(r.soc_end_median)} "
              f"(span {pct(r.soc_span_median)});  envelope {pct(r.soc_start_p95)} -> {pct(r.soc_end_p05)}")

    cb = cost_benefit(impl_all)
    if cb is not None:
        print("\ncost-benefit: SoC span needed vs association with SoH (dev set, ENG-16)")
        print(cb.round(3).to_string(index=False))

    payload = {
        "n_cells": int(win["cell_id"].nunique()),
        "n_cycles": int(win.drop_duplicates(["cell_id", "cycle"]).shape[0]),
        "soc_reference": "SoC(v) = 1 - Qdlin(v) / QD_total (total discharge capacity incl. the CV hold at 2.0 V)",
        "detect_drop_ah_per_v": 0.3,
        "pad_points": 1,
        "checks": {"synthetic": syn, "max_height_diff": max_height_diff, "area_vs_capacity": area_check},
        "per_feature": summary.to_dict(orient="records"),
        "feature_sets": sets.to_dict(orient="records"),
        "cost_benefit": None if cb is None else cb.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PARQUET.name}, {OUT_CSV.name}, {OUT_JSON.name}")

    allc = impl_all.set_index("feature")
    log_manifest("eng17_feature_soc_requirements", {
        "script": "stages/s10_soc_windows.py",
        "n_cycles": payload["n_cycles"],
        "soc_span_median": {f: float(allc.loc[f, "soc_span_median"]) for f in FEATURE_COLS},
    })


if __name__ == "__main__":
    main()
