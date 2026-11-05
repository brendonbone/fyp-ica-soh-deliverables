"""
3. Recomputed dQ/dV against the published curve; second peak (ENG-02)
=====================================================================
Does a dQ/dV curve recomputed from the raw V, Qd, I samples agree with the
published one, and is a second LFP peak (2.7-3.4 V) resolvable at 4C? Both are
answered from the status-ok rows of results/features_expanded.parquet (stage 2).
Outputs: results/eng02_second_peak.json (agreement r statistics, prominence
sweep, tail artifact); and, from one load of the raw pickles (skipped without
them or with --no-curves), the curves the illustrative figures draw:
results/dqdv_example_curves.csv (six cells at mid-life, published and
recomputed), results/ic_curve_example.npz (12 cycles of b1c20 over its life),
results/feature_geometry.npz (b1c20 cycle 1 with the nine feature landmarks).
Runtime: seconds for the JSON, minutes and ~8 GB RAM for the pickles. Not
resumable; the JSON is written last.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from src.config import PKL_DIR, RESULTS_DIR, pickles_available
from src.data_prep import log_manifest
from src.dqdv import I_DISCHARGE_MAX, recompute_dqdv
from src.features import _dominant_peak_features
from src.peak_extraction import DISTANCE, DV, PROMINENCE, ROI_MASK, V_GRID, V_ROI
from src.severson_loader import load_clean_severson

PARQUET = RESULTS_DIR / "features_expanded.parquet"  # written by stage 2
OUT_JSON = RESULTS_DIR / "eng02_second_peak.json"
CURVES_CSV = RESULTS_DIR / "dqdv_example_curves.csv"
IC_NPZ = RESULTS_DIR / "ic_curve_example.npz"
GEOMETRY_NPZ = RESULTS_DIR / "feature_geometry.npz"

# "Resolvable" uses the SAME prominence standard the pipeline applies to
# the dominant peak (0.3 Ah/V); lower thresholds are reported as a sweep.
RESOLVABLE_PROMINENCE = 0.3
SWEEP = [0.02, 0.05, 0.1, 0.2, 0.3]

EXAMPLE_CELLS = ["b1c1", "b1c30", "b2c1", "b2c30", "b3c0", "b3c20"]  # two per batch
# b1c20: its peak height and position both fall steadily over 533 cycles to
# SoH 0.82, so evenly spaced cycles fan out evenly.
IC_CELL = "b1c20"
N_CURVES = 12
GEOMETRY_CYCLE = 1  # a fresh cycle of IC_CELL


# ------------------------ the ENG-02 numbers (parquet) ------------------------
# The denominators are all status-ok cycles (not the cleaned rows), and a NaN
# prominence compares False, so a cycle without a candidate counts as "not
# resolvable".

def agreement(ok: pd.DataFrame) -> dict:
    """Per-cycle Pearson r of the recomputed against the published curve."""
    r = ok["rc_pearson_r"].dropna()
    out = {
        "n_cycles_with_recompute": int(len(r)),
        "pearson_r_median": float(r.median()),
        "pearson_r_p05": float(r.quantile(0.05)),
        "pearson_r_p01": float(r.quantile(0.01)),
        "pearson_r_min": float(r.min()),
        "frac_r_above_0.95": float((r > 0.95).mean()),
    }
    print(f"recomputed vs published Pearson r: median {out['pearson_r_median']:.4f}, "
          f"5th pct {out['pearson_r_p05']:.4f}, "
          f">0.95 in {out['frac_r_above_0.95']:.1%} of cycles")
    return out


def second_peak(ok: pd.DataFrame) -> dict:
    """How often a second peak in 2.7-3.4 V reaches each prominence threshold, and where it sits."""
    sweep = {}
    for thr in SWEEP:
        sweep[str(thr)] = {
            "published_smoothed": float((ok["prom2"] >= thr).mean()),
            "recomputed": float((ok["rc_prom2"] >= thr).mean()),
        }
        print(f"  prom2 >= {thr:>4}: published {sweep[str(thr)]['published_smoothed']:.2%}, "
              f"recomputed {sweep[str(thr)]['recomputed']:.2%}")

    resolvable = ok["prom2"] >= RESOLVABLE_PROMINENCE
    out = {
        "search_region_V": [2.7, 3.4],
        "resolvable_prominence_threshold": RESOLVABLE_PROMINENCE,
        "pct_cycles_resolvable_published": float(resolvable.mean()),
        "pct_cycles_resolvable_recomputed": float(
            (ok["rc_prom2"] >= RESOLVABLE_PROMINENCE).mean()
        ),
        "threshold_sweep": sweep,
        "n_cells_with_any_resolvable": int(ok[resolvable]["cell_id"].nunique()),
        "prom2_median": float(ok["prom2"].median()),
        "prom2_p99": float(ok["prom2"].quantile(0.99)),
        "position2_quantiles_at_0.1": {
            q: float(ok.loc[ok["prom2"] >= 0.1, "position2"].quantile(float(q)))
            for q in ("0.25", "0.5", "0.75")
        } if (ok["prom2"] >= 0.1).any() else None,
    }
    print(f"second peak resolvable (prom >= {RESOLVABLE_PROMINENCE}) in "
          f"{out['pct_cycles_resolvable_published']:.2%} of cycles (published, smoothed), "
          f"{out['pct_cycles_resolvable_recomputed']:.2%} on recomputed curves")
    return out


def tail_artifact(ok: pd.DataFrame) -> dict:
    """Candidates below 2.7 V: the discharge-knee upturn, recorded so nobody takes it for a peak."""
    out = {
        "pct_cycles_with_tail_candidate": float(ok["tail_prom2"].notna().mean()),
        "pct_tail_prom_above_0.3": float((ok["tail_prom2"] >= 0.3).mean()),
        "tail_position_median": float(ok["tail_position2"].median())
        if ok["tail_position2"].notna().any() else None,
    }
    print(f"tail artifact (<2.7 V): candidate in {out['pct_cycles_with_tail_candidate']:.1%} "
          f"of cycles, prom >= 0.3 in {out['pct_tail_prom_above_0.3']:.1%}")
    return out


def second_peak_numbers(feat: pd.DataFrame) -> dict:
    """The eng02_second_peak.json payload from the feature table."""
    ok = feat[feat["status"] == "ok"]
    print(f"{len(ok)} ok cycle-rows across {ok['cell_id'].nunique()} cells")
    return {
        "agreement_recomputed_vs_published": agreement(ok),
        "second_peak": second_peak(ok),
        "tail_artifact_below_2.7V": tail_artifact(ok),
    }


# ------------------------ the example curves (raw pickles) ------------------------

def example_curves(cells: dict) -> pd.DataFrame:
    """Published and recomputed dQ/dV of the mid-life cycle of each EXAMPLE_CELLS cell."""
    parts = []
    for cid in EXAMPLE_CELLS:
        k = len(cells[cid]["cycles"]) // 2
        cyc = cells[cid]["cycles"][str(k)]
        parts.append(pd.DataFrame({"cell_id": cid, "cycle": k, "voltage": V_GRID,
                                   "published": np.asarray(cyc["dQdV"], dtype=float),
                                   "recomputed": recompute_dqdv(cyc)}))
    return pd.concat(parts, ignore_index=True)


def ic_curve_example(cell: dict, cell_id: str, n_curves: int) -> dict:
    """V and Qd of the CC discharge leg and the recomputed dQ/dV of `n_curves` cycles spread over the life."""
    qd_summary = np.asarray(cell["summary"]["QD"], dtype=float)
    positive = qd_summary[qd_summary > 0]
    qd_ref = float(positive[0])          # same SoH reference as the pipeline
    soh_curve = qd_summary / qd_ref

    keys = sorted(cell["cycles"], key=int)
    n_total = len(keys)
    # Cycle 0 of most batch-1 cells is a 2-element zero stub, so start at 1.
    picks = np.unique(np.round(np.linspace(1, n_total - 1, n_curves)).astype(int))

    out: dict[str, np.ndarray] = {}
    kept: list[int] = []
    for j in picks:
        cyc = cell["cycles"][keys[j]]
        on_cc = np.asarray(cyc["I"], dtype=float) < I_DISCHARGE_MAX
        if on_cc.sum() < 50:
            continue
        i = len(kept)
        # The whole CC leg, including the flat foot at the 2.0 V cut-off: that
        # foot is real delivered capacity (83 mAh by end of life).
        out[f"v{i}"] = np.asarray(cyc["V"], dtype=float)[on_cc].astype(np.float32)
        out[f"q{i}"] = np.asarray(cyc["Qd"], dtype=float)[on_cc].astype(np.float32)
        out[f"dqdv{i}"] = recompute_dqdv(cyc, v_grid=V_GRID).astype(np.float32)
        kept.append(j)

    out["cycles"] = np.asarray(kept)
    out["soh"] = soh_curve[kept]
    out["soh_curve"] = soh_curve
    # The summary QD (the capacity the pipeline's SoH uses), not max(Qd) of the trace.
    out["qd"] = qd_summary[kept]
    out["qd_ref"] = np.asarray([qd_ref])
    out["n_total"] = np.asarray([n_total])
    out["cell_id"] = np.asarray([cell_id])
    print(f"  {cell_id}: cycles {[int(j) for j in kept]}, SoH {out['soh'][0]:.3f} -> {out['soh'][-1]:.3f}")
    return out


def _roi_index_to_volt(ix: float) -> float:
    """A fractional index into V_ROI (as peak_widths returns it) in volts."""
    lo, hi = int(np.floor(ix)), int(np.ceil(ix))
    if lo == hi:
        return float(V_ROI[lo])
    frac = ix - lo
    return float(V_ROI[lo] + frac * (V_ROI[hi] - V_ROI[lo]))


def feature_geometry(cell: dict, cell_id: str, cycle: int) -> dict:
    """The nine features of one published curve and the landmarks the extractor measured them from."""
    qd = np.asarray(cell["summary"]["QD"], dtype=float)
    soh = float(qd[cycle] / qd[qd > 0][0])

    # The extractor works on the PUBLISHED per-cycle dQ/dV, cropped to the ROI.
    # prev_peak_V=None is the most-prominent-peak rule (the extractor's rule on
    # a cell's first cycle; later cycles follow the previous peak position).
    dqdv_roi = np.asarray(cell["cycles"][str(cycle)]["dQdV"], dtype=float)[ROI_MASK]
    feats = _dominant_peak_features(dqdv_roi, None)

    # Re-derive the landmarks of the same find_peaks / peak_widths calls, so a
    # drawing shows the real geometry: the prominence base and the
    # half-prominence width span.
    neg = -dqdv_roi
    peaks, props = find_peaks(neg, prominence=PROMINENCE, distance=DISTANCE)
    sel = int(np.argmax(props["prominences"]))
    best = int(peaks[sel])
    _, w_height, left_ips, right_ips = peak_widths(neg, [best], rel_height=0.5)
    fwhm_v_left = _roi_index_to_volt(float(left_ips[0]))
    fwhm_v_right = _roi_index_to_volt(float(right_ips[0]))
    print(f"  {cell_id} cycle {cycle}: peak {feats['position']:.4f} V, height "
          f"{feats['height']:.4f} Ah/V, fwhm {feats['fwhm']:.4f} V, SoH {soh:.4f}")
    return {
        "v_roi": V_ROI.astype(np.float32),
        "abs_dqdv": np.abs(dqdv_roi).astype(np.float32),
        "peak_v": np.array([feats["position"]]),
        "peak_h": np.array([feats["height"]]),
        "prominence": np.array([feats["prominence"]]),
        "prom_base": np.array([float(neg[best] - props["prominences"][sel])]),
        "fwhm": np.array([feats["fwhm"]]),
        "fwhm_v": np.array([fwhm_v_left, fwhm_v_right]),
        "fwhm_y": np.array([float(w_height[0])]),   # on -dQ/dV, so it is a |dQ/dV| level
        "area": np.array([feats["area"]]),
        "area35": np.array([feats["area35"]]),
        "area25": np.array([feats["area25"]]),
        "slope_left": np.array([feats["slope_left"]]),
        "slope_right": np.array([feats["slope_right"]]),
        "soh": np.array([soh]),
        "cycle": np.array([cycle]),
        "cell_id": np.array([cell_id]),
        "dv": np.array([DV]),
    }


def write_curves() -> None:
    """One load of the raw pickles -> the three curve files."""
    print("Loading the cleaned Severson dataset (~8 GB RAM) ...", flush=True)
    cells = load_clean_severson(PKL_DIR, verbose=False)
    example_curves(cells).to_csv(CURVES_CSV, index=False, lineterminator="\n")
    print(f"Wrote {CURVES_CSV}")
    # b1c20 is neither dropped nor merged by the cleaning, so this is the same
    # record as in batch1.pkl alone.
    np.savez_compressed(IC_NPZ, **ic_curve_example(cells[IC_CELL], IC_CELL, N_CURVES))
    print(f"Wrote {IC_NPZ}")
    np.savez_compressed(GEOMETRY_NPZ, **feature_geometry(cells[IC_CELL], IC_CELL, GEOMETRY_CYCLE))
    print(f"Wrote {GEOMETRY_NPZ}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-curves", action="store_true",
                    help="only the ENG-02 numbers from the parquet; do not load the pickles")
    args = ap.parse_args()

    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (s02_extract_features) first")
    payload = second_peak_numbers(pd.read_parquet(PARQUET))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.no_curves:
        print("[skip] example curves (--no-curves)")
    elif not pickles_available():
        print(f"[skip] example curves: the raw pickles are not in {PKL_DIR}")
    else:
        write_curves()

    # Written last: run_results.py skips this stage once the JSON exists, so an
    # interrupted pickle load must not leave it behind.
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    log_manifest("eng02_second_peak", {
        "script": "stages/s03_dqdv_check.py",
        "pct_resolvable_at_0.3": payload["second_peak"]["pct_cycles_resolvable_published"],
        "recompute_agreement_median_r": payload["agreement_recomputed_vs_published"]["pearson_r_median"],
    })


if __name__ == "__main__":
    main()
