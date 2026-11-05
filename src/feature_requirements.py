"""
Data requirement of each ICA feature as a state-of-charge window (ENG-17).
==========================================================================
Every feature in `features.FEATURE_COLS` is read off the dominant peak of
one cycle's DISCHARGE dQ/dV curve, but they do not need the same amount of
that curve: `area25` integrates +/-25 mV around the peak, `area` +/-50 mV,
and `prominence` looks all the way out to the lowest point on either side.
This module turns "which part of the curve does feature f touch?" into a
state-of-charge (SoC) window, so the features can be compared by how much of
a discharge a battery-management system would have to observe.

Voltage -> SoC
--------------
The Severson cycles store `Qdlin`, the discharge capacity as a function of
voltage on the same 1000-point grid as `dQdV`. For a voltage v,

    SoC(v) = 1 - Qdlin(v) / Q_total

with Q_total the cycle's total discharge capacity (the summary `QD`, the
quantity SoH is built from). That total includes the constant-voltage hold
at 2.0 V that ends each discharge, which never appears on the voltage grid,
so SoC at 2.0 V is a few percent above zero for a fresh cell and ~10 % for
an aged one. `soc_ref="cc"` references the end of the voltage curve instead.
Discharge runs from high voltage / high SoC to low voltage / low SoC, so a
window is reported as soc_start (higher) -> soc_end (lower).

What each feature needs (exactly as `features` computes it)
--------------------------------------------------------------
Nothing can be computed before the peak is found, so every window includes
the DETECTION window: from the apex outwards on each side until |dQ/dV| has
fallen by `detect_drop` (default = the extractor's prominence threshold,
0.3 Ah/V) -- the smallest stretch of curve in which the apex is recognisable
as a peak by the pipeline's own criterion.

    height, position   detection window
    area / area35 /    detection  U  [Vp - hw, Vp + hw], hw = 50 / 35 / 25 mV
      area25
    slope_left         detection  U  [Vp - 25 mV, Vp]   (lower-voltage flank)
    slope_right        detection  U  [Vp, Vp + 25 mV]   (higher-voltage flank)
    prominence         scipy's search interval: outwards from the apex on
                       each side until the curve exceeds the apex or the ROI
                       ends. For the dominant peak that is the whole ROI.
    fwhm               the same interval, because scipy's half-PROMINENCE
                       width needs the prominence first.

For `prominence` and `fwhm` a second, smaller window is reported with
basis == "value_support": the span that fixes the VALUE once you know where
to look (base to base; half-prominence crossing to crossing). The rows with
basis == "as_implemented" are what the current extractor actually requires.

Every window is widened by `pad_points` grid points (default 1) per side for
the finite difference behind dQ/dV; any smoothing used to build dQ/dV would
widen all windows further by half its kernel.

Public names
------------
    cycle_windows(dqdv_roi, qdlin_roi, position, q_total, ...) -> list[dict]
    soc_windows_for_cell(cell, feat_rows, ...)                 -> DataFrame
    summarise_windows(win, stages=LIFE_STAGES)                 -> DataFrame
    set_requirement(win, features)                             -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import peak_prominences, peak_widths

from src.features import AREA_WINDOWS, FEATURE_COLS
from src.peak_extraction import PROMINENCE, ROI_MASK, V_GRID_POINTS, V_ROI

FLANK_V: float = 0.025          # slope_left / slope_right span (features)
DISCHARGE_CURRENT_A: float = 4.0

# (label, SoH lower bound inclusive, SoH upper bound exclusive)
LIFE_STAGES: list[tuple[str, float, float]] = [
    ("fresh (SoH >= 0.99)", 0.99, np.inf),
    ("mid (0.90-0.99)", 0.90, 0.99),
    ("aged (SoH < 0.90)", -np.inf, 0.90),
]


# ------------------------ index windows on the ROI grid ------------------------

def _detection_window(neg: np.ndarray, ip: int, drop: float) -> tuple[int, int]:
    """Smallest [i0, i1] around the apex in which it has prominence >= drop."""
    thr = neg[ip] - drop
    below_hi_v = np.nonzero(neg[:ip] <= thr)[0]          # higher-voltage side
    below_lo_v = np.nonzero(neg[ip + 1:] <= thr)[0]      # lower-voltage side
    i0 = int(below_hi_v[-1]) if below_hi_v.size else 0
    i1 = int(ip + 1 + below_lo_v[0]) if below_lo_v.size else len(neg) - 1
    return i0, i1


def _prominence_scan(neg: np.ndarray, ip: int) -> tuple[int, int]:
    """scipy's search interval: outwards until a sample exceeds the apex."""
    higher_hi_v = np.nonzero(neg[:ip] > neg[ip])[0]
    higher_lo_v = np.nonzero(neg[ip + 1:] > neg[ip])[0]
    i0 = int(higher_hi_v[-1]) if higher_hi_v.size else 0
    i1 = int(ip + 1 + higher_lo_v[0]) if higher_lo_v.size else len(neg) - 1
    return i0, i1


def _voltage_window(v_hi: float, v_lo: float) -> tuple[int, int] | None:
    """Grid indices covering [v_lo, v_hi], bracketing both ends (V_ROI descends)."""
    if v_hi > V_ROI[0] + 1e-12 or v_lo < V_ROI[-1] - 1e-12:
        return None                                   # outside the ROI -> NaN feature
    i0 = int(np.nonzero(V_ROI >= v_hi - 1e-12)[0][-1])    # grid point at/above v_hi
    i1 = int(np.nonzero(V_ROI <= v_lo + 1e-12)[0][0])     # grid point at/below v_lo
    return i0, i1


def _union(a: tuple[int, int], b: tuple[int, int] | None) -> tuple[int, int] | None:
    return None if b is None else (min(a[0], b[0]), max(a[1], b[1]))


# ------------------------ one cycle ------------------------

def cycle_windows(
    dqdv_roi: np.ndarray,
    qdlin_roi: np.ndarray,
    position: float,
    q_total: float,
    *,
    detect_drop: float = PROMINENCE,
    pad_points: int = 1,
    soc_ref: str = "total",
) -> list[dict]:
    """
    SoC window of every feature for ONE cycle.

    dqdv_roi, qdlin_roi : the cycle's dQdV and Qdlin on the ROI grid (V_ROI).
    position            : the dominant-peak voltage the extractor chose.
    q_total             : the cycle's total discharge capacity [Ah].
    Returns one dict per (feature, basis); windows that fall outside the ROI
    (the feature itself is NaN there) are returned with NaN bounds.
    """
    neg = -np.asarray(dqdv_roi, dtype=float)
    neg = np.where(np.isfinite(neg), neg, -np.inf)
    qd = np.asarray(qdlin_roi, dtype=float)
    n = len(neg)
    ip = int(np.argmin(np.abs(V_ROI - position)))

    q_ref = float(q_total) if soc_ref == "total" else float(qd[-1])
    if not np.isfinite(q_ref) or q_ref <= 0:
        q_ref = float(qd[-1])
    soc = np.clip(1.0 - qd / q_ref, 0.0, 1.0)

    detect = _detection_window(neg, ip, detect_drop)
    scan = _prominence_scan(neg, ip)
    _, lb, rb = peak_prominences(neg, [ip])
    _, _, lip, rip = peak_widths(neg, [ip], rel_height=0.5)
    bases = (int(lb[0]), int(rb[0]))
    crossings = (int(np.floor(lip[0])), int(np.ceil(rip[0])))

    windows: list[tuple[str, str, tuple[int, int] | None]] = [
        ("height", "as_implemented", detect),
        ("position", "as_implemented", detect),
    ]
    for name, hw in AREA_WINDOWS.items():
        windows.append((name, "as_implemented",
                        _union(detect, _voltage_window(position + hw, position - hw))))
    windows += [
        ("slope_left", "as_implemented",
         _union(detect, _voltage_window(position, position - FLANK_V))),
        ("slope_right", "as_implemented",
         _union(detect, _voltage_window(position + FLANK_V, position))),
        ("prominence", "as_implemented", scan),
        ("fwhm", "as_implemented", scan),
        ("prominence", "value_support", _union(detect, bases)),
        ("fwhm", "value_support", _union(detect, crossings)),
    ]

    rows = []
    for feature, basis, w in windows:
        if w is None:
            rows.append({"feature": feature, "basis": basis, "v_hi": np.nan, "v_lo": np.nan,
                         "width_mv": np.nan, "soc_start": np.nan, "soc_end": np.nan,
                         "soc_span": np.nan, "q_span_ah": np.nan})
            continue
        i0, i1 = max(0, w[0] - pad_points), min(n - 1, w[1] + pad_points)
        rows.append({
            "feature": feature, "basis": basis,
            "v_hi": float(V_ROI[i0]), "v_lo": float(V_ROI[i1]),
            "width_mv": float((V_ROI[i0] - V_ROI[i1]) * 1000.0),
            "soc_start": float(soc[i0]), "soc_end": float(soc[i1]),
            "soc_span": float(soc[i0] - soc[i1]),
            "q_span_ah": float(qd[i1] - qd[i0]),
        })
    return rows


# ------------------------ one cell / whole dataset ------------------------

def soc_windows_for_cell(
    cell: dict,
    feat_rows: pd.DataFrame,
    *,
    detect_drop: float = PROMINENCE,
    pad_points: int = 1,
    soc_ref: str = "total",
) -> pd.DataFrame:
    """
    Windows for every row of `feat_rows` (one cell's rows of the feature table:
    needs `cycle`, `position`; `cell_id`, `batch`, `SOH` are carried through).
    """
    qd_summary = np.asarray(cell["summary"]["QD"], dtype=float)
    cyc_summary = np.asarray(cell["summary"]["cycle"]).astype(int)
    q_by_cycle = dict(zip(cyc_summary.tolist(), qd_summary.tolist()))
    carry = [c for c in ("cell_id", "batch", "SOH") if c in feat_rows.columns]

    out = []
    for r in feat_rows.itertuples(index=False):
        cyc = cell["cycles"].get(str(int(r.cycle)))
        if cyc is None:
            continue
        dq, ql = np.asarray(cyc["dQdV"]), np.asarray(cyc["Qdlin"])
        if dq.size != V_GRID_POINTS or ql.size != V_GRID_POINTS or not np.isfinite(r.position):
            continue
        rows = cycle_windows(dq[ROI_MASK], ql[ROI_MASK], float(r.position),
                             q_by_cycle.get(int(r.cycle), np.nan),
                             detect_drop=detect_drop, pad_points=pad_points, soc_ref=soc_ref)
        meta = {"cycle": int(r.cycle), **{c: getattr(r, c) for c in carry}}
        out.extend({**meta, **row} for row in rows)
    df = pd.DataFrame(out)
    if not df.empty:
        df["minutes_at_4A"] = df["q_span_ah"] / DISCHARGE_CURRENT_A * 60.0
    return df


def summarise_windows(
    win: pd.DataFrame,
    stages: list[tuple[str, float, float]] = LIFE_STAGES,
) -> pd.DataFrame:
    """Median window per feature, basis and life stage, plus a 5-95 % envelope."""
    rows = []
    groups = [("all cycles", win)] + [
        (label, win[(win["SOH"] >= lo) & (win["SOH"] < hi)]) for label, lo, hi in stages]
    order = {f: i for i, f in enumerate(FEATURE_COLS)}
    for label, g in groups:
        for (feature, basis), h in g.groupby(["feature", "basis"], sort=False):
            h = h.dropna(subset=["soc_span"])
            if h.empty:
                continue
            rows.append({
                "stage": label, "feature": feature, "basis": basis, "n_cycles": int(len(h)),
                "soc_start_median": h["soc_start"].median(),
                "soc_end_median": h["soc_end"].median(),
                "soc_span_median": h["soc_span"].median(),
                # window that covers the feature in 90 % of cycles of this stage
                "soc_start_p95": h["soc_start"].quantile(0.95),
                "soc_end_p05": h["soc_end"].quantile(0.05),
                "width_mv_median": h["width_mv"].median(),
                "q_span_ah_median": h["q_span_ah"].median(),
                "minutes_at_4A_median": h["minutes_at_4A"].median(),
            })
    out = pd.DataFrame(rows)
    out["_o"] = out["feature"].map(order)
    return (out.sort_values(["stage", "basis", "_o"], kind="mergesort")
            .drop(columns="_o").reset_index(drop=True))


def set_requirement(win: pd.DataFrame, features: list[str],
                    basis: str = "as_implemented") -> pd.DataFrame:
    """Per-cycle window needed to compute ALL of `features` (union of windows)."""
    sub = win[(win["basis"] == basis) & win["feature"].isin(features)].dropna(subset=["soc_span"])
    g = sub.groupby(["cell_id", "cycle"], sort=False)
    out = g.agg(SOH=("SOH", "first"), soc_start=("soc_start", "max"),
                soc_end=("soc_end", "min"), n_features=("feature", "nunique")).reset_index()
    out = out[out["n_features"] == len(set(features))]
    out["soc_span"] = out["soc_start"] - out["soc_end"]
    return out
