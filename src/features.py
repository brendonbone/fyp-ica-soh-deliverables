"""
The nine-feature ICA pool: extraction of every feature of the dominant peak.
===========================================================================
Extends the three-feature interim extractor (src/peak_extraction.py) with:

Dominant-peak enrichment (ENG-04) — computed on the PUBLISHED dQ/dV curve
with the IDENTICAL detection + cross-cycle tracking logic as
src/peak_extraction.py, so `height`, `position`, `area` reproduce the
baseline bit-for-bit:
    height      |dQ/dV| at the dominant peak                    [Ah/V]
    position    peak voltage                                    [V]
    area        integral of |dQ/dV| over +/-50 mV               [Ah]
    area35      integral over +/-35 mV                          [Ah]
    area25      integral over +/-25 mV                          [Ah]
    fwhm        peak width at half prominence (scipy
                peak_widths, rel_height=0.5), in volts          [V]
    slope_left  (H - |dQ/dV|(Vp - 25 mV)) / 25 mV  (lower-V flank) [Ah/V^2]
    slope_right (H - |dQ/dV|(Vp + 25 mV)) / 25 mV  (higher-V flank)[Ah/V^2]
    prominence  find_peaks prominence of the dominant peak      [Ah/V]

Second-peak search (ENG-02) — run on a Savitzky-Golay-smoothed copy of the
curve (window 51, polyorder 3) so candidate peaks are not noise spikes.
All candidates from find_peaks(prominence >= 0.02, distance=20); the one
nearest the dominant position IS the dominant; the most prominent OTHER
candidate is reported (NaN columns when none exists — that is data, not an
extraction failure):
    height2, position2, prom2, peak_sep (= position2 - position), n_peaks2

Recomputed-curve check (ENG-02) — the same second-peak search on dQ/dV
rebuilt from raw V/Qd/I by src.dqdv.recompute_dqdv, plus agreement stats
with the published curve:
    rc_prom2, rc_position2, rc_n_peaks2, rc_pearson_r, rc_rms_diff

Rows keep the baseline status semantics ("ok" / "no_peak" / "bad_dqdv");
all feature columns are NaN on failed rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

from src.dqdv import compare_with_published, recompute_dqdv
from src.peak_extraction import (
    DISTANCE,
    DV,
    PROMINENCE,
    ROI_MASK,
    V_GRID_POINTS,
    V_ROI,
)

# Integration windows for the multi-scale peak areas (volts, half-width).
AREA_WINDOWS: dict[str, float] = {"area": 0.050, "area35": 0.035, "area25": 0.025}

# Second-peak search parameters. The floor collects every plausible
# candidate; analysis sweeps the "resolvable" prominence threshold later.
SECOND_PROMINENCE_FLOOR: float = 0.02
SMOOTH_WINDOW: int = 51
SMOOTH_POLYORDER: int = 3

# Voltage region where a genuine second LFP/graphite staging feature can
# live. Below ~2.7 V the discharge-knee tail upturn produces spurious
# local maxima (empirically, >99% of unrestricted candidates sat at
# 2.0-2.5 V); those are recorded separately as tail_* columns.
SECOND_V_MIN: float = 2.7
SECOND_V_MAX: float = 3.4

FEATURE_COLS: list[str] = [
    "height", "position", "area", "area35", "area25",
    "fwhm", "slope_left", "slope_right", "prominence",
]
SECOND_PEAK_COLS: list[str] = [
    "height2", "position2", "prom2", "peak_sep", "n_peaks2",
    "tail_prom2", "tail_position2",
]
RECOMPUTE_COLS: list[str] = [
    "rc_prom2", "rc_position2", "rc_n_peaks2", "rc_pearson_r", "rc_rms_diff",
]

def _interp_abs_dqdv(abs_dqdv: np.ndarray, at_v: float,
                     v_asc: np.ndarray) -> float:
    """|dQ/dV| at an arbitrary voltage inside the ROI (NaN outside).

    `v_asc` is the ROI voltage axis in ASCENDING order; `abs_dqdv` is in
    the (descending) ROI order and gets reversed to match.
    """
    if at_v < v_asc[0] or at_v > v_asc[-1]:
        return float("nan")
    return float(np.interp(at_v, v_asc, abs_dqdv[::-1]))


def _dominant_peak_features(
    dqdv_roi: np.ndarray,
    prev_peak_V: float | None,
    *,
    v_roi: np.ndarray = V_ROI,
    dv: float = DV,
    prominence: float = PROMINENCE,
    distance: int = DISTANCE,
) -> dict | None:
    """Detect + track the dominant peak; return the enriched feature dict.

    Returns None when no peak passes the baseline prominence criterion
    (-> "no_peak"). With default arguments detection is identical to the
    baseline extractor; the keyword grid/threshold overrides exist for the
    NASA cells (ENG-11), whose voltage grid and amplitudes differ.
    """
    neg = -dqdv_roi
    peaks, props = find_peaks(neg, prominence=prominence, distance=distance)
    if len(peaks) == 0:
        return None

    if prev_peak_V is None:
        sel = int(np.argmax(props["prominences"]))
    else:
        sel = int(np.argmin(np.abs(v_roi[peaks] - prev_peak_V)))
    best_idx = int(peaks[sel])

    position = float(v_roi[best_idx])
    abs_dqdv = np.abs(dqdv_roi)
    height = float(abs_dqdv[best_idx])

    out: dict = {"height": height, "position": position,
                 "prominence": float(props["prominences"][sel])}

    # Multi-scale areas (same trapezoid-with-dx convention as the baseline).
    for name, hw in AREA_WINDOWS.items():
        window = (v_roi >= position - hw) & (v_roi <= position + hw)
        out[name] = float(np.trapezoid(abs_dqdv[window], dx=dv))

    # FWHM at half prominence (scipy convention; robust to baseline offset).
    widths, _, _, _ = peak_widths(neg, [best_idx], rel_height=0.5)
    out["fwhm"] = float(widths[0] * dv)

    # Flank slopes over 25 mV each side (left = lower voltage).
    v_asc = v_roi[::-1]
    y_left = _interp_abs_dqdv(abs_dqdv, position - 0.025, v_asc)
    y_right = _interp_abs_dqdv(abs_dqdv, position + 0.025, v_asc)
    out["slope_left"] = (height - y_left) / 0.025 if np.isfinite(y_left) else float("nan")
    out["slope_right"] = (height - y_right) / 0.025 if np.isfinite(y_right) else float("nan")
    return out


_NAN_SECOND = {
    "height2": np.nan, "position2": np.nan, "prom2": np.nan,
    "peak_sep": np.nan, "n_peaks2": 0,
    "tail_prom2": np.nan, "tail_position2": np.nan,
}


def _second_peak_search(
    dqdv_roi: np.ndarray, dominant_position: float
) -> dict:
    """Find the most prominent non-dominant peak on a smoothed curve.

    Candidates inside [SECOND_V_MIN, SECOND_V_MAX] feed the height2/...
    columns; the best candidate BELOW the region is recorded as tail_*
    (discharge-knee artifact documentation). n_peaks2 counts in-region
    candidates only (dominant excluded).
    """
    neg = -dqdv_roi
    finite = np.isfinite(neg)
    if finite.sum() > SMOOTH_WINDOW:
        smoothed = neg.copy()
        smoothed[finite] = savgol_filter(
            neg[finite], SMOOTH_WINDOW, SMOOTH_POLYORDER
        )
    else:
        smoothed = neg
    # find_peaks cannot handle NaN; mask them to -inf so they never win.
    work = np.where(np.isfinite(smoothed), smoothed, -np.inf)
    peaks, props = find_peaks(
        work, prominence=SECOND_PROMINENCE_FLOOR, distance=DISTANCE
    )
    if len(peaks) == 0:
        return dict(_NAN_SECOND)

    out = dict(_NAN_SECOND)
    positions = V_ROI[peaks]
    proms = props["prominences"]

    # The candidate nearest the dominant position IS the dominant peak.
    dom = int(np.argmin(np.abs(positions - dominant_position)))
    others = [i for i in range(len(peaks)) if i != dom]

    in_region = [i for i in others
                 if SECOND_V_MIN <= positions[i] <= SECOND_V_MAX]
    below = [i for i in others if positions[i] < SECOND_V_MIN]

    out["n_peaks2"] = len(in_region)
    if in_region:
        second = max(in_region, key=lambda i: proms[i])
        pos2 = float(positions[second])
        out.update({
            "height2": float(smoothed[peaks[second]]),
            "position2": pos2,
            "prom2": float(proms[second]),
            "peak_sep": pos2 - dominant_position,
        })
    if below:
        tail = max(below, key=lambda i: proms[i])
        out.update({
            "tail_prom2": float(proms[tail]),
            "tail_position2": float(positions[tail]),
        })
    return out


def extract_all_features(
    cell: dict,
    cell_id: str | None = None,
    *,
    include_recompute: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Per-cycle extraction of the nine-feature pool (plus the second-peak and
    recomputed-curve columns) for one cell.

    Returns a DataFrame with columns:
        cycle, FEATURE_COLS, SECOND_PEAK_COLS, RECOMPUTE_COLS (optional),
        SOH, status
    """
    qd = np.asarray(cell["summary"]["QD"])
    qd_positive = qd[qd > 0]
    qd_ref = float(qd_positive[0]) if qd_positive.size > 0 else float("nan")
    summary_cycles = np.asarray(cell["summary"]["cycle"]).astype(int)

    nan_features = {c: np.nan for c in FEATURE_COLS}
    nan_second = {c: np.nan for c in SECOND_PEAK_COLS}
    nan_recompute = {c: np.nan for c in RECOMPUTE_COLS}

    records: list[dict] = []
    prev_peak_V: float | None = None

    for cycle_str in sorted(cell["cycles"].keys(), key=lambda x: int(x)):
        k = int(cycle_str)
        cyc = cell["cycles"][cycle_str]
        raw_dqdv = np.asarray(cyc["dQdV"])

        mask = summary_cycles == k
        soh = float(qd[mask][0] / qd_ref) if (mask.any() and qd_ref > 0) else float("nan")

        base = {"cycle": k, "SOH": soh}

        if raw_dqdv.size != V_GRID_POINTS:
            records.append({**base, **nan_features, **nan_second,
                            **nan_recompute, "status": "bad_dqdv"})
            continue

        dqdv_roi = raw_dqdv[ROI_MASK]
        dom = _dominant_peak_features(dqdv_roi, prev_peak_V)
        if dom is None:
            records.append({**base, **nan_features, **nan_second,
                            **nan_recompute, "status": "no_peak"})
            continue
        prev_peak_V = dom["position"]

        second = _second_peak_search(dqdv_roi, dom["position"])

        rc = dict(nan_recompute)
        if include_recompute:
            rc_curve = recompute_dqdv(cyc)
            if np.isfinite(rc_curve).sum() > 100:
                agree = compare_with_published(rc_curve, raw_dqdv)
                rc_second = _second_peak_search(
                    rc_curve[ROI_MASK], dom["position"]
                )
                rc = {
                    "rc_prom2": rc_second["prom2"],
                    "rc_position2": rc_second["position2"],
                    "rc_n_peaks2": rc_second["n_peaks2"],
                    "rc_pearson_r": agree["pearson_r"],
                    "rc_rms_diff": agree["rms_diff"],
                }

        records.append({**base, **dom, **second, **rc, "status": "ok"})

    cols = (["cycle"] + FEATURE_COLS + SECOND_PEAK_COLS
            + RECOMPUTE_COLS + ["SOH", "status"])
    df = pd.DataFrame.from_records(records, columns=cols)

    if verbose:
        n_ok = int((df["status"] == "ok").sum())
        tag = f"[{cell_id}] " if cell_id else ""
        print(f"  {tag}cycles={len(df):4d} ok={n_ok:4d} "
              f"fail_rate={(len(df) - n_ok) / max(len(df), 1):.2%}")
    return df
