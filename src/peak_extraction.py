"""
ICA peak feature extraction for SOH modelling.
==============================================
The interim model's extractor: three features of the dominant discharge
dQ/dV peak. The peak-detection logic was prototyped in
notebooks/02_peak_extraction_walkthrough.ipynb and is wrapped here as a
reusable function:

    from src.peak_extraction import extract_features

    df = extract_features(cells["b1c1"], cell_id="b1c1", verbose=True)
    # DataFrame[cycle, height, position, area, SOH, status]

Why this module exists:
- The notebook validated the approach on a single cell (b1c1, 2158 cycles);
  the full 124-cell run needs a single, tested function rather than
  copy-pasted notebook cells.
- Failed cycles produce NaN rows tagged with a `status` column so the
  caller can compute a failure rate and decide whether parameters need
  retuning (rule of thumb: >10% failure rate => retune).
- src/features.py extends this extractor to the nine-feature pool with the
  identical peak detection, so `height`, `position` and `area` are the same
  numbers in both tables.

Notes on the algorithm:
- The Severson dataset stores per-cycle dQ/dV on a 1000-point voltage grid
  from 3.5 V down to 2.0 V (descending). We crop the top 100 mV to avoid
  the CV-hold singularity, then run `find_peaks` on `-dQ/dV` (the LFP IC
  feature appears as a *minimum* in dQ/dV under this sign convention).
- Cross-cycle peak tracking uses a "previous cycle's position" prior so
  peak-1 doesn't accidentally become peak-2 as the curve ages.
- Single-peak extraction only; the second-peak search lives in src/features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ------------------------ configuration ------------------------

# The Severson per-cycle dQ/dV arrays live on a fixed 1000-point voltage grid
# from 3.5 V to 2.0 V (descending — these are charge sweeps stored in
# decreasing-voltage order in the raw .mat files).
V_GRID_MAX: float = 3.5
V_GRID_MIN: float = 2.0
V_GRID_POINTS: int = 1000
V_GRID: np.ndarray = np.linspace(V_GRID_MAX, V_GRID_MIN, V_GRID_POINTS)

# Region of interest: drop the top 100 mV to avoid the CV-hold singularity.
# Lower bound is unrestricted — peaks of interest live well above 2.0 V.
V_ROI_UPPER: float = V_GRID_MAX - 0.1
V_ROI_LOWER: float = V_GRID_MIN
ROI_MASK: np.ndarray = (V_GRID >= V_ROI_LOWER) & (V_GRID <= V_ROI_UPPER)
V_ROI: np.ndarray = V_GRID[ROI_MASK]

# Uniform voltage spacing on the grid (used as `dx` for trapezoidal area).
# Taking abs() makes the answer agnostic to the descending grid direction.
DV: float = float(abs(V_GRID[1] - V_GRID[0]))

# Peak detection parameters — tuned against b1c1 in the notebook.
# Loosen `PROMINENCE` toward 0.2 if late-life cycles drop out at >10%.
PROMINENCE: float = 0.3
DISTANCE: int = 20

# Half-width in volts for the area-under-peak integration window (±50 mV).
AREA_HALF_WIDTH: float = 0.05


# ------------------------ per-cycle extraction ------------------------

def _extract_one_cycle(
    dqdv_full: np.ndarray, prev_peak_V: float | None
) -> tuple[float, float, float]:
    """
    Extract peak height / position / area from one cycle's dQ/dV curve.

    Parameters
    ----------
    dqdv_full : np.ndarray, shape (V_GRID_POINTS,)
        The raw dQ/dV on the full voltage grid.
    prev_peak_V : float or None
        Peak voltage from the previous cycle, used as a prior to keep
        tracking the same peak across cycles. Pass None for the first cycle.

    Returns
    -------
    (height, position, area) : tuple of floats
        All three are NaN if no peak meets the prominence/distance criteria.
    """
    dqdv = dqdv_full[ROI_MASK]
    peaks, props = find_peaks(-dqdv, prominence=PROMINENCE, distance=DISTANCE)

    if len(peaks) == 0:
        return float("nan"), float("nan"), float("nan")

    # First cycle: take the most prominent peak.
    # Subsequent cycles: take whichever detected peak is closest to the
    # previous cycle's position. This prevents peak-1 silently becoming
    # peak-2 as the IC curve ages.
    if prev_peak_V is None:
        best_idx = peaks[int(np.argmax(props["prominences"]))]
    else:
        best_idx = peaks[int(np.argmin(np.abs(V_ROI[peaks] - prev_peak_V)))]

    position = float(V_ROI[best_idx])
    height = float(abs(dqdv[best_idx]))

    # Area under |dQ/dV| in a ±AREA_HALF_WIDTH window around the peak.
    # We pass `dx=DV` rather than the V-array so the result is independent
    # of grid direction (V_ROI is descending, which would otherwise give a
    # negative integral).
    window = (V_ROI >= position - AREA_HALF_WIDTH) & (
        V_ROI <= position + AREA_HALF_WIDTH
    )
    area = float(np.trapezoid(np.abs(dqdv[window]), dx=DV))

    return height, position, area


# ------------------------ full-cell extraction ------------------------

def extract_features(
    cell: dict,
    cell_id: str | None = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run peak extraction across every cycle of a cell.

    Parameters
    ----------
    cell : dict
        A cell dict from `load_clean_severson()`. Must contain:
        - `cell["cycles"]`: dict[str, dict] with per-cycle `"dQdV"` arrays
        - `cell["summary"]`: dict with `"cycle"` and `"QD"` arrays
    cell_id : str, optional
        Used only for log messages — pass it for traceability when running
        on many cells.
    verbose : bool, default False
        If True, print a one-line per-cell summary at the end.

    Returns
    -------
    pandas.DataFrame with columns:
        cycle    : int   — cycle index from the source dict
        height   : float — peak height in Ah/V (NaN on failure)
        position : float — peak voltage in V (NaN on failure)
        area     : float — area under |dQ/dV| in ±50 mV window, Ah (NaN on failure)
        SOH      : float — QD / QD_ref, where QD_ref is the first non-zero
                            discharge capacity (NaN if no valid reference)
        status   : str   — one of {"ok", "no_peak", "bad_dqdv"}

    Failed cycles are kept in the DataFrame (with NaN feature values) so the
    caller can compute a failure rate without bookkeeping. Filter with
    `df[df["status"] == "ok"]` (or `df.dropna()`) when feeding a model.
    """
    # ---- SOH reference: first non-zero QD ----
    qd = np.asarray(cell["summary"]["QD"])
    qd_positive = qd[qd > 0]
    qd_ref = float(qd_positive[0]) if qd_positive.size > 0 else float("nan")
    summary_cycles = np.asarray(cell["summary"]["cycle"]).astype(int)

    records: list[dict] = []
    prev_peak_V: float | None = None

    for cycle_str in sorted(cell["cycles"].keys(), key=lambda x: int(x)):
        k = int(cycle_str)
        raw_dqdv = np.asarray(cell["cycles"][cycle_str]["dQdV"])

        # SOH for this cycle: look up QD by matching cycle index in summary.
        mask = summary_cycles == k
        if mask.any() and qd_ref > 0:
            soh = float(qd[mask][0] / qd_ref)
        else:
            soh = float("nan")

        # Guard against dQdV arrays with unexpected length (a few cells in
        # the raw dataset have malformed cycles; the notebook skipped these
        # silently — we log them instead).
        if raw_dqdv.size != V_GRID_POINTS:
            records.append({
                "cycle": k,
                "height": np.nan,
                "position": np.nan,
                "area": np.nan,
                "SOH": soh,
                "status": "bad_dqdv",
            })
            continue

        height, position, area = _extract_one_cycle(raw_dqdv, prev_peak_V)

        if np.isnan(height):
            records.append({
                "cycle": k,
                "height": np.nan,
                "position": np.nan,
                "area": np.nan,
                "SOH": soh,
                "status": "no_peak",
            })
        else:
            prev_peak_V = position  # update prior for next cycle
            records.append({
                "cycle": k,
                "height": height,
                "position": position,
                "area": area,
                "SOH": soh,
                "status": "ok",
            })

    df = pd.DataFrame.from_records(
        records,
        columns=["cycle", "height", "position", "area", "SOH", "status"],
    )

    if verbose:
        _log_extraction_summary(df, cell_id)

    return df


# ------------------------ stats / logging ------------------------

def _log_extraction_summary(df: pd.DataFrame, cell_id: str | None) -> None:
    """Print a one-line summary of extraction stats. Used by `verbose=True`."""
    tag = f"[{cell_id}] " if cell_id else ""
    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum())
    n_no_peak = int((df["status"] == "no_peak").sum())
    n_bad = int((df["status"] == "bad_dqdv").sum())
    fail_rate = (n_total - n_ok) / n_total if n_total > 0 else float("nan")
    flag = " ⚠ >10% failure" if fail_rate > 0.10 else ""
    print(
        f"  {tag}cycles={n_total:4d}  ok={n_ok:4d}  "
        f"no_peak={n_no_peak:3d}  bad_dqdv={n_bad:3d}  "
        f"fail_rate={fail_rate:.1%}{flag}"
    )


def extraction_stats(df: pd.DataFrame) -> dict:
    """Return extraction counts as a dict — handy for programmatic checks."""
    n_total = len(df)
    n_ok = int((df["status"] == "ok").sum())
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_no_peak": int((df["status"] == "no_peak").sum()),
        "n_bad_dqdv": int((df["status"] == "bad_dqdv").sum()),
        "fail_rate": (n_total - n_ok) / n_total if n_total > 0 else float("nan"),
    }


# ------------------------ spot-check entry point ------------------------

def spot_check(cells: dict, cell_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Run `extract_features` on several cells and print a per-cell summary.

    Confirms the function works on cells beyond b1c1 (used by the test notebook).
    """
    print(f"Spot-checking {len(cell_ids)} cells:")
    results: dict[str, pd.DataFrame] = {}
    for cid in cell_ids:
        if cid not in cells:
            print(f"  [{cid}] not in dataset — skipping")
            continue
        results[cid] = extract_features(cells[cid], cell_id=cid, verbose=True)
    return results


if __name__ == "__main__":
    # Direct-run sanity check: extract on b1c1 plus three cells stratified by
    # cycle life (short / medium / long). Plotting is intentionally left to
    # an accompanying notebook — this entrypoint just confirms the function
    # runs end-to-end and prints the failure-rate stats.
    from src.severson_loader import load_clean_severson, summarise_cell

    cells = load_clean_severson()

    # Pick three batch-1 cells with short / medium / long cycle life,
    # excluding b1c1 (already validated in the notebook).
    candidates = sorted(
        (
            (cid, summarise_cell(c)["cycle_life"])
            for cid, c in cells.items()
            if cid.startswith("b1c") and cid != "b1c1"
        ),
        key=lambda x: x[1],
    )
    n = len(candidates)
    short_id = candidates[n // 8][0]
    medium_id = candidates[n // 2][0]
    long_id = candidates[-(n // 8) - 1][0]

    print(
        f"\nSelected: b1c1 (validated) + "
        f"short={short_id}, medium={medium_id}, long={long_id}\n"
    )
    spot_check(cells, ["b1c1", short_id, medium_id, long_id])
