"""
Recompute discharge dQ/dV from raw per-cycle V, Q, I signals (ENG-02).
======================================================================
The Severson pickles ship a precomputed per-cycle `dQdV` on a fixed
1000-point voltage grid (np.linspace(3.5, 2.0, 1000), descending). That
curve is unsmoothed and its provenance is opaque; to (a) verify we
understand it and (b) hunt for a second LFP peak with controlled
smoothing, this module rebuilds dQ/dV from the raw discharge samples:

    1. keep samples where I < I_DISCHARGE_MAX (CC discharge segment);
    2. drop V outside the grid span, sort by V ascending and deduplicate
       (median Q per repeated V) so Qd(V) is a function;
    3. interpolate Qd onto the fixed grid (NaN outside observed support);
    4. differentiate with np.gradient -> dQ/dV (negative on discharge,
       matching the published sign convention);
    5. optionally Savitzky-Golay smooth (window 51, polyorder 3 default —
       same style the Day-3 notebook used for charge curves).

The result is directly comparable, point-for-point, with the published
`dQdV` array.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from src.peak_extraction import V_GRID

# Discharge detection: Severson cells discharge at 4C (-4.4 A); anything
# below this threshold is comfortably inside the CC discharge segment and
# excludes rests (I ~ 0) and charge (I > 0).
I_DISCHARGE_MAX: float = -0.5

SMOOTH_WINDOW: int = 51
SMOOTH_POLYORDER: int = 3


def recompute_dqdv(
    cyc: dict,
    *,
    v_grid: np.ndarray = V_GRID,
    smooth_window: int | None = SMOOTH_WINDOW,
    polyorder: int = SMOOTH_POLYORDER,
) -> np.ndarray:
    """
    Rebuild discharge dQ/dV on `v_grid` from one cycle's raw arrays.

    Parameters
    ----------
    cyc : per-cycle dict with keys "I", "V", "Qd" (raw time series).
    v_grid : target voltage grid (descending, like the published one).
    smooth_window : Savitzky-Golay window (odd); None/0 disables smoothing.

    Returns
    -------
    np.ndarray aligned to v_grid; NaN where the discharge segment has no
    voltage support. All-NaN if the cycle has no usable discharge samples.
    """
    I = np.asarray(cyc["I"], dtype=float)
    V = np.asarray(cyc["V"], dtype=float)
    Qd = np.asarray(cyc["Qd"], dtype=float)
    if I.size < 10:
        return np.full(v_grid.shape, np.nan)

    mask = I < I_DISCHARGE_MAX
    if mask.sum() < 10:
        return np.full(v_grid.shape, np.nan)
    return dqdv_from_raw(V[mask], Qd[mask], v_grid=v_grid,
                         smooth_window=smooth_window, polyorder=polyorder)


def dqdv_from_raw(
    v: np.ndarray,
    q: np.ndarray,
    *,
    v_grid: np.ndarray,
    smooth_window: int | None = SMOOTH_WINDOW,
    polyorder: int = SMOOTH_POLYORDER,
) -> np.ndarray:
    """Chemistry-agnostic core: dQ/dV on `v_grid` from (V, Q) samples of one
    discharge segment. Also used for the NASA cells (ENG-03/ENG-11)."""
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)

    # Keep only samples inside the grid span.
    lo, hi = v_grid.min(), v_grid.max()
    keep = (v >= lo) & (v <= hi)
    v, q = v[keep], q[keep]
    if v.size < 10:
        return np.full(v_grid.shape, np.nan)

    # Sort by V ascending; collapse duplicate voltages to their median Q so
    # interpolation sees a proper function.
    order = np.argsort(v, kind="stable")
    v, q = v[order], q[order]
    uniq_v, inv = np.unique(v, return_inverse=True)
    if uniq_v.size < 10:
        return np.full(v_grid.shape, np.nan)
    sums = np.bincount(inv, weights=q)
    counts = np.bincount(inv)
    uniq_q = sums / counts

    # Interpolate Qd(V) on the ascending grid, NaN outside support.
    grid_asc = v_grid[::-1] if v_grid[0] > v_grid[-1] else v_grid
    q_interp = np.interp(grid_asc, uniq_v, uniq_q,
                         left=np.nan, right=np.nan)

    # Differentiate on the uniform grid.
    dv = float(grid_asc[1] - grid_asc[0])
    with np.errstate(invalid="ignore"):
        dqdv_asc = np.gradient(q_interp, dv)

    # Smooth the finite segment only.
    if smooth_window and smooth_window > polyorder + 1:
        finite = np.isfinite(dqdv_asc)
        n_fin = int(finite.sum())
        if n_fin > smooth_window:
            seg = dqdv_asc[finite]
            dqdv_asc = dqdv_asc.copy()
            dqdv_asc[finite] = savgol_filter(seg, smooth_window, polyorder)

    # Return in the same (descending-V) orientation as the published array.
    return dqdv_asc[::-1] if v_grid[0] > v_grid[-1] else dqdv_asc


def compare_with_published(recomputed: np.ndarray, published: np.ndarray) -> dict:
    """Pearson r and RMS difference over the common finite support."""
    ok = np.isfinite(recomputed) & np.isfinite(published)
    if ok.sum() < 10:
        return {"n_common": int(ok.sum()), "pearson_r": float("nan"),
                "rms_diff": float("nan")}
    a, b = recomputed[ok], published[ok]
    r = float(np.corrcoef(a, b)[0, 1])
    return {
        "n_common": int(ok.sum()),
        "pearson_r": r,
        "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
    }
