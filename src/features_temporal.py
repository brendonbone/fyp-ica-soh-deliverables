"""
Temporal (trajectory) feature pool (ENG-14).
============================================
Every row in the ENG-08 pipeline is one cycle's nine-feature snapshot
(`FEATURE_COLS`, the "level" features). This module appends, per base
feature f, a description of f's trajectory over the PRECEDING cycles of the
same cell, so the selection study can ask whether the rate/extent of change
of a feature outranks its current level:

    f_d{N}      change over the last N cycles:   f(k) - f(k-N)
    f_slope{N}  OLS slope of f over cycles k-N+1..k  [units per cycle]
    f_drift     f(k) - f(first clean cycle of the cell)
    f_rm{N}     rolling mean over cycles k-N+1..k, largest N only
                (a denoised level feature)

for each N in `windows` (default 10, 20).

Causality (no look-ahead)
-------------------------
Everything is computed strictly within `cell_id` and strictly from cycles
<= k. Neither SoH, cycle_life, nor any later cycle enters any derived
column. `scripts/eng14_validate_temporal.py` checks this by perturbing
later cycles and asserting earlier rows are bit-identical.

Cycle gaps
----------
After `clean_feature_table` a cell's rows are not contiguous in cycle
number (status != "ok", NaN-SoH and out-of-band SoH rows are gone; five
2-cycle gaps survive on the Severson table). Lags are therefore taken by
CYCLE NUMBER, not row position: each cell is re-indexed onto its complete
cycle range [first clean cycle .. last clean cycle], missing cycles are
forward-filled from the most recent earlier clean cycle (the backward
`merge_asof` convention -- strictly causal), and the windows are evaluated
on that complete grid. So f_d{N} is f(k) minus the value at the latest
clean cycle <= k-N, and the N-cycle windows always span exactly N cycle
numbers.

History gate
------------
`has_history` is True when cycle - first_clean_cycle >= min_history, i.e.
the cell has at least `min_history` cycle numbers of past on the grid. All
temporal columns are NaN where has_history is False, and finite everywhere
else (min_history >= max(windows) is enforced so no window is ever
partially filled on a has_history row). Rows are NOT dropped here; the
experiment script drops them and logs the counts per split.

Public names
------------
    add_temporal_features(df, windows=(10, 20), min_history=20)
        -> (df_out, temporal_cols)
    temporal_col_names(windows)          column names for any window set
    TEMPORAL_COLS                        names for the default windows
    LEVEL_COLS                           == features.FEATURE_COLS
    HISTORY_COL                          "has_history"
    history_filter_report(df)            row counts per batch / split
    base_feature_of(col)                 map a temporal column to its base
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data_prep import DEV_BATCHES, TEST_BATCH
from src.features import FEATURE_COLS

DEFAULT_WINDOWS: tuple[int, ...] = (10, 20)
DEFAULT_MIN_HISTORY: int = 20
HISTORY_COL: str = "has_history"

LEVEL_COLS: list[str] = list(FEATURE_COLS)


# ------------------------ naming ------------------------

def _kinds(windows: tuple[int, ...]) -> list[str]:
    """Suffixes, in a fixed order, for one base feature."""
    windows = tuple(sorted(int(w) for w in windows))
    kinds = [f"d{n}" for n in windows]
    kinds += [f"slope{n}" for n in windows]
    kinds += ["drift", f"rm{windows[-1]}"]
    return kinds


def temporal_col_names(
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    base_cols: list[str] = LEVEL_COLS,
) -> list[str]:
    """Deterministic column order: base feature major, kind minor."""
    return [f"{f}_{k}" for f in base_cols for k in _kinds(windows)]


TEMPORAL_COLS: list[str] = temporal_col_names(DEFAULT_WINDOWS)

_SUFFIX_RE = re.compile(r"_(d\d+|slope\d+|drift|rm\d+)$")


def base_feature_of(col: str) -> str:
    """'height_slope10' -> 'height'; a level column maps to itself."""
    return _SUFFIX_RE.sub("", col)


def is_temporal(col: str) -> bool:
    return bool(_SUFFIX_RE.search(col))


# ------------------------ per-cell computation ------------------------

def _ols_weights(n: int) -> np.ndarray:
    """Weights w such that w @ y is the OLS slope of y vs x = 0..n-1."""
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    return xc / np.sum(xc * xc)


def _cell_temporal(
    cycles: np.ndarray,
    values: np.ndarray,
    windows: tuple[int, ...],
    min_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Temporal features for ONE cell.

    cycles : (n,) int, strictly increasing clean cycle numbers.
    values : (n, p) float, base features on those rows (no NaN).
    Returns (temporal (n, p * n_kinds) float, has_history (n,) bool).
    Column order matches `temporal_col_names`.
    """
    n, p = values.shape
    first = int(cycles[0])
    pos = (cycles - first).astype(int)           # position on the full grid
    grid_len = int(pos[-1]) + 1

    # Complete cycle grid, forward-filled (strictly causal; the first grid
    # row is the first clean cycle so there is no leading NaN).
    grid = np.full((grid_len, p), np.nan)
    grid[pos] = values
    grid = pd.DataFrame(grid).ffill().to_numpy()

    n_max = windows[-1]
    blocks: list[np.ndarray] = []              # each (grid_len, p), NaN-padded

    for N in windows:                          # f(k) - f(k-N)
        d = np.full((grid_len, p), np.nan)
        if grid_len > N:
            d[N:] = grid[N:] - grid[:-N]
        blocks.append(d)

    for N in windows:                          # OLS slope over k-N+1..k
        s = np.full((grid_len, p), np.nan)
        if grid_len >= N:
            win = sliding_window_view(grid, N, axis=0)   # (grid_len-N+1, p, N)
            # Centre each window first: the weights sum to zero only up to
            # rounding, so this keeps a constant window at exactly 0 and
            # avoids cancellation on offset-heavy features (e.g. position).
            win = win - win.mean(axis=-1, keepdims=True)
            s[N - 1:] = win @ _ols_weights(N)
        blocks.append(s)

    blocks.append(grid - grid[0])              # drift from first clean cycle

    rm = np.full((grid_len, p), np.nan)        # rolling mean, largest window
    if grid_len >= n_max:
        rm[n_max - 1:] = sliding_window_view(grid, n_max, axis=0).mean(axis=-1)
    blocks.append(rm)

    # (grid_len, n_kinds, p) -> per row, base-feature-major ordering.
    stacked = np.stack(blocks, axis=1)         # (grid_len, n_kinds, p)
    stacked = np.transpose(stacked, (0, 2, 1)) # (grid_len, p, n_kinds)
    out = stacked.reshape(grid_len, p * len(blocks))[pos]

    has_history = pos >= min_history
    out[~has_history] = np.nan
    return out, has_history


# ------------------------ public API ------------------------

def add_temporal_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    min_history: int = DEFAULT_MIN_HISTORY,
    *,
    base_cols: list[str] = LEVEL_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Append trajectory features to a cleaned per-cycle feature table.

    Parameters
    ----------
    df : output of `clean_feature_table` (needs cell_id, cycle, base_cols;
        row order is preserved in the output).
    windows : lag / window lengths in cycles (sorted ascending internally).
    min_history : rows with cycle - first_clean_cycle < min_history get NaN
        temporal features and has_history == False. Must be >= max(windows).
    base_cols : the level features to derive from (default FEATURE_COLS).

    Returns
    -------
    (df_out, temporal_cols): df_out = df + temporal columns + `has_history`.
    """
    windows = tuple(sorted({int(w) for w in windows}))
    if not windows or windows[0] < 1:
        raise ValueError("windows must be positive integers")
    if min_history < windows[-1]:
        raise ValueError(
            f"min_history ({min_history}) must be >= max(windows) "
            f"({windows[-1]}) so every has_history row has full windows"
        )
    missing = [c for c in ("cell_id", "cycle", *base_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"feature table lacks columns: {missing}")
    n_nan = int(df[base_cols].isna().sum().sum())
    if n_nan:
        raise ValueError(
            f"{n_nan} NaN base-feature values; pass a cleaned table "
            "(status == 'ok' rows only) so lags are well defined"
        )
    if df.duplicated(["cell_id", "cycle"]).any():
        raise ValueError("duplicate (cell_id, cycle) rows")

    cols = temporal_col_names(windows, base_cols)
    temporal = np.full((len(df), len(cols)), np.nan)
    has_hist = np.zeros(len(df), dtype=bool)

    # groupby(sort=False) keeps the table's own cell order; rows are written
    # back by integer position so the output row order == input row order.
    positions = np.arange(len(df))
    for _, idx in df.groupby("cell_id", sort=False).indices.items():
        idx = positions[idx]
        cyc = df["cycle"].to_numpy()[idx]
        order = np.argsort(cyc, kind="stable")
        idx = idx[order]
        cyc = cyc[order]
        vals = df[base_cols].to_numpy(dtype=float)[idx]
        t, h = _cell_temporal(cyc, vals, windows, min_history)
        temporal[idx] = t
        has_hist[idx] = h

    out = df.copy()
    out[cols] = temporal
    out[HISTORY_COL] = has_hist
    return out, cols


def history_filter_report(df: pd.DataFrame) -> dict:
    """Row counts before/after dropping has_history == False, per split."""
    if HISTORY_COL not in df.columns:
        raise KeyError(f"{HISTORY_COL} missing — call add_temporal_features first")
    rep: dict = {"total": _count(df)}
    for b, g in df.groupby("batch", sort=True):
        rep[f"batch_{b}"] = _count(g)
    rep["dev"] = _count(df[df["batch"].isin(DEV_BATCHES)])
    rep["test"] = _count(df[df["batch"] == TEST_BATCH])
    per_cell = (~df[HISTORY_COL]).groupby(df["cell_id"]).sum()
    rep["dropped_per_cell"] = {
        "min": int(per_cell.min()), "max": int(per_cell.max()),
        "n_cells": int(per_cell.size),
    }
    return rep


def _count(g: pd.DataFrame) -> dict:
    keep = int(g[HISTORY_COL].sum())
    return {"rows": int(len(g)), "rows_with_history": keep,
            "rows_dropped": int(len(g)) - keep,
            "cells": int(g["cell_id"].nunique())}
