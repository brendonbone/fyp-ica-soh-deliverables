"""
Fast checks of src/ against the stored results (no raw pickles, under a minute).
================================================================================
The data-preparation conventions (cleaning, dev/test split, folds), the history
columns and the ENG-19b analysis are checked on the committed
results/features_expanded.parquet and result files; the per-cycle extraction
code (dominant-peak features, recomputed dQ/dV, SoC windows) is checked on
synthetic curves whose answer is known in closed form, and the .mat reader of
stage 0 on a small file written in the layout of the published ones.

    python -m pytest tests -q
"""
from __future__ import annotations

import json
import math
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from src.config import RESULTS_DIR
from src.data_prep import assign_folds, clean_feature_table, split_dev_test
from src.dqdv import dqdv_from_raw
from src.feature_requirements import cycle_windows
from src.features import AREA_WINDOWS, _dominant_peak_features
from src.features_temporal import (
    DEFAULT_MIN_HISTORY,
    DEFAULT_WINDOWS,
    HISTORY_COL,
    LEVEL_COLS,
    TEMPORAL_COLS,
    add_temporal_features,
    temporal_col_names,
)
from src.peak_extraction import DV, V_GRID, V_ROI
from src.severson_mat import CYCLE_FIELDS, SUMMARY_FIELDS, read_batch, write_pickle
from src.subset_search import analyse_table, eng08_rankings, mean_predictor_cv, table_from_records

PARQUET = RESULTS_DIR / "features_expanded.parquet"

# ------------------------ asserted numbers and their sources ------------------------

# results/eng00_baseline.json "clean_log" (status ok -> dropna SoH -> SoH > 0.5 -> SoH <= 1.05);
# the status counts agree with results/eng05_extraction_report.json (100,459 ok rows).
CLEAN_LOG = {"start": 100_501, "after_status_ok": 100_459, "after_dropna_soh": 100_376,
             "after_soh_min": 100_335, "after_soh_max": 100_331, "soh_clean_band": [0.5, 1.05]}

# src/severson_loader.py exclusions and merges: 41 cells in batch 1, 43 in batch 2, 40 in batch 3.
DEV_CELLS_PER_BATCH = {"b1": 41, "b2": 43}
N_DEV_CELLS = 84
N_TEST_CELLS = 40

# results/eng00_baseline.json "cell_to_fold": five folds, cells per fold.
N_FOLDS = 5
CELLS_PER_FOLD = {0: 16, 1: 16, 2: 16, 3: 18, 4: 18}

# results/eng14_validation.json check 4: 2,479 clean rows have < 20 cycles of past
# (20 rows in 123 cells, 19 in one cell with a gap), so 100,331 - 2,479 rows keep history;
# "temporal_cols": 9 features x (d10, d20, slope10, slope20, drift, rm20).
N_HISTORY_ROWS = 97_852
NO_HISTORY_ROWS_PER_CELL = {20: 123, 19: 1}
N_TEMPORAL_COLS = 54

# results/eng19b_analysis.json "v_empty_cv_rmse" (dev-CV RMSE of the training-fold mean)
# and "be_drop_order"; the analysis must reproduce the stored file to ANALYSIS_TOL.
V_EMPTY = 0.051653927838189895
BE_DROP_ORDER = ["slope_left", "area35", "height", "area25", "area", "slope_right",
                 "position", "fwhm"]
ANALYSIS_TOL = 1e-12

# results/eng10_learning_curve.json (stage 20, ENG-10) as first computed on 2026-07-07:
# n_cells -> (r2_mean, r2_std, rmse_mean), and the plateau. 66 appears because folds 3-4
# have only 66 other-fold cells. Checked to LEARNING_CURVE_TOL (the JSON is rounded to 6 dp).
LEARNING_CURVE = {10: (0.785523, 0.056236, 0.023696), 20: (0.814634, 0.040249, 0.022092),
                  30: (0.822438, 0.039663, 0.021624), 40: (0.839404, 0.030350, 0.020597),
                  50: (0.840277, 0.032038, 0.020531), 60: (0.844439, 0.028818, 0.020276),
                  66: (0.840766, 0.027830, 0.020568), 67: (0.850821, 0.027420, 0.019831)}
LEARNING_CURVE_PLATEAU = 40
LEARNING_CURVE_TOL = 1e-6 + 1e-12   # 'abs diff <= 1e-6', robust to float representation

# Synthetic dQ/dV peak, as in the ENG-17 synthetic check (OLD scripts/eng17_feature_soc_requirements.py).
PEAK_AMP, PEAK_SIGMA, PEAK_CENTRE = 6.0, 0.030, 3.1502       # Ah/V, V, V
# Synthetic linear discharge Q = Q_PER_V * (3.6 - V): dQ/dV = -Q_PER_V everywhere.
Q_PER_V = 1.1                                               # Ah/V


# ------------------------ helpers ------------------------

def assert_close(got, want, tol: float, path: str = "") -> None:
    """Nested dicts/lists equal; floats within an absolute `tol`, everything else exact."""
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            assert_close(got[k], want[k], tol, f"{path}.{k}")
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close(g, w, tol, f"{path}[{i}]")
    elif isinstance(want, float):
        assert abs(got - want) <= tol, f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def read_result(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def gaussian_peak(amp: float = PEAK_AMP, centre: float = PEAK_CENTRE,
                  sigma: float = PEAK_SIGMA) -> tuple[float, np.ndarray]:
    """(apex voltage on the ROI grid, |dQ/dV| of a Gaussian peak on V_ROI)."""
    vp = float(V_ROI[np.argmin(np.abs(V_ROI - centre))])
    return vp, amp * np.exp(-0.5 * ((V_ROI - vp) / sigma) ** 2)


def gaussian_integral(amp: float, sigma: float, half_width: float) -> float:
    """Integral of amp * exp(-x^2 / 2 sigma^2) over [-half_width, half_width]."""
    return amp * sigma * math.sqrt(2 * math.pi) * math.erf(half_width / (sigma * math.sqrt(2)))


def load_eng19_table() -> pd.DataFrame:
    """The 511-subset ENG-19 table, built as OLD scripts/eng19b_subset_analysis.py
    `load_eng19` did: the per-subset rows live in the CSV (the ENG-19 JSON holds
    only summaries), keyed by the '+'-joined pool-order feature string."""
    recs = {}
    for r in pd.read_csv(RESULTS_DIR / "eng19_exhaustive_subsets.csv").to_dict("records"):
        recs[r["features"]] = {"blocks": r["features"].split("+"), "k": int(r["k"]),
                               "n_columns": int(r["k"]),
                               "cv_pooled_rmse_mean": float(r["cv_pooled_rmse_mean"]),
                               "cv_pooled_rmse_std": float(r["cv_pooled_rmse_std"])}
    return table_from_records(recs)


# ------------------------ fixtures (the parquet is read once) ------------------------

@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@pytest.fixture(scope="module")
def clean(raw) -> pd.DataFrame:
    return clean_feature_table(raw, verbose=False)[0]


@pytest.fixture(scope="module")
def dev_test(clean) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_dev_test(clean)


@pytest.fixture(scope="module")
def folds(dev_test) -> tuple[pd.Series, dict]:
    return assign_folds(dev_test[0])


@pytest.fixture(scope="module")
def history(clean) -> tuple[pd.DataFrame, list[str]]:
    return add_temporal_features(clean, DEFAULT_WINDOWS, DEFAULT_MIN_HISTORY)


# ------------------------ (a) row cleaning ------------------------

def test_clean_feature_table_counts_and_log(raw, capsys):
    clean, log = clean_feature_table(raw)
    assert log == CLEAN_LOG
    assert len(clean) == CLEAN_LOG["after_soh_max"]
    assert (clean["status"] == "ok").all()
    assert clean["SOH"].notna().all()
    assert ((clean["SOH"] > 0.5) & (clean["SOH"] <= 1.05)).all()
    assert clean.index.equals(pd.RangeIndex(len(clean)))
    printed = capsys.readouterr().out
    assert ("clean: 100501 -> ok 100459 -> dropna 100376 -> SoH>0.5 100335 "
            "-> SoH<=1.05 100331") in printed


def test_clean_log_matches_eng00():
    assert read_result("eng00_baseline.json")["clean_log"] == CLEAN_LOG


# ------------------------ (b) dev / test split ------------------------

def test_split_dev_test(clean, dev_test):
    dev, test = dev_test
    assert dev["cell_id"].nunique() == N_DEV_CELLS
    assert test["cell_id"].nunique() == N_TEST_CELLS
    assert dev.groupby("batch")["cell_id"].nunique().to_dict() == DEV_CELLS_PER_BATCH
    assert set(test["batch"]) == {"b3"}
    assert not set(dev["cell_id"]) & set(test["cell_id"])
    assert len(dev) + len(test) == len(clean)
    assert dev.index.equals(pd.RangeIndex(len(dev)))
    assert test.index.equals(pd.RangeIndex(len(test)))


# ------------------------ (c) CV folds ------------------------

def test_assign_folds(dev_test, folds):
    dev = dev_test[0]
    fold, cell_to_fold = folds
    assert fold.index.equals(dev.index)
    assert sorted(fold.unique()) == list(range(N_FOLDS))
    folds_per_cell = fold.groupby(dev["cell_id"]).nunique()
    assert len(folds_per_cell) == N_DEV_CELLS and (folds_per_cell == 1).all()
    assert set(cell_to_fold) == set(dev["cell_id"])
    assert (dev["cell_id"].map(cell_to_fold) == fold).all()
    assert pd.Series(cell_to_fold).value_counts().sort_index().to_dict() == CELLS_PER_FOLD


def test_folds_match_eng00(folds):
    stored = read_result("eng00_baseline.json")["cell_to_fold"]
    assert {c: int(f) for c, f in folds[1].items()} == stored


# ------------------------ (d) history columns ------------------------

def test_history_columns(clean, history):
    out, cols = history
    assert cols == TEMPORAL_COLS == temporal_col_names(DEFAULT_WINDOWS)
    assert len(cols) == N_TEMPORAL_COLS
    assert list(out.columns) == list(clean.columns) + cols + [HISTORY_COL]
    assert out[clean.columns].equals(clean)            # rows and level columns untouched


def test_history_rows_and_nan_mask(clean, history):
    out, cols = history
    since_first = out["cycle"] - out.groupby("cell_id")["cycle"].transform("min")
    expect_nan = since_first < DEFAULT_MIN_HISTORY
    t = out[cols]
    assert (t.isna().all(axis=1) == expect_nan).all()
    assert (t.isna().any(axis=1) == expect_nan).all()
    assert (out[HISTORY_COL] == ~expect_nan).all()
    assert int(out[HISTORY_COL].sum()) == N_HISTORY_ROWS
    dropped = (~out[HISTORY_COL]).groupby(out["cell_id"]).sum()
    assert dropped.value_counts().to_dict() == NO_HISTORY_ROWS_PER_CELL


def test_slope_matches_polyfit(clean, history):
    """slope10 / slope20 == np.polyfit over the last N cycle numbers, with missing
    cycles forward-filled from the latest earlier clean cycle (the module's gap rule)."""
    out, _ = history
    by_cell = clean.sort_values(["cell_id", "cycle"]).groupby("cell_id")["cycle"]
    has_gap = by_cell.apply(lambda c: bool((np.diff(c.to_numpy()) > 1).any()))
    rng = np.random.default_rng(0)
    cells = [has_gap[has_gap].index[0],
             *rng.choice(sorted(clean["cell_id"].unique()), size=2, replace=False)]
    n_checked = 0
    for cid in cells:
        g = out[out["cell_id"] == cid].sort_values("cycle")
        first, last = int(g["cycle"].iloc[0]), int(g["cycle"].iloc[-1])
        grid = (g.set_index("cycle")[LEVEL_COLS].reindex(range(first, last + 1))
                .ffill().to_numpy())
        rows = g[g[HISTORY_COL]]
        for n in DEFAULT_WINDOWS:
            got = rows[[f"{f}_slope{n}" for f in LEVEL_COLS]].to_numpy()
            want = np.array([np.polyfit(np.arange(n), grid[k - first - n + 1:k - first + 1], 1)[0]
                             for k in rows["cycle"].to_numpy()])
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
            n_checked += got.size
    assert n_checked > 0


# ------------------------ (e) ENG-19b analysis of the 511-subset table ------------------------

def test_mean_predictor_reproduces_v_empty(dev_test, folds):
    v = mean_predictor_cv(dev_test[0], folds[0])
    assert_close(v, read_result("eng19b_analysis.json")["v_empty"], ANALYSIS_TOL)
    assert abs(v["cv_pooled_rmse_mean"] - V_EMPTY) <= ANALYSIS_TOL


def test_analyse_table_reproduces_eng19b():
    stored = read_result("eng19b_analysis.json")
    t = load_eng19_table()
    assert len(t) == stored["n_subsets"] == 2 ** len(LEVEL_COLS) - 1
    A = analyse_table(t, list(LEVEL_COLS), V_EMPTY, rankings=eng08_rankings())
    assert_close(A["best_per_k"], stored["best_per_k"], ANALYSIS_TOL, "best_per_k")
    assert A["be_drop_order"] == stored["be_drop_order"] == BE_DROP_ORDER
    assert A["be_last_standing"] == stored["be_last_standing"]
    assert_close(A["shapley"], stored["shapley"], ANALYSIS_TOL, "shapley")
    assert A["shapley_rank"] == stored["shapley_rank"]
    assert_close(A["shapley_efficiency_check"], stored["shapley_efficiency_check"],
                 ANALYSIS_TOL, "shapley_efficiency_check")
    assert_close(A["ranking_paths"], stored["ranking_paths"], ANALYSIS_TOL, "ranking_paths")


def test_learning_curve_matches_eng10():
    stored = read_result("eng10_learning_curve.json")
    assert stored["grid"] == [10, 20, 30, 40, 50, 60, 67] and stored["n_repeats"] == 5
    assert stored["plateau_n_cells"] == LEARNING_CURVE_PLATEAU
    rows = {r["n_cells"]: r for r in stored["aggregate"]}
    assert sorted(rows) == sorted(LEARNING_CURVE)
    for n, (r2_mean, r2_std, rmse_mean) in LEARNING_CURVE.items():
        assert_close([rows[n]["r2_mean"], rows[n]["r2_std"], rows[n]["rmse_mean"]],
                     [r2_mean, r2_std, rmse_mean], LEARNING_CURVE_TOL, f"n_cells={n}")


# ------------------------ (f) dominant-peak features ------------------------

def test_dominant_peak_features_gaussian():
    vp, absq = gaussian_peak()
    f = _dominant_peak_features(-absq, None)            # dQ/dV is negative on discharge
    assert f["position"] == vp
    assert f["height"] == PEAK_AMP
    assert f["prominence"] == pytest.approx(PEAK_AMP, abs=1e-12)
    for name, hw in AREA_WINDOWS.items():
        # the trapezoid runs over the grid points within +/- hw of the apex
        reach = math.floor(hw / DV) * DV
        assert f[name] == pytest.approx(gaussian_integral(PEAK_AMP, PEAK_SIGMA, reach), rel=5e-4)
    assert f["fwhm"] == pytest.approx(2 * math.sqrt(2 * math.log(2)) * PEAK_SIGMA, abs=2e-5)
    flank = PEAK_AMP * (1 - math.exp(-0.5 * (0.025 / PEAK_SIGMA) ** 2)) / 0.025
    assert f["slope_left"] == pytest.approx(flank, rel=1e-3)
    assert f["slope_right"] == pytest.approx(f["slope_left"], rel=1e-6)   # symmetric peak


def test_dominant_peak_tracking_and_no_peak():
    v_big, big = gaussian_peak(amp=PEAK_AMP, centre=3.15)
    v_small, small = gaussian_peak(amp=2.0, centre=2.85)
    curve = -(big + small)
    assert _dominant_peak_features(curve, None)["position"] == v_big       # most prominent
    assert _dominant_peak_features(curve, 2.86)["position"] == v_small     # nearest previous
    assert _dominant_peak_features(np.zeros_like(V_ROI), None) is None
    assert _dominant_peak_features(-gaussian_peak(amp=0.2)[1], None) is None   # below 0.3 Ah/V


# ------------------------ (g) recomputed dQ/dV ------------------------

def test_dqdv_from_raw_linear_q():
    rng = np.random.default_rng(0)
    v = np.concatenate([np.linspace(3.5, 2.0, 3001), [3.6, 3.55, 1.95, 1.9]])  # some outside the grid
    v = np.concatenate([v, v[::7]])                     # repeated voltages
    v = v[rng.permutation(len(v))]                      # unsorted samples
    q = Q_PER_V * (3.6 - v)
    for window in (51, None):
        out = dqdv_from_raw(v, q, v_grid=V_GRID, smooth_window=window)
        assert out.shape == V_GRID.shape
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, -Q_PER_V, rtol=1e-9)


def test_dqdv_from_raw_partial_support():
    v = np.linspace(3.2, 2.5, 2001)
    out = dqdv_from_raw(v, Q_PER_V * (3.6 - v), v_grid=V_GRID)
    assert np.isnan(out[(V_GRID > 3.2) | (V_GRID < 2.5)]).all()   # NaN outside, descending order kept
    inside = (V_GRID < 3.2 - 2 * DV) & (V_GRID > 2.5 + 2 * DV)
    np.testing.assert_allclose(out[inside], -Q_PER_V, rtol=1e-9)
    assert np.isnan(dqdv_from_raw(v[:5], v[:5], v_grid=V_GRID)).all()  # too few samples


# ------------------------ (h) SoC windows ------------------------

def test_cycle_windows_nested():
    vp, absq = gaussian_peak()
    qd = np.concatenate([[0.0], np.cumsum((absq[1:] + absq[:-1]) / 2 * DV)])   # Q(V), V descending
    q_total = float(qd[-1])
    rows = {(r["feature"], r["basis"]): r for r in cycle_windows(-absq, qd, vp, q_total)}
    nested = [rows[(f, "as_implemented")] for f in ("height", "area25", "area35", "area")]
    for inner, outer in zip(nested, nested[1:]):
        assert outer["v_hi"] >= inner["v_hi"] and outer["v_lo"] <= inner["v_lo"]
        assert outer["soc_start"] >= inner["soc_start"] and outer["soc_end"] <= inner["soc_end"]
        assert outer["soc_span"] > inner["soc_span"]
    for r in nested:
        assert r["v_lo"] < vp < r["v_hi"]
        assert r["soc_span"] == pytest.approx(r["q_span_ah"] / q_total, abs=1e-12)
    for name, hw in AREA_WINDOWS.items():
        # +/- hw bracketed by one grid step, plus the one-point pad, per side
        width = rows[(name, "as_implemented")]["width_mv"]
        assert 2 * hw * 1000 - 1e-6 <= width <= (2 * hw + 4 * DV) * 1000 + 1e-6
    assert rows[("position", "as_implemented")] == {**rows[("height", "as_implemented")],
                                                    "feature": "position"}
    prom = rows[("prominence", "as_implemented")]
    assert prom["v_hi"] == V_ROI[0] and prom["v_lo"] == V_ROI[-1]      # needs the whole ROI


# ------------------------ (i) the .mat files to pickles ------------------------

def write_mat_like(path, n_cells=2, n_cycles=3, policy="5.4C(40%)-3.6C"):
    """A small HDF5 file in the published layout: batch/<field> holds (n_cells, 1) references.

    Summary field k of cell i is 100 i + 10 k + (0, 1, ...); cycle field k of
    cell i, cycle j is i + k + j + linspace(0, 1, 5 + j).
    """
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")

        def ref(name, data):
            return refs.create_dataset(name, data=data).ref

        columns = {name: [] for name in ("cycle_life", "policy_readable", "summary", "cycles")}
        for i in range(n_cells):
            summary = refs.create_group(f"summary{i}")
            for k, field in enumerate(SUMMARY_FIELDS.values()):
                summary.create_dataset(field, data=(100 * i + 10 * k + np.arange(n_cycles, dtype=float))[None, :])
            cycles = refs.create_group(f"cycles{i}")
            for k, field in enumerate(CYCLE_FIELDS.values()):
                per_cycle = [ref(f"c{i}_{field}_{j}", (i + k + j + np.linspace(0, 1, 5 + j))[None, :])
                             for j in range(n_cycles)]
                cycles.create_dataset(field, data=np.array(per_cycle, dtype=h5py.ref_dtype)[:, None])
            columns["cycle_life"].append(ref(f"cycle_life{i}", np.array([[500.0 + i]])))
            utf16 = np.frombuffer(policy.encode("utf-16-le"), dtype=np.uint16)   # a MATLAB char array
            columns["policy_readable"].append(ref(f"policy{i}", utf16[:, None]))
            columns["summary"].append(summary.ref)
            columns["cycles"].append(cycles.ref)
        batch = f.create_group("batch")
        for name, column in columns.items():
            batch.create_dataset(name, data=np.array(column, dtype=h5py.ref_dtype)[:, None])


def test_read_batch_and_write_pickle(tmp_path):
    mat = tmp_path / "batch.mat"
    write_mat_like(mat)
    cells = read_batch(mat, "b9", verbose=False)
    assert list(cells) == ["b9c0", "b9c1"]
    cell = cells["b9c1"]
    assert list(cell) == ["cycle_life", "charge_policy", "summary", "cycles"]
    assert cell["cycle_life"].shape == (1, 1) and cell["cycle_life"][0, 0] == 501.0
    assert cell["charge_policy"] == "5.4C(40%)-3.6C"
    assert list(cell["summary"]) == list(SUMMARY_FIELDS)
    np.testing.assert_array_equal(cell["summary"]["QD"], 120 + np.arange(3.0))      # QDischarge, k = 2
    assert list(cell["cycles"]) == ["0", "1", "2"]
    assert list(cell["cycles"]["0"]) == list(CYCLE_FIELDS)
    np.testing.assert_array_equal(cell["cycles"]["2"]["dQdV"], 10 + np.linspace(0, 1, 7))  # discharge_dQdV, k = 7

    out = tmp_path / "batch9.pkl"
    write_pickle(cells, out)
    assert not out.with_name("batch9.pkl.part").exists()
    with open(out, "rb") as fp:
        back = pickle.load(fp)
    assert list(back) == list(cells)
    np.testing.assert_array_equal(back["b9c0"]["cycles"]["1"]["V"], cells["b9c0"]["cycles"]["1"]["V"])
