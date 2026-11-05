"""
Table VI. SoC window of the discharge that each feature needs (median over all clean cycles)
==============================================================================================
ENG-17's median state-of-charge window per feature (start -> end, span, voltage
width, minutes at 4 A), over all clean cycles of all 124 cells. Features whose
windows are identical in every life stage share a row; rows are ordered by span,
and a last row gives fwhm measured between its half-prominence crossings only.
Reads eng17_soc_windows_summary.csv and eng17_feature_soc_requirements.json. The
CSV's dev_* columns (medians over the development cells, batches 1-2) need the
per-cycle features_soc_windows.parquet that stage 10 writes but git does not keep;
without it they are left empty (the markdown table does not show them).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import (FEATURES, fmt_float, join_features, load_json, minus,
                                 result_path, write_table)
from src.config import RESULTS_DIR
from src.data_prep import split_dev_test
from src.feature_requirements import DISCHARGE_CURRENT_A, summarise_windows

LABEL = "Table VI"
TITLE = "SoC window of the discharge that each feature needs (median over all clean cycles)"
SUMMARY = "eng17_soc_windows_summary.csv"
REQ_JSON = "eng17_feature_soc_requirements.json"
SOC_PARQUET = "features_soc_windows.parquet"   # optional: only the dev_* CSV columns
SOURCES = [SUMMARY, REQ_JSON]

# stage and basis strings exactly as ENG-17 writes them
ALL, FRESH, MID, AGED = "all cycles", "fresh (SoH >= 0.99)", "mid (0.90-0.99)", "aged (SoH < 0.90)"
STAGES = [ALL, FRESH, MID, AGED]
IMPL, VALUE = "as_implemented", "value_support"
NUM = ["soc_start_median", "soc_end_median", "soc_span_median", "soc_start_p95", "soc_end_p05",
       "width_mv_median", "q_span_ah_median", "minutes_at_4A_median"]

# The cells discharge at 4C = 4.4 A; ENG-17 quotes minutes at a round 4 A (DISCHARGE_CURRENT_A).
DATASET_CURRENT_A = 4.4
VARIANT_LABEL = "fwhm, crossings only (variant)"
DEV_COLS = ["dev_n_cycles", "dev_soc_start_pct", "dev_soc_end_pct", "dev_soc_span_pct",
            "dev_width_mv", "dev_minutes_at_4A"]


def pct(x: float) -> str:
    """Fraction -> percent with one decimal ('0.1359' -> '13.6')."""
    return f"{100 * x:.1f}"


def mv(x: float) -> str:
    """Voltage width in whole mV with a thousands separator ('1,399')."""
    return minus(f"{x:,.0f}")


def load_summary() -> pd.DataFrame:
    """ENG-17 medians per stage, feature and basis.

    Read with pandas' default float parser, as the Section V script that
    produced the reference table did; round_trip moves some values by one ulp."""
    return pd.read_csv(result_path(SUMMARY))


def row(summ: pd.DataFrame, feature: str, stage: str = ALL, basis: str = IMPL) -> pd.Series:
    """The one summary row of a feature, life stage and basis."""
    r = summ[(summ["feature"] == feature) & (summ["stage"] == stage) & (summ["basis"] == basis)]
    assert len(r) == 1, (feature, stage, basis)
    return r.iloc[0]


def window_groups(summ: pd.DataFrame) -> list[tuple[str, ...]]:
    """Features whose as-implemented windows are identical (exact float equality,
    every stage and median) in pool order, sorted by the all-cycles span."""
    impl = summ[summ["basis"] == IMPL]
    sig: dict[tuple, list[str]] = {}
    for f in FEATURES:
        g = impl[impl["feature"] == f].set_index("stage").loc[STAGES, NUM]
        sig.setdefault(tuple(g.to_numpy().ravel().tolist()), []).append(f)
    return sorted((tuple(v) for v in sig.values()), key=lambda g: row(summ, g[0])["soc_span_median"])


def table_rows(summ: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(row label, feature whose window it shows, basis): the window groups, then the fwhm variant."""
    return [(join_features(g), g[0], IMPL) for g in window_groups(summ)] + [(VARIANT_LABEL, "fwhm", VALUE)]


def dev_medians() -> pd.DataFrame | None:
    """All-cycles medians over the development cells, indexed by (feature, basis);
    None when the per-cycle parquet is not on disk."""
    path = RESULTS_DIR / SOC_PARQUET
    if not path.exists():
        return None
    win_dev, _ = split_dev_test(pd.read_parquet(path))
    dev = summarise_windows(win_dev)
    return dev[dev["stage"] == ALL].set_index(["feature", "basis"])


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """The displayed table, the full-precision CSV table, the markdown formats and the notes."""
    summ = load_summary()
    req = load_json(REQ_JSON)
    dev_p = dev_medians()
    rows = table_rows(summ)

    disp, full = [], []
    for name, feature, basis in rows:
        r = {st: row(summ, feature, st, basis) for st in STAGES}
        a = r[ALL]
        disp.append({"Feature(s)": name,
                     "SoC window, start → end (%)": f"{pct(a.soc_start_median)} → {pct(a.soc_end_median)}",
                     "Span (% of discharge)": 100 * a.soc_span_median,
                     "Voltage width (mV)": a.width_mv_median,
                     "Minutes at 4 A": a.minutes_at_4A_median})
        rec = {"features": name, "basis": basis, "n_cycles": int(a.n_cycles),
               "soc_start_pct": 100 * a.soc_start_median, "soc_end_pct": 100 * a.soc_end_median,
               "soc_span_pct": 100 * a.soc_span_median, "width_mv": a.width_mv_median,
               "q_span_ah": a.q_span_ah_median, "minutes_at_4A": a.minutes_at_4A_median,
               "minutes_at_4p4A": a.minutes_at_4A_median * DISCHARGE_CURRENT_A / DATASET_CURRENT_A,
               "soc_start_p95_pct": 100 * a.soc_start_p95, "soc_end_p05_pct": 100 * a.soc_end_p05}
        for st, tag in [(FRESH, "fresh"), (AGED, "aged")]:
            rec.update({f"{tag}_soc_start_pct": 100 * r[st].soc_start_median,
                        f"{tag}_soc_end_pct": 100 * r[st].soc_end_median,
                        f"{tag}_soc_span_pct": 100 * r[st].soc_span_median,
                        f"{tag}_width_mv": r[st].width_mv_median})
        if dev_p is None:
            rec.update(dict.fromkeys(DEV_COLS))
        else:
            d = dev_p.loc[(feature, basis)]
            rec.update({"dev_n_cycles": int(d.n_cycles), "dev_soc_start_pct": 100 * d.soc_start_median,
                        "dev_soc_end_pct": 100 * d.soc_end_median, "dev_soc_span_pct": 100 * d.soc_span_median,
                        "dev_width_mv": d.width_mv_median, "dev_minutes_at_4A": d.minutes_at_4A_median})
        full.append(rec)
    formats = {"Span (% of discharge)": fmt_float(1), "Voltage width (mV)": mv,
               "Minutes at 4 A": fmt_float(1)}

    # start, end and span are separate medians, so start - end is not exactly the span
    gap = max(abs(a.soc_start_median - a.soc_end_median - a.soc_span_median)
              for a in (row(summ, f) for _, f, basis in rows if basis == IMPL))
    shorter = 1 - DISCHARGE_CURRENT_A / DATASET_CURRENT_A
    notes = [
        f"Medians over all {req['n_cycles']:,} clean cycles of the {req['n_cells']} cells (ENG-17). A window is "
        "the stretch of the discharge dQ/dV curve that the feature's computation touches, always including the "
        f"apex detection window (apex outwards until |dQ/dV| has fallen by {req['detect_drop_ah_per_v']} Ah/V, "
        f"padded by {req['pad_points']} grid point per side); SoC(v) = 1 − Q(v)/Q_total, so a window reads "
        "start → end.",
        "Rows are merged where the features need the identical window in every life stage, and ordered by span. "
        f"Start, end and span are separate medians over cycles, so start − end differs from the span by up to "
        f"{100 * gap:.1f} points.",
        "Minutes at 4 A = median charge passed in the window / 4 A; at the dataset's 4C discharge "
        f"({DATASET_CURRENT_A:g} A) every time is {100 * shorter:.0f}% shorter.",
        f"Last row, '{VARIANT_LABEL}': fwhm's window between its two half-prominence crossings only (ENG-17 basis "
        "'value support'), roughly what a fixed-baseline fwhm would need; no model uses it.",
    ]
    return pd.DataFrame(disp), pd.DataFrame(full), formats, notes


def make(stem: Path) -> list[Path]:
    df, csv_df, formats, notes = build()
    sources = SOURCES + ([SOC_PARQUET] if csv_df["dev_n_cycles"].notna().any() else [])
    return write_table(stem, df, label=LABEL, title=TITLE, formats=formats, csv_df=csv_df,
                       notes=notes, sources=sources)
