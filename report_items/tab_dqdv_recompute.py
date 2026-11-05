"""
Table B1. The dQ/dV recomputed from the raw voltage and capacity against the published curve, and the second LFP peak
Per-cycle agreement (Pearson r) of the published dQ/dV with the one recomputed from the raw V and Q,
how often a second LFP peak appears in the 2.7–3.4 V search window, and the largest maximum below
2.7 V (not searched). Values from eng02_second_peak.json; the cells at 0.1 Ah/V, the IQR and the
2.35–2.55 V share below 2.7 V are computed from the status-ok rows of features_expanded.parquet.
The NASA rows of the original table (ENG-03, cells at about 1C) are omitted: ENG-03 is not part of
this repository.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import fmt_int, fmt_pct, load_json, result_path, write_table
from report_items.fig_recomputed_vs_published import LABEL as FIG_LABEL
from src.data_prep import clean_feature_table

LABEL = "Table B1"
TITLE = ("The dQ/dV recomputed from the raw voltage and capacity against the published curve, "
         "and the second LFP peak")
SOURCES = ["eng02_second_peak.json", "features_expanded.parquet"]

# the CSV 'source' column names the stored file each value comes from (as in the thesis scripts)
S02, PARQUET = "results/eng02_second_peak.json", "results/features_expanded.parquet"
EN = "–"   # en dash in ranges
GROUPS = ("Recomputed vs published dQ/dV (LFP, 4C)", "Second peak in 2.7–3.4 V (LFP, 4C)",
          "Largest maximum below 2.7 V (LFP, 4C; not searched)")


def recorded_cycles(columns: list[str]) -> pd.DataFrame:
    """The given columns of every recorded cycle (status ok or not) of the feature table."""
    return pd.read_parquet(result_path("features_expanded.parquet"), columns=columns)


def share(x) -> str:
    """Every share in Tables B1 and S3 to two decimals."""
    return fmt_pct(x, 2)


def statistics() -> dict:
    """eng02's stored blocks plus the row counts and the numbers only the feature table has."""
    e02 = load_json("eng02_second_peak.json")
    feat = recorded_cycles(["status", "SOH", "cell_id", "prom2", "tail_position2", "tail_prom2"])
    clean, _ = clean_feature_table(feat, verbose=False)
    ok = feat[feat["status"] == "ok"]
    # the largest local maximum below 2.7 V (outside the search window), stored per cycle as tail_*
    tpos, tprom = ok["tail_position2"], ok["tail_prom2"]
    big = tprom >= 0.3
    return {
        "ag": e02["agreement_recomputed_vs_published"], "sp": e02["second_peak"],
        "tail": e02["tail_artifact_below_2.7V"],
        "n_rec": len(feat), "n_ok": len(ok), "n_clean": len(clean),
        "n_cells": int(ok["cell_id"].nunique()),
        "n_cells_01": int(ok.loc[ok["prom2"] >= 0.1, "cell_id"].nunique()),
        "knee_share": float(tpos.lt(2.35).sum() / tpos.notna().sum()),
        "band_share": float(tpos.between(2.35, 2.55, inclusive="left").sum() / tpos.notna().sum()),
        "q25": float(tpos.quantile(0.25)), "q75": float(tpos.quantile(0.75)),
        "big_cells": int(ok.loc[big, "cell_id"].nunique()),
        "either_share": float((big | (ok["prom2"] >= 0.3)).mean()),
    }


def build(st: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table (group shown once, pre-formatted values) and the CSV (raw values)."""
    ag, sp, tail = st["ag"], st["sp"], st["tail"]
    sweep, q = sp["threshold_sweep"], sp["position2_quantiles_at_0.1"]
    g1, g2, g3 = GROUPS
    n_cells = st["n_cells"]
    rows = [   # (group, measure, value, unit, shown, source)
        (g1, "Cycles compared (every status-ok cycle)", ag["n_cycles_with_recompute"], "cycles",
         fmt_int(ag["n_cycles_with_recompute"]), f"{S02}: agreement_recomputed_vs_published"),
        (g1, "Per-cycle Pearson r: median", ag["pearson_r_median"], "r", f"{ag['pearson_r_median']:.4f}", S02),
        (g1, "Per-cycle Pearson r: 5th percentile", ag["pearson_r_p05"], "r", f"{ag['pearson_r_p05']:.4f}", S02),
        (g1, "Per-cycle Pearson r: 1st percentile", ag["pearson_r_p01"], "r", f"{ag['pearson_r_p01']:.4f}", S02),
        (g1, "Per-cycle Pearson r: minimum", ag["pearson_r_min"], "r", f"{ag['pearson_r_min']:.4f}", S02),
        (g1, "Cycles with r > 0.95", ag["frac_r_above_0.95"], "share", share(ag["frac_r_above_0.95"]), S02),
        (g2, "Any second local maximum (prominence ≥ 0.02 Ah/V)", sweep["0.02"]["published_smoothed"], "share",
         share(sweep["0.02"]["published_smoothed"]), f"{S02}: second_peak.threshold_sweep"),
        (g2, "Prominence ≥ 0.1 Ah/V", sweep["0.1"]["published_smoothed"], "share",
         share(sweep["0.1"]["published_smoothed"]), S02),
        (g2, "Prominence ≥ 0.3 Ah/V (the standard of the main peak)", sp["pct_cycles_resolvable_published"],
         "share", share(sp["pct_cycles_resolvable_published"]), S02),
        (g2, "Prominence ≥ 0.3 Ah/V on the recomputed curve", sp["pct_cycles_resolvable_recomputed"], "share",
         share(sp["pct_cycles_resolvable_recomputed"]), S02),
        (g2, "Cells with at least one cycle ≥ 0.3 Ah/V", sp["n_cells_with_any_resolvable"], "cells",
         f"{sp['n_cells_with_any_resolvable']} of {n_cells}", S02),
        (g2, "Cells with at least one cycle ≥ 0.1 Ah/V", st["n_cells_01"], "cells",
         f"{st['n_cells_01']} of {n_cells}", PARQUET),
        (g2, "Position of the second peak, cycles ≥ 0.1 Ah/V (median, IQR)", q["0.5"], "V",
         f"{q['0.5']:.2f} V ({q['0.25']:.2f}{EN}{q['0.75']:.2f})", S02),
        (g3, "Any local maximum (prominence ≥ 0.02 Ah/V)", tail["pct_cycles_with_tail_candidate"], "share",
         share(tail["pct_cycles_with_tail_candidate"]), f"{S02}: tail_artifact_below_2.7V"),
        (g3, "Position (median, IQR)", tail["tail_position_median"], "V",
         f"{tail['tail_position_median']:.2f} V ({st['q25']:.2f}{EN}{st['q75']:.2f})",
         f"{S02}: tail_artifact_below_2.7V; IQR from {PARQUET}"),
        (g3, "At 2.35–2.55 V (a bump separate from the knee upturn below 2.35 V)", st["band_share"], "share",
         share(st["band_share"]), PARQUET),
        (g3, "Prominence ≥ 0.3 Ah/V", tail["pct_tail_prom_above_0.3"], "share",
         f"{share(tail['pct_tail_prom_above_0.3'])} ({st['big_cells']} cells)",
         f"{S02}: tail_artifact_below_2.7V; cells from {PARQUET}"),
    ]
    csv = pd.DataFrame(rows, columns=["group", "measure", "value", "unit", "shown", "source"])
    md = pd.DataFrame({"Group": [g if i == 0 or g != rows[i - 1][0] else "" for i, (g, *_) in enumerate(rows)],
                       "Measure": csv["measure"], "Value": csv["shown"]})
    return md, csv.drop(columns="shown")


def notes(st: dict) -> list[str]:
    """Which cycles are compared, what the second peak is, and what the maxima below 2.7 V are."""
    n_rec, n_ok, n_clean = st["n_rec"], st["n_ok"], st["n_clean"]
    return [
        "Per-cycle Pearson r between the published dQ/dV and the one recomputed from the raw V and Q "
        "of the same cycle, over every cycle whose extraction succeeded (status ok). "
        f"These {n_ok:,} cycles are the {n_clean:,} clean cycles plus {n_ok - n_clean} that cleaning "
        f"removes later (SoH label missing or out of range); {n_rec - n_ok} of the {n_rec:,} recorded "
        f"cycles failed extraction ({n_rec - n_clean} removed in all).",
        "Second peak: the largest local maximum other than the main peak in the 2.7–3.4 V staging "
        "region of the published (smoothed) curve, the window ENG-02 searched. 0.3 Ah/V is the "
        "prominence the pipeline requires of the main peak.",
        "Below 2.7 V ENG-02 recorded the largest local maximum separately and did not count it: the "
        f"upturn at the discharge knee below 2.35 V ({fmt_pct(st['knee_share'], 0)} of cycles) or a "
        f"small bump at 2.35–2.55 V ({fmt_pct(st['band_share'], 0)}), seen in both curves in "
        f"{FIG_LABEL}. Counted as a second peak, a maximum would clear the main-peak standard in "
        f"{fmt_pct(st['either_share'], 1)} of cycles, still too few to use a second peak as a feature "
        "at 4C.",
        "The original table's comparison with NASA cells at about 1C (ENG-03) is omitted; ENG-03 is "
        "not part of this repository.",
    ]


def make(stem: Path) -> list[Path]:
    st = statistics()
    md, csv = build(st)
    return write_table(stem, md, label=LABEL, title=TITLE, csv_df=csv, notes=notes(st),
                       sources=SOURCES)
