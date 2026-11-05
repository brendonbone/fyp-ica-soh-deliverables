"""
Table S3. Share of cycles with a second dQ/dV peak at or above each prominence threshold
The ENG-02 prominence sweep: share of the status-ok LFP cycles whose second local maximum in the
2.7–3.4 V window reaches each threshold, on the published curve and on the curve recomputed from
the raw V and Q. Values from eng02_second_peak.json; the cell count in the note from
features_expanded.parquet. The 'NASA ~1C' column of the original table (ENG-03) is omitted: ENG-03
is not part of this repository.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import load_json, write_table
from report_items.tab_dqdv_recompute import recorded_cycles, share

LABEL = "Table S3"
TITLE = "Share of cycles with a second dQ/dV peak at or above each prominence threshold"
SOURCES = ["eng02_second_peak.json", "features_expanded.parquet"]

THRESHOLDS = ["0.02", "0.05", "0.1", "0.2", "0.3"]   # Ah/V, the keys of eng02's threshold_sweep
COL_PUB, COL_REC = "LFP 4C, published curve", "LFP 4C, recomputed curve"


def build() -> tuple[pd.DataFrame, dict]:
    """One row per threshold (raw fractions) and the markdown formats."""
    sweep = load_json("eng02_second_peak.json")["second_peak"]["threshold_sweep"]
    df = pd.DataFrame({"Prominence threshold (Ah/V)": THRESHOLDS,
                       COL_PUB: [sweep[t]["published_smoothed"] for t in THRESHOLDS],
                       COL_REC: [sweep[t]["recomputed"] for t in THRESHOLDS]})
    return df, {COL_PUB: share, COL_REC: share}


def notes() -> list[str]:
    """Which cycles and which search window."""
    n_cycles = load_json("eng02_second_peak.json")["agreement_recomputed_vs_published"]["n_cycles_with_recompute"]
    cells = recorded_cycles(["status", "cell_id"])
    n_cells = int(cells.loc[cells["status"] == "ok", "cell_id"].nunique())
    return [f"LFP: the {n_cycles:,} status-ok cycles of the {n_cells} Severson cells, second maximum "
            "searched in 2.7–3.4 V (ENG-02). The original table's 'NASA ~1C' column (ENG-03) is omitted; "
            "ENG-03 is not part of this repository."]


def make(stem: Path) -> list[Path]:
    df, formats = build()
    return write_table(stem, df, label=LABEL, title=TITLE, formats=formats, notes=notes(),
                       sources=SOURCES)
