"""
Table XV. Dev-CV RMSE of the same three feature choices with each feature's 20-cycle history
============================================================================================
The three feature choices of Table XIV, each with its six history columns per
feature, under the same four models (ENG-28). Δ is still against the same
model on all nine level features WITHOUT history (Table XIV's first row), not
against all nine with history, so it holds the effect of the feature choice
and the history together. Reads eng28_feature_sets_models.json and
eng14_full.json (row counts for the note).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import write_table
from report_items.feature_sets import CONFIG_FORMATS, E14, E28, HIST_ROWS, MODELS_NOTE, config_table, dev_cv_note

LABEL = "Table XV"
TITLE = ("Dev-CV RMSE of the same three feature choices with each feature's 20-cycle history, and the change "
         "against the same model on all nine features without history (the reference model)")
SOURCES = [E28, E14]


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The shown table, the full-precision CSV table and the notes."""
    show, csv = config_table(HIST_ROWS)
    notes = [
        "Δ = change in dev-CV RMSE against the same model on all nine level features without history (the "
        "reference row of Table XIV), on the same rows: the effect of the feature choice and the history "
        "together. Negative = lower error (better).",
        "Each feature brings six history columns: the 20-cycle rolling mean, the drift since the cell's first "
        "clean cycle, and the slope and change over the last 20 and 10 cycles (E). A history set needs the "
        "same discharge window as its level set, in each of the previous 20 cycles as well.",
        dev_cv_note(),
        MODELS_NOTE,
    ]
    return show, csv, notes


def make(stem: Path) -> list[Path]:
    show, csv, notes = build()
    return write_table(stem, show, label=LABEL, title=TITLE, formats=CONFIG_FORMATS, csv_df=csv,
                       notes=notes, sources=SOURCES)
