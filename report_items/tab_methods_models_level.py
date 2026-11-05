"""
Table XIV. Dev-CV RMSE of three feature choices (level features only) under four models
=====================================================================================
The interim features, the best subset of D and the discharge-budget set, each
under LR, untuned GPR, untuned RF and tuned RF (ENG-28), and the change (Δ)
against the same model on all nine level features (the first row), on the same
history rows. The CSV adds the fold sd and the number of the five folds in
which the set beats the reference. Reads eng28_feature_sets_models.json and
eng14_full.json (row counts for the note).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import write_table
from report_items.feature_sets import CONFIG_FORMATS, E14, E28, LEVEL_ROWS, MODELS_NOTE, config_table, dev_cv_note

LABEL = "Table XIV"
TITLE = ("Dev-CV RMSE of three feature choices (level features only) under four models, and the change "
         "against the same model on all nine features (the reference model)")
SOURCES = [E28, E14]


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The shown table, the full-precision CSV table and the notes."""
    show, csv = config_table(LEVEL_ROWS)
    notes = [
        "Δ = change in dev-CV RMSE against the same model trained on all nine level features (the reference "
        "model of C; the first row of Table XIV), on the same rows. Negative = lower error (better).",
        dev_cv_note(),
        MODELS_NOTE,
        "Discharge budget = median over cycles of the per-cycle union of the features' SoC windows "
        "(ENG-17, Table VI).",
        "The best subset and the discharge-budget set were chosen by untuned-RF dev-CV searches on these "
        "folds (D, F), so their RF scores are mildly optimistic relative to LR and GPR.",
    ]
    return show, csv, notes


def make(stem: Path) -> list[Path]:
    show, csv, notes = build()
    return write_table(stem, show, label=LABEL, title=TITLE, formats=CONFIG_FORMATS, csv_df=csv,
                       notes=notes, sources=SOURCES)
