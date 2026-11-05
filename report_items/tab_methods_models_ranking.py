"""
Table S8. Every feature set × model combination, ranked by dev-CV RMSE
=====================================================================
The 44 ENG-28 cells (11 feature sets x 4 models), sorted by dev-CV RMSE, with
two changes: against the interim RF (ENG-28's cv_vs_interim_rf_rel, model and
features together) and against all nine level features under the same model
(the Δ of Tables XIV and XV, the features alone). Rank is ENG-28's cv_rank;
rank in model ranks the 11 sets within one model (ties share the lower rank).
Reads eng28_feature_sets_models.json and eng14_full.json (row counts for the note).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_int, fmt_pct, fmt_pm, fmt_r, write_table
from report_items.feature_sets import (E14, E28, INTERIM, REFERENCE, describe, history_rows_text, load_cells,
                                       pct)

LABEL = "Table S8"
TITLE = "Every feature set × model combination, ranked by dev-CV RMSE"
SOURCES = [E28, E14]

MODEL_LABEL = {"LR": "LR", "GPR": "GPR, untuned", "RF": "RF, untuned", "RF tuned": "RF, tuned"}
# Sets picked by an untuned-RF dev-CV search on the ENG-08 folds (ENG-19b,
# ENG-22, ENG-23) or built from such a pick (CSV column rf_selected_set).
RF_SELECTED = ("best4", "budget7", "lvl_rm_drift", "hist_rule", "hist_budget", "best4_hist", "budget7_hist")
# ENG-28 calls the all-nine set "(reference)"; here it is named after C's model.
METHOD_TABLE = {"all9": "expanded nine-feature pool (inputs of the C reference model)"}


def pct_ref(x: float) -> str:
    """Change columns: the reference cell itself reads '0 (ref.)'."""
    return "0 (ref.)" if x == 0 else pct(x)


def methods(sets: dict[str, dict]) -> dict[str, str]:
    """Method text per set; budget7's two left-out features renamed in pool order."""
    excl7 = [f for f in FEATURES if f not in sets["budget7"]["blocks"]]
    budget7 = re.sub(r"without (\w+) and (\w+)", "without " + " and ".join(excl7), sets["budget7"]["method"])
    named = dict(METHOD_TABLE, budget7=budget7)
    return {sid: f"{fs['section']}: {named.get(sid, fs['method'])}" for sid, fs in sets.items()}


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The full-precision CSV table (one row per cell, in rank order), the shown table and the notes."""
    g = load_cells()
    method = methods(g.sets)
    ranks = g.cv.rank(axis=0, method="min").astype(int)
    order = sorted(g.cells.values(), key=lambda c: (c["cv_pooled_rmse_mean"], c["set"], c["model"]))
    rows = []
    for c in order:
        fs = g.sets[c["set"]]
        k = (c["set"], c["model"])
        rows.append({
            "rank": c["cv_rank"], "set_id": c["set"], "feature_set": describe(fs),
            "method": method[c["set"]],
            "model": MODEL_LABEL[c["model"]], "model_key": c["model"],
            "n_features": len(fs["base_features"]), "n_columns": c["n_columns"],
            "soc_span": fs["soc_window"]["soc_span"],
            "cv_rmse": c["cv_pooled_rmse_mean"], "cv_rmse_fold_sd": c["cv_pooled_rmse_std"],
            "cv_r2": c["cv_pooled_r2_mean"], "cv_mae": c["cv_pooled_mae_mean"],
            "vs_interim_rf": c["cv_vs_interim_rf_rel"],
            "folds_below_interim_rf": g.wins(k, (INTERIM, "RF")),
            "vs_all9_same_model": g.own.loc[k],
            "folds_below_all9_same_model": g.wins(k, (REFERENCE, c["model"])),
            "rank_in_model": int(ranks.loc[k]),
            "rf_selected_set": c["set"] in RF_SELECTED,
        })
    full = pd.DataFrame(rows)
    show = pd.DataFrame({
        "Rank": full["rank"], "Feature set": full["feature_set"], "Subsection: method": full["method"],
        "Model": full["model"], "Features": full["n_features"], "Columns": full["n_columns"],
        "Discharge needed": full["soc_span"],
        "Dev-CV RMSE ± fold sd": [fmt_pm(a, b) for a, b in zip(full["cv_rmse"], full["cv_rmse_fold_sd"])],
        "Dev-CV R²": full["cv_r2"], "vs interim RF": full["vs_interim_rf"],
        "vs all nine, same model": full["vs_all9_same_model"],
        "Rank in model": full["rank_in_model"],
    })
    params = ", ".join(f"{k} {v}" for k, v in sorted(g.tuned_rf_params.items()))
    notes = [
        "Dev-CV = pooled RMSE of SoH averaged over the five development folds (± population sd over the "
        "folds), in SoH units (fraction of nominal capacity, p.u.); R² likewise. All "
        f"{len(full)} cells use the same rows: {history_rows_text()}, so level and history sets are compared "
        "like for like.",
        f"Models: {MODEL_LABEL['LR']} = linear regression on standardised inputs; {MODEL_LABEL['GPR']} = "
        "constant × isotropic RBF + white-noise kernel on standardised inputs, hyperparameters by marginal "
        "likelihood (no kernel search, no ARD), trained on a 2000-row per-cell stratified subsample per fold; "
        f"{MODEL_LABEL['RF']} = 300 trees, scikit-learn defaults otherwise; {MODEL_LABEL['RF tuned']} = 300 "
        f"trees, the fixed ENG-09 configuration ({params}). Seed 0.",
        "Features = ICA features measured; Columns = model inputs (a history set has 7 columns per feature, "
        "the component set 3 per feature). Discharge needed = median over cycles of the per-cycle union of "
        "the features' SoC windows (ENG-17); a history set needs that window in each of the 20 previous "
        "cycles as well.",
        "vs interim RF = change in dev-CV RMSE against the untuned RF on the interim three features "
        f"({g.interim_rf_cv:.5f}), which mixes the effect of the model with that of the features; vs all nine, "
        "same model = change against all nine level features under the same model, on the same rows (the Δ "
        "of Tables XIV and XV), the effect of the features alone. Negative is better; 0 (ref.) marks the "
        f"reference cells. Rank in model = rank of the set among the {len(g.sets)} under that model.",
        "Selection: the D, B/F, E-component and F sets (and the D and B/F sets with history) were chosen by "
        "untuned-RF dev-CV searches on these five folds, and the tuned-RF configuration by nested CV on the "
        "same development cells (ENG-09), so their RF scores are mildly optimistic relative to LR and GPR.",
    ]
    return full, show, notes


def make(stem: Path) -> list[Path]:
    full, show, notes = build()
    formats = {"Rank": fmt_int, "Features": fmt_int, "Columns": fmt_int, "Rank in model": fmt_int,
               "Discharge needed": lambda x: fmt_pct(x, 1), "Dev-CV R²": lambda x: fmt_r(x, 3),
               "vs interim RF": pct_ref, "vs all nine, same model": pct_ref}
    return write_table(stem, show, label=LABEL, title=TITLE, formats=formats, csv_df=full, bold_rows=[0],
                       notes=notes, sources=SOURCES)
