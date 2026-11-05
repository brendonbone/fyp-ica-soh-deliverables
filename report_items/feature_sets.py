"""
Shared pieces of Tables XIV, XV and S8: the ENG-28 cells and the set names.
===========================================================================
ENG-28 scores 11 feature sets x 4 models (LR, untuned GPR, untuned RF, tuned
RF) by five-fold development CV on the ENG-14 history rows, so the sets with
and without history share rows. This helper (not a report item, not in
run_report.py ORDER) loads those cells once, names each set in words
(`describe`, also used for the batch-3 tables) and builds the rows of
Tables XIV and XV (`config_table`). Reads results/eng28_feature_sets_models.json
and results/eng14_full.json (history-row counts for the notes).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from report_items.common import FEATURES, fmt_pct, fmt_rmse, join_features, load_json

E28 = "eng28_feature_sets_models.json"
E14 = "eng14_full.json"

MODELS = ["LR", "GPR", "RF", "RF tuned"]   # ENG-28's model keys (JSON / CSV)
HEAD = {"LR": "LR", "GPR": "GPR", "RF": "RF", "RF tuned": "Tuned RF"}   # column names of Tables XIV, XV
REFERENCE = "all9"      # every change in Tables XIV and XV is against this set under the same model
INTERIM = "interim3"
CHOICES = ["interim3", "best4", "budget7"]
LEVEL_ROWS = [REFERENCE] + CHOICES
HIST_ROWS = ["interim3_hist", "best4_hist", "budget7_hist"]
CONFIG = {"all9": "All nine features (reference model, C)",
          "interim3": "Interim model features", "best4": "Best subset, one-SE rule (D)",
          "budget7": "Discharge budget, 42.3% (B, F)",
          "interim3_hist": "Interim model features + history", "best4_hist": "Best subset + history",
          "budget7_hist": "Discharge budget + history"}
COMPONENT_ORDER = ["level", "rm20", "drift", "slope20", "slope10", "d20", "d10"]


def describe(fs: dict) -> str:
    """An ENG-28 feature set in words, from its blocks ('all but fwhm, prominence + history')."""
    b = fs["blocks"]
    if fs["kind"] == "component":
        return "all nine: " + " + ".join(sorted(b, key=COMPONENT_ORDER.index))
    if len(b) == 9:
        base = "all nine features" if fs["kind"] == "level" else "all nine"
    elif len(b) >= 6:
        base = "all but " + join_features([f for f in FEATURES if f not in b])
    else:
        base = join_features(b)
    return base if fs["kind"] == "level" else f"{base} + history"


def pct(x: float) -> str:
    """Signed change, 1 dp: '+4.6%', '−1.5%'."""
    return fmt_pct(x, nd=1, signed=True)


@dataclass
class Cells:
    """The ENG-28 cells indexed by set and model, with the derived frames."""
    sets: dict[str, dict]                 # set id -> feature set (blocks, kind, soc_window, ...)
    cells: dict[tuple[str, str], dict]    # (set id, model key) -> cell
    cv: pd.DataFrame                      # dev-CV RMSE, set x model
    own: pd.DataFrame                     # relative change against all9 under the same model
    interim_rf_cv: float
    tuned_rf_params: dict

    def wins(self, a: tuple[str, str], b: tuple[str, str]) -> int:
        """Folds (of 5) in which cell a has a lower RMSE than cell b (same fold map)."""
        fa = np.asarray(self.cells[a]["cv_fold_rmse"])
        fb = np.asarray(self.cells[b]["cv_fold_rmse"])
        return int((fa < fb).sum())


@lru_cache(maxsize=1)
def load_cells() -> Cells:
    """Read ENG-28 once."""
    d = load_json(E28)
    sets = {fs["id"]: fs for fs in d["feature_sets"]}
    cells = {(c["set"], c["model"]): c for c in d["cells"]}
    cv = pd.DataFrame({m: [cells[(s, m)]["cv_pooled_rmse_mean"] for s in sets] for m in MODELS},
                      index=list(sets))
    own = (cv - cv.loc[REFERENCE]) / cv.loc[REFERENCE]
    return Cells(sets, cells, cv, own, d["interim_rf_cv"], d["protocol"]["tuned_rf_params"])


def history_rows_text() -> str:
    """The rows every ENG-28 cell is scored on, in words (for the notes)."""
    hf = load_json(E14)["history_filter"]["dev"]
    return (f"the {int(hf['rows_with_history']):,} development cycles (of {int(hf['rows']):,}) at least 20 "
            "cycles after their cell's first clean cycle (ENG-14 history rows, ENG-08 cell-to-fold map)")


# ------------------------ Tables XIV and XV ------------------------

def feature_text(fs: dict) -> str:
    """Selected-features cell: the features, plus the history columns for a history set."""
    feats = join_features(fs["blocks"])
    return feats if fs["kind"] == "level" else f"{feats} + 20-cycle history ({fs['n_columns']} columns)"


def config_table(ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of Table XIV or XV: the shown table and the full-precision CSV table."""
    g = load_cells()
    show, csv_rows = [], []
    for sid in ids:
        fs = g.sets[sid]
        is_ref = sid == REFERENCE
        r = {"Configuration": CONFIG[sid], "Selected features": feature_text(fs),
             "Discharge budget (SoC span)": fs["soc_window"]["soc_span"]}
        c = {"set_id": sid, "configuration": CONFIG[sid], "features": join_features(fs["blocks"]),
             "history": fs["kind"] != "level", "n_features": len(fs["blocks"]),
             "n_columns": fs["n_columns"], "soc_span": fs["soc_window"]["soc_span"],
             "soc_start": fs["soc_window"]["soc_start"], "soc_end": fs["soc_window"]["soc_end"]}
        for m in MODELS:
            v = g.cv.loc[sid, m]
            r[f"{HEAD[m]} dev-CV"] = v
            r[f"{HEAD[m]} Δ"] = None if is_ref else g.own.loc[sid, m]
            key = m.lower().replace(" ", "_")
            c[f"{key}_cv_rmse"] = v
            c[f"{key}_cv_rmse_fold_sd"] = g.cells[(sid, m)]["cv_pooled_rmse_std"]
            c[f"{key}_reference_cv_rmse"] = g.cv.loc[REFERENCE, m]
            c[f"{key}_change_vs_reference"] = 0.0 if is_ref else g.own.loc[sid, m]
            # None on the reference row makes this column float in Table XIV's CSV ('1.0')
            c[f"{key}_folds_below_reference"] = None if is_ref else g.wins((sid, m), (REFERENCE, m))
        show.append(r)
        csv_rows.append(c)
    return pd.DataFrame(show), pd.DataFrame(csv_rows)


def delta(x) -> str:
    """Change column: 'reference' on the reference row, else '+4.6%'."""
    return "reference" if x is None or (isinstance(x, float) and np.isnan(x)) else pct(x)


CONFIG_FORMATS = {"Discharge budget (SoC span)": lambda x: fmt_pct(x, 1),
                  **{f"{HEAD[m]} dev-CV": fmt_rmse for m in MODELS},
                  **{f"{HEAD[m]} Δ": delta for m in MODELS}}


def dev_cv_note() -> str:
    """What dev-CV is and which rows it is measured on (Tables XIV and XV)."""
    return ("Dev-CV = pooled RMSE of SoH averaged over the five development folds, in SoH units (fraction "
            f"of nominal capacity, p.u.). All cells use the same rows: {history_rows_text()}, so the tables "
            "with and without history compare like for like.")


MODELS_NOTE = ("LR = linear regression on standardised inputs; GPR = untuned GPR (constant × isotropic RBF + "
               "white-noise kernel, hyperparameters by marginal likelihood, 2000-row per-cell training "
               "subsample per fold); RF = untuned random forest (300 trees, scikit-learn defaults); Tuned RF = "
               "the same forest with the fixed ENG-09 configuration (max_features 0.5, chosen once on the nine "
               "level features and not re-tuned per feature set). Seed 0.")
