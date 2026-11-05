"""
Table S4. Training without the 16 cells whose published dQ/dV is noisy (ENG-27)
Tuned RF trained on every development cell and without the 16 noisy cells, for three feature sets:
dev-CV scored on the clean validation cells only (so both training sets are scored on the same
cells) and batch-3 RMSE, mean ± sd over five forest seeds. Everything comes from
eng27_noisy_cells.json: the summary block for the rows, the per-cell height steps (cells[]) and the
training-row counts for the note that defines a noisy cell.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import fmt_pct, fmt_rmse, load_json, minus, write_table

LABEL = "Table S4"
TITLE = "Training without the 16 cells whose published dQ/dV is noisy (ENG-27)"
SOURCES = ["eng27_noisy_cells.json"]

MODELS = {"level9": "All nine level features (the reference)",
          "hp_hist": "Two features with their history: height + position",
          "all63": "All 63 columns (level + history)"}
EN = "–"   # en dash in ranges


def pct_range(lo, hi, nd: int = 1) -> str:
    """0.101, 0.259 -> '10.1–25.9%'."""
    return minus(f"{100 * lo:.{nd}f}") + EN + fmt_pct(hi, nd)


def build(e27: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table (pre-formatted strings) and the CSV (one row per model, raw values)."""
    rows, csv = [], []
    for model, label in MODELS.items():
        rec = e27["summary"][model]
        a, w = rec["all_dev"], rec["without_noisy"]
        ccv, cb3 = rec["change_cv_clean_rmse"], rec["change_b3_rmse"]
        worse = cb3["n_seeds"] - cb3["seeds_lower"]
        rows.append({
            "Model (tuned RF)": label, "Columns": rec["n_columns"],
            "Training cells (rows)":
                f"{a['train_cells']} ({a['train_rows']:,}) / {w['train_cells']} ({w['train_rows']:,})",
            "Dev-CV on the clean cells: all / without the 16":
                f"{fmt_rmse(a['cv_clean_rmse']['mean'])} / {fmt_rmse(w['cv_clean_rmse']['mean'])}",
            "Dev-CV change": fmt_pct(ccv["relative_mean"], 1, signed=True),
            "Seeds with lower dev-CV": f"{ccv['seeds_lower']} of {ccv['n_seeds']}",
            "Batch-3 RMSE, trained on all": f"{fmt_rmse(a['b3_rmse']['mean'])} ± {fmt_rmse(a['b3_rmse']['sd'])}",
            "Batch-3 RMSE, without the 16": f"{fmt_rmse(w['b3_rmse']['mean'])} ± {fmt_rmse(w['b3_rmse']['sd'])}",
            "Batch-3 change": fmt_pct(cb3["relative_mean"], 1, signed=True),
            "Seeds worse on batch 3": f"{worse} of {cb3['n_seeds']}"})
        csv.append({"model": model, "label": label, "n_columns": rec["n_columns"],
                    "train_cells_all": a["train_cells"], "train_rows_all": a["train_rows"],
                    "train_cells_without": w["train_cells"], "train_rows_without": w["train_rows"],
                    "cv_clean_all": a["cv_clean_rmse"]["mean"], "cv_clean_without": w["cv_clean_rmse"]["mean"],
                    "cv_clean_change_rel": ccv["relative_mean"], "cv_clean_seeds_lower": ccv["seeds_lower"],
                    "cv_all_cells_all": a["cv_all_rmse"]["mean"],
                    "cv_noisy_cells_all": a["cv_noisy_rmse"]["mean"],
                    "b3_rmse_all": a["b3_rmse"]["mean"], "b3_rmse_all_sd": a["b3_rmse"]["sd"],
                    "b3_rmse_without": w["b3_rmse"]["mean"], "b3_rmse_without_sd": w["b3_rmse"]["sd"],
                    "b3_change_rel": cb3["relative_mean"], "seeds_worse_b3": worse, "n_seeds": cb3["n_seeds"]})
    return pd.DataFrame(rows), pd.DataFrame(csv)


def notes(e27: dict, csv: pd.DataFrame) -> list[str]:
    """The noisy-cell rule, how the models are scored, and how much the noisy cells inflate dev-CV."""
    noisy = set(e27["noisy_cells"])
    steps = pd.Series({c["cell_id"]: c["height_step"] for c in e27["cells"]})   # all 124 cells
    noisy_steps, clean_steps = steps[steps.index.isin(noisy)], steps[~steps.index.isin(noisy)]
    lv = e27["summary"]["level9"]
    n_dev = lv["all_dev"]["train_rows"]                        # every development row
    n_noisy_rows = n_dev - lv["without_noisy"]["train_rows"]   # the rows of the noisy cells
    over = csv["cv_all_cells_all"] / csv["cv_clean_all"] - 1
    return [
        f"Noisy cell: the median change of `height` between consecutive successful cycles exceeds "
        f"{e27['rule']['threshold']:.0%} of the cell's median height (no SoH label used). {len(noisy)} cells "
        f"qualify, b1c38–b1c45 and b2c40–b2c47 ({pct_range(noisy_steps.min(), noisy_steps.max())}); the "
        f"largest value among the other {len(clean_steps)} cells is {clean_steps.max():.1%} (median "
        f"{clean_steps.median():.1%}). All {len(noisy)} are development cells ({n_noisy_rows:,} of the "
        f"{n_dev:,} development rows); batch 3 is the same 40 cells either way.",
        "Tuned RF (ENG-09 configuration), forest seeds 0–4, mean ± sd over the seeds. Dev-CV: the ENG-08 "
        f"cell-to-fold map, scored on the {lv['without_noisy']['train_cells']} clean validation cells only, "
        "so both training sets are compared on the same cells. Rows: ENG-08 rows for the nine level "
        "features, history rows for the others. Batch 3 decides nothing here.",
        "The noisy cells inflate the published dev-CV: on their own they score "
        + " / ".join(fmt_rmse(x) for x in csv["cv_noisy_cells_all"]) + ", so the dev-CV over all cells ("
        + " / ".join(fmt_rmse(x) for x in csv["cv_all_cells_all"]) + ") is "
        + pct_range(over.min(), over.max(), 0) + " above the clean-cell value.",
    ]


def make(stem: Path) -> list[Path]:
    e27 = load_json("eng27_noisy_cells.json")
    md, csv = build(e27)
    return write_table(stem, md, label=LABEL, title=TITLE, csv_df=csv, notes=notes(e27, csv),
                       sources=SOURCES)
