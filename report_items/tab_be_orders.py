"""
Table S7. Backward elimination in each partition, without and with the 20-cycle history
The ENG-25 backward-elimination path of every partition (dev-CV of its own development cells, untuned
RF): the order in which features were dropped, the dev-CV with all features and with five, and the
five-feature set. Batch 3 out starts from the study-split value (ENG-19b path, pool order; ENG-23 path,
combined-pool order) as Tables IX and XII do, not from ENG-25's stored Pearson-order value; the CSV
keeps both (cv_all, cv_all_eng25_stored). Reads eng25_partition_robustness.json, eng19b_analysis.json
and eng23_grouped_history_subsets_analysis.json.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import (PARTITION_NAMES, PARTITIONS, fmt_pct, fmt_rmse, join_features, load_json,
                                 write_table)

LABEL = "Table S7"
TITLE = "Backward elimination in each partition, without and with the 20-cycle history"
SOURCES = ["eng25_partition_robustness.json", "eng19b_analysis.json",
           "eng23_grouped_history_subsets_analysis.json"]

MODES = [("level", "Nine level features"), ("grouped", "With 20-cycle history")]
PEARSON_START = "ENG-15 k-curve, Pearson order"


def study_starts() -> dict[str, tuple[float, str]]:
    """mode -> (all-features dev-CV of the study split's BE path, its source label)."""
    a19, a23 = load_json("eng19b_analysis.json"), load_json("eng23_grouped_history_subsets_analysis.json")
    k9 = {"level": {s["k"]: s["cv"] for s in a19["backward_elimination"]}[9],
          "grouped": {s["k"]: s["cv"] for s in a23["backward_elimination"]}[9]}   # k = 9 blocks = 63 columns
    return {"level": (k9["level"], "ENG-19b BE path, pool order"),
            "grouped": (k9["grouped"], "ENG-23 BE path, combined-pool order")}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The markdown table and the CSV (one row per mode and partition, full precision)."""
    be = load_json("eng25_partition_robustness.json")["backward_elimination"]
    study = study_starts()
    rows, csv = [], []
    for mode, label in MODES:
        for p in PARTITIONS:
            b = be[mode][p]
            steps = {s["k"]: s for s in b["steps"]}
            start, start_src = study[mode] if p == "lobo_b3" else (steps[9]["cv"], PEARSON_START)
            ch = (steps[5]["cv"] - start) / start
            rows.append({"Features": label, "Partition": PARTITION_NAMES[p],
                         "Elimination order (first dropped → last standing)":
                             ", ".join(b["drop_order"]) + " → " + b["last_standing"],
                         "Dev-CV, all features": start, "Dev-CV, five features": steps[5]["cv"],
                         "Change": ch, "Five-feature set": join_features(steps[5]["blocks"])})
            csv.append({"mode": mode, "partition": p, "drop_order": " > ".join(b["drop_order"]),
                        "last_standing": b["last_standing"], "cv_all": start, "cv_all_source": start_src,
                        "cv_all_eng25_stored": steps[9]["cv"], "cv_five": steps[5]["cv"],
                        "change_five_vs_all": ch, "five_set": join_features(steps[5]["blocks"]),
                        "cv_k9": start, **{f"cv_k{k}": steps[k]["cv"] for k in range(8, 0, -1)},
                        "steps_source": b["source"]})
    return pd.DataFrame(rows), pd.DataFrame(csv)


def notes(csv: pd.DataFrame) -> list[str]:
    """What dev-CV means here, where each all-features value comes from, how history is removed."""
    b3 = csv[csv["partition"] == "lobo_b3"].set_index("mode")

    def gap(mode: str) -> str:
        """Study-split start minus ENG-25's Pearson-order start, as '5.5e-6'."""
        return f"{abs(b3.loc[mode, 'cv_all_eng25_stored'] - b3.loc[mode, 'cv_all']):.1e}".replace("e-0", "e-")

    return [
        "Dev-CV here is each partition's own five-fold CV on its development cells (untuned RF); the "
        "partition's test cells were never scored. Partitions differ in absolute RMSE because their "
        "development sets differ.",
        "All features (nine level features, or the 63-column combined pool): batch 3 out starts from the "
        f"study-split value of ENG-19 ({b3.loc['level', 'cv_all']:.5f}, pool order, as in Table IX) and ENG-23 "
        f"({b3.loc['grouped', 'cv_all']:.5f}, combined-pool order, as in Table XII); the other partitions "
        "start from the stored ENG-15 k-curve value (Pearson column order), the only all-features value "
        f"stored for them. The two orders differ by the random-forest column-order effect ({gap('level')} and "
        f"{gap('grouped')} on the study split; ENG-25 and Fig. S1 use the Pearson-order value for batch 3 out too).",
        "The reduced sets are direct backward-elimination runs (ENG-25; level features in pool order, "
        "history sets in the ENG-14 combined-pool order), except batch 3 out with history, which is read "
        "off the ENG-23 table.",
        "With history each feature is removed together with its six history columns. Untuned only, as in "
        "ENG-15.",
    ]


def make(stem: Path) -> list[Path]:
    df, csv = build()
    formats = {"Dev-CV, all features": fmt_rmse, "Dev-CV, five features": fmt_rmse,
               "Change": lambda x: fmt_pct(x, 1, signed=True)}
    return write_table(stem, df, label=LABEL, title=TITLE, formats=formats, csv_df=csv,
                       notes=notes(csv), sources=SOURCES)
