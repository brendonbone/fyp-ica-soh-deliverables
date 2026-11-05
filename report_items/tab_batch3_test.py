"""
Table XVI. Batch-3 RMSE, R² and MAE of every configuration selected in Subsections C–G, beside its dev-CV RMSE
The one-time test on the held-out batch: every configuration chosen on dev-CV in C-G (or fixed in
advance) with its dev-CV RMSE and its stored batch-3 scores. Nothing is averaged here: each number
is copied from the result file that scored it (the CSV columns dev_cv_source / b3_source say which),
and a row may take its dev-CV and its batch 3 from different files (the tuned GPR: ENG-08 dev-CV,
ENG-09 batch 3; the untuned 'level + rm20 + drift' forest: ENG-22 dev-CV, ENG-28 batch 3). The ENG-28
sets are named with feature_sets.describe, the other subsets with the D/F wording below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_pm, join_features, load_csv, load_json, write_table
from report_items.feature_sets import describe as describe_set
from src.data_prep import BASELINE_FEATURES

LABEL = "Table XVI"
TITLE = ("Batch-3 RMSE, R² and MAE of every configuration selected in Subsections C–G, beside its "
         "dev-CV RMSE")

E00 = "eng00_baseline.json"
E0607 = "eng06_07_model_table.csv"
E08T = "eng08_ranking_table_tuned.csv"
E09 = "eng09_tuning.json"
E19 = "eng19_exhaustive_subsets.csv"
A19 = "eng19b_analysis.json"
E20 = "eng20_reference_seeds.json"
T22 = "eng22_history_components.csv"
A22 = "eng22_history_components_analysis.json"
T23 = "eng23_grouped_history_subsets.csv"
A23 = "eng23_grouped_history_subsets_analysis.json"
T24 = "eng24_level_subsets_history_rows.csv"
A24 = "eng24_level_subsets_history_rows_analysis.json"
E28 = "eng28_feature_sets_models.json"
SOURCES = sorted({E00, E0607, E08T, E09, E19, A19, E20, T22, A22, T23, A23, T24, A24, E28})

MODEL_LABEL = {"RF": "RF, untuned", "RF tuned": "RF, tuned"}
NOTES = [
    "Dev-CV = pooled RMSE averaged over the five development folds (batches 1 + 2); batch 3 = one refit on "
    "all development rows, scored once on the batch-3 rows (R² and MAE are batch 3). Rows: full = all clean "
    "cycles; history = cycles with at least 20 earlier clean cycles, so the E, F and G rows are compared on "
    "the same rows.",
    "Bold: the reference model of C, the baseline the other rows are compared with (mean ± sd over 20 "
    "forests, seeds 0–9 × pool and Pearson column order, ENG-20); every other RF row is one 300-tree forest "
    "with seed 0. No row was chosen on batch 3: every configuration was selected on dev-CV (or fixed in "
    "advance) in C–G.",
    "Rule pick = the one-standard-error rule on the subset search; Budget = the best subset needing at most "
    "that share of the discharge (SoC span). † Observation, not a selection: ENG-22 scored each "
    "drop-one-component design on batch 3 as a listed design.",
]


# ------------------------ lookups ------------------------

def at(name: str, where: str = "") -> str:
    """'results/<file> <where>': the dev_cv_source / b3_source entries of the CSV."""
    return f"results/{name} {where}".rstrip()


def key_of(blocks, names=FEATURES) -> str:
    """Canonical subset key, as in the ENG-19/22/23/24 tables ('height+position+...')."""
    return "+".join(sorted(blocks, key=list(names).index))


def one(hits: list, what: str):
    """Exactly one match, or a clear error."""
    if len(hits) != 1:
        raise KeyError(f"{what}: {len(hits)} matches")
    return hits[0]


def heldout(A: dict, config: str, blocks) -> dict:
    """The analysis file's batch-3 record of a subset (config 'untuned' or 'tuned')."""
    want = frozenset(blocks)
    return one([r for r in A["heldout"] if r["config"] == config and frozenset(r["blocks"]) == want],
               f"heldout {config} {sorted(want)}")


def tuned(A: dict, blocks) -> dict:
    """The analysis file's tuned re-score of a subset (dev-CV and batch 3)."""
    want = frozenset(blocks)
    return one([r for r in A["tuned"].values() if frozenset(r["blocks"]) == want], f"tuned {sorted(want)}")


def describe(blocks) -> str:
    """A subset in words, the D/F rule: the features for up to six, 'all but ...' for seven or
    eight, 'all nine' for nine."""
    blocks = list(blocks)
    if len(blocks) == len(FEATURES):
        return "all nine"
    if len(blocks) >= 7:
        return "all but " + join_features([f for f in FEATURES if f not in blocks])
    return join_features(blocks)


def cap(s: str) -> str:
    """First letter upper case, for a label that starts a table cell."""
    return s[:1].upper() + s[1:]


def cv_of(name: str, key_col: str) -> dict:
    """A subset-search CSV reduced to key -> dev-CV RMSE."""
    t = load_csv(name)
    return dict(zip(t[key_col], t["cv_pooled_rmse_mean"]))


def b3(r: dict) -> dict:
    return {"rmse": r["b3_rmse"], "r2": r["b3_r2"], "mae": r["b3_mae"]}


def b3_pooled(p: dict) -> dict:
    return {"rmse": p["rmse"], "r2": p["r2"], "mae": p["mae"]}


# ------------------------ the table ------------------------

def build() -> pd.DataFrame:
    """One row per configuration x model in the report's order, full precision (the CSV)."""
    e00, e09, e20, e28 = load_json(E00), load_json(E09), load_json(E20), load_json(E28)
    a19, a22, a23, a24 = load_json(A19), load_json(A22), load_json(A23), load_json(A24)
    lin = load_csv(E0607).set_index("regressor")
    g8 = load_csv(E08T).query("selection_method == 'pearson' and regressor == 'gpr'").iloc[0]
    t19, t22, t23, t24 = cv_of(E19, "features"), cv_of(T22, "key"), cv_of(T23, "key"), cv_of(T24, "key")
    L = {fs["id"]: describe_set(fs) for fs in e28["feature_sets"]}
    cell = {(c["set"], c["model"]): c for c in e28["cells"]}
    comp_names = a22["names"]
    all7 = list(comp_names)
    all9 = list(FEATURES)
    pool, hist = "pool order, seed 0", "combined-pool order, seed 0"
    rows: list[dict] = []

    def add(rid, sec, config, feats, model, rows_, ncols, cv, b3_, *, cv_sd=None, b3_sd=None, r2_sd=None,
            mae_sd=None, reason="", cv_src="", b3_src="", order="", obs=False, src=""):
        rows.append({"row_id": rid, "section": sec, "configuration": config, "features": join_features(feats),
                     "model": model, "rows": rows_, "n_columns": ncols,
                     "dev_cv_rmse": cv, "dev_cv_rmse_sd": cv_sd,
                     "b3_rmse": b3_["rmse"], "b3_rmse_sd": b3_sd, "b3_r2": b3_["r2"], "b3_r2_sd": r2_sd,
                     "b3_mae": b3_["mae"], "b3_mae_sd": mae_sd, "b3_reason": reason,
                     "column_order": order, "observation": obs, "source": src,
                     "dev_cv_source": cv_src, "b3_source": b3_src})

    # ---- C: reference model ----
    add("interim", "C", f"Interim three features ({L['interim3']})", BASELINE_FEATURES,
        "RF, untuned", "full", 3, e00["cv"]["pooled_summary"]["rmse"]["mean"],
        b3_pooled(e00["batch3_ood"]["pooled"]), reason="interim baseline (ENG-00)",
        cv_src=at(E00, "cv.pooled_summary"), b3_src=at(E00, "batch3_ood"), order=pool, src="ENG-00")
    nine = cap(L["all9"])
    add("c_all9", "C", nine, all9, "LR", "full", 9, lin.loc["linear", "cv_rmse"],
        {"rmse": lin.loc["linear", "b3_rmse"], "r2": lin.loc["linear", "b3_r2"], "mae": lin.loc["linear", "b3_mae"]},
        reason="model baseline (ENG-06)", cv_src=at(E0607), b3_src=at(E0607), order="pool order",
        src="ENG-06/07")
    add("c_all9", "C", nine, all9, "GPR, untuned", "full", 9, lin.loc["gpr", "cv_rmse"],
        b3_pooled(e09["before_after_heldout"]["gpr_untuned"]["pooled"]),
        reason="before/after tuning (ENG-09)", cv_src=at(E0607),
        b3_src=at(E09, "before_after_heldout.gpr_untuned"), order="pool order", src="ENG-07, ENG-09")
    add("c_all9", "C", nine, all9, "GPR, tuned", "full", 9, g8["cv_rmse"],
        b3_pooled(e09["before_after_heldout"]["gpr_tuned"]["pooled"]),
        reason="before/after tuning (ENG-09)", cv_src=at(E08T, "pearson x gpr (k = 9)"),
        b3_src=at(E09, "before_after_heldout.gpr_tuned"),
        order="Pearson order (dev-CV); pool order (batch 3)", src="ENG-08, ENG-09")
    add("c_all9", "C", nine, all9, "RF, untuned", "full", 9, lin.loc["rf", "cv_rmse"],
        b3_pooled(e09["before_after_heldout"]["rf_untuned"]["pooled"]),
        reason="before/after tuning (ENG-09)", cv_src=at(E0607),
        b3_src=at(E09, "before_after_heldout.rf_untuned"), order=pool, src="ENG-09")
    add("c_all9", "C", nine, all9, "RF, tuned", "full", 9, tuned(a19, all9)["cv"],
        b3_pooled(e09["before_after_heldout"]["rf_tuned"]["pooled"]),
        reason="before/after tuning (ENG-09)", cv_src=at(A19, "tuned (all nine)"),
        b3_src=at(E09, "before_after_heldout.rf_tuned"), order=pool, src="ENG-09, ENG-19b")
    ref = e20["summary"]["tuned|both_orders"]
    add("ref", "C", f"Reference: {L['all9']}, 20 forests", all9, "RF, tuned", "full", 9,
        ref["cv_rmse"]["mean"],
        {"rmse": ref["b3_rmse"]["mean"], "r2": ref["b3_r2"]["mean"], "mae": ref["b3_mae"]["mean"]},
        cv_sd=ref["cv_rmse"]["sd"], b3_sd=ref["b3_rmse"]["sd"], r2_sd=ref["b3_r2"]["sd"],
        mae_sd=ref["b3_mae"]["sd"], reason="reference over seeds (ENG-20)",
        cv_src=at(E20, "summary.tuned|both_orders"), b3_src=at(E20, "summary.tuned|both_orders"),
        order="mean ± sd over seeds 0–9 × pool and Pearson order", src="ENG-20")

    # ---- D: number of features (full rows, pool order) ----
    rule = a19["selection_rules"]["one_standard_error"]["pick"]["blocks"]
    best = a19["selection_rules"]["best"]["blocks"]
    for rid, label, blocks in (("d_rule", f"Rule pick: {L['best4']}", rule),
                               ("d_best", f"Dev-CV best: {describe(best)}", best)):
        h = heldout(a19, "untuned", blocks)
        add(rid, "D", label, blocks, "RF, untuned", "full", len(blocks), t19[key_of(blocks)], b3(h),
            reason=", ".join(h["reasons"]), cv_src=at(E19), b3_src=at(A19, "heldout"), order=pool,
            src="ENG-19, ENG-19b")
        t = tuned(a19, blocks)
        add(rid, "D", label, blocks, "RF, tuned", "full", len(blocks), t["cv"], b3(t),
            reason=", ".join(heldout(a19, "tuned", blocks)["reasons"]), cv_src=at(A19, "tuned"),
            b3_src=at(A19, "tuned"), order=pool, src="ENG-19b")

    # ---- E: history (history rows, combined-pool order restricted to the set) ----
    h = heldout(a24, "untuned", all9)
    add("e_all9", "E", nine, all9, "RF, untuned", "history", 9, t24[key_of(all9)], b3(h),
        reason=", ".join(h["reasons"]), cv_src=at(T24), b3_src=at(A24, "heldout"), order=hist, src="ENG-24")
    c = cell[("all9", "RF tuned")]
    add("e_all9", "E", nine, all9, "RF, tuned", "history", 9, c["cv_pooled_rmse_mean"], b3(c),
        reason=c["b3_reason"], cv_src=at(E28, "all9 x RF tuned"), b3_src=at(E28, "all9 x RF tuned"),
        order=hist, src="ENG-28")
    h = heldout(a23, "untuned", all9)
    add("e_all63", "E", cap(L["all63"]), all9, "RF, untuned", "history", 63, t23[key_of(all9)], b3(h),
        reason=", ".join(h["reasons"]), cv_src=at(T23), b3_src=at(A23, "heldout"), order=hist, src="ENG-23")
    t = tuned(a23, all9)
    add("e_all63", "E", cap(L["all63"]), all9, "RF, tuned", "history", 63, t["cv"], b3(t),
        reason=", ".join(heldout(a23, "tuned", all9)["reasons"]), cv_src=at(A23, "tuned"),
        b3_src=at(A23, "tuned"), order=hist, src="ENG-23")
    lrd = a22["named_subsets"]["level + rm20 + drift"]["blocks"]
    for model in ("RF", "RF tuned"):
        c = cell[("lvl_rm_drift", model)]
        # the untuned forest's dev-CV is ENG-22's own; ENG-28 rescored the same design for batch 3
        cv = t22[key_of(lrd, comp_names)] if model == "RF" else c["cv_pooled_rmse_mean"]
        add("e_lrd", "E", cap(L["lvl_rm_drift"]), all9, MODEL_LABEL[model], "history", 27, cv, b3(c),
            reason=c["b3_reason"], cv_src=at(T22) if model == "RF" else at(E28, f"lvl_rm_drift x {model}"),
            b3_src=at(E28, f"lvl_rm_drift x {model}"), order=hist,
            src="ENG-22, ENG-28" if model == "RF" else "ENG-28")
    no_rm = [x for x in all7 if x != "rm20"]
    h = heldout(a22, "untuned", no_rm)
    add("e_no_rm20", "E", f"{cap(L['all63'])}, without rm20 [observation]", all9, "RF, untuned", "history", 54,
        t22[key_of(no_rm, comp_names)], b3(h), reason=", ".join(h["reasons"]), cv_src=at(T22),
        b3_src=at(A22, "heldout"), order=hist, obs=True, src="ENG-22")

    # ---- F: feature reduction with history ----
    def add_history(rid, label, blocks):
        """A history subset of the ENG-23 search: untuned from the table, tuned from its re-score."""
        h = heldout(a23, "untuned", blocks)
        add(rid, "F", label, blocks, "RF, untuned", "history", 7 * len(blocks), t23[key_of(blocks)], b3(h),
            reason=", ".join(h["reasons"]), cv_src=at(T23), b3_src=at(A23, "heldout"), order=hist, src="ENG-23")
        t = tuned(a23, blocks)
        add(rid, "F", label, blocks, "RF, tuned", "history", 7 * len(blocks), t["cv"], b3(t),
            reason=", ".join(heldout(a23, "tuned", blocks)["reasons"]), cv_src=at(A23, "tuned"),
            b3_src=at(A23, "tuned"), order=hist, src="ENG-23")

    rule23 = a23["selection_rules"]["one_standard_error"]["pick"]["blocks"]
    budget2 = min(a23["tradeoff_edge"], key=lambda e: e["soc_span"])
    add_history("f_rule", f"Rule pick: {L['hist_rule']}", rule23)
    add_history("f_budget", f"Budget {100 * budget2['soc_span']:.1f}%: {L['hist_budget']}", budget2["blocks"])
    # the level-only budget set on the same rows: the largest ENG-24 edge step under 90% of the discharge;
    # untuned from ENG-24, tuned from the ENG-28 cell
    partial = max([e for e in a24["tradeoff_edge"] if e["soc_span"] < 0.9], key=lambda e: e["soc_span"])
    seven = partial["blocks"]
    label = f"Budget {100 * partial['soc_span']:.1f}%: {L['budget7']} (level only)"
    h = heldout(a24, "untuned", seven)
    add("f_level", "F", label, seven, "RF, untuned", "history", len(seven), t24[key_of(seven)], b3(h),
        reason=", ".join(h["reasons"]), cv_src=at(T24), b3_src=at(A24, "heldout"), order=hist, src="ENG-24")
    c = cell[("budget7", "RF tuned")]
    add("f_level", "F", label, seven, "RF, tuned", "history", len(seven), c["cv_pooled_rmse_mean"], b3(c),
        reason=c["b3_reason"], cv_src=at(E28, "budget7 x RF tuned"), b3_src=at(E28, "budget7 x RF tuned"),
        order=hist, src="ENG-28")
    best23 = a23["selection_rules"]["best"]["blocks"]
    add_history("f_best", f"Dev-CV best: {describe(best23)} + history", best23)

    # ---- G: the level-only budget set of F with its history (ENG-28, Table XV) ----
    fs = next(f for f in e28["feature_sets"] if f["id"] == "budget7_hist")
    label = f"Budget {100 * partial['soc_span']:.1f}% + history: {L['budget7']}"
    for model in ("RF", "RF tuned"):
        c = cell[("budget7_hist", model)]
        add("g_budget_hist", "G", label, fs["blocks"], MODEL_LABEL[model], "history", fs["n_columns"],
            c["cv_pooled_rmse_mean"], b3(c), reason=c["b3_reason"], cv_src=at(E28, f"budget7_hist x {model}"),
            b3_src=at(E28, f"budget7_hist x {model}"), order=hist, src="ENG-28")
    return pd.DataFrame(rows)


def display(df: pd.DataFrame) -> pd.DataFrame:
    """The markdown columns (every cell pre-formatted; sd shown only where the CSV has one)."""
    return pd.DataFrame({
        "Sub.": df["section"],
        "Configuration": [c.replace(" [observation]", " †") for c in df["configuration"]],
        "Model": df["model"],
        "Rows": df["rows"],
        "Cols": df["n_columns"].astype(int).astype(str),
        "Dev-CV RMSE": [fmt_pm(m, s) for m, s in zip(df["dev_cv_rmse"], df["dev_cv_rmse_sd"])],
        "Batch-3 RMSE": [fmt_pm(m, s) for m, s in zip(df["b3_rmse"], df["b3_rmse_sd"])],
        "Batch-3 R²": [fmt_pm(m, s, 3) for m, s in zip(df["b3_r2"], df["b3_r2_sd"])],
        "Batch-3 MAE": [fmt_pm(m, s) for m, s in zip(df["b3_mae"], df["b3_mae_sd"])],
        "Source": df["source"],
    })


def make(stem: Path) -> list[Path]:
    df = build()
    bold = [int(i) for i in df.index[df["row_id"] == "ref"]]
    return write_table(stem, display(df), label=LABEL, title=TITLE, csv_df=df, bold_rows=bold, notes=NOTES,
                       sources=SOURCES)
