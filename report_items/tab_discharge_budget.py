"""
Table XIII. Best subset within each discharge budget, level only and with history
================================================================================
For each of the seven per-cycle SoC spans a subset can need (its windows CSV,
soc_span_per_cycle rounded to 0.001), the best subset needing at most that span,
without history (ENG-24) and with history (ENG-23), on the same rows. The
'same window as' column names the features whose own ENG-17 median span equals
the budget; a budget is on a search's trade-off edge when it is a step of that
search's stored tradeoff_edge (its best improves on every smaller budget).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import FEATURES, fmt_pct, fmt_rmse, join_features, load_csv, load_json, write_table
from report_items.tab_best_subsets_level_vs_history import HIST, LVL, search_table, short_subset

LABEL = "Table XIII"
TITLE = ("Best subset within each discharge budget, level features only (ENG-24) and with history "
         "(ENG-23), on the same rows (dev-CV RMSE, untuned RF)")
SOURCES = [f"{LVL}_windows.csv", f"{HIST}_windows.csv", f"{LVL}.json", f"{HIST}.json",
           "eng17_soc_windows_summary.csv", f"{LVL}_analysis.json", f"{HIST}_analysis.json"]
FORMATS = {"Level only: dev-CV": fmt_rmse, "With history: dev-CV": fmt_rmse, "Gain from history": fmt_pct}


def per_budget(t: pd.DataFrame, win: pd.DataFrame) -> tuple[list[float], dict]:
    """For each per-cycle SoC span (rounded to 0.1 % as tradeoff_edge does), the
    best subset needing at most that much, and the best needing exactly that much."""
    w = win[["key", "soc_span_per_cycle"]].merge(
        t[["key", "k", "blocks", "cv_pooled_rmse_mean", "cv_pooled_rmse_std", "cv_rank"]], on="key")
    w["budget"] = w["soc_span_per_cycle"].round(3)
    budgets = sorted(float(b) for b in w["budget"].unique())
    out = {}
    for b in budgets:
        within = w[w["budget"] <= b].sort_values(["cv_pooled_rmse_mean", "key"])
        exact = w[w["budget"] == b].sort_values(["cv_pooled_rmse_mean", "key"])
        r, e = within.iloc[0], exact.iloc[0]
        out[b] = {"blocks": list(r["blocks"]), "k": int(r["k"]), "cv": float(r["cv_pooled_rmse_mean"]),
                  "n_within": int(len(within)), "n_exact": int(len(exact)),
                  "exact_blocks": list(e["blocks"]), "exact_cv": float(e["cv_pooled_rmse_mean"])}
    return budgets, out


def single_feature_windows(budgets: list[float]) -> dict[float, str]:
    """Budget -> the features whose ENG-17 median span (all cycles) is that budget."""
    e17 = load_csv("eng17_soc_windows_summary.csv")
    e17 = e17[(e17["stage"] == "all cycles") & (e17["basis"] == "as_implemented")].set_index("feature")
    return {b: join_features([f for f in FEATURES if round(float(e17.loc[f, "soc_span_median"]), 3) == b])
            for b in budgets}


def edge_spans(stem: str) -> set[float]:
    """The budgets at which the search's stored trade-off edge steps."""
    return {e["soc_span"] for e in load_json(f"{stem}_analysis.json")["tradeoff_edge"]}


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The markdown table, the full-precision CSV table (one row per budget) and the notes."""
    # Both searches cover the same 511 subsets, so they have the same seven budgets.
    budgets, best_h = per_budget(search_table(HIST), load_csv(f"{HIST}_windows.csv"))
    _, best_l = per_budget(search_table(LVL), load_csv(f"{LVL}_windows.csv"))
    window_of = single_feature_windows(budgets)
    edge_h, edge_l = edge_spans(HIST), edge_spans(LVL)
    full = budgets[-1]

    rows = []
    for b in budgets:
        lv, hs = best_l[b], best_h[b]
        rows.append({
            "budget_soc_span": b, "same_window_as": window_of[b],
            "n_subsets_needing_exactly": lv["n_exact"], "n_subsets_within_budget": lv["n_within"],
            "level_best_subset": join_features(lv["blocks"]), "level_k": lv["k"], "level_dev_cv": lv["cv"],
            "level_on_tradeoff_edge": b in edge_l, "level_cost_vs_full": lv["cv"] / best_l[full]["cv"] - 1,
            "history_best_subset": join_features(hs["blocks"]), "history_k": hs["k"],
            "history_dev_cv": hs["cv"], "history_on_tradeoff_edge": b in edge_h,
            "history_cost_vs_full": hs["cv"] / best_h[full]["cv"] - 1,
            "gain_from_history": (lv["cv"] - hs["cv"]) / lv["cv"]})
    t = pd.DataFrame(rows)
    not_edge = [b for b in budgets if b not in edge_h]

    def cost(x: float, b: float) -> str:
        return "(full)" if b == full else fmt_pct(x, signed=True)

    md = pd.DataFrame({
        "Discharge budget (SoC span)": [fmt_pct(b) for b in budgets],
        "Level only: best subset": [short_subset(best_l[b]["blocks"]) for b in budgets],
        "Level only: dev-CV": t["level_dev_cv"],
        "Level only: vs full discharge": [cost(r["level_cost_vs_full"], r["budget_soc_span"]) for r in rows],
        "With history: best subset": [short_subset(best_h[b]["blocks"]) + ("†" if b in not_edge else "")
                                      for b in budgets],
        "With history: dev-CV": t["history_dev_cv"],
        "With history: vs full discharge": [cost(r["history_cost_vs_full"], r["budget_soc_span"]) for r in rows],
        "Gain from history": t["gain_from_history"],
    })

    notes = [
        "Budget = the SoC span a subset needs: the union of its features' windows in each cycle, median "
        "over the clean cycles, rounded to 0.1%. Every subset needs one of seven spans, the windows of "
        "single features (Table VI): " + "; ".join(f"{fmt_pct(b)} {window_of[b]}" for b in budgets) + ".",
        "With history, a subset needs the same window in each of the previous 20 cycles as well, because "
        "its history columns are built from those cycles' features.",
        "Each row gives the best subset among all subsets needing at most the budget; 'vs full discharge' "
        f"is its dev-CV relative to the best at {fmt_pct(full)}.",
    ]
    for b in not_edge:
        hs = best_h[b]
        notes.append(
            f"† No subset with history that needs exactly {fmt_pct(b)} beats the {fmt_pct(budgets[0])} set "
            f"(best at exactly {fmt_pct(b)}: {join_features(hs['exact_blocks'])}, {fmt_rmse(hs['exact_cv'])}), "
            f"so the history trade-off edge has {len(edge_h)} steps against {len(edge_l)} without history.")
    return md, t, notes


def make(stem: Path) -> list[Path]:
    md, t, notes = build()
    return write_table(stem, md, label=LABEL, title=TITLE, formats=FORMATS, csv_df=t, notes=notes,
                       sources=SOURCES)
