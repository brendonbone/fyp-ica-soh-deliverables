"""
Table VII. Dev-CV RMSE and R² of the interim model and of five regressors on all nine features
Dev-CV accuracy of the interim three-feature RF (eng00), linear regression (eng06), the RBF GPR
(eng07), the tuned Matérn GPR (eng08_full_tuned.json pearson|gpr, k = 9) and the untuned and tuned
RF (eng20_reference_seeds.json). The RF rows are means over 20 runs (seeds 0-9 x two column orders):
fold sd = mean of the runs' fold sds, seed sd = sd of their RMSEs (ddof 1). Bold: the reference model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_items.common import fmt_float, fmt_pct, fmt_pm, fmt_rmse, load_json, write_table

LABEL = "Table VII"
TITLE = "Dev-CV RMSE and R² of the interim model and of five regressors on all nine features"
SOURCES = ["eng00_baseline.json", "eng06_simple_baselines.json", "eng07_gpr.json",
           "eng08_full_tuned.json", "eng20_reference_seeds.json", "eng09_tuning.json"]

DASH = "–"            # an empty cell; negative numbers carry the minus sign U+2212
REFERENCE_ROW = 5     # the tuned RF, bolded
RF_RUNS = "seeds 0-9 x {pool, Pearson} order"
GPR_RUNS = "seed 0 (2000-row subsample)"


def harness_cv(cv: dict) -> dict:
    """RMSE, fold sd and R² of a harness `cv` block (ENG-06 / ENG-07)."""
    return {"rmse": cv["pooled_rmse_mean"], "fold_sd": cv["pooled_rmse_std"], "r2": cv["pooled_r2_mean"]}


def rf_runs(runs: pd.DataFrame, config: str) -> dict:
    """Mean over the 20 ENG-20 runs of one RF configuration ('untuned' or 'tuned')."""
    g = runs[runs["config"] == config]
    return {"rmse": float(g["cv_rmse"].mean()), "fold_sd": float(g["cv_rmse_std"].mean()),
            "r2": float(g["cv_r2"].mean()), "seed_sd": float(g["cv_rmse"].std(ddof=1)), "n": int(len(g))}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The CSV table (one row per model, full precision) and its formatted markdown view."""
    c0 = load_json("eng00_baseline.json")["cv"]["pooled_summary"]
    interim = {"rmse": c0["rmse"]["mean"], "fold_sd": c0["rmse"]["std"], "r2": c0["r2"]["mean"]}
    linear = harness_cv(load_json("eng06_simple_baselines.json")["details"]["linear"]["cv"])
    gpr = harness_cv(load_json("eng07_gpr.json")["details"]["gpr"]["cv"])
    k9 = load_json("eng08_full_tuned.json")["grid"]["pearson|gpr"]["k_curve"][-1]
    gpr_tuned = {"rmse": k9["cv_pooled_rmse_mean"], "fold_sd": k9["cv_pooled_rmse_std"],
                 "r2": k9["cv_pooled_r2_mean"]}
    runs = pd.DataFrame(load_json("eng20_reference_seeds.json")["runs"])
    rf_u, rf_t = rf_runs(runs, "untuned"), rf_runs(runs, "tuned")

    rows = [  # model, features, k, scores, seed sd, n models, runs, source
        ("Interim model: RF, untuned", "height, position, area", 3, interim, None, 1,
         "seed 0, pool order", "results/eng00_baseline.json"),
        ("Linear regression", "all nine", 9, linear, None, 1,
         "deterministic", "results/eng06_simple_baselines.json"),
        ("GPR, untuned (RBF)", "all nine", 9, gpr, None, 1, GPR_RUNS, "results/eng07_gpr.json"),
        ("GPR, tuned (Matérn ν = 1.5)", "all nine", 9, gpr_tuned, None, 1, GPR_RUNS,
         "results/eng08_full_tuned.json grid['pearson|gpr'] k=9"),
        ("RF, untuned", "all nine", 9, rf_u, rf_u["seed_sd"], rf_u["n"], RF_RUNS,
         "results/eng20_reference_seeds.json"),
        ("RF, tuned (reference)", "all nine", 9, rf_t, rf_t["seed_sd"], rf_t["n"], RF_RUNS,
         "results/eng20_reference_seeds.json"),
    ]
    csv = pd.DataFrame([{
        "model": m, "features": f, "k": k, "dev_cv_rmse": d["rmse"], "fold_sd": d["fold_sd"],
        "seed_sd": sd, "n_models": n, "dev_cv_r2": d["r2"],
        "rel_vs_interim": None if i == 0 else (d["rmse"] - interim["rmse"]) / interim["rmse"],
        "runs": r, "source": src} for i, (m, f, k, d, sd, n, r, src) in enumerate(rows)])

    disp = pd.DataFrame({
        "Model": csv["model"], "Features": csv["features"],
        "Dev-CV RMSE (± fold sd)": [fmt_pm(a, b) for a, b in zip(csv["dev_cv_rmse"], csv["fold_sd"])],
        "Seed sd": [DASH if pd.isna(x) else fmt_rmse(x) for x in csv["seed_sd"]],
        "Dev-CV R²": [fmt_float(3)(x) for x in csv["dev_cv_r2"]],
        "% vs interim": [DASH if pd.isna(x) else fmt_pct(x, signed=True) for x in csv["rel_vs_interim"]],
    })
    return csv, disp


def notes() -> list[str]:
    """What dev-CV, the sds and % vs interim mean, and how each model row was configured."""
    runs = pd.DataFrame(load_json("eng20_reference_seeds.json")["runs"])
    n = rf_runs(runs, "tuned")["n"]

    def seed0(config: str, order: str) -> str:
        r = runs[(runs["config"] == config) & (runs["order"] == order) & (runs["seed"] == 0)].iloc[0]
        return fmt_rmse(float(r["cv_rmse"]))

    e06 = load_json("eng06_simple_baselines.json")["details"]
    e09 = load_json("eng09_tuning.json")
    return [
        "Dev-CV: pooled RMSE on the validation cells of each of the five development folds "
        "(ENG-08 folds, all clean rows of batches 1 and 2), averaged over the folds; ± is the "
        "sd over the five folds (ddof 0). R² is averaged the same way. % vs interim: change in "
        "dev-CV RMSE against the interim model (negative = lower error).",
        f"RF rows: mean over ten forest seeds (0-9) × two column orders (pool and Pearson) = "
        f"{n} models each (ENG-20); the fold sd is the mean of the {n} models' fold sds, and the "
        f"seed sd is the sd of their dev-CV RMSEs (ddof 1). Other rows are single runs (seed 0). "
        f"Seed 0 alone: tuned {seed0('tuned', 'pearson')} (Pearson order, ENG-08) and "
        f"{seed0('tuned', 'pool')} (pool order); untuned {seed0('untuned', 'pearson')} (Pearson "
        f"order, ENG-08) and {seed0('untuned', 'pool')} (pool order, ENG-06 and ENG-19, the "
        f"all-nine value of the subset search in Section V-D).",
        "Interim model: the untuned 300-tree RF on height, position and area (seed 0, pool order; "
        "ENG-00, identical to the ENG-19 subset).",
        f"Linear regression: standardised inputs, ordinary least squares. Ridge (RidgeCV) gives the "
        f"same RMSE to seven decimals ({e06['ridge']['cv']['pooled_rmse_mean']:.7f}); elastic net is "
        f"worse ({fmt_rmse(e06['enet']['cv']['pooled_rmse_mean'])}).",
        "GPR: constant × isotropic kernel + white noise on standardised inputs, hyperparameters by "
        "marginal likelihood, trained on a 2000-row per-cell stratified subsample in each fold (both "
        "GPR rows). Tuned GPR: Matérn ν = 1.5 kernel, the ENG-09 fixed configuration (ENG-08 tuned "
        "grid, k = 9). The GPR does not depend on column order (< 10⁻⁹).",
        f"Tuned RF: the ENG-09 fixed configuration (max_features 0.5, max_depth None, "
        f"min_samples_leaf 1, 300 trees). Nested CV of the whole tuning procedure (ENG-09): RF "
        f"{e09['rf']['nested_rmse_mean']:.4f} (R² {e09['rf']['nested_r2_mean']:.3f}), GPR "
        f"{e09['gpr']['nested_rmse_mean']:.4f} (R² {e09['gpr']['nested_r2_mean']:.3f}).",
    ]


def make(stem: Path) -> list[Path]:
    csv, disp = build()
    return write_table(stem, disp, label=LABEL, title=TITLE, csv_df=csv, bold_rows=[REFERENCE_ROW],
                       notes=notes(), sources=SOURCES)
