"""
4. Feature pool: correlations with SoH and with each other (ENG-04, ENG-16)
===========================================================================
How strongly is each of the nine peak features associated with SoH, and how
redundant are the features among themselves?  ENG-04 characterises the pool
on the development rows (batches 1+2; batch 3 is never scored): Pearson r of
each feature with SoH, the 9x9 inter-feature Pearson and Spearman matrices and
the pairs with |r| > 0.85.  ENG-16 compares the linear (|Pearson r|) and the
monotonic (|Spearman rho|) association of every feature with SoH, for the nine
level features on all clean dev rows and for the 63-column combined pool
(level + 20-cycle history) on the dev rows that have a history.

Inputs   results/features_expanded.parquet (stage 2).
Outputs  results/eng04_feature_pool.json: pool, n_features, clean_log,
           nan_counts_clean, soh_correlations_dev, high_correlation_pairs,
           correlation_matrix (4 dp), correlation_matrix_full, spearman_matrix,
           verdict.
         results/eng16_pearson_vs_spearman.csv: pool, feature, pearson_r,
           spearman_rho, abs_r, abs_rho, pearson_rank, spearman_rank,
           rank_shift, abs_rho_minus_abs_r (9 level + 63 combined rows).
         results/eng16_spearman.json: dev_rows, nonlinear_gap_threshold,
           summary{level, combined}, caveats.
Runtime  about a minute (the history columns); nothing stochastic.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.config import RESULTS_DIR
from src.data_prep import clean_feature_table, log_manifest, split_dev_test
from src.features import FEATURE_COLS
from src.features_temporal import LEVEL_COLS
from src.subset_search import load_history_rows

PARQUET = RESULTS_DIR / "features_expanded.parquet"
POOL_JSON = RESULTS_DIR / "eng04_feature_pool.json"
SPEARMAN_CSV = RESULTS_DIR / "eng16_pearson_vs_spearman.csv"
SPEARMAN_JSON = RESULTS_DIR / "eng16_spearman.json"

HIGH_CORRELATION = 0.85  # |r| above this = redundancy candidate (ENG-04)
NONLINEAR_GAP = 0.10     # |rho| - |r| at or above this = "monotonic, not linear"

CAVEATS = [
    "rows are not independent (many cycles per cell): coefficients are ranking "
    "scores, not test statistics; no p-values are reported",
    "Spearman captures monotonic, not general, dependence; mutual information "
    "is the selector for general dependence",
    "both are univariate filters and ignore redundancy and complementarity",
]


# ------------------------ ENG-04: the pool on the dev rows ------------------------

def feature_pool(clean: pd.DataFrame, clean_log: dict, dev: pd.DataFrame) -> dict:
    """NaN audit, SoH correlations and the inter-feature matrices (pandas .corr)."""
    pool = FEATURE_COLS
    print(f"Expanded pool ({len(pool)} features): {pool}")

    # NaN audit on the cleaned table (acceptance: none beyond logged fails).
    nan_counts = {c: int(clean[c].isna().sum()) for c in pool}
    bad = {c: n for c, n in nan_counts.items() if n > 0}
    print(f"NaNs in cleaned table: {bad if bad else 'none'}")

    soh_corr = {c: float(dev[c].corr(dev["SOH"])) for c in pool}
    corr = dev[pool].corr()
    spearman = dev[pool].corr(method="spearman")
    print("\nPearson r vs SoH (dev):")
    for c, v in sorted(soh_corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {c:<12} {v:+.3f}")
    hi = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(pool) for b in pool[i + 1:]
        if abs(corr.loc[a, b]) > HIGH_CORRELATION
    ]
    print(f"\n|r| > {HIGH_CORRELATION} feature pairs (redundancy candidates):")
    for a, b, v in sorted(hi, key=lambda t: -abs(t[2])):
        print(f"  {a} - {b}: {v:+.3f}")

    return {
        "pool": pool,
        "n_features": len(pool),
        "clean_log": clean_log,
        "nan_counts_clean": nan_counts,
        "soh_correlations_dev": soh_corr,
        "high_correlation_pairs": [{"a": a, "b": b, "r": v} for a, b, v in hi],
        "correlation_matrix": corr.round(4).to_dict(),
        # Full-precision matrices so the correlation figure needs no parquet.
        "correlation_matrix_full": corr.to_dict(),
        "spearman_matrix": spearman.to_dict(),
        "verdict": "PASS" if len(pool) >= 6 and not bad else "CHECK",
    }


# ------------------------ ENG-16: Pearson vs Spearman ------------------------

def correlations(dev: pd.DataFrame, cols: list[str], pool: str) -> pd.DataFrame:
    """Per feature: scipy r and rho with SoH, their |.| ranks (ties by column order)."""
    rows = []
    for f in cols:
        rows.append({"pool": pool, "feature": f,
                     "pearson_r": float(pearsonr(dev[f], dev["SOH"])[0]),
                     "spearman_rho": float(spearmanr(dev[f], dev["SOH"])[0])})
    df = pd.DataFrame(rows)
    df["abs_r"], df["abs_rho"] = df["pearson_r"].abs(), df["spearman_rho"].abs()
    df["pearson_rank"] = df["abs_r"].rank(ascending=False, method="first").astype(int)
    df["spearman_rank"] = df["abs_rho"].rank(ascending=False, method="first").astype(int)
    df["rank_shift"] = df["spearman_rank"] - df["pearson_rank"]      # negative = climbs
    df["abs_rho_minus_abs_r"] = df["abs_rho"] - df["abs_r"]
    return df.sort_values("pearson_rank").reset_index(drop=True)


def pool_summary(df: pd.DataFrame) -> dict:
    """Kendall tau of the two rankings, their top-3s, and the curved features."""
    tau = float(kendalltau(df["pearson_rank"], df["spearman_rank"])[0])
    curved = df[df["abs_rho_minus_abs_r"] >= NONLINEAR_GAP]

    def top(col: str, n: int) -> list[str]:
        return df.sort_values(col)["feature"].head(n).tolist()

    return {
        "n_features": int(len(df)),
        "kendall_tau_pearson_vs_spearman": tau,
        "top3_pearson": top("pearson_rank", 3),
        "top3_spearman": top("spearman_rank", 3),
        "top9_overlap": len(set(top("pearson_rank", 9)) & set(top("spearman_rank", 9))),
        "monotonic_not_linear": [
            {"feature": r.feature, "abs_r": round(r.abs_r, 3), "abs_rho": round(r.abs_rho, 3)}
            for r in curved.sort_values("abs_rho_minus_abs_r", ascending=False).itertuples()],
        "biggest_climbers": [
            {"feature": r.feature, "rank_shift": int(r.rank_shift)}
            for r in df.sort_values("rank_shift").head(4).itertuples()],
        "biggest_fallers": [
            {"feature": r.feature, "rank_shift": int(r.rank_shift)}
            for r in df.sort_values("rank_shift").tail(4).itertuples()],
    }


def pearson_vs_spearman(clean: pd.DataFrame, dev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The level pool on the clean dev rows, the combined pool on the history dev rows."""
    dev_h, _, _, combined_cols, _ = load_history_rows(clean)   # ENG-14 rows
    level = correlations(dev, LEVEL_COLS, "level")
    combined = correlations(dev_h, combined_cols, "combined")
    table = pd.concat([level, combined], ignore_index=True)

    print(f"\n=== level pool ({len(dev):,} dev rows) ===")
    print(level.drop(columns=["pool", "abs_r", "abs_rho"]).round(3).to_string(index=False))

    summary = {}
    for name, df in (("level", level), ("combined", combined)):
        summary[name] = pool_summary(df)
        print(f"\n[{name}] Kendall tau between the two rankings: "
              f"{summary[name]['kendall_tau_pearson_vs_spearman']:.2f}; "
              f"|rho|-|r| >= {NONLINEAR_GAP}: "
              f"{[c['feature'] for c in summary[name]['monotonic_not_linear']][:8]}")

    # OLD ENG-16 also stored ranking_stability_across_eng15_partitions; it needs
    # the ENG-15 partitions and is not reported, so it is left out here.
    payload = {
        "dev_rows": {"level": int(len(dev)), "combined": int(len(dev_h))},
        "nonlinear_gap_threshold": NONLINEAR_GAP,
        "summary": summary,
        "caveats": CAVEATS,
    }
    return table, payload


# ------------------------ main ------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.parse_args()
    if not PARQUET.exists():
        sys.exit(f"{PARQUET} missing - run stage 2 (stages/s02_extract_features.py) first")

    clean, clean_log = clean_feature_table(pd.read_parquet(PARQUET))
    dev, _ = split_dev_test(clean)                 # batches 1+2, the ENG-08 rows

    pool = feature_pool(clean, clean_log, dev)
    table, spearman = pearson_vs_spearman(clean, dev)

    table.to_csv(SPEARMAN_CSV, index=False)        # pandas default: full float repr
    SPEARMAN_JSON.write_text(json.dumps(spearman, indent=2), encoding="utf-8")
    print(f"\nWrote {SPEARMAN_CSV} and {SPEARMAN_JSON}")
    summary = spearman["summary"]
    log_manifest("eng16_spearman_check", {
        "script": "stages/s04_feature_pool.py",
        "kendall_tau_level": summary["level"]["kendall_tau_pearson_vs_spearman"],
        "kendall_tau_combined": summary["combined"]["kendall_tau_pearson_vs_spearman"],
        "monotonic_not_linear_level": [c["feature"] for c in summary["level"]["monotonic_not_linear"]],
    })

    # Written last: run_results.py skips this stage when this file exists, so it
    # must not exist unless the ENG-16 files above were written too.
    POOL_JSON.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"Wrote {POOL_JSON}")
    log_manifest("eng04_feature_pool", {
        "script": "stages/s04_feature_pool.py",
        "n_features": pool["n_features"],
        "verdict": pool["verdict"],
    })


if __name__ == "__main__":
    main()
