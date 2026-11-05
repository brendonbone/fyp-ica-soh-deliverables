"""
Every calculation of the study, in the order the work was done.
================================================================
Each stage is one module in stages/ that reads the raw data or earlier
results and writes its result files to results/ (nothing here plots; the
tables and figures are made afterwards by run_report.py). A stage is skipped
when its main output already exists, so the script resumes where it stopped.

    python run_results.py                  # everything that is not done yet
    python run_results.py --only 3 4       # these stages (also by name)
    python run_results.py --force --only 4 # rerun a finished stage
    python run_results.py --fast           # skip the stages marked slow
    python run_results.py --list           # show the stages and stop

Stage 0 turns the three published Severson .mat files into the raw pickles,
and stages 1-3 and 10 read the pickles (see README.md, "Data"); everything
else runs from results/features_expanded.parquet, which is committed. Stage 0
never overwrites a pickle: to rebuild one, run
python -m stages.s00_build_pickles --force. The slow stages are resumable
and, where marked, shardable for a cluster (see cluster/README.md).
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import PKL_DIR, RESULTS_DIR, mats_available, pickles_available  # noqa: E402

PICKLES = "batch{1,2,3}.pkl"   # stage 0's output, written to PKL_DIR rather than results/
RAW = {"mat": (mats_available, "the published Severson .mat files"),
       "pkl": (pickles_available, "the raw Severson pickles")}

# (stage module, title, main output, raw data it reads: "mat", "pkl" or "", slow)
STAGES: list[tuple[str, str, str, str, bool]] = [
    # --- the raw data: the three published .mat files as the pickles stages 1-3 and 10 read ---
    ("s00_build_pickles",          "0. The three Severson batches from the published .mat files to pickles",
     PICKLES, "mat", False),
    # --- July: the interim model and the nine-feature pool ---
    ("s01_interim_model",          "1. Interim model: three peak features and a random forest (ENG-00)",
     "eng00_baseline.json", "pkl", False),
    ("s02_extract_features",       "2. Extract the nine peak features of every cycle (ENG-05)",
     "features_expanded.parquet", "pkl", False),
    ("s03_dqdv_check",             "3. Recomputed dQ/dV against the published curve; second peak (ENG-02)",
     "eng02_second_peak.json", "pkl", False),
    ("s04_feature_pool",           "4. Feature pool: correlations with SoH and with each other (ENG-04, ENG-16)",
     "eng04_feature_pool.json", "", False),
    ("s05_simple_models",          "5. Linear, ridge, elastic-net, GPR and RF on all nine features (ENG-06/07)",
     "eng06_07_model_table.csv", "", False),
    ("s06_tune_models",            "6. Nested-CV tuning of the RF and the GPR (ENG-09)",
     "eng09_tuned_params.json", "", True),
    ("s07_selector_grid",          "7. Ranking methods x regressors, untuned and tuned (ENG-08)",
     "eng08_full_tuned.json", "", True),
    # --- September: feature history, other train/test partitions, discharge windows ---
    ("s08_history_features",       "8. 20-cycle history of each feature: table, checks, pools (ENG-14)",
     "eng14_full.json", "", True),
    ("s09_partitions",             "9. The same comparisons under six train/test partitions (ENG-15)",
     "eng15_split_robustness.json", "", True),
    ("s10_soc_windows",            "10. How much of the discharge each feature needs (ENG-17)",
     "features_soc_windows.parquet", "pkl", False),   # per-cycle windows; the later analyses need it
    # --- October: every subset, every component, every partition, every model ---
    ("s11_level_subsets",          "11. All 511 subsets of the nine features, and what they show (ENG-19, ENG-19b)",
     "eng19b_analysis.json", "", True),
    ("s12_reference_seeds",        "12. The reference model over ten forest seeds (ENG-20)",
     "eng20_reference_seeds.json", "", False),
    ("s13_grouped_importance",     "13. Permutation importance of feature groups (ENG-21)",
     "eng21_grouped_permutation.json", "", False),
    ("s14_history_components",     "14. All 127 subsets of the seven history components (ENG-22)",
     "eng22_history_components_analysis.json", "", True),
    ("s15_level_subsets_history_rows", "15. The 511 level subsets on the history rows (ENG-24)",
     "eng24_level_subsets_history_rows_analysis.json", "", True),
    ("s16_history_subsets",        "16. All 511 subsets of the features with their history (ENG-23)",
     "eng23_grouped_history_subsets_analysis.json", "", True),
    ("s17_partition_elimination",  "17. Spearman ranking and backward elimination per partition (ENG-25)",
     "eng25_partition_robustness.json", "", True),
    ("s18_noisy_cells",            "18. Training without the 16 cells with noisy published curves (ENG-27)",
     "eng27_noisy_cells.json", "", False),
    ("s19_feature_sets_models",    "19. The chosen feature sets under four models (ENG-28)",
     "eng28_feature_sets_models.json", "", True),
    # --- added after the others; chronologically it belongs after stage 5 (July) ---
    ("s20_learning_curve", "20. Learning curve: dev-CV R2 against the number of training cells (ENG-10)",
     "eng10_learning_curve.json", "", True),      # slow: 175 forests
]


def shown(out: str) -> str:
    """A stage's main output as printed: stage 0 writes to PKL_DIR, the others to results/."""
    return f"{PKL_DIR.name}/{out}" if out == PICKLES else f"results/{out}"


def chosen(only: list[str] | None):
    items = [(int(s[0][1:3]), *s) for s in STAGES]     # the number in the module name: s00 ... s20
    if not only:
        return items
    keep = {it[0] for it in items for tok in only
            if tok == str(it[0]) or tok == it[1] or tok == it[1].split("_", 1)[1]}
    return [it for it in items if it[0] in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", help="stage numbers or module names")
    ap.add_argument("--force", action="store_true", help="rerun even if the output exists")
    ap.add_argument("--fast", action="store_true", help="skip the slow stages")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    items = chosen(args.only)
    if args.list:
        for number, module, title, out, needs, slow in items:
            flags = {"mat": ".mat files", "pkl": "raw data", "": ""}[needs] + (" slow" if slow else "")
            print(f"{title:85s} -> {shown(out)}  {flags}")
        return

    for number, module, title, out, needs, slow in items:
        done = pickles_available() if out == PICKLES else (RESULTS_DIR / out).exists()
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
        if done and not args.force:
            print(f"[skip] {shown(out)} exists", flush=True)
            continue
        if slow and args.fast:
            print("[skip] slow stage (--fast)", flush=True)
            continue
        if needs and not RAW[needs][0]():
            print(f"[skip] needs {RAW[needs][1]} (see README.md, 'Data')", flush=True)
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", "-m", f"stages.{module}"], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"stage failed: {module} (exit {r.returncode})")
        print(f"[done] {module} in {time.time() - t0:.0f} s", flush=True)

    print("\nAll stages done. Now: python run_report.py")


if __name__ == "__main__":
    main()
