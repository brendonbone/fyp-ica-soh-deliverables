"""
The report's tables and figures that come from the results, in the report's order.
====================================================================================
Run after run_results.py (or on the committed results). Nothing here fits a
model: each item reads result files and writes report/NN_<kind>_<label>_<name>.*
(csv + md for a table, pdf + png + svg for a figure). NN is the position in
ORDER below, so reordering the list renumbers the files.

Labels are the final report's numbers: Tables II and V-XVI, Figs. 1, 4 and
6-9, and in the appendices Fig. B1, Table B1, Fig. C1 and Table D1. Items
labelled Table S1-S8 and Fig. S1 are supplementary and not in the report:
S1-S4 and Fig. S1 sit in the section they belong to, S5-S8 come last. The
report's Tables I, III, IV and A1 are typed in the report, and its Figs. 2, 3
and 5 are made outside this repository.

    python run_report.py                 # everything
    python run_report.py --only 5 6      # by number
    python run_report.py --only "Table V" "Fig. 8"
    python run_report.py --list          # show the order and stop
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import REPORT_DIR  # noqa: E402

# (report label, report_items module, short file name), in the report's order.
# S-labels are supplementary items that the report does not show.
ORDER: list[tuple[str, str, str]] = [
    # --- Background II-B; Method IV-B, IV-E ---
    ("Fig. 1",     "fig_ic_curve_example",              "ic_curve_example"),
    ("Table II",   "tab_dataset",                       "dataset"),
    ("Fig. 4",     "fig_feature_geometry",              "feature_geometry"),
    ("Table S1",   "tab_partitions",                    "partitions"),               # supplementary
    # --- V-A  Data extraction and the feature pool ---
    ("Table V",    "tab_feature_soh_association",       "feature_soh_association"),
    ("Fig. 6",     "fig_feature_correlation",           "feature_correlation"),
    # --- V-B  How much of the discharge each feature needs ---
    ("Table VI",   "tab_feature_soc_windows",           "feature_soc_windows"),
    ("Fig. 7",     "fig_soc_windows_fresh_aged",        "soc_windows_fresh_aged"),
    # --- V-C  Reference model, all nine features ---
    ("Table VII",  "tab_reference_models",              "reference_models"),
    ("Table VIII", "tab_feature_importance",            "feature_importance"),
    # --- V-D  Accuracy against the number of retained features ---
    ("Table IX",   "tab_best_subsets_paths",            "best_subsets_paths"),
    ("Fig. 8",     "fig_subsets_vs_k",                  "subsets_vs_k"),
    # --- V-E  Adding feature history ---
    ("Table X",    "tab_history_gain",                  "history_gain"),
    ("Table XI",   "tab_history_components",            "history_components"),
    # --- V-F  Feature reduction with history ---
    ("Table XII",  "tab_best_subsets_level_vs_history", "best_subsets_level_vs_history"),
    ("Fig. 9",     "fig_history_subsets_vs_k",          "history_subsets_vs_k"),
    ("Table XIII", "tab_discharge_budget",              "discharge_budget"),
    # --- V-G  Accuracy gained per feature ---
    ("Table XIV",  "tab_methods_models_level",          "methods_models_level"),
    ("Table XV",   "tab_methods_models_history",        "methods_models_history"),
    # --- V-H  Verification of results (batch 3) ---
    ("Table XVI",  "tab_batch3_test",                   "batch3_test"),
    ("Table S2",   "tab_batch3_methods_models",         "batch3_methods_models"),    # supplementary
    ("Fig. S1",    "fig_be_partitions",                 "be_partitions"),            # supplementary
    # --- Appendix B  Second-peak search: graphical evidence ---
    ("Fig. B1",    "fig_recomputed_vs_published",       "recomputed_vs_published"),
    ("Table B1",   "tab_dqdv_recompute",                "dqdv_recompute"),
    ("Table S3",   "tab_second_peak",                   "second_peak"),              # supplementary
    ("Table S4",   "tab_noisy_cells",                   "noisy_cells"),              # supplementary
    # --- Appendix C  Learning curve ---
    ("Fig. C1",    "fig_learning_curve",                "learning_curve"),
    # --- Appendix D  Summary of results comparison ---
    ("Table D1",   "tab_summary_comparison",            "summary_comparison"),
    # --- Supplementary: the six train/test partitions and the full feature-set ranking ---
    ("Table S5",   "tab_partition_errors",              "partition_errors"),
    ("Table S6",   "tab_ranking_stability",             "ranking_stability"),
    ("Table S7",   "tab_be_orders",                     "be_orders"),
    ("Table S8",   "tab_methods_models_ranking",        "methods_models_ranking"),
]


def file_stem(number: int, label: str, name: str) -> Path:
    kind = "table" if label.startswith("Table") else "fig"
    tag = re.sub(r"[^A-Za-z0-9]+", "", label.split(" ", 1)[1])   # 'VI', 'B1', '8'
    return REPORT_DIR / f"{number:02d}_{kind}_{tag}_{name}"


def selected(only: list[str] | None) -> list[tuple[int, str, str, str]]:
    """The items named by --only (numbers such as 5 or 05, labels, module or file names)."""
    items = [(i + 1, *entry) for i, entry in enumerate(ORDER)]
    if not only:
        return items
    keep, unknown = set(), []
    for token in only:
        hits = {number for number, label, module, name in items
                if (token.isdigit() and int(token) == number) or token.lower() == label.lower()
                or token in (module, name)}
        keep |= hits
        if not hits:
            unknown.append(token)
    if unknown:
        sys.exit(f"--only: no item matches {unknown}; python run_report.py --list shows them")
    return [it for it in items if it[0] in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", help="numbers, labels ('Table V'), module or file names")
    ap.add_argument("--list", action="store_true", help="print the order and exit")
    args = ap.parse_args()

    items = selected(args.only)
    if args.list or not items:
        for number, label, module, name in (items or selected(None)):
            print(f"{number:02d}  {label:13s} report_items/{module}.py")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.only:   # a full run starts from a clean folder
        for old in REPORT_DIR.glob("[0-9][0-9]_*"):
            old.unlink()

    index = ["# Report items", "",
             "Written by `run_report.py` from `results/`. The items are in the report's order and carry "
             "the report's labels; items labelled S1-S8 are supplementary and not in the report.", "",
             "| No. | Label | Title | Files | Sources |", "|---|---|---|---|---|"]
    failed = []
    for number, label, module, name in items:
        t0 = time.time()
        try:
            mod = importlib.import_module(f"report_items.{module}")
            stem = file_stem(number, label, name)
            paths = mod.make(stem)
        except Exception as e:  # keep going; report at the end
            failed.append((number, label, repr(e)))
            print(f"[FAIL] {number:02d} {label}: {e!r}", flush=True)
            continue
        files = ", ".join(p.name for p in paths)
        srcs = ", ".join(getattr(mod, "SOURCES", []))
        index.append(f"| {number:02d} | {label} | {mod.TITLE} | {files} | {srcs} |")
        print(f"[ok]   {number:02d} {label}: {files} ({time.time() - t0:.1f} s)", flush=True)

    if not args.only:
        (REPORT_DIR / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    if failed:
        sys.exit(f"{len(failed)} item(s) failed: {[(n, l) for n, l, _ in failed]}")


if __name__ == "__main__":
    main()
