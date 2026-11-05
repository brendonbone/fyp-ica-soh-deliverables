"""
2. Extract the nine peak features of every cycle (ENG-05)
=========================================================
Runs src.features.extract_all_features over every cycle of the 124 cleaned
Severson cells: the nine dominant-peak features, the seven second-peak
columns, the five recomputed-curve columns and the SoH label, one row per
cycle. This table is the input of every later stage and of the report.
Inputs : the raw pickles in PKL_DIR; results/features_3feat.parquet (stage 1,
         used when present) for the check that height/position/area are the
         same numbers as the interim extractor's.
Outputs: results/features_expanded.parquet, 100,501 rows x 28 columns in the
         order cell_id, batch, charge_policy, cycle, FEATURE_COLS (9),
         SECOND_PEAK_COLS (7), RECOMPUTE_COLS (5), SOH, status, cycle_life;
         results/eng05_extraction_report.json with the keys n_cells, n_rows,
         baseline_identical, overall_fail_rate, worst_cells, status_counts,
         nan_in_ok_rows, second_peak_coverage, recompute_coverage,
         rc_pearson_r, verdict; and the manifest entry 'eng05_extraction'.
Runtime: one pass over the cells (minutes; the pickles need ~8 GB of RAM),
         not resumable. The verdict is FAIL (exit 1) when 0.5 % or more of
         the cycles fail to extract.
Smoke run: --cells b1c1,b1c3 writes results/features_expanded_smoke.parquet
         and results/eng05_extraction_report_smoke.json instead, and no
         manifest entry, so the full outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src.config import PKL_DIR, REPO_ROOT, RESULTS_DIR
from src.data_prep import log_manifest
from src.features import (
    FEATURE_COLS,
    RECOMPUTE_COLS,
    SECOND_PEAK_COLS,
    extract_all_features,
)
from src.severson_loader import load_clean_severson, summarise_cell

BASELINE_PARQUET = RESULTS_DIR / "features_3feat.parquet"  # written by stage 1
OUT_PARQUET = RESULTS_DIR / "features_expanded.parquet"
REPORT_JSON = RESULTS_DIR / "eng05_extraction_report.json"
SMOKE_PARQUET = RESULTS_DIR / "features_expanded_smoke.parquet"
SMOKE_REPORT_JSON = RESULTS_DIR / "eng05_extraction_report_smoke.json"

FAIL_RATE_LIMIT = 0.005  # fewer than 0.5 % of the cycles may fail to extract


def extract_cells(cells: dict, ids: list[str]) -> pd.DataFrame:
    """One row per cycle of every cell in `ids`, in that order."""
    t0 = time.time()
    parts = []
    for i, cid in enumerate(ids):
        df = extract_all_features(
            cells[cid], cell_id=cid, include_recompute=True, verbose=True,
        )
        meta = summarise_cell(cells[cid])
        df.insert(0, "cell_id", cid)
        df.insert(1, "batch", cid[:2])
        df.insert(2, "charge_policy", meta["policy"])
        df["cycle_life"] = meta["cycle_life"]
        parts.append(df)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(ids)} cells, "
                  f"{time.time() - t0:.0f} s elapsed", flush=True)
    feat = pd.concat(parts, ignore_index=True)
    print(f"Extracted {len(feat)} rows in {time.time() - t0:.0f} s", flush=True)
    return feat


def check_baseline(feat: pd.DataFrame) -> None:
    """Check 1: height/position/area and status equal the interim extractor's on every row."""
    ref = pd.read_parquet(BASELINE_PARQUET)
    merged = feat.merge(
        ref[["cell_id", "cycle", "height", "position", "area", "status"]],
        on=["cell_id", "cycle"], suffixes=("", "_ref"),
    )
    assert len(merged) == len(feat), "row mismatch vs baseline cache"
    for col in ["height", "position", "area"]:
        same = np.allclose(merged[col], merged[f"{col}_ref"],
                           atol=1e-12, equal_nan=True)
        assert same, f"{col} differs from baseline extraction"
    assert (merged["status"] == merged["status_ref"]).all()
    print("  check 1 OK: height/position/area identical to baseline "
          f"({len(merged)} rows)", flush=True)


def failure_rates(feat: pd.DataFrame) -> tuple[dict, str]:
    """Check 2: overall and per-cell extraction failure rates, and the PASS/FAIL verdict."""
    fail = feat["status"] != "ok"
    overall = float(fail.mean())
    per_cell = fail.groupby(feat["cell_id"]).mean().sort_values(ascending=False)
    verdict = "PASS" if overall < FAIL_RATE_LIMIT else "FAIL"
    print(f"  check 2 [{verdict}]: overall fail rate {overall:.4%} "
          f"(limit {FAIL_RATE_LIMIT:.1%}); worst cell "
          f"{per_cell.index[0]} at {per_cell.iloc[0]:.2%}", flush=True)
    return {
        "overall_fail_rate": overall,
        "worst_cells": {k: float(v) for k, v in per_cell.head(5).items()},
        "status_counts": feat["status"].value_counts().to_dict(),
    }, verdict


def coverage(ok_rows: pd.DataFrame) -> dict:
    """Check 3 (NaNs inside ok rows) and how often the second-peak / recomputed columns are filled."""
    nan_counts = {c: int(ok_rows[c].isna().sum()) for c in FEATURE_COLS}
    unexpected = {c: n for c, n in nan_counts.items() if n > 0}
    print(f"  check 3: NaNs within ok rows (dominant-peak features): "
          f"{unexpected if unexpected else 'none'}", flush=True)
    out = {
        "nan_in_ok_rows": nan_counts,
        "second_peak_coverage": {
            c: float(ok_rows[c].notna().mean()) for c in SECOND_PEAK_COLS
        },
        "recompute_coverage": {
            c: float(ok_rows[c].notna().mean()) for c in RECOMPUTE_COLS
        },
    }
    r = ok_rows["rc_pearson_r"].dropna()
    if len(r):
        out["rc_pearson_r"] = {
            "median": float(r.median()),
            "p05": float(r.quantile(0.05)),
            "min": float(r.min()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cells", help="comma-separated cell ids for a smoke run "
                                    "(writes the *_smoke files, no manifest entry)")
    args = ap.parse_args()
    smoke = args.cells is not None
    out_parquet = SMOKE_PARQUET if smoke else OUT_PARQUET
    report_json = SMOKE_REPORT_JSON if smoke else REPORT_JSON

    print("Loading cleaned Severson dataset ...", flush=True)
    cells = load_clean_severson(PKL_DIR, verbose=True)
    ids = list(cells.keys())
    if smoke:
        ids = [c.strip() for c in args.cells.split(",")]
        unknown = [c for c in ids if c not in cells]
        if unknown:
            sys.exit(f"unknown cell ids: {unknown}")

    print(f"Extracting expanded features for {len(ids)} cells ...", flush=True)
    feat = extract_cells(cells, ids)

    report: dict = {"n_cells": len(ids), "n_rows": len(feat)}
    if BASELINE_PARQUET.exists():
        check_baseline(feat)
        report["baseline_identical"] = True
    rates, verdict = failure_rates(feat)
    report.update(rates)
    report.update(coverage(feat[feat["status"] == "ok"]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)", flush=True)

    report["verdict"] = verdict
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_json}", flush=True)

    if not smoke:
        log_manifest("eng05_extraction", {
            "script": "stages/s02_extract_features.py",
            "out": out_parquet.relative_to(REPO_ROOT).as_posix(),
            "n_rows": len(feat),
            "overall_fail_rate": rates["overall_fail_rate"],
            "verdict": verdict,
        })
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
