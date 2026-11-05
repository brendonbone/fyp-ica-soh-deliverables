"""
0. The three Severson batches from the published .mat files to pickles
=======================================================================
Reads the three MATLAB files published with Severson et al. (2019) from
MAT_DIR (DATA_DIR/"Severson Dataset") and writes PKL_DIR/batch{1,2,3}.pkl, the
raw data that stages 1-3 and 10 load. The conversion is the dataset authors'
BuildPkl_Batch{1,2,3}.ipynb (src/severson_mat.py); the three pickles it
writes are byte for byte the ones the study used (same SHA-256).
Inputs:  2017-05-12_, 2017-06-30_ and 2018-04-12_batchdata_updated_struct_errorcorrect.mat
         (3.0, 2.0 and 3.2 GB; see README.md, "Data", for where to get them).
Outputs: batch1.pkl, batch2.pkl, batch3.pkl (2.9, 2.1 and 3.1 GB); one
manifest entry per batch with its cells, cycles and size.
Runtime: about 7 min for all three on the study's laptop, at most ~4 GB of
RAM. Resumable by batch: a pickle that exists is kept (--force rebuilds it),
and each is written to a .part file first, so an interrupted run leaves none
half-written.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import MAT_DIR, MAT_FILES, PKL_DIR
from src.data_prep import log_manifest
from src.severson_mat import read_batch, write_pickle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mat-dir", type=Path, default=MAT_DIR, help="folder holding the three .mat files")
    ap.add_argument("--out-dir", type=Path, default=PKL_DIR, help="where batch{1,2,3}.pkl are written")
    ap.add_argument("--batches", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    ap.add_argument("--force", action="store_true", help="rebuild a pickle that exists")
    args = ap.parse_args()

    missing = [MAT_FILES[n] for n in args.batches if not (args.mat_dir / MAT_FILES[n]).exists()]
    if missing:
        sys.exit(f"not in {args.mat_dir}: {', '.join(missing)} (see README.md, 'Data')")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for n in args.batches:
        out = args.out_dir / f"batch{n}.pkl"
        if out.exists() and not args.force:
            print(f"[skip] {out} exists", flush=True)
            continue
        t0 = time.time()
        print(f"batch {n}: {MAT_FILES[n]}", flush=True)
        cells = read_batch(args.mat_dir / MAT_FILES[n], f"b{n}")
        write_pickle(cells, out)
        n_cycles = sum(len(cell["cycles"]) for cell in cells.values())
        size = out.stat().st_size
        print(f"Wrote {out}: {len(cells)} cells, {n_cycles} cycles, {size / 1e9:.2f} GB "
              f"in {time.time() - t0:.0f} s", flush=True)
        log_manifest(f"severson_batch{n}_pickle", {
            "script": "stages/s00_build_pickles.py",
            "mat_file": MAT_FILES[n],
            "n_cells": len(cells),
            "n_cycles": n_cycles,
            "bytes": size,
        })
        del cells


if __name__ == "__main__":
    main()
