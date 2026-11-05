"""
Where things are, and the fixed conventions every stage shares.
================================================================
Every path in the project is anchored here, so the scripts can be run from
any working directory:

    REPO_ROOT    this repository
    DATA_DIR     the raw data (default REPO_ROOT/data; override with the
                 ICA_SOH_DATA environment variable, e.g. a scratch disk on a
                 cluster or a junction to another checkout)
    MAT_DIR      DATA_DIR/"Severson Dataset", holding the three published .mat
                 files (MAT_FILES) that stage 0 turns into the pickles
    PKL_DIR      DATA_DIR/"Severson Dataset"/"Severson pkl"/batch{1,2,3}.pkl
    RESULTS_DIR  the stored results that run_results.py writes and
                 run_report.py reads
    REPORT_DIR   the numbered tables and figures that run_report.py writes

The modelling conventions (cleaning band, folds, seed) live in
src/data_prep.py; the feature names in src/features.py.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = Path(os.environ.get("ICA_SOH_DATA", REPO_ROOT / "data"))
MAT_DIR: Path = DATA_DIR / "Severson Dataset"
PKL_DIR: Path = DATA_DIR / "Severson Dataset" / "Severson pkl"

# The three files published with Severson et al. (2019), by batch number
MAT_FILES: dict[int, str] = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

RESULTS_DIR: Path = REPO_ROOT / "results"
REPORT_DIR: Path = REPO_ROOT / "report"


def results(name: str) -> Path:
    """results/<name>, creating the folder on first use."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def mats_available() -> bool:
    """True when the three published Severson .mat files are on disk."""
    return all((MAT_DIR / name).exists() for name in MAT_FILES.values())


def pickles_available() -> bool:
    """True when the three Severson batch pickles are on disk."""
    return all((PKL_DIR / f"batch{n}.pkl").exists() for n in (1, 2, 3))
