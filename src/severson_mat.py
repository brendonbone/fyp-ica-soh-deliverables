"""
The published Severson .mat files as the per-batch dicts the pickles hold.
==========================================================================
A port of the dataset authors' BuildPkl_Batch{1,2,3}.ipynb
(github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation),
which stage 0 runs once per batch:

    from src.config import MAT_DIR, MAT_FILES
    from src.severson_mat import read_batch, write_pickle

    cells = read_batch(MAT_DIR / MAT_FILES[2], "b2")    # {"b2c0": {...}, ...}
    write_pickle(cells, PKL_DIR / "batch2.pkl")

The one change is the one h5py 3 needs: the notebooks read a dataset with
`.value`, which h5py 3.0 removed, and this reads it with `[()]`. The fields,
their names, the key order and the np.hstack of every array are the
notebooks', so the pickles come out equal to the ones the study used. The cell
exclusions and the batch-1/batch-2 merge are not done here: they are applied
when the pickles are loaded (src/severson_loader.py).

Each .mat file is MATLAB v7.3, that is HDF5. batch/<field> is an (n_cells, 1)
array of references; a cell's summary is a group of (1, n_cycles) arrays, and
its cycles group holds, per field, (n_cycles, 1) references to one
(1, n_samples) array per cycle.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import h5py
import numpy as np

# pickle key -> .mat field, in the notebooks' order (the pickled dicts keep it)
SUMMARY_FIELDS = {"IR": "IR", "QC": "QCharge", "QD": "QDischarge", "Tavg": "Tavg", "Tmin": "Tmin",
                  "Tmax": "Tmax", "chargetime": "chargetime", "cycle": "cycle"}
CYCLE_FIELDS = {"I": "I", "Qc": "Qc", "Qd": "Qd", "Qdlin": "Qdlin", "T": "T", "Tdlin": "Tdlin",
                "V": "V", "dQdV": "discharge_dQdV", "t": "t"}


def read_cell(f: h5py.File, batch: h5py.Group, i: int) -> dict:
    """Cell i of a batch: cycle life, charge policy, per-cycle summary and per-cycle samples."""
    summary = f[batch["summary"][i, 0]]
    cycles = f[batch["cycles"][i, 0]]
    return {
        "cycle_life": f[batch["cycle_life"][i, 0]][()],
        # a MATLAB char array is UTF-16, so every other byte is the text
        "charge_policy": f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode(),
        "summary": {key: np.hstack(summary[field][0, :].tolist())
                    for key, field in SUMMARY_FIELDS.items()},
        "cycles": {str(j): {key: np.hstack(f[cycles[field][j, 0]][()])
                            for key, field in CYCLE_FIELDS.items()}
                   for j in range(cycles["I"].shape[0])},
    }


def read_batch(mat_path: Path, prefix: str, verbose: bool = True) -> dict[str, dict]:
    """Every cell of one .mat file, keyed '<prefix>c<i>' as in the notebooks."""
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        cells: dict[str, dict] = {}
        for i in range(n):
            cells[f"{prefix}c{i}"] = read_cell(f, batch, i)
            if verbose and ((i + 1) % 10 == 0 or i == n - 1):
                print(f"  {i + 1}/{n} cells", flush=True)
    return cells


def write_pickle(cells: dict, path: Path) -> None:
    """pickle.dump as the notebooks do it, through a .part file so a stopped run leaves no half-written pickle."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fp:
        pickle.dump(cells, fp)
    os.replace(tmp, path)
