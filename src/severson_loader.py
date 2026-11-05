"""
Severson dataset loader — clean interface over the three pickled batches.
=========================================================================
Implements the cell-exclusion and batch-merge logic from the MIT-Stanford
`Load Data.ipynb` notebook, wrapped in a tidy module you can import from
anywhere in the project:

    from src.severson_loader import load_clean_severson, get_splits

    cells = load_clean_severson()          # dict[str, dict]
    train, primary_test, secondary_test = get_splits(cells)

The exclusion/merge logic follows the dataset authors' `Load Data.ipynb`
(github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation).
The pickles themselves are built from the published .mat files by the same
notebook (BuildPkl_Batch{1,2,3}.ipynb); see README.md, "Data".

Why this module exists:
- Three pickles live on disk; every analysis starts with the same load/clean
  sequence. Doing it once in a module avoids copy-pasting 40 lines of
  exclusion/merge logic into every notebook.
- Lets the rest of your project write `cells = load_clean_severson()` and
  get on with the actual work.

Memory warning:
- Loading all three pickles uses ~8 GB RAM. Close Chrome tabs etc.
- If this is a problem, use `load_single_batch(n)` to work with one at a time.
"""
from pathlib import Path

import pickle

import numpy as np

from src.config import PKL_DIR  # data/Severson Dataset/Severson pkl/batch{1,2,3}.pkl


# ------------------------ configuration ------------------------

# Cells that did NOT reach 80% of nominal capacity within the test window.
# Severson excludes these because their SOH trajectory is incomplete.
BATCH1_INCOMPLETE_CELLS = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]

# Continuation cells: these cells in batch2 are actually the SAME physical
# cells as b1c0–b1c4 from batch1, retested later. Their cycle data must be
# stitched onto the end of batch1's record. Values come from Load Data.ipynb.
CONTINUATION_MAP = {
    "b2c7": ("b1c0", 662),
    "b2c8": ("b1c1", 981),
    "b2c9": ("b1c2", 1060),
    "b2c15": ("b1c3", 208),
    "b2c16": ("b1c4", 482),
}

# Noisy cells in batch3 that Severson excludes due to measurement issues.
BATCH3_NOISY_CELLS = ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]


# ------------------------ loading ------------------------

def load_single_batch(batch_num: int, pkl_dir: Path = PKL_DIR) -> dict:
    """Load one batch pickle. No cleaning applied."""
    assert batch_num in (1, 2, 3), "batch_num must be 1, 2, or 3"
    pkl_path = pkl_dir / f"batch{batch_num}.pkl"
    with open(pkl_path, "rb") as fp:
        return pickle.load(fp)


def _merge_continuation_cell(
    batch1_cell: dict, batch2_cell: dict, add_len: int
) -> None:
    """Stitch batch2_cell's cycles onto the end of batch1_cell. In-place."""
    # Update the total cycle life — it was measured over both experimental runs.
    batch1_cell["cycle_life"] = batch1_cell["cycle_life"] + add_len

    # Concatenate each summary field. The 'cycle' field needs a cycle-number
    # offset so the numbering stays monotonic across the join.
    for field in batch1_cell["summary"].keys():
        b1_vals = batch1_cell["summary"][field]
        b2_vals = batch2_cell["summary"][field]
        if field == "cycle":
            b2_vals = b2_vals + len(b1_vals)
        batch1_cell["summary"][field] = np.hstack((b1_vals, b2_vals))

    # Append batch2's per-cycle data onto batch1's, renumbering the keys so
    # the dict stays "0", "1", ..., "N" contiguous.
    offset = len(batch1_cell["cycles"])
    for j, key in enumerate(batch2_cell["cycles"].keys()):
        batch1_cell["cycles"][str(offset + j)] = batch2_cell["cycles"][key]


def load_clean_severson(pkl_dir: Path = PKL_DIR, *, verbose: bool = True) -> dict:
    """
    Return a dict {cell_id: cell_dict} with exclusions and merges applied.

    Steps:
      1. Load batch1, drop incomplete cells.
      2. Load batch2, merge continuation cells into batch1, drop those from batch2.
      3. Load batch3, drop noisy cells.
      4. Combine all three into one dict.

    Expected final cell count: 41 (batch1) + 43 (batch2) + 40 (batch3) = 124.
    """
    # ---- batch 1 ----
    batch1 = load_single_batch(1, pkl_dir)
    for key in BATCH1_INCOMPLETE_CELLS:
        batch1.pop(key, None)
    if verbose:
        print(f"  batch1 after exclusions: {len(batch1)} cells")

    # ---- batch 2 ----
    batch2 = load_single_batch(2, pkl_dir)
    for b2_key, (b1_key, add_len) in CONTINUATION_MAP.items():
        _merge_continuation_cell(batch1[b1_key], batch2[b2_key], add_len)
    for b2_key in CONTINUATION_MAP:
        batch2.pop(b2_key, None)
    if verbose:
        print(f"  batch2 after merges + drops: {len(batch2)} cells")

    # ---- batch 3 ----
    batch3 = load_single_batch(3, pkl_dir)
    for key in BATCH3_NOISY_CELLS:
        batch3.pop(key, None)
    if verbose:
        print(f"  batch3 after exclusions: {len(batch3)} cells")

    # ---- combine ----
    # Dict union; keys are unique by batch prefix so no collisions.
    all_cells = {**batch1, **batch2, **batch3}
    if verbose:
        print(f"  TOTAL: {len(all_cells)} cells")
    return all_cells


# ------------------------ train/test splits ------------------------

def get_splits(cells: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Reproduce the Severson paper's train / primary-test / secondary-test split.

    Cells from batches 1+2 are interleaved:
      - training set      = odd-indexed cells (1, 3, 5, ...)
      - primary test set  = even-indexed cells (0, 2, 4, ...) plus cell index 83
    All of batch3 is the secondary test set.

    Returns three lists of cell_id strings (e.g. ['b1c1', 'b1c3', ...]).

    This is the CELL-LEVEL split — no cell's cycles ever appear in more than
    one set. Using this as your X_train / X_test split prevents the leakage
    trap that makes battery-ML results look artificially perfect.
    """
    cell_ids = list(cells.keys())

    # Cells are ordered batch1 first, then batch2, then batch3 because of
    # the {**batch1, **batch2, **batch3} merge order above.
    batch3_start = next(
        i for i, key in enumerate(cell_ids) if key.startswith("b3")
    )
    num_bat12 = batch3_start  # count of cells from batches 1+2

    train_idx = np.arange(1, num_bat12 - 1, 2)
    primary_test_idx = np.hstack((np.arange(0, num_bat12, 2), [83]))
    secondary_test_idx = np.arange(batch3_start, len(cell_ids))

    train_ids = [cell_ids[i] for i in train_idx]
    primary_test_ids = [cell_ids[i] for i in primary_test_idx]
    secondary_test_ids = [cell_ids[i] for i in secondary_test_idx]

    return train_ids, primary_test_ids, secondary_test_ids


# ------------------------ convenience ------------------------

def summarise_cell(cell: dict) -> dict:
    """One-line-per-cell summary useful for sanity checks.

    Note on SOH_last: a few Severson cells have qd[0] == 0 (formation-cycle
    artifact in the original .mat files). We use the first non-zero discharge
    capacity as the reference instead, and return NaN if no valid value exists.
    The Severson paper itself uses the 1.1 Ah nominal capacity as the
    reference — that's a defensible alternative if you want a constant
    denominator across all cells.
    """
    qd = np.asarray(cell["summary"]["QD"])
    valid = qd[qd > 0]
    qd_ref = float(valid[0]) if valid.size > 0 else float("nan")
    soh_last = float(qd[-1] / qd_ref) if qd_ref > 0 else float("nan")
    return {
        "policy":      cell["charge_policy"],
        "cycle_life":  float(np.atleast_1d(cell["cycle_life"]).ravel()[0]),
        "num_cycles":  len(qd),
        "QD_first":    float(qd[0]),
        "QD_last":     float(qd[-1]),
        "SOH_last":    soh_last,  # qd[-1] / first non-zero qd
    }


if __name__ == "__main__":
    # Running this file directly prints a sanity summary.
    print(f"Loading from: {PKL_DIR}")
    cells = load_clean_severson()
    print()
    train, primary, secondary = get_splits(cells)
    print(f"train cells:            {len(train)}")
    print(f"primary test cells:     {len(primary)}")
    print(f"secondary test cells:   {len(secondary)}")
    print(f"(sum = {len(train) + len(primary) + len(secondary)}, total = {len(cells)})")

    print("\n--- sample cell summaries ---")
    for cid in list(cells.keys())[:3]:
        print(f"  {cid}: {summarise_cell(cells[cid])}")
