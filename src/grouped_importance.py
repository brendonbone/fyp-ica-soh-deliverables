"""
Grouped permutation importance (ENG-21).
========================================
Permutation importance (PI) of a GROUP of columns: the same row permutation
is applied to every column of the group, so the group's joint information
is destroyed while its internal structure is kept. For near-duplicate
columns (the three areas; a feature and its rolling mean) per-column PI
under-states what the model loses, because the copy left unshuffled stands
in for the shuffled one; the group's PI does not have that problem.

The random-number handling copies sklearn.inspection.permutation_importance
(sklearn 1.8) line for line: one seed is drawn from `random_state`, every
group starts a fresh RandomState from it, `shuffling_idx` is reshuffled in
place on each repeat and applied to the already-permuted column(s). A
one-column group therefore gives exactly sklearn's importance for that
column with the same `random_state` (checked in scripts/eng21_*).
Importance = permuted RMSE - baseline RMSE (scorer neg RMSE, as
harness._sel_permutation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state

from src.data_prep import RF_SEED


def _group_scores(estimator, X: pd.DataFrame, y, idx: list[int], seed: int,
                  n_repeats: int, scorer) -> np.ndarray:
    rs = check_random_state(seed)
    Xp = X.copy()
    shuffling_idx = np.arange(Xp.shape[0])
    scores = []
    for _ in range(n_repeats):
        rs.shuffle(shuffling_idx)
        for ci in idx:                     # one permutation, every column of the group
            col = Xp.iloc[shuffling_idx, ci]
            col.index = Xp.index
            Xp[Xp.columns[ci]] = col
        scores.append(scorer(estimator, Xp, y))
    return np.array(scores)


def grouped_permutation_importance(estimator, X: pd.DataFrame, y, groups: dict[str, list[str]],
                                   *, n_repeats: int = 10, random_state=RF_SEED,
                                   scoring: str = "neg_root_mean_squared_error",
                                   n_jobs: int | None = None) -> dict[str, np.ndarray]:
    """{group name: importances over the n_repeats} for a fitted estimator."""
    rs = check_random_state(random_state)
    seed = rs.randint(np.iinfo(np.int32).max + 1)
    scorer = check_scoring(estimator, scoring=scoring)
    baseline = scorer(estimator, X, y)
    names = list(groups)
    idx = [[X.columns.get_loc(c) for c in groups[g]] for g in names]
    scores = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_group_scores)(estimator, X, y, ix, seed, n_repeats, scorer) for ix in idx)
    return {g: baseline - s for g, s in zip(names, scores)}
