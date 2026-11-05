"""
src: the extraction and modelling code (no plotting).

    config.py               paths and the data location
    severson_mat.py         the published .mat files as the per-batch pickles (stage 0)
    severson_loader.py      load + clean the 124 Severson cells
    dqdv.py                 recompute discharge dQ/dV from the raw V, Q samples
    peak_extraction.py      the interim three-feature extractor (height, position, area)
    features.py             the nine-feature pool, second-peak search, recompute check
    features_temporal.py    the 20-cycle history columns of each feature
    feature_requirements.py the state-of-charge window each feature needs
    data_prep.py            row cleaning, dev/test split, CV folds, metrics
    harness.py              regressors, ranking methods, grouped CV, held-out scoring
    subset_search.py        exhaustive block-subset searches and their derivations
    grouped_importance.py   permutation importance of a group of columns
    partitions.py           the six train/test partitions of the split-design check (stages 9, 17)

The calculation stages in stages/ call these; run_results.py runs the stages
in order and run_report.py turns the stored results into the thesis tables
and figures.
"""
