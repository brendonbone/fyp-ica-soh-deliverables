# ICA peak features for SoH estimation: the study, reproducible

Code and results of the Honours thesis *Feature Ranking Incremental Capacity
Analysis for Data-Driven State-of-Health Estimation of Lithium Iron Phosphate
Batteries* (Brendon Bone, UNSW Canberra). The study asks which features of the dominant
discharge dQ/dV peak contribute most to state-of-health (SoH) estimation on the
Severson LFP dataset, and how much accuracy each adds.

Two scripts do everything:

| Script | What it does | Writes to |
|---|---|---|
| `run_results.py` | the raw data from the published files (stage 0), then every calculation, in the order the work was done (stages 1-20) | `data/`, `results/` |
| `run_report.py` | the report's tables and figures that come from the results, in the report's order and with its labels | `report/` |

The calculations never plot and the report never fits a model: a figure can be
restyled without touching a number, and a number can be recomputed without
touching a figure.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Python 3.14; versions pinned
python run_report.py                                 # the report from the committed results (~1 min)
python run_results.py --list                         # the 21 stages and what each writes
python run_results.py --only 0                       # the raw pickles from the published .mat files (~7 min)
python run_results.py --fast                         # the quick stages (needs the raw data for 0-3, 10)
python run_results.py                                # everything (days on one machine; see cluster/)
```

`run_results.py` skips a stage whose main output exists, so it resumes where it
stopped; `--force --only N` reruns one stage. The extracted feature table
(`results/features_expanded.parquet`) is committed, so every stage after
extraction, and the whole report, run without the raw data.

## Layout

```
run_results.py       the calculations, one titled line per stage
run_report.py        the report items, one line per table or figure
src/                 extraction and modelling code (no plotting): see src/__init__.py
stages/              one module per stage (s00 ... s20)
report_items/        one module per table or figure, plus common.py (formats, style)
results/             stored results: JSON / CSV per experiment, the feature tables
report/              NN_table_<label>_<name>.csv/.md and NN_fig_<label>_<name>.pdf/.png/.svg
notebooks/           01: tests of src against the stored table; 02: the feature viewer
tests/               pytest checks of src that run without the raw data
cluster/             running the slow stages on a PBS cluster (Katana)
```

## Data

The raw data are not committed. Download the three batch files published with
Severson et al. (2019) from <https://data.matr.io/1/> into

```
data/Severson Dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat   (3.0 GB, batch 1)
data/Severson Dataset/2017-06-30_batchdata_updated_struct_errorcorrect.mat   (2.0 GB, batch 2)
data/Severson Dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat   (3.2 GB, batch 3)
```

and stage 0 (`python run_results.py --only 0`, about 7 minutes) turns them
into the pickles the loaders read:

```
data/Severson Dataset/Severson pkl/batch1.pkl   (2.9 GB)
data/Severson Dataset/Severson pkl/batch2.pkl   (2.1 GB)
data/Severson Dataset/Severson pkl/batch3.pkl   (3.1 GB)
```

The conversion is the dataset authors' `BuildPkl_Batch{1,2,3}.ipynb`
(`src/severson_mat.py`), changed only where h5py 3 needs it (`[()]` for the
removed `.value`). Rebuilt this way from the published files, the three
pickles are byte for byte the ones the study used (same SHA-256). Stage 0
never overwrites a pickle; `python -m stages.s00_build_pickles --force`
rebuilds one. Point `ICA_SOH_DATA` at another folder (or make `data/` a
symlink / junction) to keep the data elsewhere. Stage 0 needs about 4 GB of
RAM and loading all three pickles about 8 GB. Only stages 1, 2, 3 and 10 read
the pickles.

Conventions fixed for the whole study (`src/data_prep.py`): rows cleaned in the
order status ok, SoH present, SoH > 0.5, SoH <= 1.05 (100,501 -> 100,331
cycles); development set = batches 1 + 2 (84 cells), held-out test = batch 3
(40 cells); five-fold grouped CV by cell, stratified on cycle-life quartiles;
seed 0 everywhere. Batch 3 is scored once per configuration, in the stages
that the verification subsection reports.

## The stages

Experiment IDs (ENG-xx) are the ones used while the work was done; the result
files keep them so the numbers can be traced.

| Stage | Experiment | Writes | Needs | Time |
|---|---|---|---|---|
| 0 The three Severson batches from the published .mat files to pickles | | `data/Severson Dataset/Severson pkl/batch{1,2,3}.pkl` | the .mat files | ~7 min |
| 1 Interim model: three peak features and a random forest | ENG-00 | `features_3feat.parquet`, `eng00_baseline.json` | raw data | minutes |
| 2 Extract the nine peak features of every cycle | ENG-05 | `features_expanded.parquet`, `eng05_extraction_report.json` | raw data | minutes |
| 3 Recomputed dQ/dV against the published curve; second peak; example curves | ENG-02 | `eng02_second_peak.json`, `dqdv_example_curves.csv`, `ic_curve_example.npz`, `feature_geometry.npz` | raw data | minutes |
| 4 Feature pool: correlations with SoH and with each other | ENG-04, ENG-16 | `eng04_feature_pool.json`, `eng16_*` | | seconds |
| 5 Linear, ridge, elastic-net, GPR and RF on all nine features | ENG-06/07 | `eng06_*`, `eng07_gpr.json` | | 3 min |
| 6 Nested-CV tuning of the RF and the GPR | ENG-09 | `eng09_tuning.json`, `eng09_tuned_params.json` | | ~1.5 h |
| 7 Ranking methods x regressors, untuned and tuned | ENG-08 | `eng08_full*.json`, `eng08_ranking_table*.csv` | 6 | ~1.5 h |
| 8 20-cycle history of each feature: table, checks, pools | ENG-14 | `eng14_*`, `features_history.parquet` | 6 | hours |
| 9 The same comparisons under six train/test partitions | ENG-15 | `eng15_*` | 7, 8 | ~12 h |
| 10 How much of the discharge each feature needs | ENG-17 | `eng17_*`, `features_soc_windows.parquet` | raw data | minutes |
| 11 All 511 subsets of the nine features, and what they show | ENG-19, ENG-19b | `eng19_*`, `eng19b_*` | 7, 10 | ~2.5 h |
| 12 The reference model over ten forest seeds | ENG-20 | `eng20_reference_seeds.json` | 11 | 15 min |
| 13 Permutation importance of feature groups | ENG-21 | `eng21_*` | 8 | 10 min |
| 14 All 127 subsets of the seven history components | ENG-22 | `eng22_*` | 8 | ~8 h (shardable) |
| 15 The 511 level subsets on the history rows | ENG-24 | `eng24_*` | 8, 11 | ~6 h (shardable) |
| 16 All 511 subsets of the features with their history | ENG-23 | `eng23_*` | 15 | ~12 h (shardable) |
| 17 Spearman ranking and backward elimination per partition | ENG-25 | `eng25_partition_robustness.json`, `eng25_progress_*.json` | 9, 11, 16 | ~17 h (11 tasks, shardable) |
| 18 Training without the 16 cells with noisy published curves | ENG-27 | `eng27_noisy_cells.json` | 12, 16 | 25 min |
| 19 The chosen feature sets under four models | ENG-28 | `eng28_feature_sets_models.*` | 11, 14, 15, 16 | ~75 min |
| 20 Learning curve: dev-CV R2 against the number of training cells | ENG-10 | `eng10_learning_curve.{json,csv}` | 2 | ~9 min |

Stage 20 (ENG-10) was added last; in time it belongs with the July stages,
after stage 5, and it only needs the feature table of stage 2.

Times are for the 24-thread laptop the study ran on. The long searches write a
progress file after every unit and skip finished units on a rerun; stages 14,
15, 16 and 17 take `--shard I --n-shards N` for a cluster job array
(`cluster/README.md`), then `--merge` / `--analyse` on any machine. Stages 7,
8 and 9 run by default only the grid cells the report uses (RF for every
ranking method, GPR for the Pearson ranking); `--regressors rf gpr
--selectors all` restores the full grids of the original study.

Every stage that takes minutes was rerun from this repository and reproduces
its stored result file, apart from round-off in the last binary digit of some
random-forest predictions (the forest adds its trees' outputs in thread order).
Of the long searches, the first units and the linear-model cells were rerun and
match the stored tables.

## The report items

`run_report.py --list` prints the order. The items are in the order of the
final report and carry its labels; the number in each file name is the item's
position in the `ORDER` list of `run_report.py`, so reordering the list
renumbers the files. Each item module says what it shows and which result
files it reads; `report/README.md` is the index of the last run.

| In the report | Items |
|---|---|
| Background and Method (II-B, IV-B, IV-E) | Fig. 1, Table II, Fig. 4 |
| Section V | Tables V-XVI, Figs. 6-9 |
| Appendix B | Fig. B1, Table B1 |
| Appendix C | Fig. C1 (the learning curve) |
| Appendix D | Table D1 |
| Not in the report (supplementary) | Tables S1-S8, Fig. S1: the six train/test partitions, every batch-3 score of the feature sets, the second-peak shares, the noisy cells and the full feature-set ranking |

The report's figures are these items' PDF files. Its tables are typed in the
LaTeX source from these items, often with fewer columns or rows (Tables II, VI,
VIII, XI, XII, XIV, XV, XVI and B1), and every number they print is in the item
with the same label. The report's Tables I, III, IV and A1 are typed in the
report, and its Figs. 2, 3 and 5 are made outside this repository (Fig. 5 by a
script in the report's folder).

## Changing things

- **A new table or figure**: add a module to `report_items/` with `LABEL`,
  `TITLE`, `SOURCES` and `make(stem)` (see `report_items/common.py` for the
  contract and the helpers), and one line in `ORDER` in `run_report.py` at the
  place it takes in the report. Everything after it renumbers itself.
- **A restyled figure**: edit its module; the numbers it reads do not move.
- **A new calculation**: add a module to `stages/` that reads `results/` and
  the raw data through `src.config` and writes result files only, and one line
  in `STAGES` in `run_results.py`.
- **A changed extractor or model**: change `src/`, rerun the stages from the
  first one affected (`run_results.py --force --only N ...`), then the report.

## Checking the code

```bash
.venv/Scripts/python -m pytest tests -q        # src against the committed table, no raw data
```

`notebooks/01_test_src_extraction.ipynb` does the same interactively and, with
the raw data, re-extracts a few cells and compares them with the committed
table. `notebooks/02_dqdv_and_feature_extraction.ipynb` draws every feature on
any cell and cycle, lists the 170 cycles that cleaning removes, and compares the
published dQ/dV with the one recomputed from the raw samples.
