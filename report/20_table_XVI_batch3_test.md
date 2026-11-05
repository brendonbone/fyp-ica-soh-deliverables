**Table XVI. Batch-3 RMSE, R² and MAE of every configuration selected in Subsections C–G, beside its dev-CV RMSE**

| Sub. | Configuration | Model | Rows | Cols | Dev-CV RMSE | Batch-3 RMSE | Batch-3 R² | Batch-3 MAE | Source |
|---|---|---|---|---|---|---|---|---|---|
| C | Interim three features (height, position, area) | RF, untuned | full | 3 | 0.02302 | 0.01681 | 0.830 | 0.01207 | ENG-00 |
| C | All nine features | LR | full | 9 | 0.02729 | 0.01862 | 0.791 | 0.01381 | ENG-06/07 |
| C | All nine features | GPR, untuned | full | 9 | 0.02046 | 0.01681 | 0.829 | 0.01278 | ENG-07, ENG-09 |
| C | All nine features | GPR, tuned | full | 9 | 0.02028 | 0.01620 | 0.842 | 0.01230 | ENG-08, ENG-09 |
| C | All nine features | RF, untuned | full | 9 | 0.02015 | 0.01605 | 0.845 | 0.01166 | ENG-09 |
| C | All nine features | RF, tuned | full | 9 | 0.01999 | 0.01565 | 0.852 | 0.01138 | ENG-09, ENG-19b |
| **C** | **Reference: all nine features, 20 forests** | **RF, tuned** | **full** | **9** | **0.02001 ± 0.00002** | **0.01564 ± 0.00004** | **0.853 ± 0.001** | **0.01137 ± 0.00002** | **ENG-20** |
| D | Rule pick: position, fwhm, slope_right, prominence | RF, untuned | full | 4 | 0.02076 | 0.01756 | 0.814 | 0.01314 | ENG-19, ENG-19b |
| D | Rule pick: position, fwhm, slope_right, prominence | RF, tuned | full | 4 | 0.02059 | 0.01680 | 0.830 | 0.01267 | ENG-19b |
| D | Dev-CV best: all but slope_left | RF, untuned | full | 8 | 0.02013 | 0.01640 | 0.838 | 0.01198 | ENG-19, ENG-19b |
| D | Dev-CV best: all but slope_left | RF, tuned | full | 8 | 0.02001 | 0.01602 | 0.845 | 0.01174 | ENG-19b |
| E | All nine features | RF, untuned | history | 9 | 0.02038 | 0.01620 | 0.842 | 0.01183 | ENG-24 |
| E | All nine features | RF, tuned | history | 9 | 0.02029 | 0.01579 | 0.850 | 0.01158 | ENG-28 |
| E | All nine + history | RF, untuned | history | 63 | 0.01817 | 0.01413 | 0.880 | 0.00991 | ENG-23 |
| E | All nine + history | RF, tuned | history | 63 | 0.01757 | 0.01328 | 0.894 | 0.00938 | ENG-23 |
| E | All nine: level + rm20 + drift | RF, untuned | history | 27 | 0.01883 | 0.01459 | 0.872 | 0.01026 | ENG-22, ENG-28 |
| E | All nine: level + rm20 + drift | RF, tuned | history | 27 | 0.01828 | 0.01444 | 0.875 | 0.01013 | ENG-28 |
| E | All nine + history, without rm20 † | RF, untuned | history | 54 | 0.01921 | 0.01202 | 0.913 | 0.00819 | ENG-22 |
| F | Rule pick: height, fwhm + history | RF, untuned | history | 14 | 0.01846 | 0.01697 | 0.827 | 0.01092 | ENG-23 |
| F | Rule pick: height, fwhm + history | RF, tuned | history | 14 | 0.01781 | 0.01479 | 0.869 | 0.00983 | ENG-23 |
| F | Budget 13.6%: height, position + history | RF, untuned | history | 14 | 0.01934 | 0.01205 | 0.913 | 0.00863 | ENG-23 |
| F | Budget 13.6%: height, position + history | RF, tuned | history | 14 | 0.01879 | 0.01162 | 0.919 | 0.00832 | ENG-23 |
| F | Budget 42.3%: all but fwhm, prominence (level only) | RF, untuned | history | 7 | 0.02170 | 0.01634 | 0.840 | 0.01194 | ENG-24 |
| F | Budget 42.3%: all but fwhm, prominence (level only) | RF, tuned | history | 7 | 0.02147 | 0.01600 | 0.846 | 0.01173 | ENG-28 |
| F | Dev-CV best: all but area35, area25 + history | RF, untuned | history | 49 | 0.01775 | 0.01509 | 0.863 | 0.01031 | ENG-23 |
| F | Dev-CV best: all but area35, area25 + history | RF, tuned | history | 49 | 0.01695 | 0.01326 | 0.894 | 0.00933 | ENG-23 |
| G | Budget 42.3% + history: all but fwhm, prominence | RF, untuned | history | 49 | 0.01857 | 0.01335 | 0.893 | 0.00899 | ENG-28 |
| G | Budget 42.3% + history: all but fwhm, prominence | RF, tuned | history | 49 | 0.01817 | 0.01305 | 0.898 | 0.00904 | ENG-28 |

Dev-CV = pooled RMSE averaged over the five development folds (batches 1 + 2); batch 3 = one refit on all development rows, scored once on the batch-3 rows (R² and MAE are batch 3). Rows: full = all clean cycles; history = cycles with at least 20 earlier clean cycles, so the E, F and G rows are compared on the same rows.

Bold: the reference model of C, the baseline the other rows are compared with (mean ± sd over 20 forests, seeds 0–9 × pool and Pearson column order, ENG-20); every other RF row is one 300-tree forest with seed 0. No row was chosen on batch 3: every configuration was selected on dev-CV (or fixed in advance) in C–G.

Rule pick = the one-standard-error rule on the subset search; Budget = the best subset needing at most that share of the discharge (SoC span). † Observation, not a selection: ENG-22 scored each drop-one-component design on batch 3 as a listed design.

Sources: `results/eng00_baseline.json`, `results/eng06_07_model_table.csv`, `results/eng08_ranking_table_tuned.csv`, `results/eng09_tuning.json`, `results/eng19_exhaustive_subsets.csv`, `results/eng19b_analysis.json`, `results/eng20_reference_seeds.json`, `results/eng22_history_components.csv`, `results/eng22_history_components_analysis.json`, `results/eng23_grouped_history_subsets.csv`, `results/eng23_grouped_history_subsets_analysis.json`, `results/eng24_level_subsets_history_rows.csv`, `results/eng24_level_subsets_history_rows_analysis.json`, `results/eng28_feature_sets_models.json`
