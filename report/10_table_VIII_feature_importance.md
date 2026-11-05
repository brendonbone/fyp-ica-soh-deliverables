**Table VIII. Permutation importance of the reference model (tuned RF) and impurity importance of the untuned RF, beside the ranks of Table V**

| Feature | Pearson rank | Spearman rank | MI rank | PI, tuned RF (± fold sd) | PI rank | Impurity, untuned RF | Impurity rank |
|---|---|---|---|---|---|---|---|
| height | 5 | 4 | 3 | 0.0088 ± 0.0028 | 1 | 0.336 | 2 |
| area25 | 1 | 1 | 2 | 0.0064 ± 0.0015 | 2 | 0.413 | 1 |
| area | 3 | 3 | 5 | 0.0056 ± 0.0009 | 3 | 0.081 | 3 |
| fwhm | 9 | 9 | 9 | 0.0046 ± 0.0011 | 4 | 0.046 | 4 |
| position | 4 | 6 | 6 | 0.0039 ± 0.0006 | 5 | 0.034 | 5 |
| prominence | 6 | 5 | 1 | 0.0036 ± 0.0009 | 6 | 0.027 | 6 |
| area35 | 2 | 2 | 4 | 0.0026 ± 0.0002 | 7 | 0.016 | 9 |
| slope_right | 7 | 8 | 8 | 0.0024 ± 0.0006 | 8 | 0.025 | 7 |
| slope_left | 8 | 7 | 7 | 0.0015 ± 0.0005 | 9 | 0.023 | 8 |

Rows in order of permutation importance. Pearson and Spearman ranks: |r| and |ρ| with SoH on all development rows (Table V). MI: Kraskov estimate on all development rows (mutual_info_regression, random_state 0).

PI: rise in validation RMSE (SoH units) when one feature's column is shuffled; tuned RF (ENG-09 configuration) on all nine features, pool column order, seed 0; ten repeats per fold, mean over the five development folds; ± is the sd over the folds (ENG-08 tuned grid; fold values from ENG-21, which reproduces it exactly).

Impurity: mean decrease in impurity of the untuned 300-tree RF (pool order, seed 0), averaged over the five fold-training fits; it sums to 1.

Agreement of the five rankings: mean pairwise Kendall τ 0.44, mean top-three Jaccard overlap 0.54.

The CSV also holds the untuned RF's PI and the exact Shapley values of Section V-D (511-subset search, untuned RF).

Sources: `results/eng16_pearson_vs_spearman.csv`, `results/eng08_full.json`, `results/eng08_full_tuned.json`, `results/eng21_grouped_permutation.json`, `results/eng19b_analysis.json`
