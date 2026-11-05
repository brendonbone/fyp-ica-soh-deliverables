**Table S6. Stability of each feature ranking across the six train/test partitions**

| Features | Method | Kendall τ, mean | Kendall τ, min | Top-3 Jaccard, mean | Top feature (B1, B2, B3, M1, M2, M3) | Top feature, count |
|---|---|---|---|---|---|---|
| Nine level features | Pearson \|r\| | 0.97 | 0.94 | 1.00 | area25, area25, area25, area25, area25, area25 | all six: area25 |
| Nine level features | Spearman \|ρ\| | 0.81 | 0.56 | 0.62 | height, height, area25, height, area25, height | height 4, area25 2 |
| Nine level features | Mutual information | 0.91 | 0.83 | 1.00 | prominence, height, prominence, prominence, prominence, prominence | prominence 5, height 1 |
| Nine level features | RF impurity | 0.61 | 0.39 | 0.59 | height, area25, area25, height, height, height | height 4, area25 2 |
| Nine level features | Permutation importance | 0.49 | 0.28 | 0.41 | height, height, area25, height, height, height | height 5, area25 1 |
| Level + history (63 columns) | Pearson \|r\| | 0.97 | 0.94 | 0.49 | slope_right_rm20, area25_rm20, area25_rm20, area25_rm20, area25_rm20, area25_rm20 | area25_rm20 5, slope_right_rm20 1 |
| Level + history (63 columns) | Mutual information | 0.92 | 0.85 | 1.00 | height_rm20, height_rm20, area25_rm20, height_rm20, height_rm20, height_rm20 | height_rm20 5, area25_rm20 1 |
| Level + history (63 columns) | RF impurity | 0.84 | 0.73 | 0.63 | slope_right_rm20, area25_rm20, area25_rm20, area25_rm20, area25_rm20, slope_right_rm20 | area25_rm20 4, slope_right_rm20 2 |
| Level + history (63 columns) | Permutation importance | 0.63 | 0.47 | 0.55 | slope_right_rm20, area25_rm20, area25_rm20, slope_right_rm20, area25_rm20, area25_rm20 | area25_rm20 4, slope_right_rm20 2 |

Kendall τ between the rankings of each pair of partitions (15 pairs), mean and minimum; top-3 Jaccard likewise. Each ranking uses only the partition's development cells. B1, B2, B3 = batch 1, 2, 3 held out (B3 = the study split); M1–M3 = the mixed thirds.

Pearson, mutual information, RF impurity and permutation importance (untuned RF, mean over the five inner folds) from ENG-15; the first three do not depend on the regressor. Spearman from ENG-25.

With history: the 63-column combined pool on the history rows (Spearman was not run there).

Sources: `results/eng15_cells.json`, `results/eng25_partition_robustness.json`
