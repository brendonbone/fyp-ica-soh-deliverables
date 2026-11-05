**Table V. Association of each feature with SoH on the development rows**

| Feature | Pearson r | r rank | Spearman ρ | ρ rank | MI (nats) | MI rank |
|---|---|---|---|---|---|---|
| area25 | 0.825 | 1 | 0.808 | 1 | 0.673 | 2 |
| area35 | 0.810 | 2 | 0.797 | 2 | 0.637 | 4 |
| area | 0.793 | 3 | 0.789 | 3 | 0.626 | 5 |
| position | 0.743 | 4 | 0.692 | 6 | 0.458 | 6 |
| height | 0.732 | 5 | 0.786 | 4 | 0.672 | 3 |
| prominence | 0.593 | 6 | 0.753 | 5 | 0.682 | 1 |
| slope_right | 0.578 | 7 | 0.600 | 8 | 0.341 | 8 |
| slope_left | 0.334 | 8 | 0.611 | 7 | 0.353 | 7 |
| fwhm | −0.122 | 9 | −0.203 | 9 | 0.098 | 9 |

Development rows only: 59,131 clean cycles of the 84 cells in batches 1 and 2. Rows are ordered by |r|; each rank is by |r|, |ρ| or MI (1 = strongest association).

MI: k-nearest-neighbour estimate (scikit-learn mutual_info_regression, 3 neighbours, random_state 0) on all development rows, in nats; it is the score of the mutual-information selector in ENG-08.

Kendall τ between the rankings: Pearson–Spearman 0.83, Pearson–MI 0.50, Spearman–MI 0.67.

The rows are about 704 cycles from each cell and are not independent, so the coefficients are ranking scores, not test statistics; no p-values are given.

Sources: `results/eng16_pearson_vs_spearman.csv`, `results/eng08_full.json`, `results/features_expanded.parquet`
