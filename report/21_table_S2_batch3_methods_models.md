**Table S2. Batch-3 RMSE, R² and MAE of every ENG-28 feature set of Subsection G under each model (ENG-28), beside their dev-CV RMSE**

| Feature set | Model | Cols | Discharge needed | Dev-CV RMSE | Dev-CV rank | Batch-3 RMSE | Batch-3 rank | Batch-3 R² | Batch-3 MAE |
|---|---|---|---|---|---|---|---|---|---|
| height, position, area | LR | 3 | 42.3% | 0.02877 | 39 | 0.02275 | 43 | 0.689 | 0.01600 |
| height, position, area | GPR, untuned | 3 | 42.3% | 0.02220 | 29 | 0.01564 | 21 | 0.853 | 0.01113 |
| height, position, area | RF, untuned | 3 | 42.3% | 0.02329 | 31 | 0.01692 | 29 | 0.828 | 0.01223 |
| height, position, area | RF, tuned | 3 | 42.3% | 0.02307 | 30 | 0.01640 | 27 | 0.838 | 0.01192 |
| all nine features | LR | 9 | 96.4% | 0.02751 | 38 | 0.01856 | 39 | 0.793 | 0.01383 |
| all nine features | GPR, untuned | 9 | 96.4% | 0.02087 | 20 | 0.01796 | 35 | 0.806 | 0.01355 |
| all nine features | RF, untuned | 9 | 96.4% | 0.02038 | 18 | 0.01620 | 24 | 0.842 | 0.01183 |
| all nine features | RF, tuned | 9 | 96.4% | 0.02029 | 16 | 0.01579 | 22 | 0.850 | 0.01158 |
| position, fwhm, slope_right, prominence | LR | 4 | 96.4% | 0.03139 | 44 | 0.02727 | 44 | 0.553 | 0.01886 |
| position, fwhm, slope_right, prominence | GPR, untuned | 4 | 96.4% | 0.02166 | 24 | 0.01729 | 32 | 0.821 | 0.01291 |
| position, fwhm, slope_right, prominence | RF, untuned | 4 | 96.4% | 0.02103 | 22 | 0.01779 | 34 | 0.810 | 0.01338 |
| position, fwhm, slope_right, prominence | RF, tuned | 4 | 96.4% | 0.02089 | 21 | 0.01700 | 31 | 0.826 | 0.01290 |
| all but fwhm, prominence | LR | 7 | 42.3% | 0.02709 | 37 | 0.01805 | 36 | 0.804 | 0.01350 |
| all but fwhm, prominence | GPR, untuned | 7 | 42.3% | 0.02194 | 28 | 0.01665 | 28 | 0.834 | 0.01221 |
| all but fwhm, prominence | RF, untuned | 7 | 42.3% | 0.02170 | 25 | 0.01634 | 26 | 0.840 | 0.01194 |
| all but fwhm, prominence | RF, tuned | 7 | 42.3% | 0.02147 | 23 | 0.01600 | 23 | 0.846 | 0.01173 |
| all nine + history | LR | 63 | 96.4% | 0.02642 | 35 | 0.01875 | 41 | 0.789 | 0.01576 |
| all nine + history | GPR, untuned | 63 | 96.4% | 0.02191 | 27 | 0.01311 | 9 | 0.897 | 0.01044 |
| all nine + history | RF, untuned | 63 | 96.4% | 0.01817 | 4 | 0.01413 | 15 | 0.880 | 0.00991 |
| all nine + history | RF, tuned | 63 | 96.4% | 0.01757 | 1 | 0.01328 | 10 | 0.894 | 0.00938 |
| all nine: level + rm20 + drift | LR | 27 | 96.4% | 0.02651 | 36 | 0.01865 | 40 | 0.791 | 0.01565 |
| all nine: level + rm20 + drift | GPR, untuned | 27 | 96.4% | 0.02034 | 17 | 0.01621 | 25 | 0.842 | 0.01257 |
| all nine: level + rm20 + drift | RF, untuned | 27 | 96.4% | 0.01883 | 11 | 0.01459 | 17 | 0.872 | 0.01026 |
| all nine: level + rm20 + drift | RF, tuned | 27 | 96.4% | 0.01828 | 6 | 0.01444 | 16 | 0.875 | 0.01013 |
| height, fwhm + history | LR | 14 | 96.4% | 0.02903 | 41 | 0.01747 | 33 | 0.817 | 0.01376 |
| height, fwhm + history | GPR, untuned | 14 | 96.4% | 0.02499 | 34 | 0.01244 | 5 | 0.907 | 0.00826 |
| height, fwhm + history | RF, untuned | 14 | 96.4% | 0.01846 | 8 | 0.01697 | 30 | 0.827 | 0.01092 |
| height, fwhm + history | RF, tuned | 14 | 96.4% | 0.01781 | 2 | 0.01479 | 19 | 0.869 | 0.00983 |
| height, position + history | LR | 14 | 13.6% | 0.03103 | 43 | 0.01840 | 37 | 0.797 | 0.01309 |
| height, position + history | GPR, untuned | 14 | 13.6% | 0.02186 | 26 | 0.01406 | 13 | 0.881 | 0.01039 |
| height, position + history | RF, untuned | 14 | 13.6% | 0.01934 | 13 | 0.01205 | 3 | 0.913 | 0.00863 |
| height, position + history | RF, tuned | 14 | 13.6% | 0.01879 | 10 | 0.01162 | 1 | 0.919 | 0.00832 |
| height, position, area + history | LR | 21 | 42.3% | 0.02886 | 40 | 0.01891 | 42 | 0.785 | 0.01312 |
| height, position, area + history | GPR, untuned | 21 | 42.3% | 0.02072 | 19 | 0.01301 | 7 | 0.898 | 0.00937 |
| height, position, area + history | RF, untuned | 21 | 42.3% | 0.01890 | 12 | 0.01222 | 4 | 0.910 | 0.00879 |
| height, position, area + history | RF, tuned | 21 | 42.3% | 0.01826 | 5 | 0.01179 | 2 | 0.917 | 0.00844 |
| position, fwhm, slope_right, prominence + history | LR | 28 | 96.4% | 0.03095 | 42 | 0.01374 | 12 | 0.887 | 0.01030 |
| position, fwhm, slope_right, prominence + history | GPR, untuned | 28 | 96.4% | 0.02347 | 32 | 0.01464 | 18 | 0.871 | 0.01074 |
| position, fwhm, slope_right, prominence + history | RF, untuned | 28 | 96.4% | 0.01988 | 15 | 0.01510 | 20 | 0.863 | 0.01009 |
| position, fwhm, slope_right, prominence + history | RF, tuned | 28 | 96.4% | 0.01842 | 7 | 0.01408 | 14 | 0.881 | 0.00977 |
| all but fwhm, prominence + history | LR | 49 | 42.3% | 0.02427 | 33 | 0.01845 | 38 | 0.796 | 0.01583 |
| all but fwhm, prominence + history | GPR, untuned | 49 | 42.3% | 0.01963 | 14 | 0.01290 | 6 | 0.900 | 0.01003 |
| all but fwhm, prominence + history | RF, untuned | 49 | 42.3% | 0.01857 | 9 | 0.01335 | 11 | 0.893 | 0.00899 |
| all but fwhm, prominence + history | RF, tuned | 49 | 42.3% | 0.01817 | 3 | 0.01305 | 8 | 0.898 | 0.00904 |

ENG-28: every set on the history rows (cycles with at least 20 earlier clean cycles), ENG-08 cell-to-fold map, columns in the ENG-14 combined-pool order restricted to the set, seed 0; batch 3 was scored once per cell, after every dev-CV cell of that model was finished, and no choice in G is made from it. Discharge needed = the SoC span the set needs (ENG-17), as in Tables XIV and XV.

Ranks: 1 = lowest RMSE of the 44 cells. Kendall τ between the dev-CV and batch-3 RMSE of the 44 cells = 0.48; within each model (11 sets): LR −0.02; untuned GPR −0.09; untuned RF 0.35; tuned RF 0.42.

Sources: `results/eng28_feature_sets_models.json`
