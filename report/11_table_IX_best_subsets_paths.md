**Table IX. Best subset of each size against the backward-elimination path and the Pearson and PI forward paths (dev-CV RMSE, untuned RF, nine level features)**

| k | Best subset of size k | Dev-CV | BE: dropped | BE: dev-CV | Pearson: added | Pearson: dev-CV | PI (tuned RF): added | PI (tuned RF): dev-CV |
|---|---|---|---|---|---|---|---|---|
| 1 | area25 | 0.02922 | −fwhm → prominence | 0.02999 (+2.6%) | +area25 | **0.02922** | +height | 0.02950 (+1.0%) |
| 2 | height, fwhm | 0.02359 | −position → fwhm, prominence | 0.02364 (+0.2%) | +area35 | 0.02493 (+5.7%) | +area25 | 0.02545 (+7.9%) |
| 3 | height, position, fwhm | 0.02219 | −slope_right → position, fwhm, prominence | 0.02233 (+0.7%) | +area | 0.02329 (+5.0%) | +area | 0.02366 (+6.6%) |
| 4 | position, fwhm, slope_right, prominence | 0.02076 | −area | **0.02076** | +position | 0.02203 (+6.1%) | +fwhm | 0.02212 (+6.6%) |
| 5 | position, area, fwhm, slope_right, prominence | 0.02039 | −area25 | **0.02039** | +height | 0.02180 (+6.9%) | +position | 0.02078 (+1.9%) |
| 6 | position, area, area25, fwhm, slope_right, prominence | 0.02022 | −height | **0.02022** | +prominence | 0.02147 (+6.2%) | +prominence | 0.02059 (+1.8%) |
| 7 | all but area35, slope_left | 0.02015 | −area35 | **0.02015** | +slope_right | 0.02124 (+5.4%) | +area35 | 0.02045 (+1.5%) |
| 8 | all but slope_left | 0.02013 | −slope_left | **0.02013** | +slope_left | 0.02123 (+5.4%) | +slope_right | **0.02013** |
| 9 | all nine | 0.02015 | (start) | **0.02015** | +fwhm | **0.02015** | +slope_left | **0.02015** |

Dev-CV: pooled RMSE averaged over the five development folds (ENG-08 folds, 84 cells of batches 1 and 2); untuned RF, seed 0, every value read off the ENG-19 table of all 511 subsets (columns in pool order). ENG-08's Pearson k-curve (columns in ranking order) differs from the Pearson column by up to 6.8e-06.

BE starts from all nine and at each step drops the feature whose removal gives the lowest dev-CV; its column gives the feature dropped to reach size k (and the subset where it differs from the best). The Pearson and PI columns give the feature added at rank k, so the subset at k is the top k of the ranking; PI is the tuned-RF permutation-importance ranking of ENG-08, its subsets scored with the untuned RF like every other column.

Bold: the method's subset is the best subset of that size. Percentages: dev-CV above the best subset of the same size.

Row k = 9 is the search's own all-nine run (0.02015, seed 0, pool order); Table VII gives the mean over ten seeds × two column orders (0.02014, seed sd 0.00002) and the reference model, the tuned RF (0.02001).

The CSV also carries sequential forward selection (sfs_* columns), read off the same table.

Sources: `results/eng19_exhaustive_subsets.csv`, `results/eng08_full.json`, `results/eng08_full_tuned.json`, `results/eng20_reference_seeds.json`, `results/features_expanded.parquet`
