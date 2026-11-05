**Table XII. Best subset of each size: the nine level features alone (ENG-24) and each feature with its six 20-cycle history columns (ENG-23), on the same rows (dev-CV RMSE, untuned RF)**

| k | Level only: best subset | Level only: dev-CV | With history: best subset | With history: dev-CV | Gain from history | Gain over all subsets of size k: median (range) |
|---|---|---|---|---|---|---|
| 1 | area25 | 0.02958 | height | 0.02039 | 31.1% | 24.1% (10.8–42.2%) |
| 2 | height, fwhm | 0.02388 | height, fwhm | 0.01846 | 22.7% | 19.5% (11.2–39.1%) |
| 3 | height, position, fwhm | 0.02252 | height, fwhm, slope_right | 0.01806 | 19.8% | 15.3% (6.7–24.2%) |
| 4 | position, fwhm, slope_right, prominence | 0.02103 | height, fwhm, slope_left, slope_right | 0.01800 | 14.4% | 13.5% (3.6–19.6%) |
| 5 | position, area, fwhm, slope_right, prominence | 0.02065 | height, position, area, fwhm, slope_right | 0.01781 | 13.7% | 12.4% (4.7–17.5%) |
| 6 | position, area, area25, fwhm, slope_right, prominence | 0.02044 | height, position, area, fwhm, slope_right, prominence | 0.01775 | 13.2% | 11.6% (5.6–16.5%) |
| 7 | all but area35, slope_left | 0.02039 | all but area35, area25 | 0.01775 | 12.9% | 11.0% (5.9–15.4%) |
| 8 | all but slope_left | 0.02036 | all but area25 | 0.01789 | 12.2% | 10.8% (6.3–13.8%) |
| 9 | all nine | 0.02038 | all nine | 0.01817 | 10.8% | 10.8% (one subset) |

Both searches use the same 57,452 development rows (of 59,131) that have 20 cycles of history, so the level-only values are slightly higher than in Table IX (every development row). With history, a subset of k features has 7k columns (the level and its d10, d20, slope10, slope20, drift and rm20).

Gain from history = (level - history) / level for the two best subsets of that size; the last column pairs every subset of size k with itself without history (511 of the 511 pairs improved).

Sources: `results/eng24_level_subsets_history_rows_analysis.json`, `results/eng23_grouped_history_subsets_analysis.json`, `results/eng23_paired_level_history.csv`, `results/eng24_level_subsets_history_rows.json`, `results/eng23_grouped_history_subsets.json`, `results/eng14_full.json`
