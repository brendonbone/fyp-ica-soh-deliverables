**Table D1. Direct comparison between the best subset with and without history: the best subset of each size, level features only and each feature with its six history columns, on the history rows (untuned RF; development-CV and batch-3 RMSE)**

| k | Best level subset (no history) | Dev-CV, no history | Batch 3, no history | Best subset with history | Dev-CV, with history | Batch 3, with history |
|---|---|---|---|---|---|---|
| 1 | area25 | 0.02958 | 0.02280 | height | 0.02039 | 0.01323 |
| 2 | height + fwhm | 0.02388 | 0.01672 | height + fwhm | 0.01846 | 0.01697 |
| 3 | height + position + fwhm | 0.02252 | 0.01640 | height + fwhm + slope_right | 0.01806 | 0.01672 |
| 4 | position + fwhm + slope_right + prominence | 0.02103 | 0.01779 | height + fwhm + slope_left + slope_right | 0.01800 | 0.01516 |
| 5 | position + area + fwhm + slope_right + prominence | 0.02065 | 0.01657 | height + position + area + fwhm + slope_right | 0.01781 | 0.01603 |
| 6 | position + area + area25 + fwhm + slope_right + prominence | 0.02044 | 0.01651 | height + position + area + fwhm + slope_right + prominence | 0.01775 | 0.01583 |
| 7 | height + position + area + area25 + fwhm + slope_right + prominence | 0.02039 | 0.01652 | height + position + area + fwhm + slope_left + slope_right + prominence | 0.01775 | 0.01509 |
| 8 | height + position + area + area35 + area25 + fwhm + slope_right + prominence | 0.02036 | 0.01652 | height + position + area + area35 + fwhm + slope_left + slope_right + prominence | 0.01789 | 0.01408 |
| 9 | all nine | 0.02038 | 0.01620 | all nine (63 columns) | 0.01817 | 0.01413 |

Both searches use the same rows: the 57,452 development and 40,400 batch-3 cycles with at least 20 earlier clean cycles, so the level-only values differ slightly from Table IX (all 59,131 development rows).

Sources: `results/eng24_level_subsets_history_rows_analysis.json`, `results/eng23_grouped_history_subsets_analysis.json`, `results/eng14_full.json`
