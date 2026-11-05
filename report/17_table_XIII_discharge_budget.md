**Table XIII. Best subset within each discharge budget, level features only (ENG-24) and with history (ENG-23), on the same rows (dev-CV RMSE, untuned RF)**

| Discharge budget (SoC span) | Level only: best subset | Level only: dev-CV | Level only: vs full discharge | With history: best subset | With history: dev-CV | With history: vs full discharge | Gain from history |
|---|---|---|---|---|---|---|---|
| 13.6% | height, position | 0.02682 | +31.7% | height, position | 0.01934 | +9.0% | 27.9% |
| 20.0% | height, position, slope_left | 0.02458 | +20.7% | height, position† | 0.01934 | +9.0% | 21.3% |
| 20.4% | height, position, slope_right | 0.02378 | +16.8% | height, position, slope_right | 0.01905 | +7.3% | 19.9% |
| 27.0% | height, position, area25, slope_left, slope_right | 0.02267 | +11.3% | height, position, slope_left, slope_right | 0.01884 | +6.2% | 16.9% |
| 34.2% | height, position, area35, area25, slope_left, slope_right | 0.02231 | +9.5% | height, position, area35, slope_right | 0.01872 | +5.5% | 16.1% |
| 42.3% | all but fwhm, prominence | 0.02170 | +6.6% | height, position, area, area35, slope_left, slope_right | 0.01831 | +3.2% | 15.6% |
| 96.4% | all but slope_left | 0.02036 | (full) | all but area35, area25 | 0.01775 | (full) | 12.8% |

Budget = the SoC span a subset needs: the union of its features' windows in each cycle, median over the clean cycles, rounded to 0.1%. Every subset needs one of seven spans, the windows of single features (Table VI): 13.6% height, position; 20.0% slope_left; 20.4% slope_right; 27.0% area25; 34.2% area35; 42.3% area; 96.4% fwhm, prominence.

With history, a subset needs the same window in each of the previous 20 cycles as well, because its history columns are built from those cycles' features.

Each row gives the best subset among all subsets needing at most the budget; 'vs full discharge' is its dev-CV relative to the best at 96.4%.

† No subset with history that needs exactly 20.0% beats the 13.6% set (best at exactly 20.0%: height, position, slope_left, 0.01938), so the history trade-off edge has 6 steps against 7 without history.

Sources: `results/eng24_level_subsets_history_rows_windows.csv`, `results/eng23_grouped_history_subsets_windows.csv`, `results/eng24_level_subsets_history_rows.json`, `results/eng23_grouped_history_subsets.json`, `results/eng17_soc_windows_summary.csv`, `results/eng24_level_subsets_history_rows_analysis.json`, `results/eng23_grouped_history_subsets_analysis.json`
