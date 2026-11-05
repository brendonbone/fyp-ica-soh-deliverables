**Table VI. SoC window of the discharge that each feature needs (median over all clean cycles)**

| Feature(s) | SoC window, start → end (%) | Span (% of discharge) | Voltage width (mV) | Minutes at 4 A |
|---|---|---|---|---|
| height, position | 84.5 → 71.1 | 13.6 | 24 | 2.1 |
| slope_left | 84.5 → 64.4 | 20.0 | 38 | 3.1 |
| slope_right | 91.5 → 71.1 | 20.4 | 41 | 3.2 |
| area25 | 91.5 → 64.4 | 27.0 | 54 | 4.2 |
| area35 | 94.0 → 59.6 | 34.2 | 75 | 5.4 |
| area | 95.8 → 53.4 | 42.3 | 105 | 6.7 |
| fwhm, prominence | 99.7 → 3.2 | 96.4 | 1,399 | 15.1 |
| fwhm, crossings only (variant) | 92.3 → 38.1 | 54.2 | 126 | 8.5 |

Medians over all 100,331 clean cycles of the 124 cells (ENG-17). A window is the stretch of the discharge dQ/dV curve that the feature's computation touches, always including the apex detection window (apex outwards until |dQ/dV| has fallen by 0.3 Ah/V, padded by 1 grid point per side); SoC(v) = 1 − Q(v)/Q_total, so a window reads start → end.

Rows are merged where the features need the identical window in every life stage, and ordered by span. Start, end and span are separate medians over cycles, so start − end differs from the span by up to 0.2 points.

Minutes at 4 A = median charge passed in the window / 4 A; at the dataset's 4C discharge (4.4 A) every time is 9% shorter.

Last row, 'fwhm, crossings only (variant)': fwhm's window between its two half-prominence crossings only (ENG-17 basis 'value support'), roughly what a fixed-baseline fwhm would need; no model uses it.

Sources: `results/eng17_soc_windows_summary.csv`, `results/eng17_feature_soc_requirements.json`, `results/features_soc_windows.parquet`
