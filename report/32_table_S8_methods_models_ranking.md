**Table S8. Every feature set × model combination, ranked by dev-CV RMSE**

| Rank | Feature set | Subsection: method | Model | Features | Columns | Discharge needed | Dev-CV RMSE ± fold sd | Dev-CV R² | vs interim RF | vs all nine, same model | Rank in model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **all nine + history** | **E: 20-cycle feature history** | **RF, tuned** | **9** | **63** | **96.4%** | **0.01757 ± 0.00282** | **0.882** | **−24.5%** | **−13.4%** | **1** |
| 2 | height, fwhm + history | F: grouped subset search with history, one-standard-error rule | RF, tuned | 2 | 14 | 96.4% | 0.01781 ± 0.00167 | 0.881 | −23.5% | −12.2% | 2 |
| 3 | all but fwhm, prominence + history | B, E: level discharge-budget set with its 20-cycle history | RF, tuned | 7 | 49 | 42.3% | 0.01817 ± 0.00270 | 0.874 | −22.0% | −10.5% | 3 |
| 4 | all nine + history | E: 20-cycle feature history | RF, untuned | 9 | 63 | 96.4% | 0.01817 ± 0.00288 | 0.874 | −22.0% | −10.8% | 1 |
| 5 | height, position, area + history | C, E: interim three features with their 20-cycle history | RF, tuned | 3 | 21 | 42.3% | 0.01826 ± 0.00158 | 0.875 | −21.6% | −10.0% | 4 |
| 6 | all nine: level + rm20 + drift | E: history reduced to three components | RF, tuned | 9 | 27 | 96.4% | 0.01828 ± 0.00272 | 0.873 | −21.5% | −9.9% | 5 |
| 7 | position, fwhm, slope_right, prominence + history | D, E: level one-standard-error pick with its 20-cycle history | RF, tuned | 4 | 28 | 96.4% | 0.01842 ± 0.00362 | 0.869 | −20.9% | −9.2% | 6 |
| 8 | height, fwhm + history | F: grouped subset search with history, one-standard-error rule | RF, untuned | 2 | 14 | 96.4% | 0.01846 ± 0.00189 | 0.872 | −20.7% | −9.4% | 2 |
| 9 | all but fwhm, prominence + history | B, E: level discharge-budget set with its 20-cycle history | RF, untuned | 7 | 49 | 42.3% | 0.01857 ± 0.00278 | 0.869 | −20.3% | −8.9% | 3 |
| 10 | height, position + history | F: best history set within the 13.6% apex window (discharge budget) | RF, tuned | 2 | 14 | 13.6% | 0.01879 ± 0.00174 | 0.868 | −19.3% | −7.4% | 7 |
| 11 | all nine: level + rm20 + drift | E: history reduced to three components | RF, untuned | 9 | 27 | 96.4% | 0.01883 ± 0.00283 | 0.865 | −19.1% | −7.6% | 4 |
| 12 | height, position, area + history | C, E: interim three features with their 20-cycle history | RF, untuned | 3 | 21 | 42.3% | 0.01890 ± 0.00169 | 0.866 | −18.9% | −7.3% | 5 |
| 13 | height, position + history | F: best history set within the 13.6% apex window (discharge budget) | RF, untuned | 2 | 14 | 13.6% | 0.01934 ± 0.00154 | 0.860 | −17.0% | −5.1% | 6 |
| 14 | all but fwhm, prominence + history | B, E: level discharge-budget set with its 20-cycle history | GPR, untuned | 7 | 49 | 42.3% | 0.01963 ± 0.00113 | 0.856 | −15.7% | −6.0% | 1 |
| 15 | position, fwhm, slope_right, prominence + history | D, E: level one-standard-error pick with its 20-cycle history | RF, untuned | 4 | 28 | 96.4% | 0.01988 ± 0.00496 | 0.844 | −14.6% | −2.4% | 7 |
| 16 | all nine features | A, C: expanded nine-feature pool (inputs of the C reference model) | RF, tuned | 9 | 9 | 96.4% | 0.02029 ± 0.00144 | 0.846 | −12.9% | 0 (ref.) | 8 |
| 17 | all nine: level + rm20 + drift | E: history reduced to three components | GPR, untuned | 9 | 27 | 96.4% | 0.02034 ± 0.00184 | 0.845 | −12.7% | −2.6% | 2 |
| 18 | all nine features | A, C: expanded nine-feature pool (inputs of the C reference model) | RF, untuned | 9 | 9 | 96.4% | 0.02038 ± 0.00158 | 0.844 | −12.5% | 0 (ref.) | 8 |
| 19 | height, position, area + history | C, E: interim three features with their 20-cycle history | GPR, untuned | 3 | 21 | 42.3% | 0.02072 ± 0.00095 | 0.840 | −11.0% | −0.7% | 3 |
| 20 | all nine features | A, C: expanded nine-feature pool (inputs of the C reference model) | GPR, untuned | 9 | 9 | 96.4% | 0.02087 ± 0.00178 | 0.836 | −10.4% | 0 (ref.) | 4 |
| 21 | position, fwhm, slope_right, prominence | D: exhaustive subset search, one-standard-error rule | RF, tuned | 4 | 4 | 96.4% | 0.02089 ± 0.00103 | 0.837 | −10.3% | +2.9% | 9 |
| 22 | position, fwhm, slope_right, prominence | D: exhaustive subset search, one-standard-error rule | RF, untuned | 4 | 4 | 96.4% | 0.02103 ± 0.00114 | 0.835 | −9.7% | +3.2% | 9 |
| 23 | all but fwhm, prominence | B, F: best level set without fwhm and prominence (discharge budget) | RF, tuned | 7 | 7 | 42.3% | 0.02147 ± 0.00141 | 0.828 | −7.8% | +5.8% | 10 |
| 24 | position, fwhm, slope_right, prominence | D: exhaustive subset search, one-standard-error rule | GPR, untuned | 4 | 4 | 96.4% | 0.02166 ± 0.00134 | 0.825 | −7.0% | +3.8% | 5 |
| 25 | all but fwhm, prominence | B, F: best level set without fwhm and prominence (discharge budget) | RF, untuned | 7 | 7 | 42.3% | 0.02170 ± 0.00148 | 0.824 | −6.8% | +6.5% | 10 |
| 26 | height, position + history | F: best history set within the 13.6% apex window (discharge budget) | GPR, untuned | 2 | 14 | 13.6% | 0.02186 ± 0.00098 | 0.822 | −6.1% | +4.7% | 6 |
| 27 | all nine + history | E: 20-cycle feature history | GPR, untuned | 9 | 63 | 96.4% | 0.02191 ± 0.00137 | 0.821 | −5.9% | +5.0% | 7 |
| 28 | all but fwhm, prominence | B, F: best level set without fwhm and prominence (discharge budget) | GPR, untuned | 7 | 7 | 42.3% | 0.02194 ± 0.00198 | 0.819 | −5.8% | +5.1% | 8 |
| 29 | height, position, area | C: interim three-feature model (semester 1) | GPR, untuned | 3 | 3 | 42.3% | 0.02220 ± 0.00106 | 0.816 | −4.7% | +6.3% | 9 |
| 30 | height, position, area | C: interim three-feature model (semester 1) | RF, tuned | 3 | 3 | 42.3% | 0.02307 ± 0.00099 | 0.802 | −0.9% | +13.7% | 11 |
| 31 | height, position, area | C: interim three-feature model (semester 1) | RF, untuned | 3 | 3 | 42.3% | 0.02329 ± 0.00107 | 0.798 | 0 (ref.) | +14.3% | 11 |
| 32 | position, fwhm, slope_right, prominence + history | D, E: level one-standard-error pick with its 20-cycle history | GPR, untuned | 4 | 28 | 96.4% | 0.02347 ± 0.00107 | 0.795 | +0.8% | +12.5% | 10 |
| 33 | all but fwhm, prominence + history | B, E: level discharge-budget set with its 20-cycle history | LR | 7 | 49 | 42.3% | 0.02427 ± 0.00419 | 0.774 | +4.2% | −11.8% | 1 |
| 34 | height, fwhm + history | F: grouped subset search with history, one-standard-error rule | GPR, untuned | 2 | 14 | 96.4% | 0.02499 ± 0.00117 | 0.767 | +7.3% | +19.7% | 11 |
| 35 | all nine + history | E: 20-cycle feature history | LR | 9 | 63 | 96.4% | 0.02642 ± 0.00617 | 0.725 | +13.4% | −4.0% | 2 |
| 36 | all nine: level + rm20 + drift | E: history reduced to three components | LR | 9 | 27 | 96.4% | 0.02651 ± 0.00617 | 0.723 | +13.8% | −3.6% | 3 |
| 37 | all but fwhm, prominence | B, F: best level set without fwhm and prominence (discharge budget) | LR | 7 | 7 | 42.3% | 0.02709 ± 0.00402 | 0.720 | +16.3% | −1.5% | 4 |
| 38 | all nine features | A, C: expanded nine-feature pool (inputs of the C reference model) | LR | 9 | 9 | 96.4% | 0.02751 ± 0.00420 | 0.711 | +18.1% | 0 (ref.) | 5 |
| 39 | height, position, area | C: interim three-feature model (semester 1) | LR | 3 | 3 | 42.3% | 0.02877 ± 0.00428 | 0.685 | +23.5% | +4.6% | 6 |
| 40 | height, position, area + history | C, E: interim three features with their 20-cycle history | LR | 3 | 21 | 42.3% | 0.02886 ± 0.00634 | 0.676 | +23.9% | +4.9% | 7 |
| 41 | height, fwhm + history | F: grouped subset search with history, one-standard-error rule | LR | 2 | 14 | 96.4% | 0.02903 ± 0.00573 | 0.674 | +24.7% | +5.5% | 8 |
| 42 | position, fwhm, slope_right, prominence + history | D, E: level one-standard-error pick with its 20-cycle history | LR | 4 | 28 | 96.4% | 0.03095 ± 0.00750 | 0.623 | +32.9% | +12.5% | 9 |
| 43 | height, position + history | F: best history set within the 13.6% apex window (discharge budget) | LR | 2 | 14 | 13.6% | 0.03103 ± 0.00711 | 0.624 | +33.3% | +12.8% | 10 |
| 44 | position, fwhm, slope_right, prominence | D: exhaustive subset search, one-standard-error rule | LR | 4 | 4 | 96.4% | 0.03139 ± 0.00351 | 0.629 | +34.8% | +14.1% | 11 |

Dev-CV = pooled RMSE of SoH averaged over the five development folds (± population sd over the folds), in SoH units (fraction of nominal capacity, p.u.); R² likewise. All 44 cells use the same rows: the 57,452 development cycles (of 59,131) at least 20 cycles after their cell's first clean cycle (ENG-14 history rows, ENG-08 cell-to-fold map), so level and history sets are compared like for like.

Models: LR = linear regression on standardised inputs; GPR, untuned = constant × isotropic RBF + white-noise kernel on standardised inputs, hyperparameters by marginal likelihood (no kernel search, no ARD), trained on a 2000-row per-cell stratified subsample per fold; RF, untuned = 300 trees, scikit-learn defaults otherwise; RF, tuned = 300 trees, the fixed ENG-09 configuration (max_depth None, max_features 0.5, min_samples_leaf 1). Seed 0.

Features = ICA features measured; Columns = model inputs (a history set has 7 columns per feature, the component set 3 per feature). Discharge needed = median over cycles of the per-cycle union of the features' SoC windows (ENG-17); a history set needs that window in each of the 20 previous cycles as well.

vs interim RF = change in dev-CV RMSE against the untuned RF on the interim three features (0.02329), which mixes the effect of the model with that of the features; vs all nine, same model = change against all nine level features under the same model, on the same rows (the Δ of Tables XIV and XV), the effect of the features alone. Negative is better; 0 (ref.) marks the reference cells. Rank in model = rank of the set among the 11 under that model.

Selection: the D, B/F, E-component and F sets (and the D and B/F sets with history) were chosen by untuned-RF dev-CV searches on these five folds, and the tuned-RF configuration by nested CV on the same development cells (ENG-09), so their RF scores are mildly optimistic relative to LR and GPR.

Sources: `results/eng28_feature_sets_models.json`, `results/eng14_full.json`
