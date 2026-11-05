**Table XV. Dev-CV RMSE of the same three feature choices with each feature's 20-cycle history, and the change against the same model on all nine features without history (the reference model)**

| Configuration | Selected features | Discharge budget (SoC span) | LR dev-CV | LR Δ | GPR dev-CV | GPR Δ | RF dev-CV | RF Δ | Tuned RF dev-CV | Tuned RF Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| Interim model features + history | height, position, area + 20-cycle history (21 columns) | 42.3% | 0.02886 | +4.9% | 0.02072 | −0.7% | 0.01890 | −7.3% | 0.01826 | −10.0% |
| Best subset + history | position, fwhm, slope_right, prominence + 20-cycle history (28 columns) | 96.4% | 0.03095 | +12.5% | 0.02347 | +12.5% | 0.01988 | −2.4% | 0.01842 | −9.2% |
| Discharge budget + history | height, position, area, area35, area25, slope_left, slope_right + 20-cycle history (49 columns) | 42.3% | 0.02427 | −11.8% | 0.01963 | −6.0% | 0.01857 | −8.9% | 0.01817 | −10.5% |

Δ = change in dev-CV RMSE against the same model on all nine level features without history (the reference row of Table XIV), on the same rows: the effect of the feature choice and the history together. Negative = lower error (better).

Each feature brings six history columns: the 20-cycle rolling mean, the drift since the cell's first clean cycle, and the slope and change over the last 20 and 10 cycles (E). A history set needs the same discharge window as its level set, in each of the previous 20 cycles as well.

Dev-CV = pooled RMSE of SoH averaged over the five development folds, in SoH units (fraction of nominal capacity, p.u.). All cells use the same rows: the 57,452 development cycles (of 59,131) at least 20 cycles after their cell's first clean cycle (ENG-14 history rows, ENG-08 cell-to-fold map), so the tables with and without history compare like for like.

LR = linear regression on standardised inputs; GPR = untuned GPR (constant × isotropic RBF + white-noise kernel, hyperparameters by marginal likelihood, 2000-row per-cell training subsample per fold); RF = untuned random forest (300 trees, scikit-learn defaults); Tuned RF = the same forest with the fixed ENG-09 configuration (max_features 0.5, chosen once on the nine level features and not re-tuned per feature set). Seed 0.

Sources: `results/eng28_feature_sets_models.json`, `results/eng14_full.json`
