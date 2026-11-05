**Table XIV. Dev-CV RMSE of three feature choices (level features only) under four models, and the change against the same model on all nine features (the reference model)**

| Configuration | Selected features | Discharge budget (SoC span) | LR dev-CV | LR Δ | GPR dev-CV | GPR Δ | RF dev-CV | RF Δ | Tuned RF dev-CV | Tuned RF Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| All nine features (reference model, C) | height, position, area, area35, area25, fwhm, slope_left, slope_right, prominence | 96.4% | 0.02751 | reference | 0.02087 | reference | 0.02038 | reference | 0.02029 | reference |
| Interim model features | height, position, area | 42.3% | 0.02877 | +4.6% | 0.02220 | +6.3% | 0.02329 | +14.3% | 0.02307 | +13.7% |
| Best subset, one-SE rule (D) | position, fwhm, slope_right, prominence | 96.4% | 0.03139 | +14.1% | 0.02166 | +3.8% | 0.02103 | +3.2% | 0.02089 | +2.9% |
| Discharge budget, 42.3% (B, F) | height, position, area, area35, area25, slope_left, slope_right | 42.3% | 0.02709 | −1.5% | 0.02194 | +5.1% | 0.02170 | +6.5% | 0.02147 | +5.8% |

Δ = change in dev-CV RMSE against the same model trained on all nine level features (the reference model of C; the first row of Table XIV), on the same rows. Negative = lower error (better).

Dev-CV = pooled RMSE of SoH averaged over the five development folds, in SoH units (fraction of nominal capacity, p.u.). All cells use the same rows: the 57,452 development cycles (of 59,131) at least 20 cycles after their cell's first clean cycle (ENG-14 history rows, ENG-08 cell-to-fold map), so the tables with and without history compare like for like.

LR = linear regression on standardised inputs; GPR = untuned GPR (constant × isotropic RBF + white-noise kernel, hyperparameters by marginal likelihood, 2000-row per-cell training subsample per fold); RF = untuned random forest (300 trees, scikit-learn defaults); Tuned RF = the same forest with the fixed ENG-09 configuration (max_features 0.5, chosen once on the nine level features and not re-tuned per feature set). Seed 0.

Discharge budget = median over cycles of the per-cycle union of the features' SoC windows (ENG-17, Table VI).

The best subset and the discharge-budget set were chosen by untuned-RF dev-CV searches on these folds (D, F), so their RF scores are mildly optimistic relative to LR and GPR.

Sources: `results/eng28_feature_sets_models.json`, `results/eng14_full.json`
