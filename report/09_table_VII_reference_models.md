**Table VII. Dev-CV RMSE and R² of the interim model and of five regressors on all nine features**

| Model | Features | Dev-CV RMSE (± fold sd) | Seed sd | Dev-CV R² | % vs interim |
|---|---|---|---|---|---|
| Interim model: RF, untuned | height, position, area | 0.02302 ± 0.00106 | – | 0.801 | – |
| Linear regression | all nine | 0.02729 ± 0.00408 | – | 0.714 | +18.5% |
| GPR, untuned (RBF) | all nine | 0.02046 ± 0.00149 | – | 0.842 | −11.1% |
| GPR, tuned (Matérn ν = 1.5) | all nine | 0.02028 ± 0.00153 | – | 0.844 | −11.9% |
| RF, untuned | all nine | 0.02014 ± 0.00158 | 0.00002 | 0.847 | −12.5% |
| **RF, tuned (reference)** | **all nine** | **0.02001 ± 0.00142** | **0.00002** | **0.849** | **−13.1%** |

Dev-CV: pooled RMSE on the validation cells of each of the five development folds (ENG-08 folds, all clean rows of batches 1 and 2), averaged over the folds; ± is the sd over the five folds (ddof 0). R² is averaged the same way. % vs interim: change in dev-CV RMSE against the interim model (negative = lower error).

RF rows: mean over ten forest seeds (0-9) × two column orders (pool and Pearson) = 20 models each (ENG-20); the fold sd is the mean of the 20 models' fold sds, and the seed sd is the sd of their dev-CV RMSEs (ddof 1). Other rows are single runs (seed 0). Seed 0 alone: tuned 0.02001 (Pearson order, ENG-08) and 0.01999 (pool order); untuned 0.02016 (Pearson order, ENG-08) and 0.02015 (pool order, ENG-06 and ENG-19, the all-nine value of the subset search in Section V-D).

Interim model: the untuned 300-tree RF on height, position and area (seed 0, pool order; ENG-00, identical to the ENG-19 subset).

Linear regression: standardised inputs, ordinary least squares. Ridge (RidgeCV) gives the same RMSE to seven decimals (0.0272914); elastic net is worse (0.02803).

GPR: constant × isotropic kernel + white noise on standardised inputs, hyperparameters by marginal likelihood, trained on a 2000-row per-cell stratified subsample in each fold (both GPR rows). Tuned GPR: Matérn ν = 1.5 kernel, the ENG-09 fixed configuration (ENG-08 tuned grid, k = 9). The GPR does not depend on column order (< 10⁻⁹).

Tuned RF: the ENG-09 fixed configuration (max_features 0.5, max_depth None, min_samples_leaf 1, 300 trees). Nested CV of the whole tuning procedure (ENG-09): RF 0.0201 (R² 0.848), GPR 0.0204 (R² 0.843).

Sources: `results/eng00_baseline.json`, `results/eng06_simple_baselines.json`, `results/eng07_gpr.json`, `results/eng08_full_tuned.json`, `results/eng20_reference_seeds.json`, `results/eng09_tuning.json`
