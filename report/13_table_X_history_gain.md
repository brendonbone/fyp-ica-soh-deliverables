**Table X. Dev-CV RMSE of the nine level features and of the nine features with their 20-cycle history (63 columns), RF untuned and tuned**

| Model | Nine level features (9 columns) | Nine features + history (63 columns) | Change with history |
|---|---|---|---|
| RF, untuned | 0.02038 ± 0.00158 | 0.01817 ± 0.00288 | −10.8% |
| RF, tuned (reference configuration) | 0.02029 ± 0.00144 | 0.01757 ± 0.00282 | −13.4% |
| Change with tuning | −0.4% | −3.3% |  |

Dev-CV: pooled RMSE averaged over the five development folds, ± the sd over the folds (ddof 0), on the 57,452 development rows that have 20 cycles of history, so the nine-level values differ slightly from Table VII (every clean row). Change with history is relative to the nine level features, change with tuning to the untuned RF.

Tuned: the reference configuration of Table VII, fixed by the ENG-09 search (max_features 0.5, min_samples_leaf 1, no depth limit, 300 trees, seed 0); it was chosen on the nine level features and is not re-tuned for the history columns.

Sources: `results/eng24_level_subsets_history_rows.json`, `results/eng23_grouped_history_subsets.json`, `results/eng23_grouped_history_subsets_analysis.json`, `results/eng28_feature_sets_models.json`, `results/eng14_full.json`, `results/eng14_full_tuned.json`, `results/eng09_tuned_params.json`
