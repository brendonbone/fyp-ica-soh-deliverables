**Table XI. The seven history components: grouped permutation importance (PI) in the tuned 63-column RF, and dev-CV and batch-3 RMSE of each component alone and of all 63 columns without it**

| Component | Grouped PI (± fold sd) | Sum of one-column PIs | Alone: dev-CV | Alone vs all 63 | Alone: batch 3 | All 63 without it: dev-CV | Cost of dropping | All 63 without it: batch 3 | Batch-3 cost of dropping |
|---|---|---|---|---|---|---|---|---|---|
| rolling mean (rm20) | 0.04711 ± 0.00227 | 0.02884 | 0.01996 | +9.9% | 0.01590 | 0.01921 | +5.7% | 0.01202 | −14.9% |
| drift | 0.00396 ± 0.00136 | 0.00181 | 0.02413 | +32.8% | 0.01644 | 0.01848 | +1.7% | 0.01520 | +7.6% |
| level (raw) | 0.00121 ± 0.00063 | 0.00059 | 0.02038 | +12.2% | 0.01620 | 0.01823 | +0.3% | 0.01422 | +0.6% |
| slope, 20 cycles (slope20) | 0.00055 ± 0.00021 | 0.00037 | 0.03597 | +98.0% | 0.02976 | 0.01833 | +0.9% | 0.01406 | −0.5% |
| change, 20 cycles (d20) | 0.00036 ± 0.00012 | 0.00024 | 0.03628 | +99.6% | 0.02782 | 0.01825 | +0.5% | 0.01434 | +1.4% |
| change, 10 cycles (d10) | 0.00021 ± 0.00008 | 0.00013 | 0.03955 | +117.7% | 0.02684 | 0.01820 | +0.2% | 0.01425 | +0.8% |
| slope, 10 cycles (slope10) | 0.00018 ± 0.00010 | 0.00013 | 0.03975 | +118.8% | 0.02883 | 0.01823 | +0.3% | 0.01434 | +1.5% |
| All seven together (63 columns): dev-CV 0.01817, batch 3 0.01413 |  | 0.03210 |  |  |  |  |  |  |  |

Grouped PI: rise in validation RMSE when the component's nine columns (one per feature) are shuffled together; tuned RF on all 63 columns, ten repeats, mean over the five development folds ± sd over folds (ddof 0). Sum of one-column PIs: the same nine columns shuffled one at a time; the last row sums all 63.

Alone / all 63 without it: dev-CV of the untuned RF on the component's nine columns / on the other 54; percentages relative to all 63 together (dev-CV 0.01817, the untuned 63-column model of Table X). PI and RMSE in SoH units (fraction of nominal capacity).

Batch 3: the same untuned forests refitted on all development rows and scored once on the batch-3 history rows; the batch-3 cost of dropping is relative to all 63 on batch 3. Nothing was chosen on these scores.

Sources: `results/eng21_grouped_permutation.json`, `results/eng22_history_components.json`, `results/eng22_history_components_analysis.json`
