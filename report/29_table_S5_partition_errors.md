**Table S5. Held-out RMSE of the untuned random forest in the six train/test partitions**

| Partition | Test cells | Nine level features, all rows | Nine level features, history rows | Level + history | k (level + history) | Change with history |
|---|---|---|---|---|---|---|
| Batch 1 out | 41 | 0.02831 | 0.02863 | 0.02628 | 48 | −8.2% |
| Batch 2 out | 43 | 0.03781 | 0.03851 | 0.03550 | 63 | −7.8% |
| Batch 3 out (study split) | 40 | 0.01602 | 0.01621 | 0.01417 | 63 | −12.6% |
| Mixed third 1 | 42 | 0.01512 | 0.01532 | 0.01282 | 63 | −16.3% |
| Mixed third 2 | 41 | 0.01664 | 0.01681 | 0.01343 | 63 | −20.1% |
| Mixed third 3 | 41 | 0.02100 | 0.02126 | 0.01820 | 63 | −14.4% |
| Mean, batch held out |  | 0.02738 | 0.02778 | 0.02532 |  | −9.5% |
| Mean, mixed thirds |  | 0.01759 | 0.01780 | 0.01482 |  | −16.9% |

Held-out RMSE pooled over the partition's test rows; untuned RF (300 trees, seed 0), features ranked by Pearson |r| on the partition's development cells and k chosen by their own five-fold CV (ENG-15). Batch 3 out is the study split (imported from ENG-08 / ENG-14, Pearson column order; Table XVI scores the same models in the pool order, which differs by the random-forest column-order effect only). Mixed thirds: a seeded three-fold split of all 124 cells stratified on batch and cycle-life quartile.

History rows drop each cell's first 20 cycles. Level + history = the ENG-14 combined pool (the nine level features and their 54 history columns, Pearson order), with the Pearson-selected k: all 63 except for batch 1 out (k = 48); the held-out score of all 63 is not stored there. Change with history = (level + history − nine level features) / nine level features, both on the history rows; negative = history lowers the error.

Untuned models only: the tuned configuration (ENG-09) was chosen on batches 1 + 2, so it would leak in the partitions that test on them. Mean rows: plain means of the three partitions (change: mean of the three changes).

Sources: `results/eng15_cells.json`, `results/eng15_split_robustness.json`
