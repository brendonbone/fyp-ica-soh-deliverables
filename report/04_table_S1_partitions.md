**Table S1. The six development/test partitions used to check the split design**

| Partition | Kind | Test cells, batch 1 | Test cells, batch 2 | Test cells, batch 3 | Test cells | Development cells |
|---|---|---|---|---|---|---|
| Batch 1 out | leave one batch out | 41 | 0 | 0 | 41 | 83 |
| Batch 2 out | leave one batch out | 0 | 43 | 0 | 43 | 81 |
| **Batch 3 out (study split)** | **leave one batch out** | **0** | **0** | **40** | **40** | **84** |
| Mixed third 1 | mixed third (3-fold) | 14 | 14 | 14 | 42 | 82 |
| Mixed third 2 | mixed third (3-fold) | 14 | 14 | 13 | 41 | 83 |
| Mixed third 3 | mixed third (3-fold) | 13 | 15 | 13 | 41 | 83 |

Bold: the study split (batches 1 and 2 develop, batch 3 tests). Mixed thirds: an outer 3-fold split of all cells, stratified on batch and cycle-life quartile (scikit-learn StratifiedKFold, shuffled, seed 0).

Sources: `results/eng15_split_robustness.json`
