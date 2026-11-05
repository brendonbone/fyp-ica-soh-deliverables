**Table II. The Severson LFP dataset by batch: cells, cycles, charge policies, mean cycle life and role in the study**

| Batch | Role | Cells | Recorded cycles | Cycles kept after cleaning | Charge policies | Mean cycle life |
|---|---|---|---|---|---|---|
| b1 | development | 41 | 37,734 | 37,650 | 22 | 921 |
| b2 | development | 43 | 21,527 | 21,481 | 41 | 473 |
| b3 | held-out test | 40 | 41,240 | 41,200 | 8 | 1,032 |
| **Total** |  | **124** | **100,501** | **100,331** | **69** | **802** |

Cells after the loader's exclusions: 5 incomplete batch-1 cells and 6 noisy batch-3 cells dropped, and the 5 batch-2 cells that continue batch-1 cells stitched onto them.

Cleaning keeps a cycle when its peak extraction succeeded (status ok) and 0.5 < SoH ≤ 1.05. Development batches are used for ranking, tuning and cross-validation; the held-out batch is scored once. The total counts each charge policy once, so it is less than the sum when batches share a policy.

Mean cycle life: the dataset's cycle life of each cell (after the continuation merges), averaged over the cells.

Sources: `results/features_expanded.parquet`
