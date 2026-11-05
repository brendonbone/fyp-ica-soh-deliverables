**Table S4. Training without the 16 cells whose published dQ/dV is noisy (ENG-27)**

| Model (tuned RF) | Columns | Training cells (rows) | Dev-CV on the clean cells: all / without the 16 | Dev-CV change | Seeds with lower dev-CV | Batch-3 RMSE, trained on all | Batch-3 RMSE, without the 16 | Batch-3 change | Seeds worse on batch 3 |
|---|---|---|---|---|---|---|---|---|---|
| All nine level features (the reference) | 9 | 84 (59,131) / 68 (49,189) | 0.01908 / 0.01918 | +0.5% | 0 of 5 | 0.01565 ± 0.00003 | 0.01588 ± 0.00003 | +1.5% | 5 of 5 |
| Two features with their history: height + position | 14 | 84 (57,452) / 68 (47,830) | 0.01631 / 0.01624 | −0.4% | 5 of 5 | 0.01165 ± 0.00004 | 0.01172 ± 0.00003 | +0.5% | 5 of 5 |
| All 63 columns (level + history) | 63 | 84 (57,452) / 68 (47,830) | 0.01524 / 0.01527 | +0.2% | 0 of 5 | 0.01331 ± 0.00007 | 0.01348 ± 0.00015 | +1.3% | 4 of 5 |

Noisy cell: the median change of `height` between consecutive successful cycles exceeds 5% of the cell's median height (no SoH label used). 16 cells qualify, b1c38–b1c45 and b2c40–b2c47 (10.1–25.9%); the largest value among the other 108 cells is 4.7% (median 0.9%). All 16 are development cells (9,942 of the 59,131 development rows); batch 3 is the same 40 cells either way.

Tuned RF (ENG-09 configuration), forest seeds 0–4, mean ± sd over the seeds. Dev-CV: the ENG-08 cell-to-fold map, scored on the 68 clean validation cells only, so both training sets are compared on the same cells. Rows: ENG-08 rows for the nine level features, history rows for the others. Batch 3 decides nothing here.

The noisy cells inflate the published dev-CV: on their own they score 0.02345 / 0.02618 / 0.02501, so the dev-CV over all cells (0.02000 / 0.01877 / 0.01759) is 5–15% above the clean-cell value.

Sources: `results/eng27_noisy_cells.json`
