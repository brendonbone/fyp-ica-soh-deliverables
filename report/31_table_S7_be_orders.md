**Table S7. Backward elimination in each partition, without and with the 20-cycle history**

| Features | Partition | Elimination order (first dropped → last standing) | Dev-CV, all features | Dev-CV, five features | Change | Five-feature set |
|---|---|---|---|---|---|---|
| Nine level features | Batch 1 out | height, area25, area35, prominence, fwhm, slope_left, position, slope_right → area | 0.01306 | 0.01328 | +1.7% | position, area, fwhm, slope_left, slope_right |
| Nine level features | Batch 2 out | position, height, area25, prominence, area35, slope_left, fwhm, slope_right → area | 0.01491 | 0.01500 | +0.6% | area, area35, fwhm, slope_left, slope_right |
| Nine level features | Batch 3 out (study split) | slope_left, area35, height, area25, area, slope_right, position, fwhm → prominence | 0.02015 | 0.02039 | +1.2% | position, area, fwhm, slope_right, prominence |
| Nine level features | Mixed third 1 | height, area25, area, area35, slope_left, position, slope_right, fwhm → prominence | 0.01836 | 0.01844 | +0.5% | position, fwhm, slope_left, slope_right, prominence |
| Nine level features | Mixed third 2 | height, area35, area25, area, slope_left, position, slope_right, fwhm → prominence | 0.01815 | 0.01827 | +0.7% | position, fwhm, slope_left, slope_right, prominence |
| Nine level features | Mixed third 3 | height, area35, area25, prominence, slope_left, fwhm, position, slope_right → area | 0.01656 | 0.01665 | +0.6% | position, area, fwhm, slope_left, slope_right |
| With 20-cycle history | Batch 1 out | area25, area35, height, prominence, fwhm, slope_left, position, area → slope_right | 0.01233 | 0.01186 | −3.8% | position, area, fwhm, slope_left, slope_right |
| With 20-cycle history | Batch 2 out | area25, position, area35, height, slope_left, prominence, area, fwhm → slope_right | 0.01201 | 0.01089 | −9.3% | area, fwhm, slope_left, slope_right, prominence |
| With 20-cycle history | Batch 3 out (study split) | area25, area35, slope_left, prominence, position, area, slope_right, fwhm → height | 0.01817 | 0.01781 | −2.0% | height, position, area, fwhm, slope_right |
| With 20-cycle history | Mixed third 1 | area25, area35, prominence, slope_left, area, position, slope_right, fwhm → height | 0.01559 | 0.01517 | −2.7% | height, position, area, fwhm, slope_right |
| With 20-cycle history | Mixed third 2 | area25, area35, slope_left, height, prominence, position, fwhm, slope_right → area | 0.01719 | 0.01667 | −3.0% | position, area, fwhm, slope_right, prominence |
| With 20-cycle history | Mixed third 3 | area, area35, height, slope_left, prominence, fwhm, position, slope_right → area25 | 0.01402 | 0.01415 | +1.0% | position, area25, fwhm, slope_right, prominence |

Dev-CV here is each partition's own five-fold CV on its development cells (untuned RF); the partition's test cells were never scored. Partitions differ in absolute RMSE because their development sets differ.

All features (nine level features, or the 63-column combined pool): batch 3 out starts from the study-split value of ENG-19 (0.02015, pool order, as in Table IX) and ENG-23 (0.01817, combined-pool order, as in Table XII); the other partitions start from the stored ENG-15 k-curve value (Pearson column order), the only all-features value stored for them. The two orders differ by the random-forest column-order effect (5.5e-6 and 4.3e-6 on the study split; ENG-25 and Fig. S1 use the Pearson-order value for batch 3 out too).

The reduced sets are direct backward-elimination runs (ENG-25; level features in pool order, history sets in the ENG-14 combined-pool order), except batch 3 out with history, which is read off the ENG-23 table.

With history each feature is removed together with its six history columns. Untuned only, as in ENG-15.

Sources: `results/eng25_partition_robustness.json`, `results/eng19b_analysis.json`, `results/eng23_grouped_history_subsets_analysis.json`
