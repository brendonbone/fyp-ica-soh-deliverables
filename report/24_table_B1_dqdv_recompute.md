**Table B1. The dQ/dV recomputed from the raw voltage and capacity against the published curve, and the second LFP peak**

| Group | Measure | Value |
|---|---|---|
| Recomputed vs published dQ/dV (LFP, 4C) | Cycles compared (every status-ok cycle) | 100,459 |
|  | Per-cycle Pearson r: median | 0.9969 |
|  | Per-cycle Pearson r: 5th percentile | 0.9508 |
|  | Per-cycle Pearson r: 1st percentile | 0.8787 |
|  | Per-cycle Pearson r: minimum | 0.1156 |
|  | Cycles with r > 0.95 | 95.11% |
| Second peak in 2.7–3.4 V (LFP, 4C) | Any second local maximum (prominence ≥ 0.02 Ah/V) | 36.26% |
|  | Prominence ≥ 0.1 Ah/V | 1.17% |
|  | Prominence ≥ 0.3 Ah/V (the standard of the main peak) | 0.68% |
|  | Prominence ≥ 0.3 Ah/V on the recomputed curve | 0.04% |
|  | Cells with at least one cycle ≥ 0.3 Ah/V | 26 of 124 |
|  | Cells with at least one cycle ≥ 0.1 Ah/V | 35 of 124 |
|  | Position of the second peak, cycles ≥ 0.1 Ah/V (median, IQR) | 3.09 V (3.08–3.11) |
| Largest maximum below 2.7 V (LFP, 4C; not searched) | Any local maximum (prominence ≥ 0.02 Ah/V) | 99.99% |
|  | Position (median, IQR) | 2.38 V (2.21–2.43) |
|  | At 2.35–2.55 V (a bump separate from the knee upturn below 2.35 V) | 59.46% |
|  | Prominence ≥ 0.3 Ah/V | 2.52% (46 cells) |

Per-cycle Pearson r between the published dQ/dV and the one recomputed from the raw V and Q of the same cycle, over every cycle whose extraction succeeded (status ok). These 100,459 cycles are the 100,331 clean cycles plus 128 that cleaning removes later (SoH label missing or out of range); 42 of the 100,501 recorded cycles failed extraction (170 removed in all).

Second peak: the largest local maximum other than the main peak in the 2.7–3.4 V staging region of the published (smoothed) curve, the window ENG-02 searched. 0.3 Ah/V is the prominence the pipeline requires of the main peak.

Below 2.7 V ENG-02 recorded the largest local maximum separately and did not count it: the upturn at the discharge knee below 2.35 V (40% of cycles) or a small bump at 2.35–2.55 V (59%), seen in both curves in Fig. B1. Counted as a second peak, a maximum would clear the main-peak standard in 3.2% of cycles, still too few to use a second peak as a feature at 4C.

The original table's comparison with NASA cells at about 1C (ENG-03) is omitted; ENG-03 is not part of this repository.

Sources: `results/eng02_second_peak.json`, `results/features_expanded.parquet`
