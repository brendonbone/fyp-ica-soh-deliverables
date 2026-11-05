# ENG-15 — split-design robustness (auto-generated summary)

*120 of 120 grid cells finished. Held-out RMSE is pooled over the partition's test rows; untuned models; `lobo_b3` is the study's existing protocol (imported).*

## Partitions

| partition | kind | dev cells | test cells | test cells by original batch |
|---|---|---|---|---|
| lobo_b1 | lobo | 83 | 41 | {'b1': 41} |
| lobo_b2 | lobo | 81 | 43 | {'b2': 43} |
| lobo_b3 | lobo | 84 | 40 | {'b3': 40} |
| cv3_f0 | cv3 | 82 | 42 | {'b1': 14, 'b2': 14, 'b3': 14} |
| cv3_f1 | cv3 | 83 | 41 | {'b1': 14, 'b2': 14, 'b3': 13} |
| cv3_f2 | cv3 | 83 | 41 | {'b1': 13, 'b2': 15, 'b3': 13} |

Self-check (lobo_b3, Pearson × RF, recomputed through the relabel mechanism vs `results/eng08_full.json`): **PASS** (held-out RMSE 0.01601599 vs 0.01601599).

## e08 — ENG-08 grid (9 level features, full clean rows)

| regressor | selector | lobo_b1 | lobo_b2 | lobo_b3 | cv3_f0 | cv3_f1 | cv3_f2 | LOBO mean ± sd | CV3 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| rf | pearson | 0.02831 | 0.03781 | 0.01602 | 0.01512 | 0.01664 | 0.02100 | 0.02738 ± 0.01093 | 0.01759 ± 0.00305 |
| rf | mutual_info | 0.02832 | 0.03781 | 0.01602 | 0.01512 | 0.01664 | 0.02101 | 0.02738 ± 0.01093 | 0.01759 ± 0.00306 |
| rf | rf_importance | 0.02831 | 0.03782 | 0.01636 | 0.01512 | 0.01664 | 0.02096 | 0.02750 ± 0.01075 | 0.01757 ± 0.00303 |
| rf | permutation | 0.02831 | 0.03781 | 0.01636 | 0.01512 | 0.01664 | 0.02095 | 0.02749 ± 0.01075 | 0.01757 ± 0.00302 |
| gpr | pearson | 0.02687 | 0.04018 | 0.01681 | 0.01634 | 0.01802 | 0.02019 | 0.02796 ± 0.01172 | 0.01819 ± 0.00193 |
| gpr | mutual_info | 0.02687 | 0.04018 | 0.01681 | 0.01634 | 0.01802 | 0.02019 | 0.02796 ± 0.01172 | 0.01819 ± 0.00193 |
| gpr | rf_importance | 0.02798 | 0.03912 | 0.01666 | 0.01645 | 0.01827 | 0.02019 | 0.02792 ± 0.01123 | 0.01830 ± 0.00187 |
| gpr | permutation | 0.02687 | 0.03935 | 0.01655 | 0.01634 | 0.01804 | 0.02001 | 0.02759 ± 0.01142 | 0.01813 ± 0.00184 |

Chosen k and top-ranked feature per partition:

| regressor | selector | lobo_b1 | lobo_b2 | lobo_b3 | cv3_f0 | cv3_f1 | cv3_f2 |
|---|---|---|---|---|---|---|---|
| rf | pearson | k=9, `area25` | k=9, `area25` | k=9, `area25` | k=9, `area25` | k=9, `area25` | k=9, `area25` |
| rf | mutual_info | k=9, `prominence` | k=9, `height` | k=9, `prominence` | k=9, `prominence` | k=9, `prominence` | k=9, `prominence` |
| rf | rf_importance | k=8, `height` | k=9, `area25` | k=7, `area25` | k=8, `height` | k=8, `height` | k=8, `height` |
| rf | permutation | k=8, `height` | k=9, `height` | k=7, `area25` | k=9, `height` | k=8, `height` | k=8, `height` |
| gpr | pearson | k=8, `area25` | k=8, `area25` | k=9, `area25` | k=9, `area25` | k=9, `area25` | k=9, `area25` |
| gpr | mutual_info | k=8, `prominence` | k=8, `height` | k=9, `prominence` | k=9, `prominence` | k=9, `prominence` | k=9, `prominence` |
| gpr | rf_importance | k=6, `height` | k=3, `area25` | k=7, `area25` | k=4, `height` | k=5, `height` | k=9, `height` |
| gpr | permutation | k=8, `area25` | k=8, `area` | k=6, `area` | k=9, `area25` | k=7, `area25` | k=8, `area25` |

Ranking stability (Kendall τ over the 9-feature ranks):

| regressor | selector | partitions | mean τ across partitions (min) | mean top-3 Jaccard |
|---|---|---|---|---|
| rf | pearson | 6 | 0.97 (0.94) | 1.00 |
| rf | mutual_info | 6 | 0.91 (0.83) | 1.00 |
| rf | rf_importance | 6 | 0.61 (0.39) | 0.59 |
| rf | permutation | 6 | 0.49 (0.28) | 0.41 |
| gpr | pearson | 6 | 0.97 (0.94) | 1.00 |
| gpr | mutual_info | 6 | 0.91 (0.83) | 1.00 |
| gpr | rf_importance | 6 | 0.61 (0.39) | 0.59 |
| gpr | permutation | 6 | 0.54 (0.11) | 0.59 |

Cross-method agreement within each partition (mean pairwise τ, RF / GPR): lobo_b1 0.06 / 0.16; lobo_b2 0.17 / 0.14; lobo_b3 0.42 / 0.22; cv3_f0 0.05 / 0.17; cv3_f1 0.26 / 0.24; cv3_f2 0.09 / 0.13

## e14 — ENG-14 pools (history-filtered rows, untuned RF)

| selector | pool | lobo_b1 | lobo_b2 | lobo_b3 | cv3_f0 | cv3_f1 | cv3_f2 | LOBO mean ± sd | CV3 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| pearson | level | 0.02863 | 0.03851 | 0.01621 | 0.01532 | 0.01681 | 0.02126 | 0.02778 ± 0.01118 | 0.01780 ± 0.00309 |
| pearson | temporal | 0.02621 | 0.03565 | 0.01420 | 0.01289 | 0.01355 | 0.01828 | 0.02536 ± 0.01075 | 0.01491 ± 0.00294 |
| pearson | combined | 0.02628 | 0.03550 | 0.01417 | 0.01282 | 0.01343 | 0.01820 | 0.02532 ± 0.01070 | 0.01482 ± 0.00295 |
| permutation | level | 0.02862 | 0.03850 | 0.01620 | 0.01531 | 0.01680 | 0.02121 | 0.02777 ± 0.01118 | 0.01777 ± 0.00306 |
| permutation | temporal | 0.02620 | 0.03569 | 0.01433 | 0.01294 | 0.01382 | 0.01829 | 0.02541 ± 0.01070 | 0.01501 ± 0.00287 |
| permutation | combined | 0.02623 | 0.03540 | 0.01437 | 0.01284 | 0.01400 | 0.01821 | 0.02533 ± 0.01055 | 0.01502 ± 0.00282 |
| rf_importance | level | 0.02861 | 0.03852 | 0.01620 | 0.01532 | 0.01680 | 0.02121 | 0.02778 ± 0.01119 | 0.01778 ± 0.00306 |
| rf_importance | temporal | 0.02622 | 0.03566 | 0.01414 | 0.01289 | 0.01353 | 0.01831 | 0.02534 ± 0.01079 | 0.01491 ± 0.00297 |
| rf_importance | combined | 0.02623 | 0.03549 | 0.01416 | 0.01288 | 0.01340 | 0.01821 | 0.02529 ± 0.01070 | 0.01483 ± 0.00294 |
| mutual_info | level | 0.02863 | 0.03851 | 0.01617 | 0.01532 | 0.01681 | 0.02126 | 0.02777 ± 0.01119 | 0.01779 ± 0.00309 |
| mutual_info | temporal | 0.02625 | 0.03565 | 0.01424 | 0.01289 | 0.01354 | 0.01828 | 0.02538 ± 0.01073 | 0.01490 ± 0.00294 |
| mutual_info | combined | 0.02622 | 0.03552 | 0.01410 | 0.01284 | 0.01341 | 0.01820 | 0.02528 ± 0.01074 | 0.01482 ± 0.00294 |

Relative held-out RMSE vs the level pool, same partition and selector:

| selector | pool | lobo_b1 | lobo_b2 | lobo_b3 | cv3_f0 | cv3_f1 | cv3_f2 | mean |
|---|---|---|---|---|---|---|---|---|
| pearson | temporal | -8.4% | -7.4% | -12.4% | -15.8% | -19.4% | -14.0% | -12.9% |
| pearson | combined | -8.2% | -7.8% | -12.6% | -16.3% | -20.1% | -14.4% | -13.2% |
| permutation | temporal | -8.4% | -7.3% | -11.5% | -15.5% | -17.8% | -13.8% | -12.4% |
| permutation | combined | -8.4% | -8.0% | -11.3% | -16.1% | -16.7% | -14.1% | -12.4% |
| rf_importance | temporal | -8.4% | -7.4% | -12.7% | -15.9% | -19.4% | -13.7% | -12.9% |
| rf_importance | combined | -8.3% | -7.9% | -12.6% | -15.9% | -20.2% | -14.2% | -13.2% |
| mutual_info | temporal | -8.3% | -7.4% | -12.0% | -15.8% | -19.4% | -14.0% | -12.8% |
| mutual_info | combined | -8.4% | -7.8% | -12.8% | -16.2% | -20.2% | -14.4% | -13.3% |

Combined pool: where the feature kinds rank (best rank of each kind; `lvl` = best raw level feature):

| selector | partition | k | temporal share of selected | top-1 | lvl / rm20 / drift / slope20 / d20 / slope10 / d10 |
|---|---|---|---|---|---|
| pearson | lobo_b1 | 48 | 0.81 | `slope_right_rm20` | 7 / 1 / 11 / 25 / 30 / 37 / 38 |
| pearson | lobo_b2 | 63 | 0.86 | `area25_rm20` | 5 / 1 / 10 / 25 / 30 / 37 / 39 |
| pearson | lobo_b3 | 63 | 0.86 | `area25_rm20` | 7 / 1 / 12 / 25 / 30 / 37 / 39 |
| pearson | cv3_f0 | 63 | 0.86 | `area25_rm20` | 7 / 1 / 10 / 25 / 31 / 37 / 38 |
| pearson | cv3_f1 | 63 | 0.86 | `area25_rm20` | 6 / 1 / 10 / 25 / 29 / 37 / 38 |
| pearson | cv3_f2 | 63 | 0.86 | `area25_rm20` | 6 / 1 / 10 / 25 / 30 / 36 / 39 |
| permutation | lobo_b1 | 63 | 0.86 | `slope_right_rm20` | 13 / 1 / 4 / 17 / 23 / 29 / 26 |
| permutation | lobo_b2 | 36 | 0.86 | `area25_rm20` | 6 / 1 / 5 / 15 / 11 / 25 / 17 |
| permutation | lobo_b3 | 48 | 0.88 | `area25_rm20` | 13 / 1 / 8 / 17 / 24 / 26 / 23 |
| permutation | cv3_f0 | 48 | 0.88 | `slope_right_rm20` | 15 / 1 / 8 / 19 / 24 / 28 / 27 |
| permutation | cv3_f1 | 16 | 0.94 | `area25_rm20` | 14 / 1 / 4 / 13 / 16 / 22 / 23 |
| permutation | cv3_f2 | 48 | 0.90 | `area25_rm20` | 10 / 1 / 6 / 16 / 18 / 25 / 19 |
| rf_importance | lobo_b1 | 48 | 0.83 | `slope_right_rm20` | 12 / 1 / 6 / 18 / 30 / 34 / 37 |
| rf_importance | lobo_b2 | 36 | 0.86 | `area25_rm20` | 11 / 1 / 5 / 21 / 15 / 30 / 18 |
| rf_importance | lobo_b3 | 63 | 0.86 | `area25_rm20` | 18 / 1 / 10 / 20 / 26 / 30 / 28 |
| rf_importance | cv3_f0 | 36 | 0.86 | `area25_rm20` | 18 / 1 / 9 / 21 / 23 / 30 / 26 |
| rf_importance | cv3_f1 | 48 | 0.81 | `area25_rm20` | 16 / 1 / 8 / 19 / 22 / 33 / 26 |
| rf_importance | cv3_f2 | 48 | 0.83 | `slope_right_rm20` | 16 / 1 / 7 / 20 / 22 / 30 / 21 |
| mutual_info | lobo_b1 | 36 | 0.78 | `height_rm20` | 11 / 1 / 8 / 26 / 29 / 35 / 36 |
| mutual_info | lobo_b2 | 63 | 0.86 | `height_rm20` | 10 / 1 / 7 / 26 / 28 / 33 / 37 |
| mutual_info | lobo_b3 | 63 | 0.86 | `area25_rm20` | 9 / 1 / 14 / 25 / 34 / 36 / 39 |
| mutual_info | cv3_f0 | 63 | 0.86 | `height_rm20` | 8 / 1 / 11 / 26 / 29 / 35 / 38 |
| mutual_info | cv3_f1 | 63 | 0.86 | `height_rm20` | 9 / 1 / 11 / 26 / 29 / 34 / 38 |
| mutual_info | cv3_f2 | 63 | 0.86 | `height_rm20` | 8 / 1 / 12 / 26 / 30 / 36 / 39 |
