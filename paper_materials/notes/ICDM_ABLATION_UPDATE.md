# ICDM HistDistill Ablation Update

Binary component-ablation grid for budgets budget_25, budget_50 and seeds seed_0, seed_1, seed_2. Expected complete cells per method: 780.

## Performance

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.7509 | 0.1777 | 3.4333 | 780 |
| HistDistill-Density | 0.7508 | 0.1755 | 3.5904 | 780 |
| HistDistill-Refined | 0.7498 | 0.1772 | 3.5923 | 780 |
| Herding | 0.7339 | 0.1681 | 4.7019 | 780 |
| Random | 0.7260 | 0.1733 | 5.1917 | 780 |
| ImportanceSample | 0.6904 | 0.1790 | 6.2917 | 780 |
| Pure greedy selector | 0.6285 | 0.1573 | 7.4333 | 780 |
| No refinement | 0.6279 | 0.1577 | 7.4423 | 780 |
| No weight fit | 0.6163 | 0.1353 | 7.7692 | 780 |
| Root-only probes | 0.5964 | 0.1361 | 8.2865 | 780 |
| Linear coverage | 0.5957 | 0.1355 | 8.2673 | 780 |

## Paired Component Tests

| Component | Reference | Ablation | Pairs | Mean Delta | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| density term | HistDistill-Density | HistDistill-Refined | 780 | 0.0010 | 386 | 11 | 383 | 0.6859 |
| weight fit | HistDistill-Refined | No weight fit | 780 | 0.1335 | 659 | 0 | 121 | 0.0000 |
| probe regions | HistDistill-Refined | Root-only probes | 780 | 0.1534 | 671 | 0 | 109 | 0.0000 |
| saturating coverage | HistDistill-Refined | Linear coverage | 780 | 0.1541 | 672 | 0 | 108 | 0.0000 |
| local refinement | HistDistill-Refined | No refinement | 780 | 0.1219 | 651 | 1 | 128 | 0.0000 |
| warm-start/refined selector | HistDistill-Refined | Pure greedy selector | 780 | 0.1213 | 651 | 1 | 128 | 0.0000 |
| greedy vs importance | Pure greedy selector | ImportanceSample | 780 | -0.0618 | 262 | 1 | 517 | 0.0000 |

## Fidelity / SLD

| Method | Retrained Gain Corr. | Direct SLD Inf. | Direct Root Agree | Direct Rank Corr. | Margin Satisfied | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.3729 | 26.9114 | 0.1474 | 0.3302 | 0.0284 | 156 |
| Herding | 0.2863 | 54.9173 | 0.0641 | 0.2785 | 0.0123 | 156 |
| HistDistill-Density | 0.3720 | 24.7992 | 0.1538 | 0.3217 | 0.0265 | 156 |
| Pure greedy selector | 0.2048 | 76.8596 | 0.0385 | 0.1331 | 0.0000 | 156 |
| ImportanceSample | 0.2545 | 40.5315 | 0.1026 | 0.2287 | 0.0009 | 156 |
| Linear coverage | 0.0621 | 167.6732 | 0.0192 | 0.0371 | 0.0000 | 156 |
| No refinement | 0.2056 | 76.9269 | 0.0385 | 0.1336 | 0.0000 | 156 |
| No weight fit | 0.1332 | 146.1921 | 0.0192 | 0.0704 | 0.0000 | 156 |
| HistDistill-Refined | 0.3604 | 24.4979 | 0.1538 | 0.3392 | 0.0260 | 156 |
| Root-only probes | 0.0385 | 167.4566 | 0.0513 | 0.0339 | 0.0000 | 156 |
| Random | 0.3104 | 38.3088 | 0.1090 | 0.2861 | 0.0091 | 156 |

## Files

- `tables/icdm_histdistill_ablation_coverage.csv`
- `tables/icdm_histdistill_ablation_performance.csv`
- `tables/icdm_histdistill_ablation_pairwise.csv`
- `tables/icdm_histdistill_ablation_fidelity.csv`
- `figures/icdm_ablation_delta_violins.png`
- `figures/icdm_ablation_sld_tradeoff.png`
