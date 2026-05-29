# ICDM Density Pareto Diagnostic

Density sweep for Eq. (6), using beta variants of HistDistill on budgets 25/50 and seeds 0/1/2. The table reports approximate RBF MMD^2, direct SLD, and XGBoost/MLP AUROC.

| Method | Beta | MMD^2 | SLD | XGB AUROC | MLP AUROC | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| beta=0.00 | 0.0000 | 0.0077 | 24.4979 | 0.7961 | 0.6596 | 156 |
| beta=0.15 | 0.1500 | 0.0075 | 25.0760 | 0.7967 | 0.6677 | 156 |
| beta=0.35 | 0.3500 | 0.0075 | 25.0826 | 0.7959 | 0.6675 | 156 |
| beta=0.70 | 0.7000 | 0.0075 | 25.0826 | 0.7959 | 0.6675 | 156 |
| beta=1.25 | 1.2500 | 0.0074 | 25.2421 | 0.7972 | 0.6656 | 156 |

## Files

- `tables/icdm_density_pareto_cells.csv`
- `tables/icdm_density_pareto_summary.csv`
- `figures/icdm_density_pareto.png`
