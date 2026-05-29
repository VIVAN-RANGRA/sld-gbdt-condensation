# ICDM Regression Breadth

Separate breadth experiment for `regression` using XGBoost, LightGBM, and MLP downstream learners.

## Performance

| Method | Mean R2 | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Full data | 0.6257 | 0.2482 | 1.0972 | 144 |
| Herding | -0.2471 | 1.9295 | 3.8160 | 144 |
| Random | -0.6794 | 4.5557 | 3.1875 | 144 |
| HistDistill | -1.7936 | 13.8203 | 2.9201 | 144 |
| K-center | -4.0679 | 49.2544 | 3.9792 | 144 |

## Pairwise Tests

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill | Random | 144 | -1.1142 | 83 | 0 | 61 | 0.1281 |
| HistDistill | Herding | 144 | -1.5466 | 103 | 4 | 37 | 0.0000 |
| HistDistill | K-center | 144 | 2.2743 | 107 | 0 | 37 | 0.0000 |

## Files

- `tables/icdm_regression_cells.csv`
- `tables/icdm_regression_performance.csv`
- `tables/icdm_regression_pairwise.csv`
- `figures/icdm_regression_budget_profile.png`
- `figures/icdm_regression_paired_delta_violins.png`
