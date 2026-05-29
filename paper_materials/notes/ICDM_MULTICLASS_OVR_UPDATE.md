# ICDM Multiclass One-vs-Rest Breadth

Separate breadth experiment for `multiclass` using XGBoost, LightGBM, and MLP downstream learners.

## Performance

| Method | Mean Macro-AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Full data | 0.9310 | 0.0911 | 1.0590 | 144 |
| HistDistill | 0.9100 | 0.1000 | 2.8889 | 144 |
| K-center | 0.9022 | 0.1197 | 3.0382 | 144 |
| Random | 0.8971 | 0.1124 | 3.9375 | 144 |
| Herding | 0.8960 | 0.1046 | 4.0764 | 144 |

## Pairwise Tests

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill | Random | 144 | 0.0129 | 114 | 0 | 30 | 0.0000 |
| HistDistill | Herding | 144 | 0.0140 | 118 | 0 | 26 | 0.0000 |
| HistDistill | K-center | 144 | 0.0078 | 69 | 0 | 75 | 0.0999 |

## Files

- `tables/icdm_multiclass_ovr_cells.csv`
- `tables/icdm_multiclass_ovr_performance.csv`
- `tables/icdm_multiclass_ovr_pairwise.csv`
- `figures/icdm_multiclass_ovr_budget_profile.png`
- `figures/icdm_multiclass_ovr_paired_delta_violins.png`
