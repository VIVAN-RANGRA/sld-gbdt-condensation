# ICDM Theory Diagnostics

Diagnostics generated from the completed binary grid. These connect the empirical section to the SLD theory: SLD/AUROC behavior, Prop. 1 margin cells, budget scaling, transfer regret, runtime, and selector coverage.

## SLD Law

| Method | Cells | Pooled Spearman | Within Dataset Spearman | Mean SLD | Mean AUROC | Prop1 Cells | Prop1 Agree | All Root Agree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 390 | 0.2295 | -0.5777 | 27.4184 | 0.7430 | 12 | 1.0000 | 0.1615 |
| Herding | 390 | 0.0960 | -0.5250 | 54.1967 | 0.7308 | 9 | 1.0000 | 0.1000 |
| HistDistill-Density | 390 | 0.1774 | -0.5769 | 25.3561 | 0.7423 | 12 | 1.0000 | 0.1692 |
| HistDistill-Refined | 390 | 0.2093 | -0.5590 | 24.9256 | 0.7423 | 12 | 1.0000 | 0.1769 |
| Random | 390 | -0.1007 | -0.6400 | 39.9271 | 0.7248 | 8 | 1.0000 | 0.1231 |

## Budget Scaling

| Method | Mean log-log slope | Median log-log slope |
| --- | --- | --- |
| Legacy gain-path refined | -0.4265 | -0.4935 |
| Herding | -0.3627 | -0.2244 |
| HistDistill-Density | -0.4579 | -0.4825 |
| HistDistill-Refined | -0.4362 | -0.4768 |
| Random | -0.5170 | -0.5078 |

## Transfer Regret

| method_label | catboost | lightgbm | mlp | random_forest | xgboost |
| --- | --- | --- | --- | --- | --- |
| Herding | 0.1268 | 0.1099 | 0.0570 | 0.0571 | 0.0548 |
| HistDistill-Density | 0.1098 | 0.0959 | 0.0633 | 0.0419 | 0.0368 |
| HistDistill-Refined | 0.1090 | 0.0953 | 0.0642 | 0.0420 | 0.0374 |
| Legacy gain-path refined | 0.1099 | 0.0934 | 0.0639 | 0.0417 | 0.0355 |
| Random | 0.1224 | 0.1078 | 0.0761 | 0.0655 | 0.0634 |

## Runtime

| Method | Median Speedup | Mean Seconds |
| --- | --- | --- |
| Legacy gain-path refined | 3.9638 | 0.3466 |
| Herding | 3.8096 | 0.4798 |
| HistDistill-Density | 3.2067 | 0.4142 |
| HistDistill-Refined | 3.4783 | 0.4137 |
| Random | 4.0204 | 0.4305 |

## Selector Coverage

| Method | Budget | Coverage Ratio | f(S)/proxy ub | Matched Rows | Cells |
| --- | --- | --- | --- | --- | --- |
| HistDistill-Greedy | budget_25 | 0.3710 | 1.0000 | 50.0000 | 78 |
| Random | budget_25 | 0.0816 | 0.2456 | 50.0000 | 78 |
| HistDistill-Density | budget_25 | 0.0748 | 0.2233 | 50.0000 | 78 |
| HistDistill-Refined | budget_25 | 0.0748 | 0.2233 | 50.0000 | 78 |
| ImportanceSample | budget_25 | 0.0707 | 0.2226 | 50.0000 | 78 |
| Herding | budget_25 | 0.0586 | 0.1830 | 50.0000 | 78 |
| HistDistill-Greedy | budget_50 | 0.5385 | 1.0000 | 99.8462 | 78 |
| Random | budget_50 | 0.1591 | 0.2988 | 99.8462 | 78 |
| HistDistill-Density | budget_50 | 0.1524 | 0.2857 | 99.8462 | 78 |
| HistDistill-Refined | budget_50 | 0.1524 | 0.2857 | 99.8462 | 78 |
| ImportanceSample | budget_50 | 0.1424 | 0.2741 | 99.8462 | 78 |
| Herding | budget_50 | 0.1177 | 0.2286 | 99.8462 | 78 |

## Files

- `tables/icdm_sld_law_summary.csv`
- `tables/icdm_margin_distribution.csv`
- `tables/icdm_margin_summary.csv`
- `tables/icdm_budget_collapse_slopes.csv`
- `tables/icdm_budget_collapse_by_size.csv`
- `tables/icdm_transfer_regret.csv`
- `tables/icdm_runtime_summary.csv`
- `tables/icdm_selector_coverage_summary.csv`
- `figures/icdm_sld_law_scatter.png`
- `figures/icdm_margin_distribution.png`
- `figures/icdm_budget_collapse_sld.png`
- `figures/icdm_transfer_regret_profiles.png`
- `figures/icdm_runtime_speedup.png`
- `figures/icdm_selector_coverage.png`
