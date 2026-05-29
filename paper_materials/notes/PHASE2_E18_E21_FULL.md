# Phase 2 Full E18/E21 Addendum

This addendum completes the two Phase 2 items that were previously proxy-only: E18 now includes downstream AUROC under controlled split-noise dosing, and E21 now includes a regression structure-vs-leaf decomposition.

## E18 Downstream Dose-Response

| dose_ratio | auroc | delta_auc_vs_dose0 | agreement | root_agreement | split_regret_norm | n_cells |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.8153 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 234 |
| 0.2500 | 0.8153 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 234 |
| 0.4900 | 0.8153 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 234 |
| 0.5000 | 0.8153 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 234 |
| 0.7500 | 0.8131 | -0.0023 | 0.9332 | 0.9349 | 0.0109 | 234 |
| 1.0000 | 0.8112 | -0.0041 | 0.8356 | 0.8387 | 0.0302 | 234 |
| 1.5000 | 0.8110 | -0.0043 | 0.6874 | 0.6839 | 0.0695 | 234 |
| 2.0000 | 0.8077 | -0.0076 | 0.5899 | 0.5843 | 0.1011 | 234 |

## E21 Regression Decomposition

| method | method_label | deployed_r2 | leaf_refit_r2 | structure_ref_r2 | structure_error | leaf_estimate_recovery | leaf_population_l1 | n_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| histdistill | HistDistill | 0.1624 | 0.2796 | 0.6266 | 0.3471 | 0.1172 | 0.0885 | 96 |
| random | Random | 0.1569 | 0.2894 | 0.6266 | 0.3373 | 0.1325 | 0.0898 | 96 |
| herding | Herding | 0.1308 | 0.2802 | 0.6266 | 0.3465 | 0.1494 | 0.1387 | 96 |
| k_center | K-center | 0.0485 | 0.2837 | 0.6266 | 0.3429 | 0.2352 | 0.1507 | 96 |

### E21 Correlations

| x | y | pooled_spearman | n_cells |
| --- | --- | --- | --- |
| leaf_population_l1 | leaf_estimate_recovery | 0.7282 | 480 |
| leaf_population_l1 | structure_error | -0.0459 | 480 |

## Files

- `tables/phase2_e18_downstream_dose_response_cells.csv`
- `tables/phase2_e18_downstream_dose_response_summary.csv`
- `tables/phase2_e21_regression_decomposition_cells.csv`
- `tables/phase2_e21_regression_decomposition_summary.csv`
- `tables/phase2_e21_regression_decomposition_correlations.csv`
- `figures/phase2_e18_downstream_dose_response.png`
- `figures/phase2_e21_regression_decomposition.png`
