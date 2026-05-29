# Phase 2 Results

Phase 2 re-centers the paper on mechanism diagnostics: structure-vs-leaf error, continuous split regret, universal SLD mapping, and training-free selection.

## Broadened Robustness Addendum

The robustness follow-up broadens the two weakest scope points. HPO is now reported on 12 binary datasets rather than the original 5-dataset slice. E18 downstream dose-response is now reported on all 26 available binary datasets, 3 seeds, 3 repeats, and 8 dose levels (`1,872` cells total; `234` cells per dose), replacing the earlier 54-cell per-dose slice. See `notes/PHASE2_E18_E21_FULL.md` and `tables/icdm_hpo_summary.csv`.

## E16 Structure vs Leaf-Estimate Error

| method_label | deployed_auc | leaf_refit_auc | structure_ref_auc | full_auc | structure_error | leaf_estimate_recovery | leaf_population_l1 | sld_inf_norm | n_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient sampling | 0.3495 | 0.7230 | 0.7910 | 0.8258 | 0.0679 | 0.3736 | 0.1326 |  | 312 |
| HistDistill-Greedy | 0.6428 | 0.6965 | 0.7910 | 0.8258 | 0.0945 | 0.0537 | 0.2426 | 76.8596 | 312 |
| DistributionMatching | 0.7201 | 0.7560 | 0.7910 | 0.8258 | 0.0349 | 0.0360 | 0.2347 |  | 312 |
| K-center | 0.7129 | 0.7367 | 0.7910 | 0.8258 | 0.0543 | 0.0237 | 0.1956 |  | 312 |
| EL2N | 0.7146 | 0.7372 | 0.7910 | 0.8258 | 0.0537 | 0.0226 | 0.0935 | 42.2226 | 312 |
| GraNd | 0.7146 | 0.7372 | 0.7910 | 0.8258 | 0.0537 | 0.0226 | 0.0935 | 42.2226 | 312 |
| CRAIG-style | 0.7159 | 0.7341 | 0.7910 | 0.8258 | 0.0568 | 0.0182 | 0.1149 | 44.9344 | 312 |
| MVS coreset | 0.7241 | 0.7342 | 0.7910 | 0.8258 | 0.0567 | 0.0101 | 0.0877 | 40.0286 | 312 |
| GOSS coreset | 0.7284 | 0.7379 | 0.7910 | 0.8258 | 0.0530 | 0.0095 | 0.0767 | 49.7315 | 312 |
| GradMatch-style | 0.7336 | 0.7316 | 0.7910 | 0.8258 | 0.0593 | -0.0020 | 0.1127 | 68.0072 | 312 |
| Random | 0.7464 | 0.7425 | 0.7910 | 0.8258 | 0.0485 | -0.0039 | 0.0873 | 38.3088 | 312 |
| Herding | 0.7524 | 0.7323 | 0.7910 | 0.8258 | 0.0586 | -0.0201 | 0.1378 | 54.9173 | 312 |
| Gain sketch | 0.7698 | 0.7482 | 0.7910 | 0.8258 | 0.0427 | -0.0215 | 0.0688 |  | 312 |
| Gain-path sketch | 0.7716 | 0.7473 | 0.7910 | 0.8258 | 0.0437 | -0.0243 | 0.0685 |  | 312 |
| HistDistill-Refined | 0.7728 | 0.7471 | 0.7910 | 0.8258 | 0.0438 | -0.0257 | 0.0599 | 24.4979 | 312 |
| HistDistill-Density | 0.7729 | 0.7471 | 0.7910 | 0.8258 | 0.0439 | -0.0259 | 0.0611 | 24.7992 | 312 |
| Legacy gain-path refined | 0.7741 | 0.7479 | 0.7910 | 0.8258 | 0.0431 | -0.0262 | 0.0676 | 26.9114 | 312 |

### E16 Correlation Checks

| x | y | pooled_spearman | within_dataset_spearman | n_cells |
| --- | --- | --- | --- | --- |
| sld_inf_norm | structure_error | 0.0576 | 0.1958 | 3744 |
| split_regret_norm | structure_error | 0.1251 | 0.1484 | 5304 |
| leaf_population_l1 | leaf_estimate_recovery | 0.4079 | 0.3683 | 5304 |
| leaf_population_kl | leaf_estimate_recovery | 0.1897 | 0.2487 | 5304 |
| sld_inf_norm | leaf_estimate_recovery | 0.0643 | 0.1446 | 3744 |

## E17 Split-Regret Law

| metric | pooled_spearman_vs_auroc | within_dataset_spearman_vs_auroc | n_cells |
| --- | --- | --- | --- |
| split_regret_norm | 0.1074 | -0.2085 | 2652 |
| split_regret_root_norm | 0.0489 | -0.1705 | 2652 |
| sld_inf_norm | -0.0359 | -0.3353 | 1872 |

## E18 Controlled Dose-Response

This is a landscape-only controlled perturbation test: it injects calibrated split-gain noise and measures agreement/regret. It does not claim downstream AUROC causality without a full perturbed-tree retraining operator.

| dose_ratio | agreement | split_regret_norm | n_states |
| --- | --- | --- | --- |
| 0.0000 | 1.0000 | 0.0000 | 252 |
| 0.2500 | 1.0000 | 0.0000 | 252 |
| 0.4900 | 1.0000 | 0.0000 | 252 |
| 0.5000 | 1.0000 | 0.0000 | 252 |
| 0.7500 | 0.9089 | 0.0309 | 252 |
| 1.0000 | 0.7870 | 0.0726 | 252 |
| 1.5000 | 0.6300 | 0.1259 | 252 |
| 2.0000 | 0.5420 | 0.1593 | 252 |

## E19 Universal Diagnostic Map

| method_label | auroc | split_regret_norm | split_regret_agreement | line_residual_auc | leaf_estimate_recovery | structure_error | n_cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gain-path sketch | 0.7475 | 0.2512 | 0.1091 | 0.0077 | -0.0243 | 0.0437 | 156 |
| Legacy gain-path refined | 0.7509 | 0.2524 | 0.1123 | 0.0122 | -0.0262 | 0.0431 | 156 |
| HistDistill-Density | 0.7508 | 0.2529 | 0.1067 | 0.0126 | -0.0259 | 0.0439 | 156 |
| HistDistill-Refined | 0.7498 | 0.2551 | 0.1078 | 0.0135 | -0.0257 | 0.0438 | 156 |
| Random | 0.7260 | 0.2651 | 0.0809 | -0.0011 | -0.0039 | 0.0485 | 156 |
| Gain sketch | 0.7490 | 0.2684 | 0.1054 | 0.0249 | -0.0215 | 0.0427 | 156 |
| Herding | 0.7339 | 0.2787 | 0.0708 | 0.0192 | -0.0201 | 0.0586 | 156 |
| DistributionMatching | 0.7028 | 0.2982 | 0.0903 | 0.0059 | 0.0360 | 0.0349 | 156 |
| MVS coreset | 0.6967 | 0.3008 | 0.0905 | 0.0022 | 0.0101 | 0.0567 | 156 |
| K-center | 0.6967 | 0.3146 | 0.0908 | 0.0148 | 0.0237 | 0.0543 | 156 |
| GOSS coreset | 0.6997 | 0.3174 | 0.0635 | 0.0205 | 0.0095 | 0.0530 | 156 |
| GradMatch-style | 0.7150 | 0.3242 | 0.0679 | 0.0420 | -0.0020 | 0.0593 | 156 |
| Gradient sampling | 0.3656 | 0.3283 | 0.0749 | -0.3037 | 0.3736 | 0.0679 | 156 |
| EL2N | 0.6864 | 0.3287 | 0.0740 | 0.0175 | 0.0226 | 0.0537 | 156 |
| GraNd | 0.6864 | 0.3287 | 0.0740 | 0.0175 | 0.0226 | 0.0537 | 156 |
| CRAIG-style | 0.6990 | 0.3345 | 0.0723 | 0.0354 | 0.0182 | 0.0568 | 156 |
| HistDistill-Greedy | 0.6285 | 0.4373 | 0.0379 | 0.0589 | 0.0537 | 0.0945 | 156 |

### E19 Residual Checks

| quantity | value | n_methods |
| --- | --- | --- |
| method_level_split_regret_to_auroc_slope | -0.9145 | 17 |
| method_level_split_regret_to_auroc_spearman | -0.8675 | 17 |
| line_residual_vs_leaf_estimate_recovery_spearman | 0.0503 | 17 |
| line_residual_vs_leaf_population_l1_spearman | 0.2294 | 17 |
| line_residual_vs_structure_error_spearman | 0.3325 | 17 |

## E20 Training-Free Selection

| selector | top1 | top3 | rank_spearman | cells |
| --- | --- | --- | --- | --- |
| sld | 0.1218 | 0.4359 | -0.2766 | 156 |
| split_regret | 0.1026 | 0.2372 | -0.1860 | 156 |

## E21 Generality Scope

| artifact | rows | primary_metric | histdistill | random | full_data | note | tree_histdistill | tree_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiclass_ovr | 5 | macro_auroc | 0.9100 | 0.8971 | 0.9310 | supportive: HistDistill beats random on mean macro-AUROC but remains below full-data |  |  |
| regression | 5 | r2 | -1.7936 | -0.6794 | 0.6257 | limitation: all-learner mean is dominated by severe MLP failures; tree-only slice is milder | 0.1624 | 0.1569 |
| hpo | 1 | top3_jaccard | 0.0000 |  |  | negative: condensed-set HPO rankings do not recover full-data top configurations |  |  |

## Files

- `notes/PHASE2_E18_E21_FULL.md`
- `tables/phase2_e16_error_decomposition_cells.csv`
- `tables/phase2_e16_error_decomposition_summary.csv`
- `tables/phase2_e16_decomposition_correlations.csv`
- `tables/phase2_e17_split_regret_cells.csv`
- `tables/phase2_e17_metric_law_summary.csv`
- `tables/phase2_e18_dose_response_cells.csv`
- `tables/phase2_e18_dose_response_summary.csv`
- `tables/phase2_e18_downstream_dose_response_cells.csv`
- `tables/phase2_e18_downstream_dose_response_summary.csv`
- `tables/phase2_e19_universal_sld_map.csv`
- `tables/phase2_e19_residual_summary.csv`
- `tables/phase2_e20_training_free_selection.csv`
- `tables/phase2_e20_training_free_selection_summary.csv`
- `tables/phase2_e21_generality_scope.csv`
- `tables/phase2_e21_multiclass_tree_scope.csv`
- `tables/phase2_e21_regression_tree_scope.csv`
- `tables/phase2_e21_regression_decomposition_cells.csv`
- `tables/phase2_e21_regression_decomposition_summary.csv`
- `tables/phase2_e21_regression_decomposition_correlations.csv`
- `figures/phase2_e16_error_decomposition.png`
- `figures/phase2_e17_split_regret_law.png`
- `figures/phase2_e18_dose_response.png`
- `figures/phase2_e18_downstream_dose_response.png`
- `figures/phase2_e19_universal_sld_map.png`
- `figures/phase2_e20_training_free_selection.png`
- `figures/phase2_e21_regression_decomposition.png`
