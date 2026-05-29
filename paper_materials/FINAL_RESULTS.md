# Final Results Digest

This is the paper-writing digest for the current Split-Gain Landscape Distillation experiment package. The completed evidence is strongest for binary and multiclass classification; regression breadth is included but should be framed cautiously.

## Binary Headline Performance

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.7500 | 0.1778 | 2.5073 | 1300 |
| HistDistill-Refined | 0.7496 | 0.1770 | 2.6154 | 1300 |
| HistDistill-Density | 0.7494 | 0.1767 | 2.6250 | 1300 |
| Herding | 0.7339 | 0.1681 | 3.4446 | 1300 |
| Random | 0.7254 | 0.1762 | 3.8077 | 1300 |

## Binary Headline Pairwise

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill-Density | Legacy gain-path refined | 1300 | -0.0007 | 609 | 7 | 684 | 0.0501 |
| HistDistill-Density | Herding | 1300 | 0.0155 | 850 | 0 | 450 | 0.0000 |
| HistDistill-Density | Random | 1300 | 0.0239 | 971 | 1 | 328 | 0.0000 |
| HistDistill-Refined | Legacy gain-path refined | 1300 | -0.0004 | 603 | 11 | 686 | 0.0717 |
| HistDistill-Refined | Herding | 1300 | 0.0157 | 862 | 1 | 437 | 0.0000 |
| HistDistill-Refined | Random | 1300 | 0.0241 | 983 | 0 | 317 | 0.0000 |

## Five-Budget Binary Core

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.7430 | 0.1861 | 2.5985 | 3250 |
| HistDistill-Refined | 0.7425 | 0.1859 | 2.6815 | 3250 |
| HistDistill-Density | 0.7425 | 0.1859 | 2.6497 | 3250 |
| Herding | 0.7306 | 0.1760 | 3.4418 | 3250 |
| Random | 0.7244 | 0.1844 | 3.6285 | 3250 |

## Key Pairwise Tests

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill-Density | Legacy gain-path refined | 3250 | -0.0006 | 1454 | 276 | 1520 | 0.2485 |
| HistDistill-Density | Herding | 3250 | 0.0118 | 1973 | 264 | 1013 | 0.0000 |
| HistDistill-Density | Random | 3250 | 0.0181 | 2150 | 266 | 834 | 0.0000 |
| HistDistill-Refined | Legacy gain-path refined | 3250 | -0.0006 | 1410 | 282 | 1558 | 0.0391 |
| HistDistill-Refined | Herding | 3250 | 0.0118 | 1969 | 263 | 1018 | 0.0000 |
| HistDistill-Refined | Random | 3250 | 0.0181 | 2162 | 261 | 827 | 0.0000 |

## Fidelity and SLD

| Method | Retrained Gain Corr. | Direct SLD | Direct Root Agree | Direct Rank Corr. | Cells |
| --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.3937 | 18.3953 | 0.3062 | 0.4593 | 650 |
| Herding | 0.3385 | 34.5993 | 0.1923 | 0.3607 | 650 |
| HistDistill-Density | 0.3909 | 16.9979 | 0.3154 | 0.4540 | 650 |
| HistDistill-Refined | 0.3893 | 16.7257 | 0.3200 | 0.4600 | 650 |
| Random | 0.3359 | 27.7381 | 0.1923 | 0.3935 | 650 |

## Theory Diagnostics

| Method | Within-dataset SLD/AUROC | Mean SLD | Prop1 Cells | Prop1 Agree |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | -0.5777 | 27.4184 | 12 | 1.0000 |
| Herding | -0.5250 | 54.1967 | 9 | 1.0000 |
| HistDistill-Density | -0.5769 | 25.3561 | 12 | 1.0000 |
| HistDistill-Refined | -0.5590 | 24.9256 | 12 | 1.0000 |
| Random | -0.6400 | 39.9271 | 8 | 1.0000 |

## Phase 2 Mechanism: Structure vs Leaf Estimates

| Method | Deployed AUROC | Leaf-refit AUROC | Structure Ref. AUROC | Structure Error | Leaf Recovery | Leaf TV | SLD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill-Greedy | 0.6428 | 0.6965 | 0.7910 | 0.0945 | 0.0537 | 0.2426 | 76.8596 |
| Random | 0.7464 | 0.7425 | 0.7910 | 0.0485 | -0.0039 | 0.0873 | 38.3088 |
| Herding | 0.7524 | 0.7323 | 0.7910 | 0.0586 | -0.0201 | 0.1378 | 54.9173 |
| HistDistill-Refined | 0.7728 | 0.7471 | 0.7910 | 0.0438 | -0.0257 | 0.0599 | 24.4979 |
| HistDistill-Density | 0.7729 | 0.7471 | 0.7910 | 0.0439 | -0.0259 | 0.0611 | 24.7992 |
| Legacy gain-path refined | 0.7741 | 0.7479 | 0.7910 | 0.0431 | -0.0262 | 0.0676 | 26.9114 |

E16 partly falsifies the most optimistic prediction: greedy coverage has both high structure error and high leaf-population mismatch. The defensible result is not 'coverage fixes structure'; it is 'coverage alone is insufficient, and split-regret/SLD expose why.'

Phase 2 E16 correlation checks:

| x | y | pooled_spearman | within_dataset_spearman | n_cells |
| --- | --- | --- | --- | --- |
| sld_inf_norm | structure_error | 0.0576 | 0.1958 | 3744 |
| split_regret_norm | structure_error | 0.1251 | 0.1484 | 5304 |
| leaf_population_l1 | leaf_estimate_recovery | 0.4079 | 0.3683 | 5304 |
| leaf_population_kl | leaf_estimate_recovery | 0.1897 | 0.2487 | 5304 |
| sld_inf_norm | leaf_estimate_recovery | 0.0643 | 0.1446 | 3744 |

## Phase 2 Split-Regret and Universal Map

| metric | pooled_spearman_vs_auroc | within_dataset_spearman_vs_auroc | n_cells |
| --- | --- | --- | --- |
| split_regret_norm | 0.1074 | -0.2085 | 2652 |
| split_regret_root_norm | 0.0489 | -0.1705 | 2652 |
| sld_inf_norm | -0.0359 | -0.3353 | 1872 |

Universal diagnostic map:

| Method | AUROC | Split Regret | Agreement | Line Residual | Leaf Recovery | Structure Error | Cells |
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

Universal map residual checks:

| quantity | value | n_methods |
| --- | --- | --- |
| method_level_split_regret_to_auroc_slope | -0.9145 | 17 |
| method_level_split_regret_to_auroc_spearman | -0.8675 | 17 |
| line_residual_vs_leaf_estimate_recovery_spearman | 0.0503 | 17 |
| line_residual_vs_leaf_population_l1_spearman | 0.2294 | 17 |
| line_residual_vs_structure_error_spearman | 0.3325 | 17 |

## Phase 2 Controlled/Practical Scope

E18 is a landscape-only controlled perturbation test; it supports the Prop. 1 threshold behavior but does not by itself prove downstream AUROC causality.

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

E18 downstream dose-response with full-data leaf estimates:

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

Training-free selection:

| selector | top1 | top3 | rank_spearman | cells |
| --- | --- | --- | --- | --- |
| sld | 0.1218 | 0.4359 | -0.2766 | 156 |
| split_regret | 0.1026 | 0.2372 | -0.1860 | 156 |

E20 is weak as an argmin selection rule; keep it as a diagnostic ordering result, not a main practical claim.

Generality scope:

| artifact | rows | primary_metric | histdistill | random | full_data | note | tree_histdistill | tree_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multiclass_ovr | 5 | macro_auroc | 0.9100 | 0.8971 | 0.9310 | supportive: HistDistill beats random on mean macro-AUROC but remains below full-data |  |  |
| regression | 5 | r2 | -1.7936 | -0.6794 | 0.6257 | limitation: all-learner mean is dominated by severe MLP failures; tree-only slice is milder | 0.1624 | 0.1569 |
| hpo | 1 | top3_jaccard | 0.0000 |  |  | negative: condensed-set HPO rankings do not recover full-data top configurations |  |  |

E21 regression structure-vs-leaf decomposition:

| method | method_label | deployed_r2 | leaf_refit_r2 | structure_ref_r2 | structure_error | leaf_estimate_recovery | leaf_population_l1 | n_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| histdistill | HistDistill | 0.1624 | 0.2796 | 0.6266 | 0.3471 | 0.1172 | 0.0885 | 96 |
| random | Random | 0.1569 | 0.2894 | 0.6266 | 0.3373 | 0.1325 | 0.0898 | 96 |
| herding | Herding | 0.1308 | 0.2802 | 0.6266 | 0.3465 | 0.1494 | 0.1387 | 96 |
| k_center | K-center | 0.0485 | 0.2837 | 0.6266 | 0.3429 | 0.2352 | 0.1507 | 96 |

E21 regression decomposition correlations:

| x | y | pooled_spearman | n_cells |
| --- | --- | --- | --- |
| leaf_population_l1 | leaf_estimate_recovery | 0.7282 | 480 |
| leaf_population_l1 | structure_error | -0.0459 | 480 |

## Component Ablations

| Component | Reference | Ablation | Mean Delta | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| density term | HistDistill-Density | HistDistill-Refined | 0.0010 | 386 | 11 | 383 | 0.6859 |
| weight fit | HistDistill-Refined | No weight fit | 0.1335 | 659 | 0 | 121 | 0.0000 |
| probe regions | HistDistill-Refined | Root-only probes | 0.1534 | 671 | 0 | 109 | 0.0000 |
| saturating coverage | HistDistill-Refined | Linear coverage | 0.1541 | 672 | 0 | 108 | 0.0000 |
| local refinement | HistDistill-Refined | No refinement | 0.1219 | 651 | 1 | 128 | 0.0000 |
| warm-start/refined selector | HistDistill-Refined | Pure greedy selector | 0.1213 | 651 | 1 | 128 | 0.0000 |
| greedy vs importance | Pure greedy selector | ImportanceSample | -0.0618 | 262 | 1 | 517 | 0.0000 |

## Density Pareto

| Method | Beta | MMD^2 | SLD | XGB AUROC | MLP AUROC | Cells |
| --- | --- | --- | --- | --- | --- | --- |
| beta=0.00 | 0.0000 | 0.0077 | 24.4979 | 0.7961 | 0.6596 | 156 |
| beta=0.15 | 0.1500 | 0.0075 | 25.0760 | 0.7967 | 0.6677 | 156 |
| beta=0.35 | 0.3500 | 0.0075 | 25.0826 | 0.7959 | 0.6675 | 156 |
| beta=0.70 | 0.7000 | 0.0075 | 25.0826 | 0.7959 | 0.6675 | 156 |
| beta=1.25 | 1.2500 | 0.0074 | 25.2421 | 0.7972 | 0.6656 | 156 |

## Regression Breadth

| Method | Mean R2 | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Full data | 0.6257 | 0.2482 | 1.0972 | 144 |
| Herding | -0.2471 | 1.9295 | 3.8160 | 144 |
| Random | -0.6794 | 4.5557 | 3.1875 | 144 |
| HistDistill | -1.7936 | 13.8203 | 2.9201 | 144 |
| K-center | -4.0679 | 49.2544 | 3.9792 | 144 |

Regression pairwise tests:

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill | Random | 144 | -1.1142 | 83 | 0 | 61 | 0.1281 |
| HistDistill | Herding | 144 | -1.5466 | 103 | 4 | 37 | 0.0000 |
| HistDistill | K-center | 144 | 2.2743 | 107 | 0 | 37 | 0.0000 |

Regression is a mixed/weak breadth result at budgets 25/50, especially with MLP. Use tree-learner slices cautiously; do not make it the headline.

## Multiclass One-vs-Rest Breadth

| Method | Mean Macro-AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Full data | 0.9310 | 0.0911 | 1.0590 | 144 |
| HistDistill | 0.9100 | 0.1000 | 2.8889 | 144 |
| K-center | 0.9022 | 0.1197 | 3.0382 | 144 |
| Random | 0.8971 | 0.1124 | 3.9375 | 144 |
| Herding | 0.8960 | 0.1046 | 4.0764 | 144 |

Multiclass pairwise tests:

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill | Random | 144 | 0.0129 | 114 | 0 | 30 | 0.0000 |
| HistDistill | Herding | 144 | 0.0140 | 118 | 0 | 26 | 0.0000 |
| HistDistill | K-center | 144 | 0.0078 | 69 | 0 | 75 | 0.0999 |

## HPO Diagnostic

| datasets | mean_top3_jaccard | mean_top1_match | mean_score_spearman | median_hpo_speedup | mean_test_auc_gap |
| --- | --- | --- | --- | --- | --- |
| 12 | 0.0417 | 0.0833 | 0.1451 | 1.5720 | 0.0012 |

HPO is currently a weak/negative application result: the average final test-AUROC gap is small, but top-3 agreement and speedup do not meet the planned success target.

## Experiment Status

| experiment | status | note |
| --- | --- | --- |
| E1 main binary | complete | Full 26-dataset binary grid is complete for budgets 10/25/50/100/200, seeds 0..4, and five learners; the compact headline table uses budgets 25/50. |
| E2 statistical tests | complete | Wilcoxon and win/tie/loss are generated in core and ablation tables. |
| E3 SLD law | complete | Direct SLD fields regenerated for key methods; SLD law and Prop. 1 diagnostics generated. |
| E4 budget curve | complete | Key methods have budgets 10/25/50/100/200 with AUROC and SLD summaries. |
| E5 margin distribution | complete | Teacher root/probe margin distribution generated from landscape files. |
| E6 selector study | complete | Greedy/importance selector AUROC, SLD, coverage ratio, and f(S)/proxy upper-bound diagnostics are generated. |
| E7 transfer | complete | Cross-learner transfer regret table and profile figure generated. |
| E8 MMD-SLD Pareto | complete | Density beta sweep with MMD, SLD, XGBoost AUROC, and MLP AUROC is present. |
| E9 anchoring | complete | Teacher-student rho_t drift diagnostic is present for warm-start, greedy, importance, and random condensed sets. |
| E10 ablation | complete | Weight fit, root-only/probes, saturation, refinement, greedy, and importance ablations are complete for 25/50. |
| E11 regression | complete | Regression breadth suite is present with 8 datasets, budgets 25/50, seeds 0..2, and XGBoost/LightGBM/MLP; all-learner result is weak, tree-learner slice is more favorable. |
| E12 multiclass | complete | Multiclass one-vs-rest breadth suite is present with 8 datasets, budgets 25/50, seeds 0..2, and three learners; LightGBM slot uses a stable histogram-GBDT fallback. |
| E13 HPO | complete | HPO diagnostic was run; current result is a weak/negative application result rather than a headline claim. |
| E14 runtime | complete | Downstream runtime and speedup tables generated from metrics. |
| E15 sensitivity | complete | Lambda and probe-depth sensitivity summary is present; bin sensitivity remains outside the fixed-bin artifacts. |

## Evaluation Integrity

| status | count |
| --- | --- |
| pass | 28 |
| warn | 4 |

Warnings to handle in paper/re-runs:

| area | check | detail |
| --- | --- | --- |
| protocol | main preprocessing fit after split | Current binary preprocessing encodes/imputes on the full raw frame before splitting; bins are train-only. |
| protocol | breadth preprocessing fit after split | Current breadth preprocessing encodes/imputes before train/val/test splitting. |
| regression | severe negative-R2 outliers | cells below -10 R2: 11 |
| multiclass | lightgbm label caveat | 240 cells are labelled lightgbm; current code uses sklearn HistGradientBoosting fallback for this slot. |

## Paper Claim Guidance

- Strongest claim: histogram-aligned SLD gives a principled, model-aligned fidelity metric and improves SLD versus random/herding while matching legacy gain-path performance closely.
- Safe binary claim: HistDistill significantly beats random and herding, but does not beat the strongest legacy gain-path reference.
- Safe multiclass claim: the one-vs-rest extension beats random and herding on macro-AUROC and is competitive with k-center.
- Regression claim: include as breadth/appendix only; all-learner R2 is weak at 25/50 total points, though tree learners are more favorable.
- Safe theory claim: within each dataset/seed budget curve, lower SLD is associated with higher AUROC, and Prop. 1 cells show 100% root agreement.
- Phase 2 E18 claim: controlled split-noise dosing obeys the Delta/2 threshold and degrades split agreement/regret monotonically; downstream AUROC moves only mildly when leaves are re-estimated on full data.
- Phase 2 E21 claim: regression is a documented limit case with both large structure residuals and substantial leaf-estimate recovery; leaf population TV strongly predicts the leaf-estimate term.
- Unsafe claim: do not claim modern TDColER/KIP superiority, 10x HPO speedup, or a strong regression win from the current artifacts.

## Important Figures

- `figures/icdm_core_budget_profile.png`
- `figures/icdm_core_sld_scatter.png`
- `figures/icdm_ablation_delta_violins.png`
- `figures/icdm_sld_law_scatter.png`
- `figures/icdm_budget_collapse_sld.png`
- `figures/icdm_transfer_regret_profiles.png`
- `figures/icdm_density_pareto.png`
- `figures/icdm_anchor_drift.png`
- `figures/icdm_sensitivity.png`
- `figures/icdm_hpo_speedup_agreement.png`
- `figures/icdm_binary_headline_5seed_performance.png`
- `figures/icdm_regression_budget_profile.png`
- `figures/icdm_multiclass_ovr_budget_profile.png`
- `figures/phase2_e16_error_decomposition.png`
- `figures/phase2_e17_split_regret_law.png`
- `figures/phase2_e18_dose_response.png`
- `figures/phase2_e18_downstream_dose_response.png`
- `figures/phase2_e19_universal_sld_map.png`
- `figures/phase2_e20_training_free_selection.png`
- `figures/phase2_e21_regression_decomposition.png`

## Main Notes

- `notes/PHASE2_RESULTS.md`
- `notes/PHASE2_E18_E21_FULL.md`
- `notes/ICDM_EXPERIMENT_AUDIT.md`
- `notes/EVAL_INTEGRITY_AUDIT.md`
- `notes/ICDM_CORE_UPDATE.md`
- `notes/ICDM_ABLATION_UPDATE.md`
- `notes/ICDM_THEORY_DIAGNOSTICS.md`
- `notes/ICDM_DENSITY_PARETO.md`
- `notes/ICDM_ANCHORING_DIAGNOSTIC.md`
- `notes/ICDM_SENSITIVITY_DIAGNOSTIC.md`
- `notes/ICDM_HPO_DIAGNOSTIC.md`
- `notes/ICDM_BINARY_HEADLINE_5SEED_UPDATE.md`
- `notes/ICDM_REGRESSION_UPDATE.md`
- `notes/ICDM_MULTICLASS_OVR_UPDATE.md`
