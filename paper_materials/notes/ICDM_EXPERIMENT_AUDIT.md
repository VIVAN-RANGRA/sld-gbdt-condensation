# ICDM Experiment Audit

This audit checks whether the completed runs match `FINAL_EXPERIMENTS.md` and flags anything that would be unsafe to claim.

## Dataset Scope

Processed datasets: 27 total, 26 real binary datasets plus smoke. All real processed datasets have 2-2 classes.

## Key Grid Coverage

| method | n_cells | expected_cells | datasets | budgets | seeds | learners | complete | auroc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gain_path_refined | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7430 |
| herding | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7306 |
| histdistill_density | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7425 |
| histdistill_refined | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7425 |
| random | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7244 |

## Key Fidelity Coverage

| method | n_cells | expected_cells | direct_sld_cells | direct_sld_complete |
| --- | --- | --- | --- | --- |
| gain_path_refined | 650 | 650 | 650 | True |
| herding | 650 | 650 | 650 | True |
| histdistill_density | 650 | 650 | 650 | True |
| histdistill_refined | 650 | 650 | 650 | True |
| random | 650 | 650 | 650 | True |

## Ablation Grid Coverage

| method | n_cells | expected_cells | datasets | budgets | seeds | learners | complete | auroc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| histdistill_density | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.7508 |
| histdistill_greedy | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.6285 |
| histdistill_importance | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.6904 |
| histdistill_linear_coverage | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.5957 |
| histdistill_no_refine | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.6279 |
| histdistill_no_weight_fit | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.6163 |
| histdistill_refined | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.7498 |
| histdistill_root_only | 780 | 780 | 26 | 2 | 3 | 5 | True | 0.5964 |

## FINAL_EXPERIMENTS Status

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

## Main Audit Conclusions

- The completed binary results are on the full 26 real datasets, not the smoke dataset and not a four-dataset subset.
- The stale direct-SLD issue has been fixed for the key 5-budget grid.
- Regression and multiclass breadth suites are present as separate experiments; multiclass is supportive, while regression is weak/mixed and should be appendix-framed.
- The safest current paper framing is a strong binary and multiclass classification study with theory diagnostics, transfer, runtime, and ablations; avoid claiming a strong regression or HPO win.
