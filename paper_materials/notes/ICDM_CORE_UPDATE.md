# ICDM Core Experiment Update

Core binary benchmark summary for budgets budget_10, budget_100, budget_200, budget_25, budget_50 and seeds seed_0, seed_1, seed_2, seed_3, seed_4. Methods are included only when complete on the paired grid. Expected complete cells per method: 3250.

## Performance

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

## Fidelity / SLD

| Method | Retrained Gain Corr. | Direct SLD Inf. | Direct Root Agree | Direct Rank Corr. | Cells |
| --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.3937 | 18.3953 | 0.3062 | 0.4593 | 650 |
| Herding | 0.3385 | 34.5993 | 0.1923 | 0.3607 | 650 |
| HistDistill-Density | 0.3909 | 16.9979 | 0.3154 | 0.4540 | 650 |
| HistDistill-Refined | 0.3893 | 16.7257 | 0.3200 | 0.4600 | 650 |
| Random | 0.3359 | 27.7381 | 0.1923 | 0.3935 | 650 |

## Coverage

| method | n_cells | expected_cells | datasets | budgets | seeds | learners | complete | auroc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gain_herding | 1560 | 3250 | 26 | 4 | 3 | 5 | False | 0.7275 |
| gain_path_herding | 1560 | 3250 | 26 | 4 | 3 | 5 | False | 0.7274 |
| gain_path_refined | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7430 |
| gain_path_safeguarded | 780 | 3250 | 26 | 2 | 3 | 5 | False | 0.7504 |
| herding | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7306 |
| histdistill_density | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7425 |
| histdistill_greedy | 780 | 3250 | 26 | 2 | 3 | 5 | False | 0.6285 |
| histdistill_importance | 780 | 3250 | 26 | 2 | 3 | 5 | False | 0.6904 |
| histdistill_refined | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7425 |
| random | 3250 | 3250 | 26 | 5 | 5 | 5 | True | 0.7244 |

CSV outputs:

- `tables/icdm_core_coverage.csv`
- `tables/icdm_core_performance.csv`
- `tables/icdm_core_by_budget.csv`
- `tables/icdm_core_by_learner.csv`
- `tables/icdm_core_pairwise.csv`
- `tables/icdm_core_fidelity.csv`

Figures:

- `figures/icdm_core_rank_profile.png`
- `figures/icdm_core_budget_profile.png`
- `figures/icdm_core_sld_scatter.png`
