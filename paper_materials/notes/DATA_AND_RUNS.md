# Data And Runs

## Processed Data Inventory

The workspace currently contains 27 processed binary datasets, including the focused suite and extra appendix-scale datasets. Smoke data is present for engineering checks but should not be used in paper claims.

Focused paper suite:

| Dataset | Source | Train | Val | Test | Total | Features | Train Positive Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | pmlb | 12000 | 4000 | 4000 | 20000 | 14 | 0.765 |
| australian | openml | 414 | 138 | 138 | 690 | 14 | 0.447 |
| bank_marketing | openml | 12000 | 4000 | 4000 | 20000 | 16 | 0.117 |
| breast_w | openml_suite | 419 | 140 | 140 | 699 | 9 | 0.346 |
| credit_g | openml_suite | 600 | 200 | 200 | 1000 | 20 | 0.700 |
| diabetes | openml_suite | 460 | 154 | 154 | 768 | 8 | 0.348 |
| pc1 | openml | 665 | 222 | 222 | 1109 | 21 | 0.069 |
| spambase | openml_suite | 2760 | 920 | 921 | 4601 | 57 | 0.394 |

Full processed-data inventory is in `tables/dataset_inventory.csv`.

## Focused Evaluation Design

- Budgets: 25 and 50 samples per class.
- Seeds: 0, 1, 2.
- Learners: XGBoost, LightGBM, CatBoost, Random Forest, MLP.
- Metric: AUROC, with accuracy and log loss retained in CSVs.
- Main statistical unit: dataset x budget x seed x learner.

## Method Coverage

| Method | n_cells | datasets | budgets | seeds | learners | complete | auroc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_data | 240 | 8 | 2 | 3 | 5 | True | 0.8704 |
| gain_path_refined | 240 | 8 | 2 | 3 | 5 | True | 0.8350 |
| gain_path_safeguarded | 240 | 8 | 2 | 3 | 5 | True | 0.8336 |
| gain_herding | 240 | 8 | 2 | 3 | 5 | True | 0.8290 |
| gain_path_herding | 240 | 8 | 2 | 3 | 5 | True | 0.8289 |
| gain_path_prior_weighted | 240 | 8 | 2 | 3 | 5 | True | 0.8241 |
| herding | 240 | 8 | 2 | 3 | 5 | True | 0.8146 |
| random | 240 | 8 | 2 | 3 | 5 | True | 0.8027 |
| distribution_matching | 240 | 8 | 2 | 3 | 5 | True | 0.7939 |
| k_center | 240 | 8 | 2 | 3 | 5 | True | 0.7510 |
| gaindistill | 240 | 8 | 2 | 3 | 5 | True | 0.5133 |

## Broad Result Coverage

The appendix includes historical and exploratory runs. These are useful for robustness checks and ablations, but the primary paper tables should use complete methods only.

| Method | n_rows | datasets | budgets | seeds | learners | auroc_mean |
| --- | --- | --- | --- | --- | --- | --- |
| full_data | 1560 | 26 | 4 | 3 | 5 | 0.8119 |
| gain_herding | 1560 | 26 | 4 | 3 | 5 | 0.7275 |
| gain_path_herding | 1560 | 26 | 4 | 3 | 5 | 0.7274 |
| gaindistill_path_hybrid | 1560 | 26 | 4 | 3 | 5 | 0.7215 |
| gain_weighted_coreset | 1560 | 26 | 4 | 3 | 5 | 0.7185 |
| gaindistill_hybrid | 1560 | 26 | 4 | 3 | 5 | 0.7185 |
| herding | 1560 | 26 | 4 | 3 | 5 | 0.7167 |
| random | 1560 | 26 | 4 | 3 | 5 | 0.7075 |
| k_center | 1560 | 26 | 4 | 3 | 5 | 0.6833 |
| distribution_matching | 1560 | 26 | 4 | 3 | 5 | 0.6826 |
| gaindistill | 1560 | 26 | 4 | 3 | 5 | 0.5034 |
| gradient_sampling | 1560 | 26 | 4 | 3 | 5 | 0.4043 |
| tree_region | 1560 | 26 | 4 | 3 | 5 | 0.4043 |
| gain_path_refined | 780 | 26 | 2 | 3 | 5 | 0.7509 |
| gain_path_safeguarded | 780 | 26 | 2 | 3 | 5 | 0.7504 |
| refined_no_gain_weight | 240 | 8 | 2 | 3 | 5 | 0.8350 |
| refined_no_hess | 240 | 8 | 2 | 3 | 5 | 0.8332 |
| refined_no_grad | 240 | 8 | 2 | 3 | 5 | 0.8318 |
| refined_no_conf | 240 | 8 | 2 | 3 | 5 | 0.8314 |
| refined_path_occupancy_only | 240 | 8 | 2 | 3 | 5 | 0.8302 |
