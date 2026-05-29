# Limited HistDistill Update

This is the first limited subset for the new histogram-aligned plan: 4 datasets, budget 25/class, seeds 0/1, and learners xgboost/lightgbm/mlp. The goal is debugging and direction checking, not a final paper table.

## Downstream

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.8117 | 0.1113 | 2.3750 | 24 |
| HistDistill-Refined | 0.8067 | 0.1112 | 2.7917 | 24 |
| Herding | 0.8021 | 0.1210 | 2.8333 | 24 |
| HistDistill-Density | 0.7943 | 0.1271 | 2.8333 | 24 |
| Random | 0.7571 | 0.1354 | 4.1667 | 24 |

## Paired Tests

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill-Refined | Legacy gain-path refined | 24 | -0.0050 | 10 | 0 | 14 | 0.1974 |
| HistDistill-Refined | Herding | 24 | 0.0045 | 11 | 0 | 13 | 0.8334 |
| HistDistill-Refined | Random | 24 | 0.0496 | 20 | 0 | 4 | 0.0004 |
| HistDistill-Density | Legacy gain-path refined | 24 | -0.0174 | 9 | 0 | 15 | 0.1434 |
| HistDistill-Density | Herding | 24 | -0.0078 | 12 | 0 | 12 | 0.9441 |
| HistDistill-Density | Random | 24 | 0.0372 | 19 | 0 | 5 | 0.0035 |

## Fidelity / SLD

| Method | Retrained Gain Corr. | Direct SLD Inf. | Direct Root Agree | Direct Rank Corr. | Cells |
| --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.4285 | 1.8198 | 0.2500 | 0.4762 | 8 |
| Herding | 0.3594 | 2.4374 | 0.2500 | 0.3790 | 8 |
| HistDistill-Density | 0.4474 | 1.6017 | 0.7500 | 0.4575 | 8 |
| HistDistill-Refined | 0.4514 | 1.5302 | 0.7500 | 0.4511 | 8 |
| Random | 0.3162 | 2.0107 | 0.2500 | 0.4000 | 8 |

## Readout

- `HistDistill-Refined` is now close to the legacy gain-path method and above herding/random on this limited slice.
- The pure coverage selector remains weak; the deployed variant currently needs gain-path warm start plus bounded histogram weights.
- Direct SLD/root agreement improved for the deployed HistDistill variants, which supports the new theory direction, but the margin-satisfied criterion is not green yet.
- Do not scale to broad26 until the next patch improves pure greedy support or proves the warm-start formulation is the paper method.

CSV outputs:

- `tables/limited_histdistill_performance.csv`
- `tables/limited_histdistill_pairwise.csv`
- `tables/limited_histdistill_fidelity.csv`
- `tables/limited_histdistill_by_learner.csv`
- `tables/limited_histdistill_by_dataset.csv`

Figures:

- `figures/limited_histdistill_transfer_profiles.png`
- `figures/limited_histdistill_delta_violins.png`
- `figures/limited_histdistill_sld_scatter.png`
