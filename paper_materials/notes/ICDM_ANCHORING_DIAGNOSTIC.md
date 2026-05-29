# ICDM Anchoring Drift Diagnostic

Teacher-student raw-margin drift on the condensed support. This is a measurable proxy for Assumption A: lower rho_t means the student trajectory stays closer to the teacher trajectory.

Final checkpoint shown below: 50.

| Method | Budget | rho max | rho mean | rho p90 | Cells |
| --- | --- | --- | --- | --- | --- |
| HistDistill-Refined | budget_25 | 2.1705 | 0.8880 | 1.5663 | 78 |
| ImportanceSample | budget_25 | 2.3546 | 0.8389 | 1.5776 | 78 |
| Random | budget_25 | 2.7624 | 1.1091 | 1.9752 | 78 |
| Pure greedy selector | budget_25 | 3.2579 | 1.2317 | 2.3917 | 78 |
| HistDistill-Refined | budget_50 | 2.1690 | 0.8020 | 1.3768 | 78 |
| ImportanceSample | budget_50 | 2.1940 | 0.6893 | 1.3246 | 78 |
| Random | budget_50 | 2.6485 | 0.9844 | 1.7458 | 78 |
| Pure greedy selector | budget_50 | 3.5797 | 1.2598 | 2.5194 | 78 |

## Files

- `tables/icdm_anchor_drift_cells.csv`
- `tables/icdm_anchor_drift_summary.csv`
- `figures/icdm_anchor_drift.png`
