# ICDM Sensitivity Diagnostic

Appendix robustness diagnostic on the current binary pipeline. It varies evaluation lambda and max probe depth at budget 50. Bin-count sensitivity is not included because the current processed artifacts fix bins during preprocessing.

| Method | lambda | Probe Depth | SLD | Root Agree | Cells |
| --- | --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.1000 | 2 | 3.9600 | 0.3974 | 78 |
| Legacy gain-path refined | 1.0000 | 0 | 1.3585 | 0.4872 | 78 |
| Legacy gain-path refined | 1.0000 | 1 | 2.9353 | 0.4872 | 78 |
| Legacy gain-path refined | 1.0000 | 2 | 3.9600 | 0.4872 | 78 |
| Legacy gain-path refined | 10.0000 | 2 | 3.9714 | 0.5000 | 78 |
| HistDistill-Density | 0.1000 | 2 | 3.6682 | 0.3462 | 78 |
| HistDistill-Density | 1.0000 | 0 | 1.4886 | 0.3974 | 78 |
| HistDistill-Density | 1.0000 | 1 | 2.7845 | 0.3974 | 78 |
| HistDistill-Density | 1.0000 | 2 | 3.6385 | 0.3974 | 78 |
| HistDistill-Density | 10.0000 | 2 | 3.5655 | 0.4615 | 78 |
| HistDistill-Refined | 0.1000 | 2 | 3.6928 | 0.3333 | 78 |
| HistDistill-Refined | 1.0000 | 0 | 1.3192 | 0.4615 | 78 |
| HistDistill-Refined | 1.0000 | 1 | 2.7971 | 0.4615 | 78 |
| HistDistill-Refined | 1.0000 | 2 | 3.6542 | 0.4615 | 78 |
| HistDistill-Refined | 10.0000 | 2 | 3.5631 | 0.5256 | 78 |
| Random | 0.1000 | 2 | 6.0874 | 0.2692 | 78 |
| Random | 1.0000 | 0 | 3.9070 | 0.3077 | 78 |
| Random | 1.0000 | 1 | 5.3471 | 0.3077 | 78 |
| Random | 1.0000 | 2 | 6.2061 | 0.3077 | 78 |
| Random | 10.0000 | 2 | 6.1406 | 0.3205 | 78 |

## Files

- `tables/icdm_sensitivity_cells.csv`
- `tables/icdm_sensitivity_summary.csv`
- `figures/icdm_sensitivity.png`
