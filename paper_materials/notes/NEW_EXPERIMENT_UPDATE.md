# New Experiment Update

## Broad 26-Dataset Core Benchmark

This block expands the core comparison to 26 real binary datasets, budgets 25/50, seeds 0/1/2, and five downstream learners. Each complete method has 780 paired cells.

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Gain-path refined | 0.7509 | 0.1777 | 3.0404 | 780 |
| Gain-path safeguarded | 0.7504 | 0.1779 | 3.0090 | 780 |
| Gain sketch | 0.7490 | 0.1763 | 3.1885 | 780 |
| Gain-path sketch | 0.7475 | 0.1804 | 3.0506 | 780 |
| Herding | 0.7339 | 0.1681 | 4.1788 | 780 |
| Random | 0.7260 | 0.1733 | 4.5327 | 780 |

Key broad paired tests:

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gain sketch | Random | 780 | 0.0230 | 584 | 0 | 196 | 0.0000 |
| Gain sketch | Herding | 780 | 0.0151 | 514 | 0 | 266 | 0.0000 |
| Gain-path sketch | Random | 780 | 0.0215 | 572 | 0 | 208 | 0.0000 |
| Gain-path sketch | Herding | 780 | 0.0137 | 524 | 0 | 256 | 0.0000 |
| Gain-path refined | Random | 780 | 0.0248 | 585 | 2 | 193 | 0.0000 |
| Gain-path refined | Herding | 780 | 0.0170 | 542 | 1 | 237 | 0.0000 |
| Gain-path safeguarded | Random | 780 | 0.0244 | 586 | 2 | 192 | 0.0000 |
| Gain-path safeguarded | Herding | 780 | 0.0165 | 545 | 1 | 234 | 0.0000 |

CSV outputs:

- `tables/broad26_core_performance.csv`
- `tables/broad26_core_pairwise.csv`
- `tables/broad26_core_budget_means.csv`
- `tables/broad26_core_dataset_means.csv`

## Focused Refined-Method Ablations

This block tests which components of the refined gain/path method matter on the focused 8-dataset suite.

| Variant | Mean AUROC | Avg. Rank | Cells |
| --- | --- | --- | --- |
| No gain weighting | 0.8350 | 4.7771 | 240 |
| Gain-path refined | 0.8350 | 4.7917 | 240 |
| No Hessian mass | 0.8332 | 4.9208 | 240 |
| No gradient mass | 0.8318 | 4.7979 | 240 |
| No teacher confidence | 0.8314 | 4.9938 | 240 |
| Path occupancy only | 0.8302 | 5.2542 | 240 |
| No path features | 0.8290 | 5.2667 | 240 |
| No local refinement | 0.8289 | 4.9562 | 240 |
| No raw features | 0.8284 | 5.2417 | 240 |

Ablation deltas are measured as variant minus full `Gain-path refined`; negative values mean the ablation hurts.

| Variant | Diff vs Full | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| No path features | -0.0059 | 106 | 1 | 133 | 0.0033 |
| No local refinement | -0.0061 | 118 | 2 | 120 | 0.2911 |
| No gradient mass | -0.0031 | 117 | 3 | 120 | 0.2735 |
| No Hessian mass | -0.0018 | 115 | 2 | 123 | 0.4018 |
| No teacher confidence | -0.0035 | 60 | 114 | 66 | 0.1693 |
| No raw features | -0.0065 | 102 | 1 | 137 | 0.0023 |
| No gain weighting | 0.0000 | 2 | 238 | 0 | 0.1797 |
| Path occupancy only | -0.0047 | 110 | 0 | 130 | 0.0143 |

CSV outputs:

- `tables/focused_refined_ablation_performance.csv`
- `tables/focused_refined_ablation_pairwise.csv`
- `tables/focused_refined_ablation_fidelity.csv`
