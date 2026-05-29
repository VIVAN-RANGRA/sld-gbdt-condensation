# Results Interpretation

## What Holds Up

The clean focused benchmark supports the claim that split-gain/path-aware condensation improves compact tabular training sets.

The primary method, **Gain-path refined**, improves over:

- Random selection by roughly 0.032 AUROC on average.
- Raw-feature herding by roughly 0.020 AUROC on average.
- Gain-only sketching by roughly 0.006 AUROC on average.

The gains over random and herding are statistically strong under paired Wilcoxon tests. The gain over `gain_path_herding` is positive on mean AUROC but not yet statistically decisive.

## What Does Not Hold Up Yet

The phrase "compact synthetic dataset" should be avoided for the main method. The current empirical winner is selected-row condensation with local sketch refinement. The synthetic optimizer is implemented and useful as a math ablation, but it should not be the headline result unless it improves.

## Recommended Claims

Safe claims:

- Tree-mechanism-aligned condensation outperforms generic coresets on binary tabular benchmarks.
- Preserving split-gain/path statistics improves both downstream AUROC and split-gain fidelity.
- The method transfers beyond the teacher family to LightGBM, CatBoost, Random Forest, and MLP.

Claims to avoid:

- Fully synthetic data beats all baselines.
- The method is proven for multiclass tasks.
- Every math loss term is necessary.

## Main Quantitative Evidence

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Full data | 0.8704 | 0.1228 | 2.1396 | 240 |
| Gain-path refined | 0.8350 | 0.1176 | 4.4583 | 240 |
| Gain-path safeguarded | 0.8336 | 0.1192 | 4.5417 | 240 |
| Gain sketch | 0.8290 | 0.1228 | 4.6312 | 240 |
| Gain-path sketch | 0.8289 | 0.1278 | 4.6625 | 240 |
| Gain-path prior weighted | 0.8241 | 0.1310 | 5.1792 | 240 |
| Herding | 0.8146 | 0.1231 | 6.2104 | 240 |
| Random | 0.8027 | 0.1332 | 7.2250 | 240 |
| Distribution matching | 0.7939 | 0.1269 | 7.9792 | 240 |
| K-center | 0.7510 | 0.1539 | 8.4000 | 240 |
| Root-only synthetic | 0.5133 | 0.1419 | 10.5729 | 240 |

## Key Pairwise Tests

| Method | Comparator | Pairs | Mean Diff | Median Diff | Wins | Ties | Losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gain-path refined | Gain sketch | 240 | 0.0060 | 0.0000 | 120 | 1 | 119 | 0.0444 |
| Gain-path safeguarded | Gain sketch | 240 | 0.0047 | -0.0002 | 117 | 1 | 122 | 0.0772 |
| Gain-path sketch | Gain sketch | 240 | -0.0001 | 0.0003 | 121 | 3 | 116 | 0.2573 |
| Gain-path refined | Gain-path sketch | 240 | 0.0061 | 0.0000 | 120 | 2 | 118 | 0.2911 |
| Gain-path safeguarded | Gain-path sketch | 240 | 0.0048 | 0.0000 | 103 | 32 | 105 | 0.2561 |
| Gain sketch | Gain-path sketch | 240 | 0.0001 | -0.0003 | 116 | 3 | 121 | 0.2573 |
| Gain-path refined | Herding | 240 | 0.0204 | 0.0122 | 165 | 1 | 74 | 3.21e-12 |
| Gain-path safeguarded | Herding | 240 | 0.0191 | 0.0122 | 164 | 1 | 75 | 1.37e-11 |
| Gain sketch | Herding | 240 | 0.0144 | 0.0080 | 171 | 0 | 69 | 1.94e-08 |
| Gain-path sketch | Herding | 240 | 0.0143 | 0.0100 | 155 | 0 | 85 | 9.63e-08 |
| Gain-path refined | Random | 240 | 0.0323 | 0.0189 | 199 | 2 | 39 | 1.75e-25 |
| Gain-path safeguarded | Random | 240 | 0.0309 | 0.0173 | 195 | 2 | 43 | 3.58e-24 |
| Gain sketch | Random | 240 | 0.0263 | 0.0185 | 186 | 0 | 54 | 1.12e-18 |
| Gain-path sketch | Random | 240 | 0.0262 | 0.0189 | 192 | 0 | 48 | 9.36e-20 |
