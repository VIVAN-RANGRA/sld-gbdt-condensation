# ICDM Binary Headline 5-Seed Update

Headline binary grid for budgets budget_25, budget_50 and seeds seed_0, seed_1, seed_2, seed_3, seed_4. Expected cells per complete method: 1300.

## Performance

| Method | Mean AUROC | Std | Avg. Rank | Cells |
| --- | --- | --- | --- | --- |
| Legacy gain-path refined | 0.7500 | 0.1778 | 2.5073 | 1300 |
| HistDistill-Refined | 0.7496 | 0.1770 | 2.6154 | 1300 |
| HistDistill-Density | 0.7494 | 0.1767 | 2.6250 | 1300 |
| Herding | 0.7339 | 0.1681 | 3.4446 | 1300 |
| Random | 0.7254 | 0.1762 | 3.8077 | 1300 |

## Pairwise Tests

| Method | Comparator | Pairs | Mean Diff | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistDistill-Density | Legacy gain-path refined | 1300 | -0.0007 | 609 | 7 | 684 | 0.0501 |
| HistDistill-Density | Herding | 1300 | 0.0155 | 850 | 0 | 450 | 0.0000 |
| HistDistill-Density | Random | 1300 | 0.0239 | 971 | 1 | 328 | 0.0000 |
| HistDistill-Refined | Legacy gain-path refined | 1300 | -0.0004 | 603 | 11 | 686 | 0.0717 |
| HistDistill-Refined | Herding | 1300 | 0.0157 | 862 | 1 | 437 | 0.0000 |
| HistDistill-Refined | Random | 1300 | 0.0241 | 983 | 0 | 317 | 0.0000 |

## Files

- `tables/icdm_binary_headline_5seed_performance.csv`
- `tables/icdm_binary_headline_5seed_pairwise.csv`
- `figures/icdm_binary_headline_5seed_performance.png`
