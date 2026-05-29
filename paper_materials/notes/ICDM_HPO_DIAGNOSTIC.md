# ICDM HPO Diagnostic

XGBoost random-search HPO on 12 datasets with 100 shared configurations. Distilled search uses `histdistill_refined` at `budget_50` / `seed_0`.

## Overall

| datasets | mean_top3_jaccard | mean_top1_match | mean_score_spearman | median_hpo_speedup | mean_test_auc_gap |
| --- | --- | --- | --- | --- | --- |
| 12 | 0.0417 | 0.0833 | 0.1451 | 1.5720 | 0.0012 |

## Per Dataset

| dataset | top3_jaccard | top1_match | score_spearman | hpo_speedup | full_best_test_auc | distilled_selected_test_auc | test_auc_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adult | 0.0000 | 0.0000 | 0.0496 | 3.2175 | 0.9270 | 0.9141 | 0.0129 |
| australian | 0.0000 | 0.0000 | 0.0997 | 1.1980 | 0.9461 | 0.9425 | 0.0036 |
| bank_marketing | 0.0000 | 0.0000 | 0.1081 | 3.3778 | 0.9179 | 0.9001 | 0.0178 |
| breast_w | 0.0000 | 0.0000 | 0.3701 | 1.0814 | 0.9880 | 0.9900 | -0.0020 |
| credit_g | 0.0000 | 0.0000 | 0.3195 | 1.2167 | 0.8345 | 0.8451 | -0.0106 |
| diabetes | 0.0000 | 0.0000 | 0.3005 | 1.1600 | 0.8580 | 0.8683 | -0.0104 |
| electricity | 0.0000 | 0.0000 | 0.1155 | 3.2708 | 0.9354 | 0.9247 | 0.0107 |
| kr_vs_kp | 0.0000 | 0.0000 | 0.1361 | 1.5424 | 0.9999 | 0.9973 | 0.0026 |
| pc1 | 0.0000 | 0.0000 | 0.3157 | 1.3393 | 0.9184 | 0.9264 | -0.0081 |
| phoneme | 0.5000 | 1.0000 | 0.2094 | 1.6880 | 0.9443 | 0.9461 | -0.0018 |
| qsar_biodeg | 0.0000 | 0.0000 | -0.4075 | 1.6016 | 0.9077 | 0.9088 | -0.0011 |
| spambase | 0.0000 | 0.0000 | 0.1248 | 3.0903 | 0.9898 | 0.9885 | 0.0013 |

## Files

- `tables/icdm_hpo_summary.csv`
- `tables/icdm_hpo_overall.csv`
- `tables/icdm_hpo_trials_<dataset>.csv`
- `figures/icdm_hpo_speedup_agreement.png`
