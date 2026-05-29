# Evaluation Integrity Audit

This audit checks source-level assumptions and generated metric invariants for the current experiment package.

## Status Counts

| status | count |
| --- | --- |
| pass | 28 |
| warn | 4 |

## Checks

| area | check | status | detail |
| --- | --- | --- | --- |
| source | expected files present | pass | all present |
| source | scripts\06_train_downstream.py parses | pass |  |
| source | scripts\07_eval_split_fidelity.py parses | pass |  |
| source | scripts\24_run_hpo_experiment.py parses | pass |  |
| source | scripts\29_run_breadth_experiments.py parses | pass |  |
| source | gaindistill\data.py parses | pass |  |
| source | gaindistill\models.py parses | pass |  |
| source | gaindistill\landscape.py parses | pass |  |
| source | gaindistill\baselines.py parses | pass |  |
| source | binary downstream evaluates held-out test | pass | scripts/06_train_downstream.py |
| source | breadth evaluates held-out test | pass | scripts/29_run_breadth_experiments.py |
| protocol | main preprocessing fit after split | warn | Current binary preprocessing encodes/imputes on the full raw frame before splitting; bins are train-only. |
| protocol | breadth preprocessing fit after split | warn | Current breadth preprocessing encodes/imputes before train/val/test splitting. |
| binary downstream | unique metric cells | pass | duplicate keys: 0 |
| binary downstream | AUROC range | pass | bad values: 0 |
| binary downstream | accuracy range | pass | bad values: 0 |
| binary downstream | log-loss finite | pass | bad values: 0 |
| binary downstream | key 5-budget grid coverage | pass | expected/method: 3250; observed: {'gain_path_refined': 3250, 'herding': 3250, 'histdistill_density': 3250, 'histdistill_refined': 3250, 'random': 3250} |
| split fidelity | unique fidelity cells | pass | duplicate keys: 0 |
| split fidelity | root_agreement range | pass | bad values: 0; missing: 0 |
| split fidelity | top5_overlap range | pass | bad values: 0; missing: 0 |
| split fidelity | sld_root_agreement_direct range | pass | bad values: 0; missing: 4267 |
| split fidelity | sld_margin_satisfied_rate range | pass | bad values: 0; missing: 4267 |
| split fidelity | gain_rank_correlation range | pass | bad values: 0; missing: 0 |
| split fidelity | sld_rank_correlation_direct range | pass | bad values: 0; missing: 4267 |
| regression | unique cells | pass | duplicate keys: 0 |
| regression | r2 finite | pass | bad values: 0 |
| regression | severe negative-R2 outliers | warn | cells below -10 R2: 11 |
| multiclass | unique cells | pass | duplicate keys: 0 |
| multiclass | macro_auroc finite | pass | bad values: 0 |
| multiclass | macro_auroc range | pass | bad values: 0 |
| multiclass | lightgbm label caveat | warn | 240 cells are labelled lightgbm; current code uses sklearn HistGradientBoosting fallback for this slot. |

## Interpretation

- `fail` means a generated artifact or code invariant is broken.
- `warn` means results may still be usable, but the paper must disclose the caveat or the experiment should be rerun after a protocol fix.
- The current high-risk warnings are preprocessing-before-split in binary/breadth data preparation, severe regression outliers, and the multiclass LightGBM fallback label.
