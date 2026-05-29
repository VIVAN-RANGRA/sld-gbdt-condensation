# GainDistill Experiment Pipeline

This repository implements the binary GainDistill research pipeline: teacher split-gain landscapes, multi-probe synthetic optimization, gain/path coreset baselines, downstream transfer evaluation, split-fidelity metrics, and focused statistical reports.

## Setup

```bash
python -m pip install -r requirements.txt
```

The scripts use `project.n_jobs: half` by default, which resolves to half of the machine's logical cores. You can override it with `--n-jobs 8`, `--n-jobs half`, or an integer.

## Dataset Downloads

Smoke dataset:

```bash
python scripts/00_download_datasets.py --source smoke --out data/raw/smoke
```

OpenML curated fallback list:

```bash
python scripts/00_download_datasets.py --source openml_curated --max-datasets 5 --out data/raw/openml_curated
```

OpenML benchmark suite:

```bash
python scripts/00_download_datasets.py --source openml_cc18 --suite-id 99 --max-datasets 30 --out data/raw/openml_cc18
```

PMLB:

```bash
python scripts/00_download_datasets.py --source pmlb --max-datasets 20 --out data/raw/pmlb
```

If remote downloads fail, the downloader creates a local smoke dataset so the pipeline remains testable.

## Quick Verification

```bash
python scripts/run_smoke_pipeline.py --budget-per-class 5 --steps 30 --n-jobs 8
```

## One Real Dataset

```bash
python scripts/01_preprocess.py --raw-dir data/raw/openml_curated --all --n-jobs 8
python scripts/02_train_teacher.py --dataset credit_g --seed 0 --n-jobs 8
python scripts/03_compute_landscape.py --dataset credit_g --seed 0 --n-jobs 8 --force
python scripts/04_distill.py --dataset credit_g --budget-per-class 10 --steps 80 --seed 0 --n-jobs 8 --method-name gaindistill_anchor --force
python scripts/05_run_baselines.py --dataset credit_g --budget-per-class 10 --seed 0 --n-jobs 8
python scripts/06_train_downstream.py --dataset credit_g --learners xgboost,lightgbm,catboost,random_forest,mlp --seed 0 --n-jobs 8
python scripts/07_eval_split_fidelity.py --dataset credit_g --seed 0 --n-jobs 8
python scripts/09_aggregate_results.py --results-dir results --out paper_tables
python scripts/13_focused_report.py --tables-dir paper_tables --out paper_tables
```

## Larger Sweep

Run downloads and preprocessing once, then repeat teacher, landscape, distillation, baselines, downstream evaluation, split fidelity, and aggregation for multiple seeds and budgets:

```bash
python scripts/00_download_datasets.py --source openml_cc18 --suite-id 99 --max-datasets 30 --out data/raw/openml_cc18
python scripts/01_preprocess.py --raw-dir data/raw/openml_cc18 --all --n-jobs 8

python scripts/02_train_teacher.py --all --seed 0 --n-jobs 8
python scripts/03_compute_landscape.py --all --seed 0 --n-jobs 8
python scripts/04_distill.py --all --budget-per-class 50 --seed 0 --n-jobs 8
python scripts/05_run_baselines.py --all --budget-per-class 50 --seed 0 --n-jobs 8
python scripts/06_train_downstream.py --all --seed 0 --n-jobs 8
python scripts/07_eval_split_fidelity.py --all --seed 0 --n-jobs 8
python scripts/09_aggregate_results.py --results-dir results --out paper_tables
```

The baseline, downstream, and fidelity runners support targeted clean reruns:

```bash
python scripts/05_run_baselines.py --dataset credit_g --budget-per-class 50 --seed 0 --methods gain_path_refined --n-jobs 8 --force
python scripts/06_train_downstream.py --dataset credit_g --methods gain_path_refined --budgets 50 --distilled-seeds 0 --learners xgboost,lightgbm,catboost,random_forest,mlp --n-jobs 8 --force
python scripts/07_eval_split_fidelity.py --dataset credit_g --methods gain_path_refined --budgets 50 --distilled-seeds 0 --n-jobs 8 --force
```

## Focused Paper Loop

For fast research iteration, use the 8-dataset binary suite in `configs/focused.yaml`:

```bash
python scripts/11_run_focused_research.py --config configs/focused.yaml --n-jobs 8 --run-ablations
```

This runs:

- selected binary datasets only,
- budgets 25 and 50,
- seeds 0, 1, and 2,
- `gaindistill_anchor`,
- key baselines,
- optional math ablations,
- downstream evaluation,
- split fidelity,
- Wilcoxon and win/tie/loss aggregation.

For a clean focused report that filters to complete methods:

```bash
python scripts/13_focused_report.py --tables-dir paper_tables --out paper_tables --methods full_data,random,herding,k_center,distribution_matching,gain_herding,gain_path_herding,gain_path_refined,gain_path_safeguarded,gain_path_prior_weighted,gaindistill
```

## Current Scope

Implemented:

- Dataset download and caching for smoke, OpenML, and PMLB.
- Binary classification preprocessing with train/validation/test splits.
- XGBoost teacher when installed, with sklearn fallback.
- Root and probe-region split-gain landscapes with candidate subsampling.
- Differentiable soft-bin GainDistill optimizer with soft teacher routing.
- Random, k-center, herding, gradient sampling, distribution matching, and tree-region-style baselines.
- Gain-sketch coreset/hybrid variants: `gain_herding`, `gain_path_herding`, `gain_path_refined`, `gain_path_safeguarded`, `gain_path_prior_weighted`, `gain_weighted_coreset`, and `gaindistill_hybrid`.
- Multi-probe synthetic variants: `gaindistill_full` and `gaindistill_anchor`.
- Downstream evaluation and split-fidelity aggregation.
- Loss-component ablation generation.

Still research-critical before final large claims:

- Expand the clean focused gain-sketch result to additional binary datasets if time allows.
- Complete broader math ablations for the synthetic optimizer before making strong claims about every loss term.
- Keep multiclass one-vs-rest as future work unless it is implemented and evaluated end to end.
- Treat generated smoke or single-dataset checks as engineering validation only.
