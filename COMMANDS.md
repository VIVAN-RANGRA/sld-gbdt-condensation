# Reproducibility Commands

Run these from the repository root:

```powershell
cd sld-gbdt-condensation
```

## 1. Environment

```powershell
python --version
python -m pip install -r requirements.txt
```

## 2. Smoke Pipeline

```powershell
python scripts\run_smoke_pipeline.py --budget-per-class 5 --steps 30 --n-jobs half
```

## 3. Focused Binary Replication

```powershell
python scripts\00_download_datasets.py --source openml_curated --max-datasets 10 --out data\raw\openml_curated
python scripts\01_preprocess.py --raw-dir data\raw\openml_curated --all --n-jobs 8
python scripts\11_run_focused_research.py --config configs\focused.yaml --n-jobs 8 --run-ablations
python scripts\13_focused_report.py --tables-dir paper_tables --out paper_tables --methods full_data,random,herding,k_center,distribution_matching,gain_herding,gain_path_herding,gain_path_refined,gain_path_safeguarded,gain_path_prior_weighted,gaindistill
```

## 4. Core Paper Tables

```powershell
python scripts\19_run_icdm_core.py --config configs\focused.yaml --budgets 10,25,50,100,200 --seeds 0,1,2 --n-jobs 8
python scripts\20_summarize_icdm_core.py --paper-tables paper_tables --paper-materials paper_materials --budgets 10,25,50,100,200 --seeds 0,1,2
python scripts\21_summarize_icdm_ablations.py --paper-tables paper_tables --paper-materials paper_materials --budgets 25,50 --seeds 0,1,2
python scripts\22_summarize_theory_diagnostics.py --config configs\focused.yaml --paper-tables paper_tables --paper-materials paper_materials --budgets 10,25,50,100,200 --seeds 0,1,2
python scripts\25_summarize_density_pareto.py --config configs\focused.yaml --paper-tables paper_tables --paper-materials paper_materials --budgets 25,50 --seeds 0,1,2
python scripts\30_summarize_binary_headline_5seed.py --paper-tables paper_tables --paper-materials paper_materials --budgets 25,50 --seeds 0,1,2,3,4
python scripts\26_build_final_results_digest.py --paper-materials paper_materials
```

## 5. Phase-2 and Breadth Diagnostics

```powershell
python scripts\29_run_breadth_experiments.py --config configs\focused.yaml --paper-materials paper_materials --task both --budgets 25,50 --seeds 0,1,2 --n-jobs 8
python scripts\32_run_phase2_experiments.py --config configs\focused.yaml --paper-tables paper_tables --paper-materials paper_materials --budgets 25,50 --seeds 0,1,2 --n-jobs 8
python scripts\33_run_phase2_full_e18_e21.py --config configs\focused.yaml --paper-materials paper_materials --seeds 0,1,2 --n-jobs 8
python scripts\31_audit_eval_integrity.py
```

## 6. Large-Scale Audit

Quick debug run:

```powershell
python scripts\34_run_large_scale_experiment.py --quick --n-jobs 8
```

Full/near-full run, if disk and network budget allow:

```powershell
python scripts\34_run_large_scale_experiment.py --datasets covertype,higgs,susy --methods histdistill_refined,histdistill_density,herding,gradient_sampling,random --budget-rows 400 --seed 0 --n-jobs 8
```

## 7. Expected Output Locations

```text
data\processed\<dataset>\arrays.npz
results\teachers\<dataset>\seed_<seed>\
results\landscapes\<dataset>\seed_<seed>\
results\distilled\<dataset>\<method>\budget_<budget>\seed_<seed>\
results\downstream\<dataset>\<method>\budget_<budget>\seed_<seed>\<learner>\
results\split_fidelity\<dataset>\<method>\budget_<budget>\seed_<seed>\
paper_tables\
paper_materials\tables\
paper_materials\figures\
paper_materials\FINAL_RESULTS.md
paper\figures\
paper\main.pdf
```
