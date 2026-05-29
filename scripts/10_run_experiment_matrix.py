from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openml-datasets", type=int, default=10)
    ap.add_argument("--pmlb-datasets", type=int, default=5)
    ap.add_argument("--budgets", default="10,25,50,100")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-jobs", default="8")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--learners", default="xgboost,lightgbm,catboost,random_forest,mlp")
    args = ap.parse_args()
    py = sys.executable
    budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    if not args.skip_download:
        run([py, "scripts/00_download_datasets.py", "--source", "openml_curated", "--max-datasets", str(args.openml_datasets), "--out", "data/raw/openml_curated"])
        run([py, "scripts/00_download_datasets.py", "--source", "pmlb", "--max-datasets", str(args.pmlb_datasets), "--out", "data/raw/pmlb"])

    for raw_dir in ["data/raw/openml_curated", "data/raw/openml_cc18", "data/raw/pmlb"]:
        if Path(raw_dir).exists():
            run([py, "scripts/01_preprocess.py", "--raw-dir", raw_dir, "--all", "--n-jobs", args.n_jobs])

    for seed in seeds:
        run([py, "scripts/02_train_teacher.py", "--all", "--seed", seed, "--n-jobs", args.n_jobs])
        run([py, "scripts/03_compute_landscape.py", "--all", "--seed", seed, "--n-jobs", args.n_jobs])
        for budget in budgets:
            run([py, "scripts/04_distill.py", "--all", "--budget-per-class", budget, "--steps", str(args.steps), "--seed", seed, "--n-jobs", args.n_jobs])
            run([py, "scripts/05_run_baselines.py", "--all", "--budget-per-class", budget, "--seed", seed, "--n-jobs", args.n_jobs])

    run([py, "scripts/06_train_downstream.py", "--all", "--learners", args.learners, "--n-jobs", args.n_jobs])
    run([py, "scripts/07_eval_split_fidelity.py", "--all", "--n-jobs", args.n_jobs])

    if not args.skip_ablations:
        run([py, "scripts/08_run_ablations.py", "--datasets", "selected_5", "--budget-per-class", "50", "--seed", "0", "--n-jobs", args.n_jobs])
        run([py, "scripts/06_train_downstream.py", "--all", "--learners", args.learners, "--n-jobs", args.n_jobs])
        run([py, "scripts/07_eval_split_fidelity.py", "--all", "--n-jobs", args.n_jobs])

    run([py, "scripts/09_aggregate_results.py", "--results-dir", "results", "--out", "paper_tables"])


if __name__ == "__main__":
    main()
