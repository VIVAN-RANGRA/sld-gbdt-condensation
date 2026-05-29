from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-per-class", type=int, default=10)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default="half")
    args = ap.parse_args()
    py = sys.executable
    run([py, "scripts/00_download_datasets.py", "--source", "smoke", "--out", "data/raw/smoke", "--seed", str(args.seed)])
    run([py, "scripts/01_preprocess.py", "--raw-dir", "data/raw/smoke", "--all", "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/02_train_teacher.py", "--dataset", "smoke_binary", "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/03_compute_landscape.py", "--dataset", "smoke_binary", "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/04_distill.py", "--dataset", "smoke_binary", "--budget-per-class", str(args.budget_per_class), "--steps", str(args.steps), "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/05_run_baselines.py", "--dataset", "smoke_binary", "--budget-per-class", str(args.budget_per_class), "--seed", str(args.seed), "--n-jobs", args.n_jobs, "--methods", "random,k_center,distribution_matching,tree_region"])
    run([py, "scripts/06_train_downstream.py", "--dataset", "smoke_binary", "--learners", "xgboost,hist_gradient_boosting", "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/07_eval_split_fidelity.py", "--dataset", "smoke_binary", "--seed", str(args.seed), "--n-jobs", args.n_jobs])
    run([py, "scripts/09_aggregate_results.py", "--results-dir", "results", "--out", "paper_tables"])


if __name__ == "__main__":
    main()
