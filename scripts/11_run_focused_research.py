from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--datasets", default="adult,australian,bank_marketing,breast_w,credit_g,diabetes,pc1,spambase")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", default="80")
    ap.add_argument("--n-jobs", default="8")
    ap.add_argument("--run-ablations", action="store_true")
    ap.add_argument("--force-landscape", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    budgets = [x.strip() for x in args.budgets.split(",") if x.strip()]
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]

    for seed in seeds:
        for ds in datasets:
            run([py, "scripts/02_train_teacher.py", "--config", args.config, "--dataset", ds, "--seed", seed, "--n-jobs", args.n_jobs])
            cmd = [py, "scripts/03_compute_landscape.py", "--config", args.config, "--dataset", ds, "--seed", seed, "--n-jobs", args.n_jobs]
            if args.force_landscape:
                cmd.append("--force")
            run(cmd)
        for budget in budgets:
            for ds in datasets:
                run(
                    [
                        py,
                        "scripts/04_distill.py",
                        "--config",
                        args.config,
                        "--dataset",
                        ds,
                        "--budget-per-class",
                        budget,
                        "--steps",
                        args.steps,
                        "--seed",
                        seed,
                        "--n-jobs",
                        args.n_jobs,
                        "--method-name",
                        "gaindistill_anchor",
                    ]
                )
                run(
                    [
                        py,
                        "scripts/05_run_baselines.py",
                        "--config",
                        args.config,
                        "--dataset",
                        ds,
                        "--budget-per-class",
                        budget,
                        "--seed",
                        seed,
                        "--n-jobs",
                        args.n_jobs,
                        "--methods",
                        "full_data,random,herding,k_center,distribution_matching,gain_path_herding,gain_herding",
                    ]
                )

    if args.run_ablations:
        run([py, "scripts/12_run_math_ablations.py", "--config", args.config, "--datasets", args.datasets, "--budgets", "50", "--seeds", "0", "--n-jobs", args.n_jobs])

    run([py, "scripts/06_train_downstream.py", "--config", args.config, "--all", "--learners", "xgboost,lightgbm,catboost,random_forest,mlp", "--n-jobs", args.n_jobs])
    run([py, "scripts/07_eval_split_fidelity.py", "--config", args.config, "--all", "--n-jobs", args.n_jobs])
    run([py, "scripts/09_aggregate_results.py", "--results-dir", "results", "--out", "paper_tables"])


if __name__ == "__main__":
    main()
