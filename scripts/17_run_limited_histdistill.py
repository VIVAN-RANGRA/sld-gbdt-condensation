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
    ap.add_argument("--datasets", default="adult,australian,diabetes,spambase")
    ap.add_argument("--budgets", default="25")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--learners", default="xgboost,lightgbm,mlp")
    ap.add_argument(
        "--methods",
        default=(
            "histdistill_greedy,histdistill_refined,histdistill_density,"
            "histdistill_importance,gain_path_refined,herding,random"
        ),
    )
    ap.add_argument("--n-jobs", default="half")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    budgets = [x.strip() for x in args.budgets.split(",") if x.strip()]
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]

    for seed in seeds:
        for ds in datasets:
            run([py, "scripts/02_train_teacher.py", "--config", args.config, "--dataset", ds, "--seed", seed, "--n-jobs", args.n_jobs])
            run([py, "scripts/03_compute_landscape.py", "--config", args.config, "--dataset", ds, "--seed", seed, "--n-jobs", args.n_jobs])
        for budget in budgets:
            for ds in datasets:
                cmd = [
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
                    args.methods,
                ]
                if args.force:
                    cmd.append("--force")
                run(cmd)

    common = ["--config", args.config, "--all", "--methods", args.methods, "--distilled-seeds", args.seeds, "--budgets", args.budgets, "--n-jobs", args.n_jobs]
    downstream_cmd = [py, "scripts/06_train_downstream.py", *common, "--learners", args.learners]
    fidelity_cmd = [py, "scripts/07_eval_split_fidelity.py", *common]
    if args.force:
        downstream_cmd.append("--force")
        fidelity_cmd.append("--force")
    run(downstream_cmd)
    run(fidelity_cmd)
    run([py, "scripts/09_aggregate_results.py", "--results-dir", "results", "--out", "paper_tables"])


if __name__ == "__main__":
    main()
