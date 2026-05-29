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
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--learners", default="xgboost,lightgbm,catboost,random_forest,mlp")
    ap.add_argument(
        "--methods",
        default="histdistill_refined,histdistill_density,histdistill_importance,histdistill_greedy",
    )
    ap.add_argument("--n-jobs", default="8")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-teachers", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    budgets = [x.strip() for x in args.budgets.split(",") if x.strip()]
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]

    for seed in seeds:
        if not args.skip_teachers:
            run([py, "scripts/02_train_teacher.py", "--config", args.config, "--all", "--seed", seed, "--n-jobs", args.n_jobs])
            run([py, "scripts/03_compute_landscape.py", "--config", args.config, "--all", "--seed", seed, "--n-jobs", args.n_jobs])
        for budget in budgets:
            cmd = [
                py,
                "scripts/05_run_baselines.py",
                "--config",
                args.config,
                "--all",
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

    common = [
        "--config",
        args.config,
        "--all",
        "--methods",
        args.methods,
        "--distilled-seeds",
        args.seeds,
        "--budgets",
        args.budgets,
        "--n-jobs",
        args.n_jobs,
    ]
    downstream = [py, "scripts/06_train_downstream.py", *common, "--learners", args.learners]
    fidelity = [py, "scripts/07_eval_split_fidelity.py", *common]
    if args.force:
        downstream.append("--force")
        fidelity.append("--force")
    run(downstream)
    run(fidelity)
    run([py, "scripts/09_aggregate_results.py", "--results-dir", "results", "--out", "paper_tables"])


if __name__ == "__main__":
    main()
