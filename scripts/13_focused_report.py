from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import wilcoxon

from gaindistill.utils import ensure_dir


FOCUSED_DATASETS = [
    "adult",
    "australian",
    "bank_marketing",
    "breast_w",
    "credit_g",
    "diabetes",
    "pc1",
    "spambase",
]
PAPER_LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]


def _parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _seed_names(value: str) -> list[str]:
    out = []
    for seed in _parse_csv_arg(value):
        out.append(seed if seed.startswith("seed_") else f"seed_{seed}")
    return out


def _budget_names(value: str) -> list[str]:
    out = []
    for budget in _parse_csv_arg(value):
        out.append(budget if budget.startswith("budget_") else f"budget_{budget}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="paper_tables")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--datasets", default=",".join(FOCUSED_DATASETS))
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--learners", default=",".join(PAPER_LEARNERS))
    ap.add_argument("--methods", default=None)
    ap.add_argument("--comparators", default="random,herding,gain_herding,gain_path_herding")
    args = ap.parse_args()

    tables = Path(args.tables_dir)
    out = ensure_dir(args.out)
    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    fidelity_path = tables / "all_split_fidelity_metrics.csv"
    fidelity = pd.read_csv(fidelity_path) if fidelity_path.exists() else pd.DataFrame()

    datasets = set(_parse_csv_arg(args.datasets))
    budgets = set(_budget_names(args.budgets))
    seeds = set(_seed_names(args.seeds))
    learners = set(_parse_csv_arg(args.learners))
    comparators = _parse_csv_arg(args.comparators)
    methods = set(_parse_csv_arg(args.methods)) if args.methods else None
    expected = len(datasets) * len(budgets) * len(seeds) * len(learners)

    df = downstream[
        downstream["dataset"].isin(datasets)
        & downstream["budget"].isin(budgets)
        & downstream["seed"].isin(seeds)
        & downstream["learner"].isin(learners)
    ].copy()
    if methods is not None:
        df = df[df["method"].isin(methods)].copy()
    coverage = (
        df.groupby("method")
        .agg(
            n_cells=("auroc", "count"),
            datasets=("dataset", "nunique"),
            budgets=("budget", "nunique"),
            seeds=("seed", "nunique"),
            learners=("learner", "nunique"),
            auroc_mean=("auroc", "mean"),
        )
        .reset_index()
    )
    coverage["expected_cells"] = expected
    coverage["complete"] = coverage["n_cells"] == expected
    coverage = coverage.sort_values(["complete", "auroc_mean"], ascending=[False, False])
    coverage.to_csv(out / "focused_coverage.csv", index=False)

    complete_methods = coverage.loc[coverage["complete"], "method"].tolist()
    complete = df[df["method"].isin(complete_methods)].copy()
    if complete.empty:
        print("No complete focused methods found.")
        return

    pivot = complete.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean().rename("avg_rank")
    summary = (
        complete.groupby("method")
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            log_loss_mean=("log_loss", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            n_cells=("auroc", "count"),
        )
        .join(ranks)
        .reset_index()
        .sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])
    )
    summary.to_csv(out / "focused_complete_performance.csv", index=False)

    complete.pivot_table(index="method", columns="budget", values="auroc", aggfunc="mean").reset_index().to_csv(
        out / "focused_budget_means.csv", index=False
    )
    complete.pivot_table(index="method", columns="learner", values="auroc", aggfunc="mean").reset_index().to_csv(
        out / "focused_learner_means.csv", index=False
    )
    complete.pivot_table(index="dataset", columns="method", values="auroc", aggfunc="mean").reset_index().to_csv(
        out / "focused_dataset_means.csv", index=False
    )

    stats = []
    for method in complete_methods:
        for comp in comparators:
            if method == comp or method not in pivot.columns or comp not in pivot.columns:
                continue
            pair = pivot[[method, comp]].dropna()
            if pair.empty:
                continue
            diff = pair[method] - pair[comp]
            try:
                p_value = float(wilcoxon(diff).pvalue) if (diff.abs() > 1e-12).any() else 1.0
            except ValueError:
                p_value = 1.0
            stats.append(
                {
                    "method": method,
                    "comparator": comp,
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wins": int((diff > 1e-6).sum()),
                    "ties": int((diff.abs() <= 1e-6).sum()),
                    "losses": int((diff < -1e-6).sum()),
                    "wilcoxon_p": p_value,
                }
            )
    pd.DataFrame(stats).sort_values(["comparator", "mean_diff"], ascending=[True, False]).to_csv(
        out / "focused_pairwise_stats.csv", index=False
    )

    if not fidelity.empty:
        fdf = fidelity[
            fidelity["dataset"].isin(datasets)
            & fidelity["budget"].isin(budgets)
            & fidelity["seed"].isin(seeds)
            & fidelity["method"].isin(complete_methods)
        ].copy()
        if not fdf.empty:
            (
                fdf.groupby("method")
                .agg(
                    root_agreement=("root_agreement", "mean"),
                    top5_overlap=("top5_overlap", "mean"),
                    gain_rank_correlation=("gain_rank_correlation", "mean"),
                    n_cells=("gain_rank_correlation", "count"),
                )
                .reset_index()
                .sort_values("gain_rank_correlation", ascending=False)
                .to_csv(out / "focused_fidelity.csv", index=False)
            )

    print(f"Wrote focused report outputs to {out}.")


if __name__ == "__main__":
    main()
