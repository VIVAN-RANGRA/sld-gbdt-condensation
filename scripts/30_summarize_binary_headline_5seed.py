from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gaindistill.utils import ensure_dir


METHODS = ["histdistill_density", "histdistill_refined", "gain_path_refined", "herding", "random"]
LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
LABELS = {
    "histdistill_density": "HistDistill-Density",
    "histdistill_refined": "HistDistill-Refined",
    "gain_path_refined": "Legacy gain-path refined",
    "herding": "Herding",
    "random": "Random",
}


def label(method: str) -> str:
    return LABELS.get(method, method.replace("_", " "))


def md_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{float_digits}f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def pairwise(pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["histdistill_density", "histdistill_refined"]:
        for comp in ["gain_path_refined", "herding", "random"]:
            pair = pivot[[method, comp]].dropna()
            diff = pair[method] - pair[comp]
            try:
                p_value = float(wilcoxon(diff).pvalue) if (diff.abs() > 1e-12).any() else 1.0
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "method": method,
                    "method_label": label(method),
                    "comparator": comp,
                    "comparator_label": label(comp),
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wins": int((diff > 1e-6).sum()),
                    "ties": int((diff.abs() <= 1e-6).sum()),
                    "losses": int((diff < -1e-6).sum()),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    budgets = {b if b.startswith("budget_") else f"budget_{b}" for b in args.budgets.split(",") if b.strip()}
    seeds = {s if s.startswith("seed_") else f"seed_{s}" for s in args.seeds.split(",") if s.strip()}
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")
    df = pd.read_csv(Path(args.paper_tables) / "all_downstream_metrics.csv")
    df = df[
        (df["dataset"] != "smoke_binary")
        & df["method"].isin(METHODS)
        & df["budget"].isin(budgets)
        & df["seed"].isin(seeds)
        & df["learner"].isin(LEARNERS)
    ].copy()
    expected = df["dataset"].nunique() * len(budgets) * len(seeds) * len(LEARNERS)
    coverage = (
        df.groupby("method")
        .agg(n_cells=("auroc", "count"), datasets=("dataset", "nunique"), budgets=("budget", "nunique"), seeds=("seed", "nunique"), learners=("learner", "nunique"))
        .reset_index()
    )
    coverage["expected_cells"] = expected
    coverage["complete"] = coverage["n_cells"].eq(expected)
    complete_methods = coverage.loc[coverage["complete"], "method"].tolist()
    complete = df[df["method"].isin(complete_methods)].copy()
    pivot = complete.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean().rename("avg_rank")
    perf = (
        complete.groupby("method")
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            log_loss_mean=("log_loss", "mean"),
            n_cells=("auroc", "count"),
        )
        .join(ranks)
        .reset_index()
    )
    perf["method_label"] = perf["method"].map(label)
    perf = perf.sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])
    stats = pairwise(pivot)
    by_learner = complete.pivot_table(index="method", columns="learner", values="auroc", aggfunc="mean").reset_index()
    by_learner["method_label"] = by_learner["method"].map(label)
    coverage.to_csv(out_tables / "icdm_binary_headline_5seed_coverage.csv", index=False)
    perf.to_csv(out_tables / "icdm_binary_headline_5seed_performance.csv", index=False)
    stats.to_csv(out_tables / "icdm_binary_headline_5seed_pairwise.csv", index=False)
    by_learner.to_csv(out_tables / "icdm_binary_headline_5seed_by_learner.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = perf["method"].tolist()
    x = np.arange(len(order))
    means = [perf.loc[perf["method"] == m, "auroc_mean"].iloc[0] for m in order]
    sem = [complete.loc[complete["method"] == m, "auroc"].sem() for m in order]
    ax.errorbar(x, means, yerr=[1.96 * s for s in sem], fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in order], rotation=25, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Binary headline 5-seed performance")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_figs / "icdm_binary_headline_5seed_performance.png", dpi=220)
    plt.close(fig)

    lines = [
        "# ICDM Binary Headline 5-Seed Update",
        "",
        f"Headline binary grid for budgets {', '.join(sorted(budgets))} and seeds {', '.join(sorted(seeds))}. Expected cells per complete method: {expected}.",
        "",
        "## Performance",
        "",
        md_table(
            perf[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
                columns={"method_label": "Method", "auroc_mean": "Mean AUROC", "auroc_std": "Std", "avg_rank": "Avg. Rank", "n_cells": "Cells"}
            )
        ),
        "",
        "## Pairwise Tests",
        "",
        md_table(
            stats[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                columns={"method_label": "Method", "comparator_label": "Comparator", "n_pairs": "Pairs", "mean_diff": "Mean Diff", "wilcoxon_p": "Wilcoxon p"}
            )
        ),
        "",
        "## Files",
        "",
        "- `tables/icdm_binary_headline_5seed_performance.csv`",
        "- `tables/icdm_binary_headline_5seed_pairwise.csv`",
        "- `figures/icdm_binary_headline_5seed_performance.png`",
        "",
    ]
    (out_notes / "ICDM_BINARY_HEADLINE_5SEED_UPDATE.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote binary headline 5-seed summaries.")


if __name__ == "__main__":
    main()
