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


METHODS = [
    "histdistill_refined",
    "histdistill_density",
    "histdistill_importance",
    "histdistill_greedy",
    "gain_path_refined",
    "gain_path_safeguarded",
    "gain_herding",
    "gain_path_herding",
    "herding",
    "random",
]
LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
LABELS = {
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_density": "HistDistill-Density",
    "histdistill_importance": "ImportanceSample",
    "histdistill_greedy": "HistDistill-Greedy",
    "gain_path_refined": "Legacy gain-path refined",
    "gain_path_safeguarded": "Legacy gain-path safeguarded",
    "gain_herding": "Gain sketch",
    "gain_path_herding": "Gain-path sketch",
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


def paired_stats(pivot: pd.DataFrame, methods: list[str], comparators: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
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


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_figures(df: pd.DataFrame, fid: pd.DataFrame, out_figs: Path) -> None:
    order = [m for m in METHODS if m in df["method"].unique()]
    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_rank = ranks.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    methods = [m for m in mean_rank.index if m in order]
    ax.scatter(mean_rank[methods], np.arange(len(methods)), s=42)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([label(m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("ICDM core average ranks")
    ax.grid(axis="x", alpha=0.2)
    save_fig(fig, out_figs / "icdm_core_rank_profile.png")

    budget = df.groupby(["method", "budget"])["auroc"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in order:
        sub = budget[budget["method"] == method].copy()
        if sub.empty:
            continue
        x = sub["budget"].str.replace("budget_", "", regex=False).astype(int)
        y = sub["mean"].astype(float)
        sem = sub["sem"].fillna(0.0).astype(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=label(method))
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.08)
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("AUROC")
    ax.set_title("ICDM core budget profile")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, out_figs / "icdm_core_budget_profile.png")

    if not fid.empty and "sld_inf_norm" in fid.columns:
        perf = df.groupby(["dataset", "budget", "seed", "method"])["auroc"].mean().reset_index()
        merged = perf.merge(
            fid[["dataset", "budget", "seed", "method", "sld_inf_norm", "sld_root_agreement_direct"]],
            on=["dataset", "budget", "seed", "method"],
            how="inner",
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6.6, 4.5))
            for method, sub in merged.groupby("method"):
                ax.scatter(sub["sld_inf_norm"], sub["auroc"], s=24, alpha=0.55, label=label(method))
            ax.set_xscale("log")
            ax.set_xlabel("Normalized SLD infinity")
            ax.set_ylabel("Mean AUROC across learners")
            ax.set_title("SLD vs downstream quality")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)
            save_fig(fig, out_figs / "icdm_core_sld_scatter.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    budgets = {b if b.startswith("budget_") else f"budget_{b}" for b in args.budgets.split(",") if b.strip()}
    seeds = {s if s.startswith("seed_") else f"seed_{s}" for s in args.seeds.split(",") if s.strip()}
    tables = Path(args.paper_tables)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    df = downstream[
        (downstream["dataset"] != "smoke_binary")
        & downstream["method"].isin(METHODS)
        & downstream["budget"].isin(budgets)
        & downstream["seed"].isin(seeds)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
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
    expected = df["dataset"].nunique() * len(budgets) * len(seeds) * len(LEARNERS)
    coverage["expected_cells"] = expected
    coverage["complete"] = coverage["n_cells"].eq(expected)
    complete_methods = coverage.loc[coverage["complete"], "method"].tolist()
    complete = df[df["method"].isin(complete_methods)].copy()
    pivot = complete.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
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
    )
    summary["method_label"] = summary["method"].map(label)
    summary = summary.sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])
    by_budget = complete.pivot_table(index="method", columns="budget", values="auroc", aggfunc="mean").reset_index()
    by_budget["method_label"] = by_budget["method"].map(label)
    by_learner = complete.pivot_table(index="method", columns="learner", values="auroc", aggfunc="mean").reset_index()
    by_learner["method_label"] = by_learner["method"].map(label)
    stats = paired_stats(pivot, complete_methods, ["gain_path_refined", "herding", "random"])

    fidelity_path = tables / "all_split_fidelity_metrics.csv"
    if fidelity_path.exists():
        fid_all = pd.read_csv(fidelity_path)
        fid = fid_all[
            (fid_all["dataset"] != "smoke_binary")
            & fid_all["method"].isin(complete_methods)
            & fid_all["budget"].isin(budgets)
            & fid_all["seed"].isin(seeds)
        ].copy()
        fid_summary = (
            fid.groupby("method")
            .agg(
                root_agreement=("root_agreement", "mean"),
                gain_rank_correlation=("gain_rank_correlation", "mean"),
                root_sld_inf_norm=("root_sld_inf_norm", "mean"),
                sld_inf_norm=("sld_inf_norm", "mean"),
                sld_root_agreement_direct=("sld_root_agreement_direct", "mean"),
                sld_rank_correlation_direct=("sld_rank_correlation_direct", "mean"),
                sld_margin_satisfied_rate=("sld_margin_satisfied_rate", "mean"),
                n_cells=("gain_rank_correlation", "count"),
            )
            .reset_index()
        )
        fid_summary["method_label"] = fid_summary["method"].map(label)
    else:
        fid = pd.DataFrame()
        fid_summary = pd.DataFrame()

    coverage.to_csv(out_tables / "icdm_core_coverage.csv", index=False)
    summary.to_csv(out_tables / "icdm_core_performance.csv", index=False)
    by_budget.to_csv(out_tables / "icdm_core_by_budget.csv", index=False)
    by_learner.to_csv(out_tables / "icdm_core_by_learner.csv", index=False)
    stats.to_csv(out_tables / "icdm_core_pairwise.csv", index=False)
    if not fid_summary.empty:
        fid_summary.to_csv(out_tables / "icdm_core_fidelity.csv", index=False)
    make_figures(complete, fid, out_figs)

    perf_md = summary[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={"method_label": "Method", "auroc_mean": "Mean AUROC", "auroc_std": "Std", "avg_rank": "Avg. Rank", "n_cells": "Cells"}
    )
    key_stats = stats[
        stats["method"].isin(["histdistill_refined", "histdistill_density"])
        & stats["comparator"].isin(["gain_path_refined", "herding", "random"])
    ][["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={"method_label": "Method", "comparator_label": "Comparator", "n_pairs": "Pairs", "mean_diff": "Mean Diff", "wilcoxon_p": "Wilcoxon p"}
    )
    fid_md = (
        fid_summary[["method_label", "gain_rank_correlation", "sld_inf_norm", "sld_root_agreement_direct", "sld_rank_correlation_direct", "n_cells"]]
        .rename(
            columns={
                "method_label": "Method",
                "gain_rank_correlation": "Retrained Gain Corr.",
                "sld_inf_norm": "Direct SLD Inf.",
                "sld_root_agreement_direct": "Direct Root Agree",
                "sld_rank_correlation_direct": "Direct Rank Corr.",
                "n_cells": "Cells",
            }
        )
        if not fid_summary.empty
        else pd.DataFrame()
    )

    note = f"""# ICDM Core Experiment Update

Core binary benchmark summary for budgets {', '.join(sorted(budgets))} and seeds {', '.join(sorted(seeds))}. Methods are included only when complete on the paired grid. Expected complete cells per method: {expected}.

## Performance

{md_table(perf_md)}

## Key Pairwise Tests

{md_table(key_stats)}

## Fidelity / SLD

{md_table(fid_md)}

## Coverage

{md_table(coverage[['method', 'n_cells', 'expected_cells', 'datasets', 'budgets', 'seeds', 'learners', 'complete', 'auroc_mean']])}

CSV outputs:

- `tables/icdm_core_coverage.csv`
- `tables/icdm_core_performance.csv`
- `tables/icdm_core_by_budget.csv`
- `tables/icdm_core_by_learner.csv`
- `tables/icdm_core_pairwise.csv`
- `tables/icdm_core_fidelity.csv`

Figures:

- `figures/icdm_core_rank_profile.png`
- `figures/icdm_core_budget_profile.png`
- `figures/icdm_core_sld_scatter.png`
"""
    (out_notes / "ICDM_CORE_UPDATE.md").write_text(note, encoding="utf-8")
    print(f"Wrote ICDM core summaries to {materials}.")


if __name__ == "__main__":
    main()
