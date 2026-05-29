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


DATASETS = ["adult", "australian", "diabetes", "spambase"]
METHODS = ["histdistill_refined", "histdistill_density", "gain_path_refined", "herding", "random"]
LEARNERS = ["xgboost", "lightgbm", "mlp"]
SEEDS = ["seed_0", "seed_1"]
BUDGETS = ["budget_25"]

LABELS = {
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_density": "HistDistill-Density",
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
    for method in METHODS:
        for comp in ["gain_path_refined", "herding", "random"]:
            if method == comp or method not in pivot.columns or comp not in pivot.columns:
                continue
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


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_figures(df: pd.DataFrame, fid: pd.DataFrame, out_figs: Path) -> None:
    order = ["histdistill_refined", "histdistill_density", "gain_path_refined", "herding", "random"]
    by_learner = df.groupby(["method", "learner"])["auroc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xlabels = ["xgboost", "lightgbm", "mlp"]
    x = np.arange(len(xlabels))
    for method in order:
        vals = []
        for learner in xlabels:
            sub = by_learner[(by_learner["method"] == method) & (by_learner["learner"] == learner)]
            vals.append(float(sub["auroc"].iloc[0]) if not sub.empty else np.nan)
        ax.plot(x, vals, marker="o", linewidth=1.8, label=label(method))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Limited transfer profile")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, out_figs / "limited_histdistill_transfer_profiles.png")

    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
    rows = []
    for method in ["histdistill_refined", "histdistill_density", "herding", "random"]:
        if method not in pivot.columns or "gain_path_refined" not in pivot.columns:
            continue
        diff = (pivot[method] - pivot["gain_path_refined"]).dropna()
        for val in diff:
            rows.append({"method_label": label(method), "delta": float(val)})
    delta = pd.DataFrame(rows)
    if not delta.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        groups = [g["delta"].values for _, g in delta.groupby("method_label", sort=False)]
        names = [name for name, _ in delta.groupby("method_label", sort=False)]
        parts = ax.violinplot(groups, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("AUROC delta vs legacy gain-path refined")
        ax.set_title("Paired limited-run deltas")
        ax.grid(axis="y", alpha=0.2)
        save_fig(fig, out_figs / "limited_histdistill_delta_violins.png")

    if not fid.empty and "sld_inf_norm" in fid.columns:
        perf = df.groupby(["dataset", "budget", "seed", "method"])["auroc"].mean().reset_index()
        merged = perf.merge(
            fid[["dataset", "budget", "seed", "method", "sld_inf_norm", "sld_root_agreement_direct"]],
            on=["dataset", "budget", "seed", "method"],
            how="inner",
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for method, sub in merged.groupby("method"):
                ax.scatter(sub["sld_inf_norm"], sub["auroc"], s=34, alpha=0.65, label=label(method))
            ax.set_xscale("log")
            ax.set_xlabel("Mean normalized SLD infinity")
            ax.set_ylabel("Mean AUROC across learners")
            ax.set_title("SLD vs downstream quality")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
            save_fig(fig, out_figs / "limited_histdistill_sld_scatter.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    args = ap.parse_args()

    tables = Path(args.paper_tables)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    df = downstream[
        downstream["dataset"].isin(DATASETS)
        & downstream["method"].isin(METHODS)
        & downstream["budget"].isin(BUDGETS)
        & downstream["seed"].isin(SEEDS)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
    df["method_label"] = df["method"].map(label)

    summary = (
        df.groupby("method")
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            log_loss_mean=("log_loss", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            n_cells=("auroc", "count"),
        )
        .reset_index()
    )
    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean().rename("avg_rank")
    summary = summary.join(ranks, on="method")
    summary["method_label"] = summary["method"].map(label)
    summary = summary.sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])

    by_learner = df.pivot_table(index="method", columns="learner", values="auroc", aggfunc="mean").reset_index()
    by_learner["method_label"] = by_learner["method"].map(label)
    by_dataset = df.pivot_table(index="dataset", columns="method", values="auroc", aggfunc="mean").reset_index()
    stats = pairwise(pivot)

    fidelity_path = tables / "all_split_fidelity_metrics.csv"
    if fidelity_path.exists():
        fid_all = pd.read_csv(fidelity_path)
        fid = fid_all[
            fid_all["dataset"].isin(DATASETS)
            & fid_all["method"].isin(METHODS)
            & fid_all["budget"].isin(BUDGETS)
            & fid_all["seed"].isin(SEEDS)
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

    summary.to_csv(out_tables / "limited_histdistill_performance.csv", index=False)
    by_learner.to_csv(out_tables / "limited_histdistill_by_learner.csv", index=False)
    by_dataset.to_csv(out_tables / "limited_histdistill_by_dataset.csv", index=False)
    stats.to_csv(out_tables / "limited_histdistill_pairwise.csv", index=False)
    if not fid_summary.empty:
        fid_summary.to_csv(out_tables / "limited_histdistill_fidelity.csv", index=False)
    make_figures(df, fid, out_figs)

    perf_md = summary[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={
            "method_label": "Method",
            "auroc_mean": "Mean AUROC",
            "auroc_std": "Std",
            "avg_rank": "Avg. Rank",
            "n_cells": "Cells",
        }
    )
    pair_md = stats[
        stats["method"].isin(["histdistill_refined", "histdistill_density"])
        & stats["comparator"].isin(["gain_path_refined", "herding", "random"])
    ][["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={
            "method_label": "Method",
            "comparator_label": "Comparator",
            "n_pairs": "Pairs",
            "mean_diff": "Mean Diff",
            "wilcoxon_p": "Wilcoxon p",
        }
    )
    fid_md = (
        fid_summary[
            [
                "method_label",
                "gain_rank_correlation",
                "sld_inf_norm",
                "sld_root_agreement_direct",
                "sld_rank_correlation_direct",
                "n_cells",
            ]
        ].rename(
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

    note = f"""# Limited HistDistill Update

This is the first limited subset for the new histogram-aligned plan: {len(DATASETS)} datasets, budget 25/class, seeds 0/1, and learners xgboost/lightgbm/mlp. The goal is debugging and direction checking, not a final paper table.

## Downstream

{md_table(perf_md)}

## Paired Tests

{md_table(pair_md)}

## Fidelity / SLD

{md_table(fid_md)}

## Readout

- `HistDistill-Refined` is now close to the legacy gain-path method and above herding/random on this limited slice.
- The pure coverage selector remains weak; the deployed variant currently needs gain-path warm start plus bounded histogram weights.
- Direct SLD/root agreement improved for the deployed HistDistill variants, which supports the new theory direction, but the margin-satisfied criterion is not green yet.
- Do not scale to broad26 until the next patch improves pure greedy support or proves the warm-start formulation is the paper method.

CSV outputs:

- `tables/limited_histdistill_performance.csv`
- `tables/limited_histdistill_pairwise.csv`
- `tables/limited_histdistill_fidelity.csv`
- `tables/limited_histdistill_by_learner.csv`
- `tables/limited_histdistill_by_dataset.csv`

Figures:

- `figures/limited_histdistill_transfer_profiles.png`
- `figures/limited_histdistill_delta_violins.png`
- `figures/limited_histdistill_sld_scatter.png`
"""
    (out_notes / "LIMITED_HISTDISTILL_UPDATE.md").write_text(note, encoding="utf-8")
    print(f"Wrote limited HistDistill summaries to {materials}.")


if __name__ == "__main__":
    main()
