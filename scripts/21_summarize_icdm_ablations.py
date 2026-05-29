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
    "histdistill_density",
    "histdistill_refined",
    "histdistill_no_weight_fit",
    "histdistill_root_only",
    "histdistill_linear_coverage",
    "histdistill_no_refine",
    "histdistill_greedy",
    "histdistill_importance",
    "gain_path_refined",
    "herding",
    "random",
]
LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
LABELS = {
    "histdistill_density": "HistDistill-Density",
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_no_weight_fit": "No weight fit",
    "histdistill_root_only": "Root-only probes",
    "histdistill_linear_coverage": "Linear coverage",
    "histdistill_no_refine": "No refinement",
    "histdistill_greedy": "Pure greedy selector",
    "histdistill_importance": "ImportanceSample",
    "gain_path_refined": "Legacy gain-path refined",
    "herding": "Herding",
    "random": "Random",
}
ABLATION_TESTS = [
    ("histdistill_density", "histdistill_refined", "density term"),
    ("histdistill_refined", "histdistill_no_weight_fit", "weight fit"),
    ("histdistill_refined", "histdistill_root_only", "probe regions"),
    ("histdistill_refined", "histdistill_linear_coverage", "saturating coverage"),
    ("histdistill_refined", "histdistill_no_refine", "local refinement"),
    ("histdistill_refined", "histdistill_greedy", "warm-start/refined selector"),
    ("histdistill_greedy", "histdistill_importance", "greedy vs importance"),
]


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


def paired_row(pivot: pd.DataFrame, better: str, worse: str, component: str) -> dict[str, object] | None:
    if better not in pivot.columns or worse not in pivot.columns:
        return None
    pair = pivot[[better, worse]].dropna()
    if pair.empty:
        return None
    diff = pair[better] - pair[worse]
    try:
        p_value = float(wilcoxon(diff).pvalue) if (diff.abs() > 1e-12).any() else 1.0
    except ValueError:
        p_value = 1.0
    return {
        "component": component,
        "better_method": better,
        "better_label": label(better),
        "ablation_method": worse,
        "ablation_label": label(worse),
        "n_pairs": int(len(diff)),
        "mean_delta": float(diff.mean()),
        "median_delta": float(diff.median()),
        "wins": int((diff > 1e-6).sum()),
        "ties": int((diff.abs() <= 1e-6).sum()),
        "losses": int((diff < -1e-6).sum()),
        "wilcoxon_p": p_value,
    }


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_figures(df: pd.DataFrame, fid: pd.DataFrame, out_figs: Path) -> None:
    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])["auroc"].mean().unstack()
    delta_frames = []
    for better, worse, component in ABLATION_TESTS:
        if better not in pivot.columns or worse not in pivot.columns:
            continue
        diff = (pivot[better] - pivot[worse]).dropna()
        if diff.empty:
            continue
        delta_frames.append(pd.DataFrame({"component": component, "delta": diff.to_numpy()}))
    if delta_frames:
        deltas = pd.concat(delta_frames, ignore_index=True)
        order = [c for _, _, c in ABLATION_TESTS if c in set(deltas["component"])]
        data = [deltas.loc[deltas["component"] == c, "delta"].to_numpy() for c in order]
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Paired AUROC delta")
        ax.set_title("Component ablation deltas")
        ax.grid(axis="y", alpha=0.2)
        save_fig(fig, out_figs / "icdm_ablation_delta_violins.png")

    if not fid.empty and "sld_inf_norm" in fid.columns:
        perf = df.groupby(["dataset", "budget", "seed", "method"])["auroc"].mean().reset_index()
        merged = perf.merge(
            fid[["dataset", "budget", "seed", "method", "sld_inf_norm", "sld_rank_correlation_direct"]],
            on=["dataset", "budget", "seed", "method"],
            how="inner",
        ).dropna(subset=["sld_inf_norm"])
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for method in [m for m in METHODS if m in set(merged["method"])]:
                sub = merged[merged["method"] == method]
                ax.scatter(sub["sld_inf_norm"], sub["auroc"], s=24, alpha=0.5, label=label(method))
            ax.set_xscale("log")
            ax.set_xlabel("Normalized SLD infinity")
            ax.set_ylabel("Mean AUROC across learners")
            ax.set_title("Ablation SLD-performance tradeoff")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)
            save_fig(fig, out_figs / "icdm_ablation_sld_tradeoff.png")


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
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    df = downstream[
        (downstream["dataset"] != "smoke_binary")
        & downstream["method"].isin(METHODS)
        & downstream["budget"].isin(budgets)
        & downstream["seed"].isin(seeds)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
    expected = df["dataset"].nunique() * len(budgets) * len(seeds) * len(LEARNERS)
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

    pairwise_rows = [paired_row(pivot, better, worse, component) for better, worse, component in ABLATION_TESTS]
    pairwise = pd.DataFrame([row for row in pairwise_rows if row is not None])

    fidelity_path = tables / "all_split_fidelity_metrics.csv"
    fid = pd.DataFrame()
    fid_summary = pd.DataFrame()
    if fidelity_path.exists():
        fid_all = pd.read_csv(fidelity_path)
        fid = fid_all[
            (fid_all["dataset"] != "smoke_binary")
            & fid_all["method"].isin(complete_methods)
            & fid_all["budget"].isin(budgets)
            & fid_all["seed"].isin(seeds)
        ].copy()
        if not fid.empty:
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

    coverage.to_csv(out_tables / "icdm_histdistill_ablation_coverage.csv", index=False)
    summary.to_csv(out_tables / "icdm_histdistill_ablation_performance.csv", index=False)
    pairwise.to_csv(out_tables / "icdm_histdistill_ablation_pairwise.csv", index=False)
    if not fid_summary.empty:
        fid_summary.to_csv(out_tables / "icdm_histdistill_ablation_fidelity.csv", index=False)
    make_figures(complete, fid, out_figs)

    perf_md = summary[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={
            "method_label": "Method",
            "auroc_mean": "Mean AUROC",
            "auroc_std": "Std",
            "avg_rank": "Avg. Rank",
            "n_cells": "Cells",
        }
    )
    pair_md = pairwise[
        ["component", "better_label", "ablation_label", "n_pairs", "mean_delta", "wins", "ties", "losses", "wilcoxon_p"]
    ].rename(
        columns={
            "component": "Component",
            "better_label": "Reference",
            "ablation_label": "Ablation",
            "n_pairs": "Pairs",
            "mean_delta": "Mean Delta",
            "wilcoxon_p": "Wilcoxon p",
        }
    )
    lines = [
        "# ICDM HistDistill Ablation Update",
        "",
        f"Binary component-ablation grid for budgets {', '.join(sorted(budgets))} and seeds {', '.join(sorted(seeds))}. "
        f"Expected complete cells per method: {expected}.",
        "",
        "## Performance",
        "",
        md_table(perf_md),
        "",
        "## Paired Component Tests",
        "",
        md_table(pair_md),
    ]
    if not fid_summary.empty:
        fid_md = fid_summary[
            [
                "method_label",
                "gain_rank_correlation",
                "sld_inf_norm",
                "sld_root_agreement_direct",
                "sld_rank_correlation_direct",
                "sld_margin_satisfied_rate",
                "n_cells",
            ]
        ].rename(
            columns={
                "method_label": "Method",
                "gain_rank_correlation": "Retrained Gain Corr.",
                "sld_inf_norm": "Direct SLD Inf.",
                "sld_root_agreement_direct": "Direct Root Agree",
                "sld_rank_correlation_direct": "Direct Rank Corr.",
                "sld_margin_satisfied_rate": "Margin Satisfied",
                "n_cells": "Cells",
            }
        )
        lines.extend(["", "## Fidelity / SLD", "", md_table(fid_md)])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tables/icdm_histdistill_ablation_coverage.csv`",
            "- `tables/icdm_histdistill_ablation_performance.csv`",
            "- `tables/icdm_histdistill_ablation_pairwise.csv`",
            "- `tables/icdm_histdistill_ablation_fidelity.csv`",
            "- `figures/icdm_ablation_delta_violins.png`",
            "- `figures/icdm_ablation_sld_tradeoff.png`",
            "",
        ]
    )
    (out_notes / "ICDM_ABLATION_UPDATE.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote ICDM ablation summaries to paper_materials.")


if __name__ == "__main__":
    main()
