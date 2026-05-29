from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gaindistill.utils import ensure_dir


FOCUSED = [
    "adult",
    "australian",
    "bank_marketing",
    "breast_w",
    "credit_g",
    "diabetes",
    "pc1",
    "spambase",
]

LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
BUDGETS = ["budget_25", "budget_50"]
SEEDS = ["seed_0", "seed_1", "seed_2"]

MATH_METHODS = [
    "math_full",
    "math_no_probes",
    "math_root_only",
    "math_no_soft_routing",
    "math_no_anchor",
    "math_no_newton",
    "math_no_hist",
]

REFERENCE_METHODS = ["gain_path_refined", "herding", "random"]

METHOD_LABELS = {
    "math_full": "Full synthetic math",
    "math_no_probes": "No probes",
    "math_root_only": "Root only",
    "math_no_soft_routing": "No soft routing",
    "math_no_anchor": "No anchor",
    "math_no_newton": "No Newton",
    "math_no_hist": "No histogram",
    "gain_path_refined": "Gain-path refined",
    "herding": "Herding",
    "random": "Random",
}


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


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


def summarize_downstream(downstream: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keep = MATH_METHODS + REFERENCE_METHODS
    df = downstream[
        downstream["dataset"].isin(FOCUSED)
        & downstream["method"].isin(keep)
        & downstream["budget"].isin(BUDGETS)
        & downstream["seed"].isin(SEEDS)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
    expected = len(FOCUSED) * len(BUDGETS) * len(SEEDS) * len(LEARNERS)
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
    coverage["method_label"] = coverage["method"].map(label)

    complete_methods = coverage.loc[coverage["complete"], "method"].tolist()
    complete = df[df["method"].isin(complete_methods)].copy()
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
    )
    summary["method_label"] = summary["method"].map(label)
    summary = summary.sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])
    budget = complete.pivot_table(index="method", columns="budget", values="auroc", aggfunc="mean").reset_index()
    budget["method_label"] = budget["method"].map(label)
    stats = paired_stats(pivot, complete_methods, ["math_full", "gain_path_refined", "herding", "random"])
    return coverage, summary, budget, stats


def summarize_fidelity(fidelity: pd.DataFrame) -> pd.DataFrame:
    keep = MATH_METHODS + ["gain_path_refined"]
    fid = fidelity[
        fidelity["dataset"].isin(FOCUSED)
        & fidelity["method"].isin(keep)
        & fidelity["budget"].isin(BUDGETS)
        & fidelity["seed"].isin(SEEDS)
    ].copy()
    if fid.empty:
        return pd.DataFrame()
    out = (
        fid.groupby("method")
        .agg(
            root_agreement=("root_agreement", "mean"),
            top5_overlap=("top5_overlap", "mean"),
            gain_rank_correlation=("gain_rank_correlation", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            n_cells=("gain_rank_correlation", "count"),
        )
        .reset_index()
    )
    out["method_label"] = out["method"].map(label)
    return out.sort_values("gain_rank_correlation", ascending=False)


def summarize_traces(results_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset in FOCUSED:
        for method in MATH_METHODS:
            for budget in BUDGETS:
                for seed in SEEDS:
                    meta_path = results_dir / "distilled" / dataset / method / budget / seed / "metadata.json"
                    if not meta_path.exists():
                        continue
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        continue
                    trace = meta.get("trace") or []
                    if not trace:
                        continue
                    last = trace[-1]
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "method_label": label(method),
                            "budget": budget,
                            "seed": seed,
                            "rows": meta.get("rows"),
                            "final_loss": last.get("loss"),
                            "final_gain_loss": last.get("gain"),
                            "final_margin_loss": last.get("margin"),
                            "final_anchor_loss": last.get("anchor"),
                            "probes": last.get("probes"),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby("method")
        .agg(
            method_label=("method_label", "first"),
            final_loss_mean=("final_loss", "mean"),
            final_gain_loss_mean=("final_gain_loss", "mean"),
            final_margin_loss_mean=("final_margin_loss", "mean"),
            final_anchor_loss_mean=("final_anchor_loss", "mean"),
            probes_mean=("probes", "mean"),
            n_cells=("final_loss", "count"),
        )
        .reset_index()
    )


def make_figures(downstream: pd.DataFrame, fidelity: pd.DataFrame, out_figs: Path) -> None:
    ensure_dir(out_figs)
    keep = MATH_METHODS + ["gain_path_refined", "herding", "random"]
    df = downstream[
        downstream["dataset"].isin(FOCUSED)
        & downstream["method"].isin(keep)
        & downstream["budget"].isin(BUDGETS)
        & downstream["seed"].isin(SEEDS)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
    if df.empty:
        return

    order = [m for m in ["gain_path_refined", "math_full"] + MATH_METHODS[1:] + ["herding", "random"] if m in df["method"].unique()]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in order:
        sub = df[df["method"] == method]
        agg = sub.groupby("budget")["auroc"].agg(["mean", "sem"]).reset_index()
        x = agg["budget"].str.replace("budget_", "", regex=False).astype(int)
        y = agg["mean"].astype(float)
        sem = agg["sem"].fillna(0.0).astype(float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=label(method))
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.10)
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("AUROC")
    ax.set_title("Synthetic math ablations under compression")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_figs / "math_ablation_budget_profiles.png", dpi=220)
    plt.close(fig)

    pivot = df.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    if "math_full" in pivot.columns:
        rows = []
        for method in MATH_METHODS[1:] + ["gain_path_refined", "herding", "random"]:
            if method not in pivot.columns:
                continue
            diff = (pivot[method] - pivot["math_full"]).dropna()
            for val in diff:
                rows.append({"method": method, "method_label": label(method), "delta": float(val)})
        delta = pd.DataFrame(rows)
        if not delta.empty:
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            groups = [g["delta"].values for _, g in delta.groupby("method_label", sort=False)]
            names = [name for name, _ in delta.groupby("method_label", sort=False)]
            parts = ax.violinplot(groups, showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_alpha(0.35)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(np.arange(1, len(names) + 1))
            ax.set_xticklabels(names, rotation=35, ha="right")
            ax.set_ylabel("AUROC delta vs full synthetic math")
            ax.set_title("Paired downstream deltas")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_figs / "math_ablation_delta_violins.png", dpi=220)
            plt.close(fig)

    if not fidelity.empty:
        perf = df.groupby(["dataset", "budget", "seed", "method"])["auroc"].mean().reset_index()
        fid = fidelity[
            fidelity["dataset"].isin(FOCUSED)
            & fidelity["method"].isin(MATH_METHODS + ["gain_path_refined"])
            & fidelity["budget"].isin(BUDGETS)
            & fidelity["seed"].isin(SEEDS)
        ].copy()
        merged = perf.merge(
            fid[["dataset", "budget", "seed", "method", "gain_rank_correlation"]],
            on=["dataset", "budget", "seed", "method"],
            how="inner",
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            for method, sub in merged.groupby("method"):
                ax.scatter(sub["gain_rank_correlation"], sub["auroc"], s=24, alpha=0.55, label=label(method))
            ax.set_xlabel("Gain-rank correlation")
            ax.set_ylabel("Mean AUROC across learners")
            ax.set_title("Split fidelity vs downstream performance")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(out_figs / "math_ablation_fidelity_scatter.png", dpi=220)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    tables = Path(args.paper_tables)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    fidelity_path = tables / "all_split_fidelity_metrics.csv"
    fidelity = pd.read_csv(fidelity_path) if fidelity_path.exists() else pd.DataFrame()

    coverage, summary, budget, stats = summarize_downstream(downstream)
    fid_summary = summarize_fidelity(fidelity) if not fidelity.empty else pd.DataFrame()
    trace_summary = summarize_traces(Path(args.results_dir))

    coverage.to_csv(out_tables / "math_ablation_coverage.csv", index=False)
    summary.to_csv(out_tables / "math_ablation_performance.csv", index=False)
    budget.to_csv(out_tables / "math_ablation_budget_means.csv", index=False)
    stats.to_csv(out_tables / "math_ablation_pairwise.csv", index=False)
    if not fid_summary.empty:
        fid_summary.to_csv(out_tables / "math_ablation_fidelity.csv", index=False)
    if not trace_summary.empty:
        trace_summary.to_csv(out_tables / "math_ablation_trace_summary.csv", index=False)

    make_figures(downstream, fidelity, out_figs)

    expected = len(FOCUSED) * len(BUDGETS) * len(SEEDS) * len(LEARNERS)
    perf_md = summary[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={
            "method_label": "Method",
            "auroc_mean": "Mean AUROC",
            "auroc_std": "Std",
            "avg_rank": "Avg. Rank",
            "n_cells": "Cells",
        }
    )
    key_stats = stats[
        (
            stats["comparator"].eq("math_full")
            & stats["method"].isin([m for m in MATH_METHODS if m != "math_full"])
        )
        | (stats["method"].eq("math_full") & stats["comparator"].isin(["gain_path_refined", "herding", "random"]))
    ][["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={
            "method_label": "Method",
            "comparator_label": "Comparator",
            "n_pairs": "Pairs",
            "mean_diff": "Mean Diff",
            "wilcoxon_p": "Wilcoxon p",
        }
    )
    cov_md = coverage[["method_label", "n_cells", "expected_cells", "complete", "auroc_mean"]].rename(
        columns={
            "method_label": "Method",
            "n_cells": "Cells",
            "expected_cells": "Expected",
            "complete": "Complete",
            "auroc_mean": "Current Mean AUROC",
        }
    )

    note = f"""# Math Ablation Update

This block evaluates the differentiable synthetic objective components on the focused 8-dataset binary suite. The target complete grid is {expected} paired downstream cells per method: 8 datasets x 2 budgets x 3 distilled seeds x 5 learners.

## Coverage

{md_table(cov_md)}

## Downstream Performance

{md_table(perf_md)}

## Paired Tests

Positive mean differences mean the row method is better than the comparator.

{md_table(key_stats)}

## Interpretation

- These tables are meant to support the math section, not replace the main selected-row condensation result.
- If `math_full` trails `gain_path_refined`, the paper should frame the synthetic optimizer as a faithful model-aligned extension/ablation and keep `gain_path_refined` as the empirical headline.
- If removing probes, soft routing, Newton, histogram, or anchors barely changes AUROC, the writing should claim those terms improve landscape matching diagnostics only when the corresponding fidelity table supports it.

CSV outputs:

- `tables/math_ablation_coverage.csv`
- `tables/math_ablation_performance.csv`
- `tables/math_ablation_budget_means.csv`
- `tables/math_ablation_pairwise.csv`
- `tables/math_ablation_fidelity.csv`
- `tables/math_ablation_trace_summary.csv`

Figures:

- `figures/math_ablation_budget_profiles.png`
- `figures/math_ablation_delta_violins.png`
- `figures/math_ablation_fidelity_scatter.png`
"""
    (out_notes / "MATH_ABLATION_UPDATE.md").write_text(note, encoding="utf-8")

    final_path = materials / "FINAL_RESULTS.md"
    if final_path.exists():
        text = final_path.read_text(encoding="utf-8")
        marker = "\n## Math Ablation Update\n"
        text = text.split(marker)[0].rstrip() + marker + note.split("# Math Ablation Update", 1)[1]
        final_path.write_text(text, encoding="utf-8")

    print(f"Wrote math ablation summaries to {materials}.")


if __name__ == "__main__":
    main()
