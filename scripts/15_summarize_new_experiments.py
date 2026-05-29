from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gaindistill.utils import ensure_dir


CORE_METHODS = [
    "random",
    "herding",
    "gain_herding",
    "gain_path_herding",
    "gain_path_refined",
    "gain_path_safeguarded",
]

ABLATION_METHODS = [
    "gain_path_refined",
    "refined_no_path",
    "refined_no_refine",
    "refined_no_grad",
    "refined_no_hess",
    "refined_no_conf",
    "refined_no_raw",
    "refined_no_gain_weight",
    "refined_path_occupancy_only",
]

METHOD_LABELS = {
    "random": "Random",
    "herding": "Herding",
    "gain_herding": "Gain sketch",
    "gain_path_herding": "Gain-path sketch",
    "gain_path_refined": "Gain-path refined",
    "gain_path_safeguarded": "Gain-path safeguarded",
    "refined_no_path": "No path features",
    "refined_no_refine": "No local refinement",
    "refined_no_grad": "No gradient mass",
    "refined_no_hess": "No Hessian mass",
    "refined_no_conf": "No teacher confidence",
    "refined_no_raw": "No raw features",
    "refined_no_gain_weight": "No gain weighting",
    "refined_path_occupancy_only": "Path occupancy only",
}

FOCUSED = {
    "adult",
    "australian",
    "bank_marketing",
    "breast_w",
    "credit_g",
    "diabetes",
    "pc1",
    "spambase",
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


def summarize_block(df: pd.DataFrame, methods: list[str], expected: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[df["method"].isin(methods)].copy()
    coverage = sub.groupby("method").agg(n_cells=("auroc", "count")).reset_index()
    complete_methods = coverage.loc[coverage["n_cells"] == expected, "method"].tolist()
    sub = sub[sub["method"].isin(complete_methods)].copy()
    pivot = sub.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean().rename("avg_rank")
    summary = (
        sub.groupby("method")
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
    return summary, pivot, sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    args = ap.parse_args()

    tables = Path(args.paper_tables)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    downstream = downstream[
        (downstream["dataset"] != "smoke_binary")
        & downstream["budget"].isin(["budget_25", "budget_50"])
        & downstream["seed"].isin(["seed_0", "seed_1", "seed_2"])
        & downstream["learner"].isin(["xgboost", "lightgbm", "catboost", "random_forest", "mlp"])
    ].copy()

    broad_datasets = sorted(d for d in downstream["dataset"].unique() if d != "smoke_binary")
    broad_expected = len(broad_datasets) * 2 * 3 * 5
    broad_summary, broad_pivot, broad_sub = summarize_block(downstream, CORE_METHODS, broad_expected)
    broad_stats = paired_stats(broad_pivot, CORE_METHODS, ["random", "herding", "gain_herding", "gain_path_herding"])
    broad_budget = broad_sub.pivot_table(index="method", columns="budget", values="auroc", aggfunc="mean").reset_index()
    broad_budget["method_label"] = broad_budget["method"].map(label)
    broad_dataset = broad_sub.pivot_table(index="dataset", columns="method", values="auroc", aggfunc="mean").reset_index()

    broad_summary.to_csv(out_tables / "broad26_core_performance.csv", index=False)
    broad_stats.to_csv(out_tables / "broad26_core_pairwise.csv", index=False)
    broad_budget.to_csv(out_tables / "broad26_core_budget_means.csv", index=False)
    broad_dataset.to_csv(out_tables / "broad26_core_dataset_means.csv", index=False)

    focused = downstream[downstream["dataset"].isin(FOCUSED)].copy()
    ab_expected = len(FOCUSED) * 2 * 3 * 5
    ab_summary, ab_pivot, ab_sub = summarize_block(focused, ABLATION_METHODS, ab_expected)
    ab_stats = paired_stats(ab_pivot, ABLATION_METHODS, ["gain_path_refined"])
    ab_fidelity_path = tables / "all_split_fidelity_metrics.csv"
    if ab_fidelity_path.exists():
        fidelity = pd.read_csv(ab_fidelity_path)
        fid = fidelity[
            fidelity["dataset"].isin(FOCUSED)
            & fidelity["budget"].isin(["budget_25", "budget_50"])
            & fidelity["seed"].isin(["seed_0", "seed_1", "seed_2"])
            & fidelity["method"].isin(ABLATION_METHODS)
        ].copy()
        ab_fid = (
            fid.groupby("method")
            .agg(
                root_agreement=("root_agreement", "mean"),
                top5_overlap=("top5_overlap", "mean"),
                gain_rank_correlation=("gain_rank_correlation", "mean"),
                n_cells=("gain_rank_correlation", "count"),
            )
            .reset_index()
        )
        ab_fid["method_label"] = ab_fid["method"].map(label)
    else:
        ab_fid = pd.DataFrame()

    ab_summary.to_csv(out_tables / "focused_refined_ablation_performance.csv", index=False)
    ab_stats.to_csv(out_tables / "focused_refined_ablation_pairwise.csv", index=False)
    if not ab_fid.empty:
        ab_fid.to_csv(out_tables / "focused_refined_ablation_fidelity.csv", index=False)

    broad_md = broad_summary[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={"method_label": "Method", "auroc_mean": "Mean AUROC", "auroc_std": "Std", "avg_rank": "Avg. Rank", "n_cells": "Cells"}
    )
    broad_key = broad_stats[
        broad_stats["method"].isin(["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding"])
        & broad_stats["comparator"].isin(["random", "herding"])
    ][["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={"method_label": "Method", "comparator_label": "Comparator", "n_pairs": "Pairs", "mean_diff": "Mean Diff", "wilcoxon_p": "Wilcoxon p"}
    )
    ab_md = ab_summary[["method_label", "auroc_mean", "avg_rank", "n_cells"]].rename(
        columns={"method_label": "Variant", "auroc_mean": "Mean AUROC", "avg_rank": "Avg. Rank", "n_cells": "Cells"}
    )
    ab_key = ab_stats[["method_label", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={"method_label": "Variant", "mean_diff": "Diff vs Full", "wilcoxon_p": "Wilcoxon p"}
    )

    note = f"""# New Experiment Update

## Broad 26-Dataset Core Benchmark

This block expands the core comparison to {len(broad_datasets)} real binary datasets, budgets 25/50, seeds 0/1/2, and five downstream learners. Each complete method has {broad_expected} paired cells.

{md_table(broad_md)}

Key broad paired tests:

{md_table(broad_key)}

CSV outputs:

- `tables/broad26_core_performance.csv`
- `tables/broad26_core_pairwise.csv`
- `tables/broad26_core_budget_means.csv`
- `tables/broad26_core_dataset_means.csv`

## Focused Refined-Method Ablations

This block tests which components of the refined gain/path method matter on the focused 8-dataset suite.

{md_table(ab_md)}

Ablation deltas are measured as variant minus full `Gain-path refined`; negative values mean the ablation hurts.

{md_table(ab_key)}

CSV outputs:

- `tables/focused_refined_ablation_performance.csv`
- `tables/focused_refined_ablation_pairwise.csv`
- `tables/focused_refined_ablation_fidelity.csv`
"""
    (out_notes / "NEW_EXPERIMENT_UPDATE.md").write_text(note, encoding="utf-8")

    final_path = materials / "FINAL_RESULTS.md"
    if final_path.exists():
        text = final_path.read_text(encoding="utf-8")
        marker = "\n## New Experiment Update\n"
        text = text.split(marker)[0].rstrip() + marker + note.split("# New Experiment Update", 1)[1]
        final_path.write_text(text, encoding="utf-8")

    print(f"Wrote new experiment summaries to {materials}.")


if __name__ == "__main__":
    main()
