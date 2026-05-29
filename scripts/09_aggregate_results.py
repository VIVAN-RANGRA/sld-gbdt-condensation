from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from gaindistill.utils import ensure_dir, read_json


def collect_metrics(root: Path, filename: str, kind: str) -> pd.DataFrame:
    rows = []
    for p in root.glob(f"**/{filename}"):
        rel = p.relative_to(root)
        try:
            vals = read_json(p)
        except Exception as exc:
            print(f"Skipping malformed metric file {p}: {exc}")
            continue
        parts = rel.parts
        row = {"kind": kind, "path": str(rel)}
        if kind == "downstream" and len(parts) >= 6:
            row.update({"dataset": parts[0], "method": parts[1], "budget": parts[2], "seed": parts[3], "learner": parts[4]})
        elif kind == "split_fidelity" and len(parts) >= 5:
            row.update({"dataset": parts[0], "method": parts[1], "budget": parts[2], "seed": parts[3]})
        row.update(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_tests(downstream: pd.DataFrame, comparators: list[str]) -> pd.DataFrame:
    rows = []
    if downstream.empty:
        return pd.DataFrame(rows)
    paired = downstream.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    methods = [m for m in paired.columns if isinstance(m, str)]
    for method in methods:
        for comp in comparators:
            if method == comp or comp not in paired.columns:
                continue
            sub = paired[[method, comp]].dropna()
            if sub.empty:
                continue
            diff = sub[method] - sub[comp]
            wins = int((diff > 1e-6).sum())
            ties = int((diff.abs() <= 1e-6).sum())
            losses = int((diff < -1e-6).sum())
            try:
                p_value = float(wilcoxon(diff).pvalue) if (diff.abs() > 1e-12).any() else 1.0
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "method": method,
                    "comparator": comp,
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()
    results = Path(args.results_dir)
    out = ensure_dir(args.out)
    downstream = collect_metrics(results / "downstream", "metrics.json", "downstream")
    fidelity = collect_metrics(results / "split_fidelity", "split_fidelity.json", "split_fidelity")
    paper_learners = {"xgboost", "lightgbm", "catboost", "random_forest", "mlp"}
    if not downstream.empty:
        downstream = downstream[(downstream["dataset"] != "smoke_binary") & (downstream["learner"].isin(paper_learners))].copy()
    if not fidelity.empty:
        fidelity = fidelity[fidelity["dataset"] != "smoke_binary"].copy()
    if not downstream.empty:
        downstream.to_csv(out / "all_downstream_metrics.csv", index=False)
        group_cols = ["dataset", "method", "budget", "learner"]
        agg = downstream.groupby(group_cols).agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), accuracy_mean=("accuracy", "mean"), train_seconds_mean=("train_seconds", "mean")).reset_index()
        agg.to_csv(out / "table_1_main_performance.csv", index=False)
        rank = agg.copy()
        rank["rank"] = rank.groupby(["dataset", "budget", "learner"])["auroc_mean"].rank(ascending=False, method="average")
        rank.groupby(["method", "learner"])["rank"].mean().reset_index().to_csv(out / "average_ranks.csv", index=False)
        low = agg[agg["budget"].isin(["budget_5", "budget_10", "budget_25", "budget_50", "budget_100"])]
        low.pivot_table(index=["budget", "learner"], columns="method", values="auroc_mean", aggfunc="mean").reset_index().to_csv(
            out / "table_3_low_budget.csv", index=False
        )
        agg.pivot_table(index="method", columns="learner", values="auroc_mean", aggfunc="mean").reset_index().to_csv(
            out / "table_4_transfer.csv", index=False
        )
        runtime = downstream.groupby(["dataset", "method", "budget"]).agg(
            distill_train_rows=("train_rows", "mean"),
            downstream_train_seconds=("train_seconds", "mean"),
            auroc_mean=("auroc", "mean"),
        ).reset_index()
        runtime.to_csv(out / "table_6_runtime.csv", index=False)
    if not fidelity.empty:
        fidelity.to_csv(out / "all_split_fidelity_metrics.csv", index=False)
        fidelity_agg = fidelity.groupby(["dataset", "method", "budget"]).agg(
            root_agreement_mean=("root_agreement", "mean"),
            top5_overlap_mean=("top5_overlap", "mean"),
            gain_rank_correlation_mean=("gain_rank_correlation", "mean"),
        ).reset_index()
        fidelity_agg.to_csv(out / "table_2_split_fidelity.csv", index=False)
    if not downstream.empty and not fidelity.empty:
        ab_mask = (
            downstream["method"].str.startswith("ablation_", na=False)
            | downstream["method"].str.startswith("sketch_", na=False)
            | downstream["method"].str.startswith("math_", na=False)
        )
        fid_mask = (
            fidelity["method"].str.startswith("ablation_", na=False)
            | fidelity["method"].str.startswith("sketch_", na=False)
            | fidelity["method"].str.startswith("math_", na=False)
        )
        ab_down = downstream[ab_mask].copy()
        ab_fid = fidelity[fid_mask].copy()
        if not ab_down.empty:
            ab_down["variant"] = ab_down["method"].str.replace("ablation_", "", regex=False)
            ab_down["variant"] = ab_down["variant"].str.replace("sketch_", "sketch_", regex=False)
            ab_down["variant"] = ab_down["variant"].str.replace("math_", "math_", regex=False)
            ab_summary = ab_down.groupby("variant").agg(downstream_score=("auroc", "mean")).reset_index()
            if not ab_fid.empty:
                ab_fid["variant"] = ab_fid["method"].str.replace("ablation_", "", regex=False)
                ab_fid["variant"] = ab_fid["variant"].str.replace("sketch_", "sketch_", regex=False)
                ab_fid["variant"] = ab_fid["variant"].str.replace("math_", "math_", regex=False)
                ab_fid_summary = ab_fid.groupby("variant").agg(
                    split_agreement=("root_agreement", "mean"),
                    gain_corr=("gain_rank_correlation", "mean"),
                    top5_overlap=("top5_overlap", "mean"),
                ).reset_index()
                ab_summary = ab_summary.merge(ab_fid_summary, on="variant", how="left")
            ab_summary.to_csv(out / "table_5_ablation.csv", index=False)
    if not downstream.empty:
        stats = statistical_tests(downstream, ["random", "herding", "gain_path_herding", "gain_herding"])
        if not stats.empty:
            stats.to_csv(out / "table_7_statistical_tests.csv", index=False)
            stats[["method", "comparator", "wins", "ties", "losses", "mean_diff", "wilcoxon_p"]].to_csv(out / "table_8_win_tie_loss.csv", index=False)
    if not downstream.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = downstream.groupby(["method", "budget"])["auroc"].mean().reset_index()
        for method, sub in plot_df.groupby("method"):
            x = sub["budget"].str.replace("budget_", "", regex=False).astype(int)
            ax.plot(x, sub["auroc"], marker="o", label=method)
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figure_1_compression_curves.png", dpi=180)
        plt.close(fig)
    if not downstream.empty and not fidelity.empty:
        perf = downstream.groupby(["dataset", "method", "budget", "seed"])["auroc"].mean().reset_index()
        fid = fidelity.groupby(["dataset", "method", "budget", "seed"])["gain_rank_correlation"].mean().reset_index()
        merged = perf.merge(fid, on=["dataset", "method", "budget", "seed"], how="inner")
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            for method, sub in merged.groupby("method"):
                ax.scatter(sub["gain_rank_correlation"], sub["auroc"], s=14, alpha=0.55, label=method)
            ax.set_xlabel("Gain-rank correlation")
            ax.set_ylabel("AUROC")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(out / "figure_2_split_fidelity_vs_accuracy.png", dpi=180)
            plt.close(fig)
    ab_path = out / "table_5_ablation.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        if not ab.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ab.sort_values("downstream_score").plot.barh(x="variant", y="downstream_score", ax=ax, legend=False)
            ax.set_xlabel("Mean AUROC")
            ax.set_ylabel("")
            fig.tight_layout()
            fig.savefig(out / "figure_3_ablation_barplot.png", dpi=180)
            plt.close(fig)
    print(f"Wrote aggregate outputs to {out}.")


if __name__ == "__main__":
    main()
