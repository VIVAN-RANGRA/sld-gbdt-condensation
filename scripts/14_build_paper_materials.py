from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PRIMARY_METHODS = [
    "full_data",
    "gain_path_refined",
    "gain_path_safeguarded",
    "gain_herding",
    "gain_path_herding",
    "gain_path_prior_weighted",
    "herding",
    "random",
    "distribution_matching",
    "k_center",
    "gaindistill",
]

METHOD_LABELS = {
    "full_data": "Full data",
    "gain_path_refined": "Gain-path refined",
    "gain_path_safeguarded": "Gain-path safeguarded",
    "gain_herding": "Gain sketch",
    "gain_path_herding": "Gain-path sketch",
    "gain_path_prior_weighted": "Gain-path prior weighted",
    "herding": "Herding",
    "random": "Random",
    "distribution_matching": "Distribution matching",
    "k_center": "K-center",
    "gaindistill": "Root-only synthetic",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def as_percent(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def p_value_text(x: float) -> str:
    if pd.isna(x):
        return ""
    if float(x) < 1e-3:
        return f"{float(x):.2e}"
    return f"{float(x):.4f}"


def md_table(df: pd.DataFrame, max_rows: int | None = None, float_digits: int = 4) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{float_digits}f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_dataset_inventory(processed_dir: Path, focused: set[str]) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted(processed_dir.iterdir()):
        arr_path = ds_dir / "arrays.npz"
        meta_path = ds_dir / "metadata.json"
        if not arr_path.exists() or not meta_path.exists():
            continue
        arr = np.load(arr_path)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        y_train = arr["y_train"]
        y_val = arr["y_val"]
        y_test = arr["y_test"]
        rows.append(
            {
                "dataset": ds_dir.name,
                "in_focused_suite": ds_dir.name in focused,
                "source": meta.get("source", "unknown"),
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "n_total": int(len(y_train) + len(y_val) + len(y_test)),
                "n_features": int(arr["X_train"].shape[1]),
                "positive_rate_train": float(np.mean(y_train)),
                "positive_rate_val": float(np.mean(y_val)),
                "positive_rate_test": float(np.mean(y_test)),
            }
        )
    return pd.DataFrame(rows)


def build_tables(
    tables_dir: Path,
    processed_dir: Path,
    out_tables: Path,
    appendix_dir: Path,
) -> dict[str, pd.DataFrame]:
    focused = set(FOCUSED_DATASETS)
    downstream = pd.read_csv(tables_dir / "all_downstream_metrics.csv")
    fidelity = pd.read_csv(tables_dir / "all_split_fidelity_metrics.csv")
    complete = pd.read_csv(tables_dir / "focused_complete_performance.csv")
    pairwise = pd.read_csv(tables_dir / "focused_pairwise_stats.csv")
    budget = pd.read_csv(tables_dir / "focused_budget_means.csv")
    learner = pd.read_csv(tables_dir / "focused_learner_means.csv")
    dataset_means = pd.read_csv(tables_dir / "focused_dataset_means.csv")
    focused_fidelity = pd.read_csv(tables_dir / "focused_fidelity.csv")
    coverage = pd.read_csv(tables_dir / "focused_coverage.csv")

    complete = complete[complete["method"].isin(PRIMARY_METHODS)].copy()
    complete["method_label"] = complete["method"].map(method_label)
    complete = complete.sort_values(["auroc_mean", "avg_rank"], ascending=[False, True])

    pairwise = pairwise[pairwise["method"].isin(PRIMARY_METHODS)].copy()
    pairwise["method_label"] = pairwise["method"].map(method_label)
    pairwise["comparator_label"] = pairwise["comparator"].map(method_label)
    pairwise = pairwise.sort_values(["comparator", "mean_diff"], ascending=[True, False])

    for df in [budget, learner, dataset_means, focused_fidelity, coverage]:
        if "method" in df.columns:
            df["method_label"] = df["method"].map(method_label)

    inventory = load_dataset_inventory(processed_dir, focused)
    inventory.to_csv(out_tables / "dataset_inventory.csv", index=False)
    inventory[inventory["in_focused_suite"]].to_csv(out_tables / "focused_dataset_inventory.csv", index=False)

    broad_coverage = (
        downstream.groupby("method")
        .agg(
            n_rows=("auroc", "count"),
            datasets=("dataset", "nunique"),
            budgets=("budget", "nunique"),
            seeds=("seed", "nunique"),
            learners=("learner", "nunique"),
            auroc_mean=("auroc", "mean"),
        )
        .reset_index()
        .sort_values(["datasets", "n_rows", "auroc_mean"], ascending=[False, False, False])
    )
    broad_coverage.to_csv(out_tables / "broad_result_coverage.csv", index=False)

    broad_complete_like = downstream[
        downstream["dataset"].isin(FOCUSED_DATASETS)
        & downstream["budget"].isin(["budget_25", "budget_50"])
        & downstream["seed"].isin(["seed_0", "seed_1", "seed_2"])
        & downstream["method"].isin(PRIMARY_METHODS)
    ].copy()
    broad_complete_like.to_csv(appendix_dir / "focused_cell_level_downstream.csv", index=False)

    complete.to_csv(out_tables / "main_performance.csv", index=False)
    pairwise.to_csv(out_tables / "main_pairwise_wilcoxon.csv", index=False)
    budget.to_csv(out_tables / "main_budget_means.csv", index=False)
    learner.to_csv(out_tables / "main_transfer_by_learner.csv", index=False)
    dataset_means.to_csv(out_tables / "main_dataset_means.csv", index=False)
    focused_fidelity.to_csv(out_tables / "main_split_fidelity.csv", index=False)
    coverage.to_csv(out_tables / "focused_method_coverage.csv", index=False)

    ablation_path = tables_dir / "table_5_ablation.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation.to_csv(out_tables / "appendix_ablation_summary.csv", index=False)
    else:
        ablation = pd.DataFrame()

    for src_name in [
        "all_downstream_metrics.csv",
        "all_split_fidelity_metrics.csv",
        "table_1_main_performance.csv",
        "table_2_split_fidelity.csv",
        "table_3_low_budget.csv",
        "table_4_transfer.csv",
        "table_5_ablation.csv",
        "table_6_runtime.csv",
        "table_7_statistical_tests.csv",
        "table_8_win_tie_loss.csv",
    ]:
        src = tables_dir / src_name
        if src.exists():
            shutil.copy2(src, appendix_dir / src_name)

    return {
        "downstream": downstream,
        "fidelity": fidelity,
        "complete": complete,
        "pairwise": pairwise,
        "budget": budget,
        "learner": learner,
        "dataset_means": dataset_means,
        "focused_fidelity": focused_fidelity,
        "coverage": coverage,
        "inventory": inventory,
        "broad_coverage": broad_coverage,
        "ablation": ablation,
    }


def build_figures(dfs: dict[str, pd.DataFrame], out_figs: Path) -> None:
    complete = dfs["complete"].copy()
    complete["label"] = complete["method"].map(method_label)
    plot_complete = complete.sort_values("auroc_mean", ascending=True)
    colors = ["#737373" if m == "full_data" else "#2A6FBB" if "gain" in m else "#B35C32" for m in plot_complete["method"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(plot_complete["label"], plot_complete["auroc_mean"], color=colors)
    ax.set_xlabel("Mean AUROC")
    ax.set_xlim(max(0.45, plot_complete["auroc_mean"].min() - 0.03), min(0.91, plot_complete["auroc_mean"].max() + 0.02))
    ax.set_title("Focused Binary Suite: Complete-Method Performance")
    save_fig(fig, out_figs / "main_mean_auroc.png")

    budget = dfs["budget"].copy()
    if "method_label" not in budget.columns:
        budget["method_label"] = budget["method"].map(method_label)
    long_budget = budget.melt(id_vars=["method", "method_label"], value_vars=[c for c in budget.columns if c.startswith("budget_")], var_name="budget", value_name="auroc")
    long_budget = long_budget[long_budget["method"].isin(PRIMARY_METHODS)]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method in ["full_data", "gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]:
        sub = long_budget[long_budget["method"] == method].copy()
        if sub.empty:
            continue
        x = sub["budget"].str.replace("budget_", "", regex=False).astype(int)
        ax.plot(x, sub["auroc"], marker="o", linewidth=2.0, label=method_label(method))
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Low-Budget Compression Curves")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, out_figs / "main_budget_curves.png")

    pairwise = dfs["pairwise"].copy()
    key = pairwise[
        (pairwise["method"].isin(["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding"]))
        & (pairwise["comparator"].isin(["random", "herding"]))
    ].copy()
    if not key.empty:
        key["label"] = key["method"].map(method_label) + " vs " + key["comparator"].map(method_label)
        key = key.sort_values("mean_diff", ascending=True)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.barh(key["label"], key["mean_diff"], color="#3B7A57")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Paired mean AUROC improvement")
        ax.set_title("Paired Improvements Over Generic Coresets")
        save_fig(fig, out_figs / "main_pairwise_improvements.png")

        key2 = key.sort_values("wins", ascending=True)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        left = np.zeros(len(key2))
        ax.barh(key2["label"], key2["wins"], left=left, label="Wins", color="#2A6FBB")
        left += key2["wins"].to_numpy()
        ax.barh(key2["label"], key2["ties"], left=left, label="Ties", color="#BDBDBD")
        left += key2["ties"].to_numpy()
        ax.barh(key2["label"], key2["losses"], left=left, label="Losses", color="#B35C32")
        ax.set_xlabel("Paired evaluation cells")
        ax.set_title("Win/Tie/Loss Counts")
        ax.legend(fontsize=8, ncol=3)
        save_fig(fig, out_figs / "main_win_tie_loss.png")

    fidelity = dfs["focused_fidelity"].copy()
    fidelity = fidelity[fidelity["method"].isin(PRIMARY_METHODS)].copy()
    fidelity["label"] = fidelity["method"].map(method_label)
    fidelity = fidelity.sort_values("gain_rank_correlation", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(fidelity["label"], fidelity["gain_rank_correlation"], color="#6B5B95")
    ax.set_xlabel("Gain-rank correlation")
    ax.set_title("Split-Gain Landscape Fidelity")
    save_fig(fig, out_figs / "main_split_fidelity.png")

    learner = dfs["learner"].copy()
    learner = learner[learner["method"].isin(["full_data", "gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"])].copy()
    learner = learner.set_index("method")[[c for c in learner.columns if c in ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]]]
    learner = learner.rename(index=method_label)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(learner.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=float(np.nanmin(learner.to_numpy())), vmax=float(np.nanmax(learner.to_numpy())))
    ax.set_xticks(np.arange(len(learner.columns)))
    ax.set_xticklabels([c.replace("_", " ").title() for c in learner.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(learner.index)))
    ax.set_yticklabels(learner.index)
    ax.set_title("Cross-Model Transfer")
    fig.colorbar(im, ax=ax, label="Mean AUROC")
    save_fig(fig, out_figs / "main_transfer_heatmap.png")

    dataset_means = dfs["dataset_means"].copy()
    keep_methods = ["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]
    cols = [m for m in keep_methods if m in dataset_means.columns]
    heat = dataset_means.set_index("dataset")[cols].rename(columns=method_label)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=float(np.nanmin(heat.to_numpy())), vmax=float(np.nanmax(heat.to_numpy())))
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Dataset-Level Mean AUROC")
    fig.colorbar(im, ax=ax, label="Mean AUROC")
    save_fig(fig, out_figs / "main_dataset_heatmap.png")


def build_enhanced_figures(dfs: dict[str, pd.DataFrame], out_figs: Path) -> None:
    downstream = dfs["downstream"].copy()
    fidelity = dfs["fidelity"].copy()
    primary = [
        "full_data",
        "gain_path_refined",
        "gain_path_safeguarded",
        "gain_herding",
        "gain_path_herding",
        "herding",
        "random",
    ]
    focused_mask = (
        downstream["dataset"].isin(FOCUSED_DATASETS)
        & downstream["budget"].isin(["budget_25", "budget_50"])
        & downstream["seed"].isin(["seed_0", "seed_1", "seed_2"])
        & downstream["method"].isin(primary)
    )
    df = downstream[focused_mask].copy()
    df["method_label"] = df["method"].map(method_label)
    method_colors = {
        "full_data": "#333333",
        "gain_path_refined": "#1B6CA8",
        "gain_path_safeguarded": "#2A9D8F",
        "gain_herding": "#5E60CE",
        "gain_path_herding": "#577590",
        "herding": "#C65D32",
        "random": "#8D6E63",
    }

    # Figure A: rank profile as a dot/slope plot, not a bar chart.
    piv = df.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    ranks = piv.rank(axis=1, ascending=False, method="average")
    rank_mean = ranks.mean().sort_values()
    rank_se = ranks.std() / np.sqrt(ranks.count())
    methods_ranked = [m for m in rank_mean.index if m in primary]
    y = np.arange(len(methods_ranked))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, method in enumerate(methods_ranked):
        ax.errorbar(
            rank_mean[method],
            i,
            xerr=1.96 * rank_se[method],
            fmt="o",
            color=method_colors.get(method, "#555555"),
            capsize=3,
            markersize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in methods_ranked])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank, lower is better")
    ax.set_title("Rank Profile Across 240 Paired Evaluation Cells")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, out_figs / "paper_rank_profile.png")

    # Figure B: budget curves with 95% CI over paired cells.
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method in ["full_data", "gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("budget")
            .agg(mean=("auroc", "mean"), std=("auroc", "std"), n=("auroc", "count"))
            .reset_index()
        )
        agg["x"] = agg["budget"].str.replace("budget_", "", regex=False).astype(int)
        agg = agg.sort_values("x")
        ci = 1.96 * agg["std"] / np.sqrt(agg["n"])
        ax.plot(agg["x"], agg["mean"], marker="o", linewidth=2.2, label=method_label(method), color=method_colors.get(method))
        ax.fill_between(agg["x"], agg["mean"] - ci, agg["mean"] + ci, color=method_colors.get(method), alpha=0.12, linewidth=0)
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Compression Curves With 95% Confidence Bands")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, out_figs / "paper_budget_curves_ci.png")

    # Figure C: paired AUROC delta distributions against random and herding.
    pivot = df.groupby(["dataset", "budget", "learner", "seed", "method"])["auroc"].mean().unstack()
    violin_rows = []
    for method in ["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding"]:
        for comp in ["random", "herding"]:
            if method not in pivot.columns or comp not in pivot.columns:
                continue
            diff = (pivot[method] - pivot[comp]).dropna()
            for value in diff:
                violin_rows.append({"comparison": f"{method_label(method)}\nvs {method_label(comp)}", "diff": float(value)})
    vdf = pd.DataFrame(violin_rows)
    if not vdf.empty:
        comparisons = list(dict.fromkeys(vdf["comparison"].tolist()))
        data = [vdf.loc[vdf["comparison"] == c, "diff"].to_numpy() for c in comparisons]
        fig, ax = plt.subplots(figsize=(10.0, 5.2))
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#4C78A8")
            body.set_alpha(0.35)
            body.set_edgecolor("#2F4B7C")
        parts["cmedians"].set_color("#111111")
        rng = np.random.default_rng(0)
        for i, values in enumerate(data, start=1):
            sample = values if len(values) <= 160 else rng.choice(values, size=160, replace=False)
            jitter = rng.normal(0.0, 0.035, size=len(sample))
            ax.scatter(np.full(len(sample), i) + jitter, sample, s=8, color="#1B1B1B", alpha=0.28, linewidths=0)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(np.arange(1, len(comparisons) + 1))
        ax.set_xticklabels(comparisons, rotation=30, ha="right")
        ax.set_ylabel("Paired AUROC difference")
        ax.set_title("Distribution of Paired Improvements")
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, out_figs / "paper_paired_delta_violins.png")

    # Figure D: method-by-method mean delta matrix.
    matrix_methods = ["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]
    mat = np.zeros((len(matrix_methods), len(matrix_methods)), dtype=float)
    for i, a in enumerate(matrix_methods):
        for j, b in enumerate(matrix_methods):
            if a == b:
                mat[i, j] = 0.0
            elif a in pivot.columns and b in pivot.columns:
                pair = pivot[[a, b]].dropna()
                mat[i, j] = float((pair[a] - pair[b]).mean()) if not pair.empty else np.nan
            else:
                mat[i, j] = np.nan
    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix_methods)))
    ax.set_xticklabels([method_label(m) for m in matrix_methods], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix_methods)))
    ax.set_yticklabels([method_label(m) for m in matrix_methods])
    ax.set_title("Pairwise Mean AUROC Difference")
    for i in range(len(matrix_methods)):
        for j in range(len(matrix_methods)):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Row method minus column method")
    save_fig(fig, out_figs / "paper_pairwise_delta_matrix.png")

    # Figure E: split-fidelity versus downstream AUROC scatter.
    perf = (
        df.groupby(["dataset", "budget", "seed", "method"])
        .agg(auroc=("auroc", "mean"))
        .reset_index()
    )
    fid = fidelity[
        fidelity["dataset"].isin(FOCUSED_DATASETS)
        & fidelity["budget"].isin(["budget_25", "budget_50"])
        & fidelity["seed"].isin(["seed_0", "seed_1", "seed_2"])
        & fidelity["method"].isin(primary)
    ].copy()
    merged = perf.merge(fid[["dataset", "budget", "seed", "method", "gain_rank_correlation"]], on=["dataset", "budget", "seed", "method"], how="inner")
    if not merged.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        for method in ["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]:
            sub = merged[merged["method"] == method]
            if sub.empty:
                continue
            ax.scatter(
                sub["gain_rank_correlation"],
                sub["auroc"],
                s=36,
                alpha=0.62,
                label=method_label(method),
                color=method_colors.get(method),
                edgecolors="none",
            )
        x = merged["gain_rank_correlation"].to_numpy()
        yvals = merged["auroc"].to_numpy()
        if len(x) > 2 and np.nanstd(x) > 1e-8:
            coef = np.polyfit(x, yvals, deg=1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#111111", linewidth=1.4, linestyle="--", label="Linear trend")
        ax.set_xlabel("Split-gain rank correlation")
        ax.set_ylabel("Mean downstream AUROC")
        ax.set_title("Fidelity-Performance Relationship")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_fig(fig, out_figs / "paper_fidelity_performance_scatter.png")

    # Figure F: transfer profiles as parallel line plot.
    learner_means = dfs["learner"].copy()
    learners = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(learners))
    for method in ["full_data", "gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding", "herding", "random"]:
        sub = learner_means[learner_means["method"] == method]
        if sub.empty:
            continue
        vals = sub.iloc[0][learners].to_numpy(dtype=float)
        ax.plot(x, vals, marker="o", linewidth=2.0, label=method_label(method), color=method_colors.get(method))
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", " ").title() for l in learners])
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Cross-Model Transfer Profiles")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, out_figs / "paper_transfer_profiles.png")

    # Figure G: dataset slope plot from random to herding to gain-path refined.
    dataset_means = dfs["dataset_means"].copy()
    needed = ["dataset", "random", "herding", "gain_path_refined"]
    if all(c in dataset_means.columns for c in needed):
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        x = np.array([0, 1, 2])
        labels = ["Random", "Herding", "Gain-path refined"]
        for _, row in dataset_means.iterrows():
            vals = [row["random"], row["herding"], row["gain_path_refined"]]
            ax.plot(x, vals, marker="o", linewidth=1.6, alpha=0.72)
            ax.text(2.04, vals[-1], str(row["dataset"]), va="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean AUROC")
        ax.set_title("Dataset-Level Improvement Trajectories")
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, out_figs / "paper_dataset_slopegraph.png")


def build_markdown(dfs: dict[str, pd.DataFrame], out_root: Path) -> None:
    notes = ensure_dir(out_root / "notes")
    figs_rel = "figures"
    tables_rel = "tables"
    appendix_rel = "appendix"

    inventory = dfs["inventory"]
    focused_inventory = inventory[inventory["in_focused_suite"]].copy()
    complete = dfs["complete"]
    pairwise = dfs["pairwise"]
    budget = dfs["budget"]
    learner = dfs["learner"]
    fidelity = dfs["focused_fidelity"]
    coverage = dfs["coverage"]
    broad_coverage = dfs["broad_coverage"]

    top_complete = complete[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
        columns={
            "method_label": "Method",
            "auroc_mean": "Mean AUROC",
            "auroc_std": "Std",
            "avg_rank": "Avg. Rank",
            "n_cells": "Cells",
        }
    )
    key_pairs = pairwise[
        (pairwise["method"].isin(["gain_path_refined", "gain_path_safeguarded", "gain_herding", "gain_path_herding"]))
        & (pairwise["comparator"].isin(["random", "herding", "gain_herding", "gain_path_herding"]))
    ][["method_label", "comparator_label", "n_pairs", "mean_diff", "median_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
        columns={
            "method_label": "Method",
            "comparator_label": "Comparator",
            "n_pairs": "Pairs",
            "mean_diff": "Mean Diff",
            "median_diff": "Median Diff",
            "wins": "Wins",
            "ties": "Ties",
            "losses": "Losses",
            "wilcoxon_p": "Wilcoxon p",
        }
    )
    key_pairs = key_pairs.sort_values(["Comparator", "Mean Diff"], ascending=[True, False])
    key_pairs["Wilcoxon p"] = key_pairs["Wilcoxon p"].map(p_value_text)

    dataset_md = focused_inventory[
        ["dataset", "source", "n_train", "n_val", "n_test", "n_total", "n_features", "positive_rate_train"]
    ].rename(
        columns={
            "dataset": "Dataset",
            "source": "Source",
            "n_train": "Train",
            "n_val": "Val",
            "n_test": "Test",
            "n_total": "Total",
            "n_features": "Features",
            "positive_rate_train": "Train Positive Rate",
        }
    )

    budget_display = budget.rename(columns={"method_label": "Method"}).drop(columns=[c for c in ["method"] if c in budget.columns])
    budget_display = budget_display[["Method"] + [c for c in budget_display.columns if c != "Method"]]
    learner_display = learner.rename(columns={"method_label": "Method"}).drop(columns=[c for c in ["method"] if c in learner.columns])
    learner_display = learner_display[["Method"] + [c for c in learner_display.columns if c != "Method"]]
    fidelity_display = fidelity.rename(columns={"method_label": "Method"}).drop(columns=[c for c in ["method"] if c in fidelity.columns])
    fidelity_display = fidelity_display[["Method"] + [c for c in fidelity_display.columns if c != "Method"]]

    final = f"""# Final Results Package

This folder is the paper-writing package for the current Split-Gain Landscape Condensation direction. It separates the primary focused benchmark from appendix-scale historical runs.

## Main Claim Supported By Current Results

Tree-gain/path-aware condensation gives substantially better compact tabular training sets than generic random selection and raw-feature herding on the focused binary benchmark.

The strongest complete method is **Gain-path refined**. It is the recommended primary method for the paper. The synthetic root-only method is not competitive and should be presented only as an ablation or early negative result.

## Focused Benchmark

- Datasets: {len(focused_inventory)} binary tabular datasets.
- Budgets: 25 and 50 samples per class.
- Seeds: 0, 1, 2.
- Downstream learners: XGBoost, LightGBM, CatBoost, Random Forest, MLP.
- Complete-method paired cells: 240 per method.

{md_table(dataset_md, float_digits=3)}

## Table 1: Main Performance

{md_table(top_complete, float_digits=4)}

Source CSV: `{tables_rel}/main_performance.csv`

## Table 2: Paired Statistical Tests

{md_table(key_pairs, float_digits=4)}

Source CSV: `{tables_rel}/main_pairwise_wilcoxon.csv`

## Table 3: Budget Sensitivity

{md_table(budget_display, float_digits=4)}

Source CSV: `{tables_rel}/main_budget_means.csv`

## Table 4: Cross-Model Transfer

{md_table(learner_display, float_digits=4)}

Source CSV: `{tables_rel}/main_transfer_by_learner.csv`

## Table 5: Split-Gain Fidelity

{md_table(fidelity_display, float_digits=4)}

Source CSV: `{tables_rel}/main_split_fidelity.csv`

## Preferred Paper Figures

![Rank profile](figures/paper_rank_profile.png)

![Budget curves with confidence intervals](figures/paper_budget_curves_ci.png)

![Paired delta distributions](figures/paper_paired_delta_violins.png)

![Pairwise delta matrix](figures/paper_pairwise_delta_matrix.png)

![Fidelity-performance scatter](figures/paper_fidelity_performance_scatter.png)

![Transfer profiles](figures/paper_transfer_profiles.png)

![Dataset slopegraph](figures/paper_dataset_slopegraph.png)

## Diagnostic Figures

![Main mean AUROC]({figs_rel}/main_mean_auroc.png)

![Budget curves]({figs_rel}/main_budget_curves.png)

![Paired improvements]({figs_rel}/main_pairwise_improvements.png)

![Win tie loss]({figs_rel}/main_win_tie_loss.png)

![Split fidelity]({figs_rel}/main_split_fidelity.png)

![Transfer heatmap]({figs_rel}/main_transfer_heatmap.png)

![Dataset heatmap]({figs_rel}/main_dataset_heatmap.png)

## Appendix Tables

- `{appendix_rel}/all_downstream_metrics.csv`: all collected downstream metric rows.
- `{appendix_rel}/all_split_fidelity_metrics.csv`: all collected split-fidelity rows.
- `{appendix_rel}/table_1_main_performance.csv`: aggregate performance by dataset, method, budget, learner.
- `{appendix_rel}/table_2_split_fidelity.csv`: aggregate split-fidelity metrics.
- `{appendix_rel}/table_5_ablation.csv`: ablation summary where available.
- `{appendix_rel}/table_7_statistical_tests.csv`: broad paired Wilcoxon output.
- `{appendix_rel}/table_8_win_tie_loss.csv`: broad win/tie/loss output.

## Paper Positioning

Use the title direction **Split-Gain Landscape Condensation for Gradient-Boosted Tabular Models** unless the final paper centers the synthetic optimizer. The present empirical winner is a gain/path-aware condensed coreset, not a fully synthetic distilled dataset.
"""
    (out_root / "FINAL_RESULTS.md").write_text(final, encoding="utf-8")

    data_runs = f"""# Data And Runs

## Processed Data Inventory

The workspace currently contains {len(inventory)} processed binary datasets, including the focused suite and extra appendix-scale datasets. Smoke data is present for engineering checks but should not be used in paper claims.

Focused paper suite:

{md_table(dataset_md, float_digits=3)}

Full processed-data inventory is in `{tables_rel}/dataset_inventory.csv`.

## Focused Evaluation Design

- Budgets: 25 and 50 samples per class.
- Seeds: 0, 1, 2.
- Learners: XGBoost, LightGBM, CatBoost, Random Forest, MLP.
- Metric: AUROC, with accuracy and log loss retained in CSVs.
- Main statistical unit: dataset x budget x seed x learner.

## Method Coverage

{md_table(coverage[["method", "n_cells", "datasets", "budgets", "seeds", "learners", "complete", "auroc_mean"]].rename(columns={"method": "Method"}), float_digits=4)}

## Broad Result Coverage

The appendix includes historical and exploratory runs. These are useful for robustness checks and ablations, but the primary paper tables should use complete methods only.

{md_table(broad_coverage.head(20).rename(columns={"method": "Method"}), float_digits=4)}
"""
    (notes / "DATA_AND_RUNS.md").write_text(data_runs, encoding="utf-8")

    interpretation = f"""# Results Interpretation

## What Holds Up

The clean focused benchmark supports the claim that split-gain/path-aware condensation improves compact tabular training sets.

The primary method, **Gain-path refined**, improves over:

- Random selection by roughly 0.032 AUROC on average.
- Raw-feature herding by roughly 0.020 AUROC on average.
- Gain-only sketching by roughly 0.006 AUROC on average.

The gains over random and herding are statistically strong under paired Wilcoxon tests. The gain over `gain_path_herding` is positive on mean AUROC but not yet statistically decisive.

## What Does Not Hold Up Yet

The phrase "compact synthetic dataset" should be avoided for the main method. The current empirical winner is selected-row condensation with local sketch refinement. The synthetic optimizer is implemented and useful as a math ablation, but it should not be the headline result unless it improves.

## Recommended Claims

Safe claims:

- Tree-mechanism-aligned condensation outperforms generic coresets on binary tabular benchmarks.
- Preserving split-gain/path statistics improves both downstream AUROC and split-gain fidelity.
- The method transfers beyond the teacher family to LightGBM, CatBoost, Random Forest, and MLP.

Claims to avoid:

- Fully synthetic data beats all baselines.
- The method is proven for multiclass tasks.
- Every math loss term is necessary.

## Main Quantitative Evidence

{md_table(top_complete, float_digits=4)}

## Key Pairwise Tests

{md_table(key_pairs, float_digits=4)}
"""
    (notes / "RESULTS_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    writing_notes = f"""# Writing Notes

## Recommended Paper Story

Most dataset distillation objectives were designed around neural-network training signals. Gradient-boosted tabular models are built through feature-threshold split gains and leaf-wise first- and second-order statistics. The paper should argue that condensation for tree-dominated tabular learning should preserve those training mechanisms directly.

## Suggested Title

Split-Gain Landscape Condensation for Gradient-Boosted Tabular Models

Use "Distillation" only if the final paper centers synthetic optimization. The current results support "Condensation" more strongly.

## Suggested Contributions

1. A split-gain landscape view of tabular data condensation for gradient-boosted trees.
2. A gain/path-aware condensed coreset method that preserves candidate split behavior, gradient/Hessian mass, teacher confidence, and tree-path occupancy.
3. A focused binary benchmark across datasets, budgets, seeds, and downstream learners.
4. Mechanism analysis through split-fidelity, ablations, and paired statistical tests.

## Suggested Main Tables

- Table 1: Main complete-method AUROC and rank.
- Table 2: Paired Wilcoxon and win/tie/loss against random and herding.
- Table 3: Budget sensitivity.
- Table 4: Cross-model transfer.
- Table 5: Split-gain fidelity.
- Appendix: dataset inventory, method coverage, all downstream metrics, all split-fidelity metrics, ablations.

## Reviewer Risk Notes

- Add stronger 2026-era tabular condensation baselines if time permits.
- Keep binary scope explicit.
- Be transparent that the synthetic variant is not the current empirical winner.
- Avoid overclaiming fidelity as a complete explanation: the fidelity improvement is real but not perfectly aligned with downstream AUROC.
"""
    (notes / "WRITING_NOTES.md").write_text(writing_notes, encoding="utf-8")

    readme = f"""# Paper Materials

This folder contains the current paper-writing materials.

- `FINAL_RESULTS.md`: primary tables and figures for the paper.
- `notes/DATA_AND_RUNS.md`: dataset inventory, run design, and coverage.
- `notes/RESULTS_INTERPRETATION.md`: what the results support and what they do not.
- `notes/WRITING_NOTES.md`: paper framing, contributions, and reviewer risks.
- `tables/`: clean CSV tables for main text and appendix.
- `figures/`: paper-ready PNG figures.
- `appendix/`: larger raw/aggregate CSVs for appendix checks.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", default="paper_tables")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out", default="paper_materials")
    args = ap.parse_args()

    out_root = ensure_dir(args.out)
    out_tables = ensure_dir(out_root / "tables")
    out_figs = ensure_dir(out_root / "figures")
    appendix_dir = ensure_dir(out_root / "appendix")

    dfs = build_tables(Path(args.tables_dir), Path(args.processed_dir), out_tables, appendix_dir)
    build_figures(dfs, out_figs)
    build_enhanced_figures(dfs, out_figs)
    build_markdown(dfs, out_root)
    print(f"Wrote paper materials to {out_root}.")


if __name__ == "__main__":
    main()
