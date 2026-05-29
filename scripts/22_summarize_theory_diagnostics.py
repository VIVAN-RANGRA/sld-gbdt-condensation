from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gaindistill.baselines import _histdistill_matrices
from gaindistill.data import load_processed
from gaindistill.landscape import load_landscape
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config


KEY_METHODS = [
    "histdistill_density",
    "histdistill_refined",
    "gain_path_refined",
    "herding",
    "random",
]
SELECTOR_METHODS = [
    "histdistill_greedy",
    "histdistill_importance",
    "histdistill_refined",
    "histdistill_density",
    "herding",
    "random",
]
LEARNERS = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
LABELS = {
    "histdistill_density": "HistDistill-Density",
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_greedy": "HistDistill-Greedy",
    "histdistill_importance": "ImportanceSample",
    "gain_path_refined": "Legacy gain-path refined",
    "herding": "Herding",
    "random": "Random",
    "full_data": "Full data",
}


def label(method: str) -> str:
    return LABELS.get(method, method.replace("_", " "))


def budget_int(value: str) -> int:
    return int(str(value).replace("budget_", ""))


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


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def dataset_sizes(processed_dir: Path) -> pd.DataFrame:
    rows = []
    for ds_dir in list_dataset_dirs(processed_dir):
        if ds_dir.name == "smoke_binary":
            continue
        bundle = load_processed(ds_dir)
        rows.append(
            {
                "dataset": ds_dir.name,
                "n_train": int(len(bundle.y_train)),
                "n_features": int(bundle.X_train.shape[1]),
                "positive_rate": float(np.mean(bundle.y_train)),
            }
        )
    return pd.DataFrame(rows)


def build_margin_distribution(cfg: dict, seeds: set[str]) -> pd.DataFrame:
    rows = []
    root = Path(cfg["project"]["results_dir"]) / "landscapes"
    for ds_path in root.iterdir() if root.exists() else []:
        if not ds_path.is_dir() or ds_path.name == "smoke_binary":
            continue
        for seed_path in ds_path.iterdir():
            if not seed_path.is_dir() or seed_path.name not in seeds:
                continue
            if not (seed_path / "landscape.npz").exists():
                continue
            landscape = load_landscape(seed_path)
            states = [("root", 0, np.asarray(landscape.gains, dtype=np.float64))]
            for probe in landscape.probes:
                states.append((f"probe_d{probe.depth}", int(probe.depth), np.asarray(probe.gains, dtype=np.float64)))
            for name, depth, gains in states:
                if len(gains) < 2:
                    continue
                order = np.argsort(-gains)
                rows.append(
                    {
                        "dataset": ds_path.name,
                        "seed": seed_path.name,
                        "state": name,
                        "depth": int(depth),
                        "margin": float(gains[order[0]] - gains[order[1]]),
                        "best_gain": float(gains[order[0]]),
                        "num_candidates": int(len(gains)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_sld_law(downstream: pd.DataFrame, fidelity: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = (
        downstream[downstream["method"].isin(methods)]
        .groupby(["dataset", "budget", "seed", "method"])["auroc"]
        .mean()
        .reset_index()
    )
    fid = fidelity[fidelity["method"].isin(methods)].copy()
    cols = [
        "dataset",
        "budget",
        "seed",
        "method",
        "sld_inf_norm",
        "sld_root_margin_satisfied",
        "sld_root_agreement_direct",
        "sld_margin_satisfied_rate",
        "sld_rank_correlation_direct",
    ]
    merged = perf.merge(fid[[c for c in cols if c in fid.columns]], on=["dataset", "budget", "seed", "method"], how="inner")
    merged = merged.dropna(subset=["sld_inf_norm", "auroc"])
    rows = []
    for method, sub in merged.groupby("method"):
        corr = spearmanr(sub["sld_inf_norm"], sub["auroc"]).correlation if len(sub) > 2 else np.nan
        within_corrs = []
        for _, group in sub.groupby(["dataset", "seed"]):
            if len(group) < 3 or group["sld_inf_norm"].nunique() < 2 or group["auroc"].nunique() < 2:
                continue
            wcorr = spearmanr(group["sld_inf_norm"], group["auroc"]).correlation
            if not pd.isna(wcorr):
                within_corrs.append(float(wcorr))
        ok = sub[sub.get("sld_root_margin_satisfied", 0.0).astype(float) > 0.5]
        rows.append(
            {
                "method": method,
                "method_label": label(method),
                "n_cells": int(len(sub)),
                "pooled_spearman_sld_vs_auroc": float(0.0 if pd.isna(corr) else corr),
                "within_dataset_spearman_sld_vs_auroc": float(np.mean(within_corrs)) if within_corrs else np.nan,
                "mean_sld_inf_norm": float(sub["sld_inf_norm"].mean()),
                "mean_auroc": float(sub["auroc"].mean()),
                "prop1_cells": int(len(ok)),
                "prop1_root_agreement": float(ok["sld_root_agreement_direct"].mean()) if len(ok) else np.nan,
                "all_root_agreement": float(sub["sld_root_agreement_direct"].mean()) if "sld_root_agreement_direct" in sub else np.nan,
            }
        )
    return pd.DataFrame(rows), merged


def summarize_budget_collapse(fidelity: pd.DataFrame, sizes: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fid = fidelity[fidelity["method"].isin(methods)].dropna(subset=["sld_inf_norm"]).copy()
    fid["budget_n"] = fid["budget"].map(budget_int)
    merged = fid.merge(sizes[["dataset", "n_train"]], on="dataset", how="left")
    merged["size_group"] = pd.qcut(merged["n_train"].rank(method="first"), q=3, labels=["small N", "medium N", "large N"])
    rows = []
    for (method, dataset), sub in merged.groupby(["method", "dataset"]):
        by_b = sub.groupby("budget_n")["sld_inf_norm"].mean().reset_index()
        by_b = by_b[(by_b["sld_inf_norm"] > 0) & (by_b["budget_n"] > 0)]
        slope = np.nan
        if len(by_b) >= 3:
            slope = float(np.polyfit(np.log(by_b["budget_n"]), np.log(by_b["sld_inf_norm"]), 1)[0])
        rows.append(
            {
                "method": method,
                "method_label": label(method),
                "dataset": dataset,
                "n_train": int(sub["n_train"].iloc[0]) if pd.notna(sub["n_train"].iloc[0]) else np.nan,
                "loglog_sld_budget_slope": slope,
            }
        )
    slopes = pd.DataFrame(rows)
    by_group = (
        merged.groupby(["method", "size_group", "budget_n"], observed=False)
        .agg(sld_inf_norm=("sld_inf_norm", "mean"), root_agreement=("sld_root_agreement_direct", "mean"))
        .reset_index()
    )
    return slopes, by_group


def summarize_transfer_regret(downstream: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    full = downstream[downstream["method"] == "full_data"].copy()
    ref = (
        full.groupby(["dataset", "seed", "learner"])
        .agg(full_auroc=("auroc", "mean"), full_seconds=("train_seconds", "mean"))
        .reset_index()
    )
    comp = downstream[downstream["method"].isin(methods)].copy()
    merged = comp.merge(ref, on=["dataset", "seed", "learner"], how="inner")
    merged["regret"] = merged["full_auroc"] - merged["auroc"]
    merged["speedup_vs_full"] = merged["full_seconds"] / merged["train_seconds"].clip(lower=1e-9)
    out = (
        merged.groupby(["method", "learner"])
        .agg(
            auroc=("auroc", "mean"),
            full_auroc=("full_auroc", "mean"),
            transfer_regret=("regret", "mean"),
            speedup_vs_full=("speedup_vs_full", "median"),
            n_cells=("auroc", "count"),
        )
        .reset_index()
    )
    out["method_label"] = out["method"].map(label)
    return out


def summarize_runtime(downstream: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    full = (
        downstream[downstream["method"] == "full_data"]
        .groupby(["dataset", "seed", "learner"])
        .agg(full_seconds=("train_seconds", "mean"))
        .reset_index()
    )
    comp = downstream[downstream["method"].isin(methods)].merge(full, on=["dataset", "seed", "learner"], how="left")
    comp["speedup_vs_full"] = comp["full_seconds"] / comp["train_seconds"].clip(lower=1e-9)
    out = (
        comp.groupby(["method", "budget"])
        .agg(
            downstream_seconds_mean=("train_seconds", "mean"),
            downstream_seconds_median=("train_seconds", "median"),
            speedup_vs_full_median=("speedup_vs_full", "median"),
            train_rows=("train_rows", "mean"),
            n_cells=("train_seconds", "count"),
        )
        .reset_index()
    )
    out["budget_n"] = out["budget"].map(budget_int)
    out["method_label"] = out["method"].map(label)
    return out.sort_values(["method", "budget_n"])


def match_selected_indices(X: np.ndarray, y: np.ndarray, X_sel: np.ndarray, y_sel: np.ndarray) -> list[int]:
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for i, (row, cls) in enumerate(zip(X, y)):
        buckets.setdefault((int(cls), np.ascontiguousarray(row).tobytes()), []).append(i)
    used: set[int] = set()
    out: list[int] = []
    for row, cls in zip(X_sel, y_sel):
        key = (int(cls), np.ascontiguousarray(row.astype(X.dtype, copy=False)).tobytes())
        choices = buckets.get(key, [])
        pick = next((i for i in choices if i not in used), None)
        if pick is None and choices:
            pick = choices[0]
        if pick is not None:
            used.add(int(pick))
            out.append(int(pick))
    return out


def summarize_selector_coverage(cfg: dict, budgets: set[str], seeds: set[str], methods: list[str], max_jobs_note: bool = False) -> pd.DataFrame:
    del max_jobs_note
    rows = []
    processed = {p.name: p for p in list_dataset_dirs(cfg["project"]["processed_dir"]) if p.name != "smoke_binary"}
    results = Path(cfg["project"]["results_dir"])
    for dataset, ds_dir in processed.items():
        bundle = load_processed(ds_dir)
        for seed in sorted(seeds):
            seed_n = int(seed.replace("seed_", ""))
            teacher_path = results / "teachers" / dataset / seed / "teacher.joblib"
            landscape_path = results / "landscapes" / dataset / seed
            if not teacher_path.exists() or not (landscape_path / "landscape.npz").exists():
                continue
            teacher = joblib.load(teacher_path)
            landscape = load_landscape(landscape_path)
            coverage, caps, omega, _design, _target = _histdistill_matrices(
                bundle.X_train,
                bundle.y_train,
                bundle.meta,
                teacher,
                landscape,
                include_density=False,
            )
            full_value = float(np.sum(omega * caps))
            for budget in sorted(budgets, key=budget_int):
                budget_n = budget_int(budget)
                singleton = np.minimum(coverage, caps[None, :]) @ omega
                proxy_upper_bound = 0.0
                for cls in np.unique(bundle.y_train):
                    cls_scores = np.sort(singleton[bundle.y_train == cls])[::-1]
                    proxy_upper_bound += float(np.sum(cls_scores[: min(budget_n, len(cls_scores))]))
                proxy_upper_bound = min(float(proxy_upper_bound), full_value)
                for method in methods:
                    distilled_path = results / "distilled" / dataset / method / budget / seed / "distilled.npz"
                    if not distilled_path.exists():
                        continue
                    arr = np.load(distilled_path)
                    idx = match_selected_indices(bundle.X_train, bundle.y_train, arr["X"], arr["y"])
                    if not idx:
                        continue
                    selected_mass = coverage[idx].sum(axis=0)
                    value = float(np.sum(omega * np.minimum(selected_mass, caps)))
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "budget": budget,
                            "method": method,
                            "method_label": label(method),
                            "coverage_value": value,
                            "coverage_ratio": value / max(full_value, 1e-12),
                            "coverage_proxy_upper_bound": proxy_upper_bound,
                            "coverage_to_proxy_upper_bound": value / max(proxy_upper_bound, 1e-12),
                            "matched_rows": int(len(idx)),
                            "seed_n": seed_n,
                        }
                    )
    return pd.DataFrame(rows)


def make_figures(
    sld_merged: pd.DataFrame,
    margins: pd.DataFrame,
    collapse: pd.DataFrame,
    transfer: pd.DataFrame,
    runtime: pd.DataFrame,
    selector: pd.DataFrame,
    out_figs: Path,
) -> None:
    if not sld_merged.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for method in KEY_METHODS:
            sub = sld_merged[sld_merged["method"] == method]
            if sub.empty:
                continue
            ax.scatter(sub["sld_inf_norm"], sub["auroc"], s=24, alpha=0.48, label=label(method))
        ax.set_xscale("log")
        ax.set_xlabel("Normalized SLD infinity")
        ax.set_ylabel("Mean AUROC across learners")
        ax.set_title("SLD law diagnostic")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, out_figs / "icdm_sld_law_scatter.png")

    if not margins.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for depth in sorted(margins["depth"].unique()):
            vals = np.sort(margins.loc[margins["depth"] == depth, "margin"].clip(lower=1e-12).to_numpy())
            if len(vals) == 0:
                continue
            y = np.linspace(0, 1, len(vals), endpoint=True)
            ax.plot(vals, y, linewidth=1.8, label=f"depth {depth}")
        ax.set_xscale("log")
        ax.set_xlabel("Gain margin")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("Teacher margin distribution")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        save_fig(fig, out_figs / "icdm_margin_distribution.png")

    if not collapse.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for (method, group), sub in collapse.groupby(["method", "size_group"], observed=False):
            if method not in ["histdistill_refined", "histdistill_density", "gain_path_refined"]:
                continue
            sub = sub.sort_values("budget_n")
            ax.plot(
                sub["budget_n"],
                sub["sld_inf_norm"],
                marker="o",
                linewidth=1.4,
                label=f"{label(method)} / {group}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("Normalized SLD infinity")
        ax.set_title("Budget collapse by dataset size")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.5, ncol=2)
        save_fig(fig, out_figs / "icdm_budget_collapse_sld.png")

    if not transfer.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        x = np.arange(len(LEARNERS))
        for method in KEY_METHODS:
            sub = transfer[transfer["method"] == method].set_index("learner").reindex(LEARNERS)
            if sub["transfer_regret"].isna().all():
                continue
            ax.plot(x, sub["transfer_regret"], marker="o", linewidth=1.7, label=label(method))
        ax.set_xticks(x)
        ax.set_xticklabels(LEARNERS, rotation=20, ha="right")
        ax.set_ylabel("Full-data AUROC minus distilled AUROC")
        ax.set_title("Cross-learner transfer regret")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, out_figs / "icdm_transfer_regret_profiles.png")

    if not runtime.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for method in KEY_METHODS:
            sub = runtime[runtime["method"] == method].sort_values("budget_n")
            if sub.empty:
                continue
            ax.plot(sub["budget_n"], sub["speedup_vs_full_median"], marker="o", linewidth=1.7, label=label(method))
        ax.set_xscale("log")
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("Median downstream speedup vs full data")
        ax.set_title("Runtime speedup profile")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, out_figs / "icdm_runtime_speedup.png")

    if not selector.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        grouped = selector.groupby(["method", "budget"])["coverage_ratio"].mean().reset_index()
        grouped["budget_n"] = grouped["budget"].map(budget_int)
        for method in SELECTOR_METHODS:
            sub = grouped[grouped["method"] == method].sort_values("budget_n")
            if sub.empty:
                continue
            ax.plot(sub["budget_n"], sub["coverage_ratio"], marker="o", linewidth=1.7, label=label(method))
        ax.set_xlabel("Samples per class")
        ax.set_ylabel("Coverage objective / full cap")
        ax.set_title("Selector coverage objective")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, out_figs / "icdm_selector_coverage.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--budgets", default="10,25,50,100,200")
    ap.add_argument("--selector-budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-selector", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    budgets = {b if b.startswith("budget_") else f"budget_{b}" for b in args.budgets.split(",") if b.strip()}
    selector_budgets = {
        b if b.startswith("budget_") else f"budget_{b}" for b in args.selector_budgets.split(",") if b.strip()
    }
    seeds = {s if s.startswith("seed_") else f"seed_{s}" for s in args.seeds.split(",") if s.strip()}
    tables = Path(args.paper_tables)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")

    downstream = pd.read_csv(tables / "all_downstream_metrics.csv")
    downstream = downstream[
        (downstream["dataset"] != "smoke_binary")
        & downstream["seed"].isin(seeds)
        & downstream["learner"].isin(LEARNERS)
    ].copy()
    fidelity = pd.read_csv(tables / "all_split_fidelity_metrics.csv")
    fidelity = fidelity[
        (fidelity["dataset"] != "smoke_binary")
        & fidelity["seed"].isin(seeds)
        & fidelity["budget"].isin(budgets)
    ].copy()
    sizes = dataset_sizes(Path(cfg["project"]["processed_dir"]))

    sld_summary, sld_merged = summarize_sld_law(downstream[downstream["budget"].isin(budgets)], fidelity, KEY_METHODS)
    margins = build_margin_distribution(cfg, seeds)
    slopes, collapse = summarize_budget_collapse(fidelity, sizes, KEY_METHODS)
    transfer = summarize_transfer_regret(downstream[downstream["budget"].isin(budgets)], KEY_METHODS)
    runtime = summarize_runtime(downstream[downstream["budget"].isin(budgets)], KEY_METHODS)
    selector = pd.DataFrame()
    if not args.skip_selector:
        selector = summarize_selector_coverage(cfg, selector_budgets, seeds, SELECTOR_METHODS)

    sld_summary.to_csv(out_tables / "icdm_sld_law_summary.csv", index=False)
    sld_merged.to_csv(out_tables / "icdm_sld_law_cells.csv", index=False)
    margins.to_csv(out_tables / "icdm_margin_distribution.csv", index=False)
    if not margins.empty:
        (
            margins.groupby("depth")
            .agg(
                nodes=("margin", "count"),
                median_margin=("margin", "median"),
                mean_margin=("margin", "mean"),
                p10_margin=("margin", lambda x: float(np.quantile(x, 0.10))),
                p90_margin=("margin", lambda x: float(np.quantile(x, 0.90))),
            )
            .reset_index()
            .to_csv(out_tables / "icdm_margin_summary.csv", index=False)
        )
    slopes.to_csv(out_tables / "icdm_budget_collapse_slopes.csv", index=False)
    collapse.to_csv(out_tables / "icdm_budget_collapse_by_size.csv", index=False)
    transfer.to_csv(out_tables / "icdm_transfer_regret.csv", index=False)
    runtime.to_csv(out_tables / "icdm_runtime_summary.csv", index=False)
    if not selector.empty:
        selector.to_csv(out_tables / "icdm_selector_coverage_cells.csv", index=False)
        (
            selector.groupby(["method", "budget"])
            .agg(
                coverage_ratio=("coverage_ratio", "mean"),
                coverage_to_proxy_upper_bound=("coverage_to_proxy_upper_bound", "mean"),
                matched_rows=("matched_rows", "mean"),
                n_cells=("coverage_ratio", "count"),
            )
            .reset_index()
            .assign(method_label=lambda x: x["method"].map(label), budget_n=lambda x: x["budget"].map(budget_int))
            .sort_values(["budget_n", "coverage_ratio"], ascending=[True, False])
            .to_csv(out_tables / "icdm_selector_coverage_summary.csv", index=False)
        )

    make_figures(sld_merged, margins, collapse, transfer, runtime, selector, out_figs)

    sld_md = sld_summary[
        [
            "method_label",
            "n_cells",
            "pooled_spearman_sld_vs_auroc",
            "within_dataset_spearman_sld_vs_auroc",
            "mean_sld_inf_norm",
            "mean_auroc",
            "prop1_cells",
            "prop1_root_agreement",
            "all_root_agreement",
        ]
    ].rename(
        columns={
            "method_label": "Method",
            "n_cells": "Cells",
            "pooled_spearman_sld_vs_auroc": "Pooled Spearman",
            "within_dataset_spearman_sld_vs_auroc": "Within Dataset Spearman",
            "mean_sld_inf_norm": "Mean SLD",
            "mean_auroc": "Mean AUROC",
            "prop1_cells": "Prop1 Cells",
            "prop1_root_agreement": "Prop1 Agree",
            "all_root_agreement": "All Root Agree",
        }
    )
    slope_md = (
        slopes.groupby("method")
        .agg(mean_loglog_slope=("loglog_sld_budget_slope", "mean"), median_loglog_slope=("loglog_sld_budget_slope", "median"))
        .reset_index()
    )
    slope_md["Method"] = slope_md["method"].map(label)
    slope_md = slope_md[["Method", "mean_loglog_slope", "median_loglog_slope"]].rename(
        columns={"mean_loglog_slope": "Mean log-log slope", "median_loglog_slope": "Median log-log slope"}
    )
    transfer_md = transfer.pivot_table(index="method_label", columns="learner", values="transfer_regret", aggfunc="mean").reset_index()
    runtime_md = (
        runtime.groupby("method")
        .agg(median_speedup=("speedup_vs_full_median", "median"), mean_seconds=("downstream_seconds_mean", "mean"))
        .reset_index()
    )
    runtime_md["Method"] = runtime_md["method"].map(label)
    runtime_md = runtime_md[["Method", "median_speedup", "mean_seconds"]].rename(
        columns={"median_speedup": "Median Speedup", "mean_seconds": "Mean Seconds"}
    )
    lines = [
        "# ICDM Theory Diagnostics",
        "",
        "Diagnostics generated from the completed binary grid. These connect the empirical section to the SLD theory: "
        "SLD/AUROC behavior, Prop. 1 margin cells, budget scaling, transfer regret, runtime, and selector coverage.",
        "",
        "## SLD Law",
        "",
        md_table(sld_md),
        "",
        "## Budget Scaling",
        "",
        md_table(slope_md),
        "",
        "## Transfer Regret",
        "",
        md_table(transfer_md),
        "",
        "## Runtime",
        "",
        md_table(runtime_md),
    ]
    selector_summary_path = out_tables / "icdm_selector_coverage_summary.csv"
    if selector_summary_path.exists():
        selector_summary = pd.read_csv(selector_summary_path)
        selector_md = selector_summary[
            ["method_label", "budget", "coverage_ratio", "coverage_to_proxy_upper_bound", "matched_rows", "n_cells"]
        ].rename(
            columns={
                "method_label": "Method",
                "budget": "Budget",
                "coverage_ratio": "Coverage Ratio",
                "coverage_to_proxy_upper_bound": "f(S)/proxy ub",
                "matched_rows": "Matched Rows",
                "n_cells": "Cells",
            }
        )
        lines.extend(["", "## Selector Coverage", "", md_table(selector_md)])
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `tables/icdm_sld_law_summary.csv`",
            "- `tables/icdm_margin_distribution.csv`",
            "- `tables/icdm_margin_summary.csv`",
            "- `tables/icdm_budget_collapse_slopes.csv`",
            "- `tables/icdm_budget_collapse_by_size.csv`",
            "- `tables/icdm_transfer_regret.csv`",
            "- `tables/icdm_runtime_summary.csv`",
            "- `tables/icdm_selector_coverage_summary.csv`",
            "- `figures/icdm_sld_law_scatter.png`",
            "- `figures/icdm_margin_distribution.png`",
            "- `figures/icdm_budget_collapse_sld.png`",
            "- `figures/icdm_transfer_regret_profiles.png`",
            "- `figures/icdm_runtime_speedup.png`",
            "- `figures/icdm_selector_coverage.png`",
            "",
        ]
    )
    (out_notes / "ICDM_THEORY_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote ICDM theory diagnostics to paper_materials.")


if __name__ == "__main__":
    main()
