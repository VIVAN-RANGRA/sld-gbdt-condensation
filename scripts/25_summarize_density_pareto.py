from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from gaindistill.data import load_processed
from gaindistill.utils import ensure_dir, load_config


DENSITY_METHODS = {
    "histdistill_refined": 0.0,
    "histdistill_density_weak": 0.15,
    "histdistill_density_mid": 0.35,
    "histdistill_density_strong": 0.70,
    "histdistill_density_xstrong": 1.25,
}
LABELS = {
    "histdistill_refined": "beta=0.00",
    "histdistill_density_weak": "beta=0.15",
    "histdistill_density_mid": "beta=0.35",
    "histdistill_density_strong": "beta=0.70",
    "histdistill_density_xstrong": "beta=1.25",
}


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


def rbf_mmd2(X: np.ndarray, Y: np.ndarray, seed: int, max_full: int = 2000, y_weight: np.ndarray | None = None) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) > max_full:
        X = X[rng.choice(len(X), size=max_full, replace=False)]
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-6)
    Xs = (X - mean) / std
    Ys = (Y - mean) / std
    probe = np.vstack([Xs[: min(len(Xs), 500)], Ys[: min(len(Ys), 500)]])
    if len(probe) < 2:
        return float("nan")
    wy = np.ones(len(Ys), dtype=np.float64) if y_weight is None else np.asarray(y_weight, dtype=np.float64)
    wy = np.maximum(wy, 1e-12)
    wy = wy / max(float(wy.sum()), 1e-12)
    wx = np.full(len(Xs), 1.0 / max(1, len(Xs)), dtype=np.float64)
    d = pairwise_distances(probe, metric="sqeuclidean")
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, 1e-6)
    Kxx_mat = np.exp(-gamma * pairwise_distances(Xs, Xs, metric="sqeuclidean"))
    Kyy_mat = np.exp(-gamma * pairwise_distances(Ys, Ys, metric="sqeuclidean"))
    Kxy_mat = np.exp(-gamma * pairwise_distances(Xs, Ys, metric="sqeuclidean"))
    Kxx = float(wx @ Kxx_mat @ wx)
    Kyy = float(wy @ Kyy_mat @ wy)
    Kxy = float(wx @ Kxy_mat @ wy)
    return float(Kxx + Kyy - 2.0 * Kxy)


def compute_mmd_cells(cfg: dict, methods: list[str], budgets: set[str], seeds: set[str]) -> pd.DataFrame:
    rows = []
    results = Path(cfg["project"]["results_dir"])
    processed = Path(cfg["project"]["processed_dir"])
    for dataset_dir in processed.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "smoke_binary":
            continue
        bundle = load_processed(dataset_dir)
        for method in methods:
            for budget in budgets:
                for seed in seeds:
                    p = results / "distilled" / dataset_dir.name / method / budget / seed / "distilled.npz"
                    if not p.exists():
                        continue
                    arr = np.load(p)
                    seed_n = int(seed.replace("seed_", "")) + 1000 * budget_int(budget)
                    weights = arr["weights"] if "weights" in arr.files else None
                    rows.append(
                        {
                            "dataset": dataset_dir.name,
                            "method": method,
                            "method_label": LABELS.get(method, method),
                            "density_beta": DENSITY_METHODS[method],
                            "budget": budget,
                            "seed": seed,
                            "mmd2": rbf_mmd2(bundle.X_train, arr["X"], seed_n, y_weight=weights),
                            "rows": int(len(arr["y"])),
                        }
                    )
    return pd.DataFrame(rows)


def make_figures(cells: pd.DataFrame, out_figs: Path) -> None:
    if cells.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    for method, sub in cells.groupby("method"):
        ax.scatter(sub["mmd2"], sub["sld_inf_norm"], s=28, alpha=0.55, label=LABELS.get(method, method))
    ax.set_xlabel("RBF MMD^2")
    ax.set_ylabel("Normalized SLD infinity")
    ax.set_title("Density-SLD Pareto cells")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    summary = cells.groupby(["density_beta", "method_label"]).agg(xgb_auroc=("xgboost_auroc", "mean"), mlp_auroc=("mlp_auroc", "mean")).reset_index()
    summary = summary.sort_values("density_beta")
    ax.plot(summary["density_beta"], summary["xgb_auroc"], marker="o", linewidth=1.8, label="XGBoost")
    ax.plot(summary["density_beta"], summary["mlp_auroc"], marker="o", linewidth=1.8, label="MLP")
    ax.set_xlabel("Density beta")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Learner response to density weight")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "icdm_density_pareto.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")
    budgets = {b if b.startswith("budget_") else f"budget_{b}" for b in args.budgets.split(",") if b.strip()}
    seeds = {s if s.startswith("seed_") else f"seed_{s}" for s in args.seeds.split(",") if s.strip()}
    methods = list(DENSITY_METHODS)

    mmd = compute_mmd_cells(cfg, methods, budgets, seeds)
    downstream = pd.read_csv(Path(args.paper_tables) / "all_downstream_metrics.csv")
    perf = (
        downstream[
            (downstream["dataset"] != "smoke_binary")
            & downstream["method"].isin(methods)
            & downstream["budget"].isin(budgets)
            & downstream["seed"].isin(seeds)
            & downstream["learner"].isin(["xgboost", "mlp"])
        ]
        .pivot_table(index=["dataset", "method", "budget", "seed"], columns="learner", values="auroc", aggfunc="mean")
        .reset_index()
        .rename(columns={"xgboost": "xgboost_auroc", "mlp": "mlp_auroc"})
    )
    fidelity = pd.read_csv(Path(args.paper_tables) / "all_split_fidelity_metrics.csv")
    fid = fidelity[
        (fidelity["dataset"] != "smoke_binary")
        & fidelity["method"].isin(methods)
        & fidelity["budget"].isin(budgets)
        & fidelity["seed"].isin(seeds)
    ][["dataset", "method", "budget", "seed", "sld_inf_norm", "sld_rank_correlation_direct"]]
    cells = mmd.merge(perf, on=["dataset", "method", "budget", "seed"], how="left").merge(
        fid, on=["dataset", "method", "budget", "seed"], how="left"
    )
    cells.to_csv(out_tables / "icdm_density_pareto_cells.csv", index=False)
    summary = (
        cells.groupby(["method", "method_label", "density_beta"])
        .agg(
            mmd2=("mmd2", "mean"),
            sld_inf_norm=("sld_inf_norm", "mean"),
            xgboost_auroc=("xgboost_auroc", "mean"),
            mlp_auroc=("mlp_auroc", "mean"),
            n_cells=("mmd2", "count"),
        )
        .reset_index()
        .sort_values("density_beta")
    )
    summary.to_csv(out_tables / "icdm_density_pareto_summary.csv", index=False)
    make_figures(cells.dropna(subset=["mmd2", "sld_inf_norm"]), out_figs)
    lines = [
        "# ICDM Density Pareto Diagnostic",
        "",
        "Density sweep for Eq. (6), using beta variants of HistDistill on budgets 25/50 and seeds 0/1/2. "
        "The table reports approximate RBF MMD^2, direct SLD, and XGBoost/MLP AUROC.",
        "",
        md_table(
            summary[
                [
                    "method_label",
                    "density_beta",
                    "mmd2",
                    "sld_inf_norm",
                    "xgboost_auroc",
                    "mlp_auroc",
                    "n_cells",
                ]
            ].rename(
                columns={
                    "method_label": "Method",
                    "density_beta": "Beta",
                    "mmd2": "MMD^2",
                    "sld_inf_norm": "SLD",
                    "xgboost_auroc": "XGB AUROC",
                    "mlp_auroc": "MLP AUROC",
                    "n_cells": "Cells",
                }
            )
        ),
        "",
        "## Files",
        "",
        "- `tables/icdm_density_pareto_cells.csv`",
        "- `tables/icdm_density_pareto_summary.csv`",
        "- `figures/icdm_density_pareto.png`",
        "",
    ]
    (out_notes / "ICDM_DENSITY_PARETO.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote density Pareto diagnostics to paper_materials.")


if __name__ == "__main__":
    main()
