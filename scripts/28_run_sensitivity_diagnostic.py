from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.landscape import compute_landscape_from_arrays, load_teacher, split_landscape_discrepancy
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits


LABELS = {
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_density": "HistDistill-Density",
    "gain_path_refined": "Legacy gain-path refined",
    "random": "Random",
}


def run_one(
    cfg: dict,
    dataset_dir: Path,
    method: str,
    budget: str,
    seed: str,
    reg_lambda: float,
    probe_depth: int,
) -> dict | None:
    bundle = load_processed(dataset_dir)
    results = Path(cfg["project"]["results_dir"])
    teacher_path = results / "teachers" / bundle.name / seed / "teacher.joblib"
    dist_path = results / "distilled" / bundle.name / method / budget / seed / "distilled.npz"
    if not teacher_path.exists() or not dist_path.exists():
        return None
    seed_n = int(seed.replace("seed_", ""))
    teacher = load_teacher(teacher_path)
    full = compute_landscape_from_arrays(
        bundle.X_train,
        bundle.y_train,
        bundle.meta,
        teacher,
        float(reg_lambda),
        float(cfg["teacher"]["gamma"]),
        cfg,
        seed_n,
    )
    if probe_depth < 2:
        full = replace(full, probes=[p for p in full.probes if int(p.depth) <= probe_depth])
    arr = np.load(dist_path)
    weights = arr["weights"] if "weights" in arr.files else None
    metrics = split_landscape_discrepancy(
        full,
        arr["X"],
        arr["y"],
        bundle.meta,
        teacher,
        float(reg_lambda),
        float(cfg["teacher"]["gamma"]),
        sample_weight=weights,
    )
    return {
        "dataset": bundle.name,
        "method": method,
        "method_label": LABELS.get(method, method),
        "budget": budget,
        "seed": seed,
        "reg_lambda": float(reg_lambda),
        "probe_depth": int(probe_depth),
        **metrics,
    }


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


def make_figure(summary: pd.DataFrame, out_figs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    lambda_view = summary[summary["probe_depth"] == 2].copy()
    for method, sub in lambda_view.groupby("method"):
        by_lam = sub.groupby("reg_lambda")["sld_inf_norm"].mean().reset_index().sort_values("reg_lambda")
        axes[0].plot(by_lam["reg_lambda"], by_lam["sld_inf_norm"], marker="o", linewidth=1.7, label=LABELS.get(method, method))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Evaluation lambda")
    axes[0].set_ylabel("Normalized SLD infinity")
    axes[0].set_title("Lambda sensitivity")
    axes[0].grid(alpha=0.2)

    depth_view = summary[summary["reg_lambda"] == 1.0].copy()
    for method, sub in depth_view.groupby("method"):
        by_depth = sub.groupby("probe_depth")["sld_inf_norm"].mean().reset_index().sort_values("probe_depth")
        axes[1].plot(by_depth["probe_depth"], by_depth["sld_inf_norm"], marker="o", linewidth=1.7, label=LABELS.get(method, method))
    axes[1].set_xlabel("Max probe depth")
    axes[1].set_ylabel("Normalized SLD infinity")
    axes[1].set_title("Probe-depth sensitivity")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "icdm_sensitivity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--methods", default="histdistill_refined,histdistill_density,gain_path_refined,random")
    ap.add_argument("--budget", default="budget_50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--reg-lambdas", default="0.1,1.0,10.0")
    ap.add_argument("--probe-depths", default="0,1,2")
    ap.add_argument("--n-jobs", default="6")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs)
    set_thread_limits(n_jobs)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [s if s.startswith("seed_") else f"seed_{s.strip()}" for s in args.seeds.split(",") if s.strip()]
    reg_lambdas = [float(x.strip()) for x in args.reg_lambdas.split(",") if x.strip()]
    probe_depths = [int(x.strip()) for x in args.probe_depths.split(",") if x.strip()]
    datasets = [p for p in list_dataset_dirs(cfg["project"]["processed_dir"]) if p.name != "smoke_binary"]
    tasks = [(d, m, s, lam, depth) for d in datasets for m in methods for s in seeds for lam in reg_lambdas for depth in probe_depths]
    rows = Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(
        delayed(run_one)(cfg, dataset_dir, method, args.budget, seed, reg_lambda, probe_depth)
        for dataset_dir, method, seed, reg_lambda, probe_depth in tasks
    )
    cells = pd.DataFrame([r for r in rows if r is not None])
    cells.to_csv(out_tables / "icdm_sensitivity_cells.csv", index=False)
    summary = (
        cells.groupby(["method", "method_label", "reg_lambda", "probe_depth"])
        .agg(
            sld_inf_norm=("sld_inf_norm", "mean"),
            sld_root_agreement_direct=("sld_root_agreement_direct", "mean"),
            sld_rank_correlation_direct=("sld_rank_correlation_direct", "mean"),
            n_cells=("sld_inf_norm", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_tables / "icdm_sensitivity_summary.csv", index=False)
    make_figure(summary, out_figs)
    compact = summary[(summary["reg_lambda"] == 1.0) | (summary["probe_depth"] == 2)][
        ["method_label", "reg_lambda", "probe_depth", "sld_inf_norm", "sld_root_agreement_direct", "n_cells"]
    ].rename(
        columns={
            "method_label": "Method",
            "reg_lambda": "lambda",
            "probe_depth": "Probe Depth",
            "sld_inf_norm": "SLD",
            "sld_root_agreement_direct": "Root Agree",
            "n_cells": "Cells",
        }
    )
    lines = [
        "# ICDM Sensitivity Diagnostic",
        "",
        "Appendix robustness diagnostic on the current binary pipeline. It varies evaluation lambda and max probe depth at budget 50. Bin-count sensitivity is not included because the current processed artifacts fix bins during preprocessing.",
        "",
        md_table(compact),
        "",
        "## Files",
        "",
        "- `tables/icdm_sensitivity_cells.csv`",
        "- `tables/icdm_sensitivity_summary.csv`",
        "- `figures/icdm_sensitivity.png`",
        "",
    ]
    (out_notes / "ICDM_SENSITIVITY_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote sensitivity diagnostics to paper_materials.")


if __name__ == "__main__":
    main()
