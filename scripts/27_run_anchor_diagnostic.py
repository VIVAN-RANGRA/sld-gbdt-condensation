from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from gaindistill.landscape import load_teacher, raw_margin_at_checkpoint
from gaindistill.models import fit_teacher
from gaindistill.utils import ensure_dir, load_config, resolve_n_jobs, set_thread_limits


LABELS = {
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_greedy": "Pure greedy selector",
    "histdistill_importance": "ImportanceSample",
    "random": "Random",
}


def budget_int(value: str) -> int:
    return int(str(value).replace("budget_", ""))


def run_one(cfg: dict, dataset: str, method: str, budget: str, seed: str, checkpoints: list[int], worker_threads: int) -> list[dict]:
    set_thread_limits(worker_threads)
    results = Path(cfg["project"]["results_dir"])
    dist_path = results / "distilled" / dataset / method / budget / seed / "distilled.npz"
    teacher_path = results / "teachers" / dataset / seed / "teacher.joblib"
    if not dist_path.exists() or not teacher_path.exists():
        return []
    arr = np.load(dist_path)
    X, y = arr["X"], arr["y"]
    weights = arr["weights"] if "weights" in arr.files else None
    teacher = load_teacher(teacher_path)
    student = fit_teacher(X, y, cfg, worker_threads, sample_weight=weights).model
    rows = []
    for checkpoint in checkpoints:
        tea = raw_margin_at_checkpoint(teacher, X, y, checkpoint)
        stu = raw_margin_at_checkpoint(student, X, y, checkpoint)
        diff = np.abs(stu - tea)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": LABELS.get(method, method),
                "budget": budget,
                "seed": seed,
                "checkpoint": int(checkpoint),
                "rho_max": float(np.max(diff)),
                "rho_mean": float(np.mean(diff)),
                "rho_p90": float(np.quantile(diff, 0.90)),
                "rows": int(len(y)),
            }
        )
    return rows


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


def make_figure(cells: pd.DataFrame, out_figs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    summary = cells.groupby(["method", "method_label", "checkpoint"]).agg(rho_max=("rho_max", "mean"), rho_p90=("rho_p90", "mean")).reset_index()
    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("checkpoint")
        ax.plot(sub["checkpoint"], sub["rho_max"], marker="o", linewidth=1.7, label=sub["method_label"].iloc[0])
    ax.set_xlabel("Boosting checkpoint")
    ax.set_ylabel("Mean max margin drift rho_t")
    ax.set_title("Teacher-student anchoring drift")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_figs / "icdm_anchor_drift.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--methods", default="histdistill_refined,histdistill_greedy,histdistill_importance,random")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--checkpoints", default="1,5,10,25,50")
    ap.add_argument("--n-jobs", default="6")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs)
    set_thread_limits(n_jobs)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [b if b.startswith("budget_") else f"budget_{b.strip()}" for b in args.budgets.split(",") if b.strip()]
    seeds = [s if s.startswith("seed_") else f"seed_{s.strip()}" for s in args.seeds.split(",") if s.strip()]
    checkpoints = [int(c.strip()) for c in args.checkpoints.split(",") if c.strip()]
    dist_root = Path(cfg["project"]["results_dir"]) / "distilled"
    datasets = sorted([p.name for p in dist_root.iterdir() if p.is_dir() and p.name != "smoke_binary"])
    tasks = [(d, m, b, s) for d in datasets for m in methods for b in budgets for s in seeds]
    worker_threads = max(1, n_jobs // max(1, min(n_jobs, len(tasks))))
    nested = Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(
        delayed(run_one)(cfg, dataset, method, budget, seed, checkpoints, worker_threads)
        for dataset, method, budget, seed in tasks
    )
    rows = [row for group in nested for row in group]
    cells = pd.DataFrame(rows)
    cells.to_csv(out_tables / "icdm_anchor_drift_cells.csv", index=False)
    summary = (
        cells.groupby(["method", "method_label", "budget", "checkpoint"])
        .agg(rho_max=("rho_max", "mean"), rho_mean=("rho_mean", "mean"), rho_p90=("rho_p90", "mean"), n_cells=("rho_max", "count"))
        .reset_index()
        .assign(budget_n=lambda x: x["budget"].map(budget_int))
        .sort_values(["budget_n", "checkpoint", "rho_max"])
    )
    summary.to_csv(out_tables / "icdm_anchor_drift_summary.csv", index=False)
    make_figure(cells, out_figs)
    final_checkpoint = max(checkpoints)
    md = summary[summary["checkpoint"] == final_checkpoint][["method_label", "budget", "rho_max", "rho_mean", "rho_p90", "n_cells"]].rename(
        columns={
            "method_label": "Method",
            "budget": "Budget",
            "rho_max": "rho max",
            "rho_mean": "rho mean",
            "rho_p90": "rho p90",
            "n_cells": "Cells",
        }
    )
    lines = [
        "# ICDM Anchoring Drift Diagnostic",
        "",
        "Teacher-student raw-margin drift on the condensed support. This is a measurable proxy for Assumption A: lower rho_t means the student trajectory stays closer to the teacher trajectory.",
        "",
        f"Final checkpoint shown below: {final_checkpoint}.",
        "",
        md_table(md),
        "",
        "## Files",
        "",
        "- `tables/icdm_anchor_drift_cells.csv`",
        "- `tables/icdm_anchor_drift_summary.csv`",
        "- `figures/icdm_anchor_drift.png`",
        "",
    ]
    (out_notes / "ICDM_ANCHORING_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote anchoring diagnostics to paper_materials.")


if __name__ == "__main__":
    main()
