from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.landscape import compute_gains_for_candidates, load_landscape, load_teacher, split_fidelity, split_landscape_discrepancy
from gaindistill.models import fit_teacher
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits, write_json


def run_one(bundle_dir: Path, distilled_dir: Path, cfg: dict, default_seed: int, worker_threads: int, force: bool = False) -> None:
    set_thread_limits(worker_threads)
    rel = distilled_dir.relative_to(Path(cfg["project"]["results_dir"]) / "distilled")
    out = ensure_dir(Path(cfg["project"]["results_dir"]) / "split_fidelity" / rel)
    if (out / "split_fidelity.json").exists() and not force:
        return
    try:
        seed = int(distilled_dir.name.split("_")[1])
    except (IndexError, ValueError):
        seed = default_seed
    bundle = load_processed(bundle_dir)
    full = load_landscape(Path(cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}")
    arr = np.load(distilled_dir / "distilled.npz")
    weights = arr["weights"] if "weights" in arr.files else None
    teacher_path = Path(cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}" / "teacher.joblib"
    teacher = load_teacher(teacher_path) if teacher_path.exists() else None
    fit = fit_teacher(arr["X"], arr["y"], cfg, worker_threads, sample_weight=weights)
    gains = compute_gains_for_candidates(
        arr["X"],
        arr["y"],
        bundle.meta,
        fit.model,
        full.candidates,
        float(cfg["teacher"]["reg_lambda"]),
        float(cfg["teacher"]["gamma"]),
        sample_weight=weights,
    )
    metrics = split_fidelity(full, gains)
    if teacher is not None:
        metrics.update(
            split_landscape_discrepancy(
                full,
                arr["X"],
                arr["y"],
                bundle.meta,
                teacher,
                float(cfg["teacher"]["reg_lambda"]),
                float(cfg["teacher"]["gamma"]),
                sample_weight=weights,
            )
        )
    metrics.update({"train_seconds": fit.train_seconds, "backend": fit.backend})
    write_json(out / "split_fidelity.json", metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--distilled-seeds", default=None)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    processed = {p.name: p for p in list_dataset_dirs(cfg["project"]["processed_dir"])}
    dist_root = Path(cfg["project"]["results_dir"]) / "distilled"
    dirs = [p for p in dist_root.glob("*/*/budget_*/seed_*") if (p / "distilled.npz").exists()]
    if args.dataset:
        dirs = [p for p in dirs if p.parts[-4] == args.dataset]
    if args.methods:
        methods = {m.strip() for m in args.methods.split(",") if m.strip()}
        dirs = [p for p in dirs if p.parts[-3] in methods]
    if args.distilled_seeds:
        seeds = {s.strip() if s.strip().startswith("seed_") else f"seed_{s.strip()}" for s in args.distilled_seeds.split(",") if s.strip()}
        dirs = [p for p in dirs if p.parts[-1] in seeds]
    if args.budgets:
        budgets = {b.strip() if b.strip().startswith("budget_") else f"budget_{b.strip()}" for b in args.budgets.split(",") if b.strip()}
        dirs = [p for p in dirs if p.parts[-2] in budgets]
    tasks = [(processed[p.parts[-4]], p) for p in dirs if p.parts[-4] in processed]
    worker_threads = max(1, n_jobs // max(1, min(len(tasks), n_jobs)))
    Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(delayed(run_one)(d, p, cfg, args.seed, worker_threads, args.force) for d, p in tasks)
    print(f"Computed split-fidelity metrics for {len(tasks)} distilled dataset(s).")


if __name__ == "__main__":
    main()
