from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from gaindistill.landscape import load_teacher
from gaindistill.data import load_processed
from gaindistill.distill import distill_multi_probe, distill_root, save_distilled
from gaindistill.landscape import load_landscape
from gaindistill.utils import list_dataset_dirs, load_config, resolve_n_jobs, set_seed, set_thread_limits


def run_one(ds_dir: Path, cfg: dict, seed: int, budget: int, worker_threads: int, method_name: str, force: bool) -> None:
    set_thread_limits(worker_threads)
    set_seed(seed)
    bundle = load_processed(ds_dir)
    out = Path(cfg["project"]["results_dir"]) / "distilled" / bundle.name / method_name / f"budget_{budget}" / f"seed_{seed}"
    if (out / "distilled.npz").exists() and not force:
        return
    landscape = load_landscape(Path(cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}")
    teacher_path = Path(cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}" / "teacher.joblib"
    teacher = load_teacher(teacher_path) if teacher_path.exists() else None
    if landscape.probes and teacher is not None and method_name != "gaindistill_root":
        distilled = distill_multi_probe(bundle.X_train, bundle.y_train, bundle.meta, landscape, cfg, seed, budget, teacher=teacher)
    else:
        distilled = distill_root(bundle.X_train, bundle.y_train, bundle.meta, landscape, cfg, seed, budget)
    save_distilled(out, distilled, bundle.meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget-per-class", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--method-name", default="gaindistill_full")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["distill"]["steps"] = args.steps
    budget = int(args.budget_per_class or cfg["distill"]["budget_per_class"])
    cfg["project"]["seed"] = args.seed
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    dirs = list_dataset_dirs(cfg["project"]["processed_dir"])
    if args.dataset:
        dirs = [Path(cfg["project"]["processed_dir"]) / args.dataset]
    worker_threads = max(1, n_jobs // max(1, len(dirs)))
    Parallel(n_jobs=min(n_jobs, len(dirs) or 1))(delayed(run_one)(d, cfg, args.seed, budget, worker_threads, args.method_name, args.force) for d in dirs)
    print(f"Distilled {len(dirs)} dataset(s) as {args.method_name} at budget {budget}/class using up to {n_jobs} cores.")


if __name__ == "__main__":
    main()
