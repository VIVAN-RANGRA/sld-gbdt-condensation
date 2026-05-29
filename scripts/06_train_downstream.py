from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.models import evaluate_binary, fit_downstream
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits, write_json


def run_one(bundle_dir: Path, distilled_dir: Path, cfg: dict, learner: str, seed: int, worker_threads: int, force: bool = False) -> None:
    set_thread_limits(worker_threads)
    rel = distilled_dir.relative_to(Path(cfg["project"]["results_dir"]) / "distilled")
    out = ensure_dir(Path(cfg["project"]["results_dir"]) / "downstream" / rel / learner)
    if (out / "metrics.json").exists() and not force:
        return
    bundle = load_processed(bundle_dir)
    arr = np.load(distilled_dir / "distilled.npz")
    X, y = arr["X"], arr["y"]
    weights = arr["weights"] if "weights" in arr.files else None
    fit = fit_downstream(learner, X, y, cfg, worker_threads, seed, sample_weight=weights)
    metrics = evaluate_binary(fit.model, bundle.X_test, bundle.y_test)
    metrics.update({"train_seconds": fit.train_seconds, "backend": fit.backend, "train_rows": int(len(y))})
    write_json(out / "metrics.json", metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--learners", default=None)
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
    learners = [x.strip() for x in (args.learners or ",".join(cfg["downstream"]["learners"])).split(",") if x.strip()]
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
    tasks = [(processed[p.parts[-4]], p, l) for p in dirs for l in learners if p.parts[-4] in processed]
    worker_threads = max(1, n_jobs // max(1, min(len(tasks), n_jobs)))
    Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(delayed(run_one)(d, p, cfg, l, args.seed, worker_threads, args.force) for d, p, l in tasks)
    print(f"Evaluated {len(tasks)} downstream learner run(s).")


if __name__ == "__main__":
    main()
